// champ_session [PV_MODEL.onnx]: a stateless battle sandbox for the Champ viewer (gui/web/champ/).
//
// One JSON request per stdin line, one JSON response per stdout line. Every request carries the whole sandbox:
//   {"start": <combat_v4 start>, "ops": [op...], "query": ...}
// The battle is rebuilt from scratch for each request: BattleContext::init(start_game(start)), then every op in
// order. Undo is the caller dropping the last op. Ops:
//   {"act": BITS}                                 execute a legal sts::search::Action
//   {"set": {"player.hp": 50, "champ.strength": 4, "champ.intent": "THE_CHAMP_HEAVY_SLASH", ...}}
//   {"move": {"from": PILE, "index": I, "to": PILE}}      PILE: hand | draw | discard | exhaust
//   {"add": {"id": CARD_ID, "upgraded": BOOL, "to": PILE}}
//   {"remove": {"from": PILE, "index": I}}
// A card is addressed by "index" or by "match": [CARD_ID, UPGRADED] (the first such card; for the draw pile).
// Edits are only allowed at a normal play decision (PLAYER_NORMAL). New / moved draw-pile cards go to a
// deterministic pseudo-random position: the order is hidden from the player and every search resamples it.
// Queries:
//   view                 public state (what a player sees; never draw order or RNG) + legal moves
//   pv                   view + the PV network's value (100 × P(win)) and per-move policy priors (needs PV_MODEL)
//   (search / playout are no longer used by gui/champ_server.py, which only asks view / pv)
//   search SIMS [SALT]   view + one teacher search (guided rollout, 8 particles): visits / mean value per move
//   playout SIMS FROM N  win/HP of teacher fights from public-belief particles FROM..FROM+N-1 of this state
//                        (the hidden draw order / RNG resampled per playout; search salt = particle index + 1)
#include "agents/combat/pv/search.hpp"
#include "agents/combat/teacher/teacher_search.hpp"
#include "environments/combat/environment.hpp"
#include "environments/combat/record_v4.hpp"
#include "environments/combat/view.hpp"
#include "constants/Cards.h"
#include "constants/MonsterMoves.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/PlayerStatusEffects.h"
#include "constants/Relics.h"
#include "sim/search/Action.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

using Json = nlohmann::json;
namespace S = sts;
using sts::search::Action;

namespace {

S::Monster& champ(S::BattleContext& bc) {
    for (int i = 0; i < bc.monsters.monsterCount; ++i)
        if (bc.monsters.arr[i].id == S::MonsterId::THE_CHAMP) return bc.monsters.arr[i];
    throw std::runtime_error{"no Champ in this battle"};
}

S::MMID move_by_name(const std::string& name) {
    for (int i = 0; i <= static_cast<int>(S::MMID::WRITHING_MASS_STRONG_STRIKE); ++i)
        if (S::monsterMoveStrings[i] == name) return static_cast<S::MMID>(i);
    throw std::runtime_error{"unknown monster move " + name};
}

// ---------------- edits ----------------

void set_player_status(S::Player& p, const std::string& name, int v) {
    if (name == "strength") { p.strength = v; return; }
    if (name == "dexterity") { p.dexterity = v; return; }
    if (name == "artifact") { p.artifact = v; return; }
    auto set = [&]<::PlayerStatus s>() {
        if (v == 0) { p.statusMap.erase(s); p.setHasStatus<s>(false); }
        else { p.setStatusValueNoChecks<s>(v); p.setHasStatus<s>(true); }
    };
    if (name == "vulnerable") set.operator()<::PlayerStatus::VULNERABLE>();
    else if (name == "weak") set.operator()<::PlayerStatus::WEAK>();
    else if (name == "frail") set.operator()<::PlayerStatus::FRAIL>();
    else if (name == "metallicize") set.operator()<::PlayerStatus::METALLICIZE>();
    else if (name == "demon_form") set.operator()<::PlayerStatus::DEMON_FORM>();
    else if (name == "plated_armor") set.operator()<::PlayerStatus::PLATED_ARMOR>();
    else throw std::runtime_error{"cannot set player status " + name};
}

void set_champ_status(S::Monster& m, const std::string& name, int v) {
    if (name == "strength") { m.strength = v; return; }
    auto set = [&]<S::MonsterStatus s>() { m.setStatus<s>(v); m.setHasStatus<s>(v != 0); };
    if (name == "vulnerable") set.operator()<S::MonsterStatus::VULNERABLE>();
    else if (name == "weak") set.operator()<S::MonsterStatus::WEAK>();
    else if (name == "metallicize") set.operator()<S::MonsterStatus::METALLICIZE>();
    else if (name == "artifact") set.operator()<S::MonsterStatus::ARTIFACT>();
    else throw std::runtime_error{"cannot set champ status " + name};
}

void apply_set(S::BattleContext& bc, const std::string& key, const Json& value) {
    auto& p = bc.player;
    auto& m = champ(bc);
    if (key == "champ.intent" || key == "champ.last_move") {
        m.moveHistory[key == "champ.intent" ? 0 : 1] = move_by_name(value.get<std::string>());
        return;
    }
    const int v = value.get<int>();
    if (key == "player.hp") p.curHp = std::clamp(v, 1, p.maxHp);
    else if (key == "player.max_hp") { p.maxHp = std::max(1, v); p.curHp = std::min(p.curHp, p.maxHp); }
    else if (key == "player.block") p.block = std::max(0, v);
    else if (key == "player.energy") p.energy = std::max(0, v);
    else if (key.starts_with("player.")) set_player_status(p, key.substr(7), v);
    else if (key == "champ.hp") m.curHp = std::clamp(v, 1, m.maxHp);
    else if (key == "champ.max_hp") { m.maxHp = std::max(1, v); m.curHp = std::min(m.curHp, m.maxHp); }
    else if (key == "champ.block") m.block = std::max(0, v);
    else if (key == "champ.phase2") m.miscInfo = (m.miscInfo & 0x3) | (v ? 0x4 : 0);
    else if (key == "champ.stances_used") m.miscInfo = (m.miscInfo & 0x4) | std::clamp(v, 0, 2);
    else if (key.starts_with("champ.")) set_champ_status(m, key.substr(6), v);
    else if (key == "turn") bc.turn = std::max(0, v);
    else throw std::runtime_error{"cannot set " + key};
}

std::size_t pile_size(const S::CardManager& c, const std::string& pile) {
    if (pile == "hand") return c.cardsInHand;
    if (pile == "draw") return c.drawPile.size();
    if (pile == "discard") return c.discardPile.size();
    if (pile == "exhaust") return c.exhaustPile.size();
    throw std::runtime_error{"unknown pile " + pile};
}

S::CardInstance take(S::CardManager& c, const std::string& pile, int i) {
    if (i < 0 || static_cast<std::size_t>(i) >= pile_size(c, pile)) throw std::runtime_error{"no card at " + pile};
    S::CardInstance card;
    if (pile == "hand") { card = c.hand[i]; c.removeFromHandAtIdx(i); }
    else if (pile == "draw") { card = c.drawPile[i]; c.removeFromDrawPileAtIdx(i); }
    else if (pile == "discard") { card = c.discardPile[i]; c.removeFromDiscard(i); }
    else { card = c.exhaustPile[i]; c.removeFromExhaustPile(i); c.notifyAddCardToCombat(card); }
    return card;
}

void put(S::BattleContext& bc, const std::string& pile, const S::CardInstance& card, std::size_t salt) {
    auto& c = bc.cards;
    if (pile == "hand") {
        if (c.cardsInHand >= S::CardManager::MAX_HAND_SIZE) throw std::runtime_error{"hand is full"};
        c.moveToHand(card);
    } else if (pile == "draw") {
        c.insertToDrawPile(static_cast<int>((salt * 2654435761u) % (c.drawPile.size() + 1)), card);
    } else if (pile == "discard") c.moveToDiscardPile(card);
    else if (pile == "exhaust") c.moveToExhaustPile(card);
    else throw std::runtime_error{"unknown pile " + pile};
}

// A card op's index: "index", or "match": [card id, upgraded] (first such card; the draw pile's order is hidden).
int card_index(const S::CardManager& c, const Json& at) {
    if (!at.contains("match")) return at.at("index").get<int>();
    const auto id = static_cast<S::CardId>(at.at("match")[0].get<int>());
    const bool up = at.at("match")[1].get<int>() != 0;
    const std::string pile = at.at("from");
    for (std::size_t i = 0; i < pile_size(c, pile); ++i) {
        const auto& card = pile == "hand" ? c.hand[i] : pile == "draw" ? c.drawPile[i]
                         : pile == "discard" ? c.discardPile[i] : c.exhaustPile[i];
        if (card.id == id && card.isUpgraded() == up) return static_cast<int>(i);
    }
    throw std::runtime_error{"no such card in " + pile};
}

void apply(S::BattleContext& bc, const Json& op, std::size_t index) {
    if (op.contains("act")) {
        const Action action{op.at("act").get<std::uint32_t>()};
        if (bc.outcome != S::Outcome::UNDECIDED || !action.isValidAction(bc)) throw std::runtime_error{"illegal action"};
        action.execute(bc);
        return;
    }
    if (bc.outcome != S::Outcome::UNDECIDED || bc.inputState != S::InputState::PLAYER_NORMAL)
        throw std::runtime_error{"edits are only allowed at a normal play decision"};
    if (op.contains("set")) {
        for (const auto& [key, value] : op.at("set").items()) apply_set(bc, key, value);
    } else if (op.contains("move")) {
        const auto& m = op.at("move");
        const auto card = take(bc.cards, m.at("from"), card_index(bc.cards, m));
        put(bc, m.at("to"), card, index);
    } else if (op.contains("add")) {
        const auto& a = op.at("add");
        S::CardInstance card{static_cast<S::CardId>(a.at("id").get<int>()), a.value("upgraded", false)};
        card.uniqueId = static_cast<std::int16_t>(bc.cards.nextUniqueCardId++);
        bc.cards.notifyAddCardToCombat(card);
        put(bc, a.at("to"), card, index);
    } else if (op.contains("remove")) {
        const auto& r = op.at("remove");
        const auto card = take(bc.cards, r.at("from"), card_index(bc.cards, r));
        bc.cards.notifyRemoveFromCombat(card);
    } else throw std::runtime_error{"unknown op " + op.dump()};
}

S::BattleContext build(const Json& request) {
    S::BattleContext bc;
    bc.init(stsrl::combat_v4::start_game(request.at("start")));
    std::size_t i = 0;
    for (const auto& op : request.value("ops", Json::array())) apply(bc, op, ++i);
    return bc;
}

// ---------------- public view ----------------

// The shared view plus the Champ's phase and stances, under "champ" for the Champ page.
Json champ_view(const S::BattleContext& bc) {
    auto v = stsrl::combat::view(bc);
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        const auto& m = bc.monsters.arr[i];
        if (m.id != S::MonsterId::THE_CHAMP) continue;
        v["champ"] = v["monsters"][i];
        v["champ"]["phase2"] = (m.miscInfo >> 2) & 1;
        v["champ"]["stances_used"] = m.miscInfo & 0x3;
    }
    return v;
}

// ---------------- agents ----------------

double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

Json pv_query(const S::BattleContext& bc, stsrl::pv::Evaluator* evaluator) {
    if (!evaluator) throw std::runtime_error{"no PV model loaded"};
    if (bc.outcome != S::Outcome::UNDECIDED) throw std::runtime_error{"the fight is over"};
    const auto moves = stsrl::pv::legal_actions(bc);
    const auto prediction = stsrl::pv::evaluate_root(*evaluator, bc);
    const auto priors = stsrl::pv::policy_priors(prediction, moves.size());
    Json out{{"value", prediction.value / 100.0}, {"priors", Json::object()}};
    for (std::size_t i = 0; i < moves.size(); ++i) out["priors"][std::to_string(moves[i].bits)] = priors[i];
    return out;
}

Json search_query(const S::BattleContext& bc, std::int64_t sims, std::uint64_t salt) {
    if (bc.outcome != S::Outcome::UNDECIDED) throw std::runtime_error{"the fight is over"};
    stsrl::CombatEnvironment env{bc};
    const auto n = env.legal_action_count();
    const auto searcher = stsrl::teacher::leaf_search({"guided_rollout", 0, 0}, nullptr, {sims, stsrl::teacher::particles});
    const auto t = std::chrono::steady_clock::now();
    stsrl::teacher::tweaks().search_salt = salt;  // 0: the bench teacher's own seeds; others: independent reruns
    const auto d = stsrl::teacher::search_decision(env, n, searcher, false, stsrl::teacher::particles);
    stsrl::teacher::tweaks().search_salt = 0;
    Json out{{"simulations", d.used}, {"root_value", d.value}, {"chosen", env.action_bits(d.chosen)},
             {"seconds", since(t)}, {"moves", Json::object()}};
    for (const auto& c : d.tried)
        out["moves"][std::to_string(env.action_bits(c.index))] = {{"visits", c.visits}, {"value", c.value}};
    return out;
}

Json playout_query(const S::BattleContext& bc, std::int64_t sims, int from, int n) {
    if (bc.outcome != S::Outcome::UNDECIDED) throw std::runtime_error{"the fight is over"};
    const auto particles = stsrl::teacher::public_particles(bc, from + n);
    const auto searcher = stsrl::teacher::leaf_search({"guided_rollout", 0, 0}, nullptr, {sims, stsrl::teacher::particles});
    Json out = Json::array();
    for (int i = from; i < from + n; ++i) {
        stsrl::teacher::tweaks().search_salt = static_cast<std::uint64_t>(i) + 1;
        const auto t = std::chrono::steady_clock::now();
        const auto end = stsrl::teacher::play_fight(particles[i], searcher, false, stsrl::teacher::particles, false);
        out.push_back({{"particle", i}, {"won", end.outcome == S::Outcome::PLAYER_VICTORY}, {"hp", end.player.curHp},
                       {"turns", end.turn}, {"seconds", since(t)}});
    }
    stsrl::teacher::tweaks().search_salt = 0;
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::unique_ptr<stsrl::pv::Evaluator> evaluator;
    if (argc > 1) evaluator = std::make_unique<stsrl::pv::Evaluator>(argv[1]);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        Json out;
        try {
            const auto request = Json::parse(line);
            const auto bc = build(request);
            const auto query = request.value("query", std::string{"view"});
            out["view"] = champ_view(bc);
            Json relics = Json::array();  // the battle keeps only relic bits; names come from the start
            for (const auto& r : request.at("start").at("relics")) relics.push_back(S::relicNames[r.at("id").get<int>()]);
            out["view"]["relics"] = relics;
            try {  // a failed query (e.g. on a finished fight) still returns the view
                if (query == "pv") out["pv"] = pv_query(bc, evaluator.get());
                else if (query == "search") out["search"] = search_query(bc, request.value("sims", 20000), request.value("salt", std::uint64_t{0}));
                else if (query == "playout")
                    out["playouts"] = playout_query(bc, request.value("sims", 20000), request.value("from", 0), request.value("n", 1));
                else if (query != "view") throw std::runtime_error{"unknown query " + query};
            } catch (const std::exception& e) {
                out["query_error"] = e.what();
            }
        } catch (const std::exception& e) {
            out = {{"error", e.what()}};
        }
        std::cout << out.dump() << std::endl;
    }
    return 0;
}
