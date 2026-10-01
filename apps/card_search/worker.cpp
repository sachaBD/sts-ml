// Card-pick search worker (slop_docs/card-policy-loop/README.md, step A). Long-running; JSON lines on stdin/stdout.
//
// Python -> worker, one line per job:
//   {"seed": S, "decision": j, "rollouts": N, "root_seed": R, "rollout_seed": Q}
//   Plays Ironclad A20 act 1 from seed S (SimpleAgent out of combat, fights and card picks by the model) up to the
//   j-th card-reward decision (0-based) = the ROOT. Then for every option (each offered card, then skip) runs N
//   rollouts to the end of act 1. Rollout r uses the same game seed hash(Q, r) (future card offers, events,
//   encounters via macro_sim::crn_reseed) and the same fight random numbers for every option: common random numbers.
// worker -> Python, while a job runs: model requests, one line per batch, answered by one line:
//   {"type": "batch", "fights": [{"k", "encounter", "state", "member", "u": [u1, u2]}],
//                     "picks":  [{"k", "state", "boss", "options": [card...]}]}
//   <- {"fights": [{"k", "won", "hp"}], "picks": [{"k", "choice"}]}     choice = option index, len(options) = skip
// worker -> Python, when the job ends:
//   {"type": "result", "seed", "decision", "status": "ok" | "no_decision", "root": state, "options": [card...],
//    "rollouts": [[{"won", "hp", "fights", "boss_hp"}, ...] per option],
//    "root_run": the root sim's own play (decision -1 / simple_picks: a whole run, e.g. for sim-vs-real checks)}
#include "apps/common/game_state.hpp"
#include "apps/common/macro_sim.hpp"
#include "constants/MonsterEncounters.h"
#include "sim/search/GameAction.h"
#include "sim/search/SimpleAgent.h"

#include <cctype>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {
namespace ms = stsrl::macro_sim;
using Json = nlohmann::json;
using sts::GameContext;

std::uint64_t mix(std::uint64_t x) {  // splitmix64
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
double unit(std::uint64_t x) { return static_cast<double>(mix(x) >> 11) * 0x1.0p-53; }

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string encounter_name(sts::MonsterEncounter e) { return lower(sts::monsterEncounterEnumNames[static_cast<int>(e)]); }

Json card_json(const sts::Card& c) {
    return {{"card_id", static_cast<int>(c.id)}, {"upgraded", c.getUpgraded()}, {"misc", c.misc},
            {"name", lower(sts::cardEnumStrings[static_cast<int>(c.id)])}};
}

// A card-reward decision is pending: SimpleAgent would take potions / relics / keys first, so wait for those.
bool card_decision(const GameContext& gc) {
    if (gc.screenState != sts::ScreenState::REWARDS) return false;
    const auto& r = gc.info.rewardsContainer;
    if (r.potionCount > 0 && gc.potionCount < gc.potionCapacity) return false;
    return r.relicCount == 0 && !r.sapphireKey && !r.emeraldKey && r.cardRewardCount > 0;
}

const auto& last_reward(const GameContext& gc) {
    const auto& r = gc.info.rewardsContainer;
    return r.cardRewards[r.cardRewardCount - 1];
}

enum class Need { Fight, Pick, Root, Done };

struct Sim {
    GameContext gc;
    sts::search::SimpleAgent agent;
    std::uint64_t fight_seed = 0;
    int member = 0, option = -1, rollout = -1, fights = 0, decisions = 0, target = -1;
    bool won = false, simple_picks = false;
    int boss_hp = -1;

    // Advance until the model is needed or the act is over.
    Need advance(const ms::Options& opt) {
        while (true) {
            if (gc.outcome != sts::GameOutcome::UNDECIDED || gc.act > 1) return Need::Done;
            ms::prepare(gc, opt);
            if (gc.screenState == sts::ScreenState::BATTLE) return Need::Fight;
            if (card_decision(gc) && !simple_picks) return decisions == target ? Need::Root : Need::Pick;
            agent.curGameContext = &gc;
            agent.stepOutOfCombat(gc);
        }
    }
    void fight(bool w, int hp) {
        const bool boss = gc.curRoom == sts::Room::BOSS;
        if (boss) boss_hp = gc.curHp;
        ms::BattleResult r;
        r.won = w;
        r.hp = hp;
        ms::apply_battle_result(gc, r);
        ++fights;
        if (boss && w && gc.outcome == sts::GameOutcome::UNDECIDED) won = true;
        if (boss && w) gc.outcome = sts::GameOutcome::PLAYER_VICTORY;  // stop at the end of act 1
    }
    void pick(int choice) {
        const auto& r = gc.info.rewardsContainer;
        const int idx = r.cardRewardCount - 1;
        if (choice < static_cast<int>(last_reward(gc).size()))
            sts::search::GameAction(sts::search::GameAction::RewardsActionType::CARD, idx, choice).execute(gc);
        else
            sts::search::GameAction(sts::search::GameAction::RewardsActionType::SKIP).execute(gc);
        ++decisions;
    }
};

Json fight_request(const Sim& s, int k) {
    const auto base = s.fight_seed * 1000003ULL + static_cast<std::uint64_t>(s.fights) * 2;
    return {{"k", k}, {"encounter", encounter_name(s.gc.info.encounter)}, {"state", stsrl::game_state::state(s.gc)},
            {"member", s.member}, {"u", {unit(base), unit(base + 1)}}};
}

Json pick_request(const Sim& s, int k) {
    Json options = Json::array();
    for (const auto& c : last_reward(s.gc)) options.push_back(card_json(c));
    return {{"k", k}, {"state", stsrl::game_state::state(s.gc)}, {"boss", encounter_name(s.gc.boss)}, {"options", options}};
}

// Run all sims to completion (or to the root), answering model requests in batches over stdio.
void drive(std::vector<Sim>& sims, const ms::Options& opt) {
    std::vector<Need> need(sims.size());
    for (std::size_t i = 0; i < sims.size(); ++i) need[i] = sims[i].advance(opt);
    while (true) {
        Json fights = Json::array(), picks = Json::array();
        for (std::size_t i = 0; i < sims.size(); ++i) {
            if (need[i] == Need::Fight) fights.push_back(fight_request(sims[i], static_cast<int>(i)));
            if (need[i] == Need::Pick) picks.push_back(pick_request(sims[i], static_cast<int>(i)));
        }
        if (fights.empty() && picks.empty()) return;
        std::cout << Json{{"type", "batch"}, {"fights", fights}, {"picks", picks}}.dump() << '\n' << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) throw std::runtime_error("stdin closed during a batch");
        const auto reply = Json::parse(line);
        for (const auto& f : reply.at("fights")) sims[f.at("k").get<int>()].fight(f.at("won"), f.at("hp"));
        for (const auto& p : reply.at("picks")) sims[p.at("k").get<int>()].pick(p.at("choice"));
        for (const auto& f : reply.at("fights")) need[f.at("k").get<int>()] = sims[f.at("k").get<int>()].advance(opt);
        for (const auto& p : reply.at("picks")) need[p.at("k").get<int>()] = sims[p.at("k").get<int>()].advance(opt);
    }
}

Json run_job(const Json& job) {
    const ms::Options opt{.crn = true, .filter = true};
    const auto seed = job.at("seed").get<std::uint64_t>();
    const int decision = job.at("decision"), n = job.at("rollouts");
    const auto root_seed = job.at("root_seed").get<std::uint64_t>(), rollout_seed = job.at("rollout_seed").get<std::uint64_t>();
    std::vector<Sim> root(1, Sim{GameContext{sts::CharacterClass::IRONCLAD, seed, 20}, {}, root_seed, 0, -1, -1, 0, 0, decision});
    root[0].simple_picks = job.value("simple_picks", false);
    drive(root, opt);
    Json out{{"type", "result"}, {"seed", seed}, {"decision", decision}};
    auto& r = root[0];
    out["root_run"] = {{"won", r.won}, {"boss_hp", r.boss_hp}, {"fights", r.fights}, {"floor", r.gc.floorNum},
                       {"hp", r.gc.curHp}, {"deck", static_cast<int>(r.gc.deck.cards.size())}};
    if (!card_decision(r.gc) || r.decisions != decision || r.gc.outcome != sts::GameOutcome::UNDECIDED) {
        out["status"] = "no_decision";
        return out;
    }
    out["status"] = "ok";
    out["root"] = ms::state_json(r.gc);
    out["root"]["boss"] = encounter_name(r.gc.boss);
    Json options = Json::array();
    for (const auto& c : last_reward(r.gc)) options.push_back(card_json(c));
    out["options"] = options;
    const int k = static_cast<int>(options.size()) + 1;
    std::vector<Sim> sims;
    sims.reserve(static_cast<std::size_t>(k * n));
    for (int o = 0; o < k; ++o)
        for (int i = 0; i < n; ++i) {
            Sim s = r;
            s.target = -1;
            s.option = o;
            s.rollout = i;
            s.member = i % 3;
            s.fights = 0;
            s.fight_seed = mix(rollout_seed * 7919ULL + static_cast<std::uint64_t>(i));
            s.gc.seed = mix(rollout_seed ^ (0xabcdef12345ULL + static_cast<std::uint64_t>(i)));  // shared by all options
            s.pick(o);
            sims.push_back(std::move(s));
        }
    drive(sims, opt);
    Json rollouts = Json::array();
    for (int o = 0; o < k; ++o) {
        Json per = Json::array();
        for (int i = 0; i < n; ++i) {
            const auto& s = sims[static_cast<std::size_t>(o * n + i)];
            per.push_back({{"won", s.won}, {"hp", s.won ? s.gc.curHp : 0}, {"fights", s.fights}, {"boss_hp", s.boss_hp}});
        }
        rollouts.push_back(per);
    }
    out["rollouts"] = rollouts;
    return out;
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::cout << run_job(Json::parse(line)).dump() << '\n' << std::flush;
    }
}
