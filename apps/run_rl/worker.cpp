// Real-run RL worker (slop_docs/run-rl/README.md). Long-running; JSON lines on stdin/stdout.
// Plays real Ironclad act 1 runs: the MCTS teacher (guided-rollout leaves, no recording) plays every fight,
// SimpleAgent plays everything out of combat EXCEPT card rewards, which are asked of Python.
//
// Python -> worker, one line per run:
//   {"seed": S, "ascension": A, "simulations": {easy, hard, elite, event, boss}}
// worker -> Python, at every card reward (answered by one line {"choice": i}; i = len(options) means skip):
//   {"type": "pick", "state": macro_sim::state_json, "boss": name, "options": [card...], "simple": i}
//   simple = SimpleAgent's choice for this reward (the baseline policy).
// worker -> Python, when the run ends:
//   {"type": "done", "seed", "boss", "status": died | act_complete, "floor", "fights", "final_hp", "seconds",
//    "steps": [...]}   steps in play order:
//     {"kind": "start", "state"}                                         before the first floor
//     {"kind": "fight", "state", "encounter", "category", "won", "hp_before"}   state after exitBattle
//     {"kind": "pick", "state", "options", "choice", "simple"}           state BEFORE the pick
//     {"kind": "decide", "decision": rest | path, "options": [label...], "choice", "simple", "after"}
//
// Optional request key "decide": ["rest", "path", "shop", "neow"] (default none = SimpleAgent). Each listed decision is asked of Python
// as {"type": "decide", "decision", "state", "options": [label...], "after": [state...], "simple": i}: one AFTER-STATE
// per option, computed exactly on a copy of the game. Reply {"choice": i}; -1 = let SimpleAgent decide (only sent
// back when "simple" is -1, i.e. SimpleAgent did something outside the option list).
//   rest: options rest / smith <card> (one per distinct resulting deck) / lift (Girya). Recall, dig, toke: never offered.
//   shop: options leave / buy card i / buy potion i (free slot) / buy relic i / remove <card> (one per distinct deck).
//         Asked again after every purchase until leave. Card / potion / removal after-states are computed on a copy
//         (deterministic); a relic's is BUILT (relic added, gold paid), since some relics roll on pickup (no peeking).
//   neow: the 4 offered Neow options as arms {"index", "bonus", "drawback", "arm" = "<bonus>|<drawback>"}; NO
//         after-states (outcomes are random: a bandit decides, apps/run_rl/neow.py). simple = 0 (SimpleAgent always
//         takes the first). Follow-up screens (card reward, card select) are handled as usual afterwards.
//   path: one option per next map node (only when there are >= 2). After-state = this state with only the remaining
//         routes that start at that node (macro_sim paths, first_xs); nothing is simulated.
#include "agents/teacher_search.hpp"
#include "apps/common/game_state.hpp"
#include "apps/common/macro_sim.hpp"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "sim/search/GameAction.h"
#include "sim/search/SimpleAgent.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace {
using Json = nlohmann::json;
using sts::GameContext;
namespace teacher = stsrl::teacher;

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string encounter_name(sts::MonsterEncounter e) { return lower(sts::monsterEncounterEnumNames[static_cast<int>(e)]); }

// As scenarios/act1_run.cpp (budget category of a fight).
std::string category(const GameContext& game, sts::MonsterEncounter encounter) {
    namespace pool = sts::MonsterEncounterPool;
    if (game.curRoom == sts::Room::BOSS) return "boss";
    if (game.curRoom == sts::Room::ELITE) return "elite";
    if (game.curRoom == sts::Room::MONSTER) {
        const auto act = game.act - 1;
        const auto in = [&](const auto* list, int count) { return std::find(list, list + count, encounter) != list + count; };
        if (in(pool::weakEnemies[act], pool::weakCount[act])) return "easy";
        if (in(pool::strongEnemies[act], pool::strongCount[act])) return "hard";
    }
    return "event";
}

Json card_json(const sts::Card& c) {
    return {{"card_id", static_cast<int>(c.id)}, {"upgraded", c.getUpgraded()}, {"misc", c.misc},
            {"name", lower(sts::cardEnumStrings[static_cast<int>(c.id)])}};
}

// A card-reward decision is pending: SimpleAgent would take potions / relics / keys first (as card_search).
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

void take(GameContext& gc, int choice) {
    const auto& r = gc.info.rewardsContainer;
    if (choice < static_cast<int>(last_reward(gc).size()))
        sts::search::GameAction(sts::search::GameAction::RewardsActionType::CARD, r.cardRewardCount - 1, choice).execute(gc);
    else
        sts::search::GameAction(sts::search::GameAction::RewardsActionType::SKIP).execute(gc);
}

// SimpleAgent's choice at this reward: run its card step on a copy and see which offered card entered the deck.
int simple_choice(const GameContext& gc) {
    GameContext copy = gc;
    sts::search::SimpleAgent agent;
    agent.curGameContext = &copy;
    agent.stepCardReward(copy);
    const auto& offered = last_reward(gc);
    const int n = static_cast<int>(offered.size());
    if (copy.deck.size() == gc.deck.size()) return n;  // skipped
    std::map<int, int> before;
    for (const auto& c : gc.deck.cards) ++before[static_cast<int>(c.id)];
    for (const auto& c : copy.deck.cards)
        if (before[static_cast<int>(c.id)]-- <= 0)
            for (int i = 0; i < n; ++i)
                if (offered[i].id == c.id) return i;
    return n;
}

// Deck + HP + relics: two after-states that agree here are the same decision outcome.
std::string outcome_key(const GameContext& gc) {
    std::vector<std::pair<int, int>> deck;
    for (const auto& c : gc.deck.cards) deck.emplace_back(static_cast<int>(c.id), c.getUpgraded());
    std::sort(deck.begin(), deck.end());
    Json k = {{"deck", deck}, {"hp", gc.curHp}, {"max_hp", gc.maxHp}};
    for (const auto& r : gc.relics.relics) k["relics"].push_back({static_cast<int>(r.id), r.data});
    return k.dump();
}

struct Option {
    Json label;
    std::vector<int> actions;  // GameActions executed in order
    Json after;
};

std::vector<Option> rest_options(const GameContext& gc, std::vector<std::string>& keys) {
    std::vector<Option> out;
    for (int a : {0, 1, 3}) {
        if (!sts::search::GameAction(a).isValidAction(gc)) continue;
        GameContext c = gc;
        sts::search::GameAction(a).execute(c);
        if (a != 1) {
            const auto k = outcome_key(c);
            if (std::find(keys.begin(), keys.end(), k) != keys.end()) continue;
            keys.push_back(k);
            out.push_back({{{"action", a == 0 ? "rest" : "lift"}}, {a}, stsrl::macro_sim::state_json(c)});
            continue;
        }
        if (c.screenState != sts::ScreenState::CARD_SELECT) continue;
        for (int i = 0; i < static_cast<int>(c.info.toSelectCards.size()); ++i) {
            GameContext c2 = c;
            sts::search::GameAction(i).execute(c2);
            const auto k = outcome_key(c2);
            if (std::find(keys.begin(), keys.end(), k) != keys.end()) continue;
            keys.push_back(k);
            const auto& card = c.info.toSelectCards[i].card;
            out.push_back({{{"action", "smith"}, {"card", lower(sts::cardEnumStrings[static_cast<int>(card.id)])},
                            {"upgraded", card.getUpgraded()}}, {1, i}, stsrl::macro_sim::state_json(c2)});
        }
    }
    return out;
}

// sts::Neow::Bonus / Drawback names (Neow.h order).
constexpr const char* neow_bonus_names[] = {
    "three_cards", "one_random_rare_card", "remove_card", "upgrade_card", "transform_card", "random_colorless",
    "three_small_potions", "random_common_relic", "ten_percent_hp_bonus", "three_enemy_kill", "hundred_gold",
    "random_colorless_2", "remove_two", "one_rare_relic", "three_rare_cards", "two_fifty_gold",
    "transform_two_cards", "twenty_percent_hp_bonus", "boss_relic", "invalid"};
constexpr const char* neow_drawback_names[] = {
    "invalid", "none", "ten_percent_hp_loss", "no_gold", "curse", "percent_damage", "lose_starter_relic"};

std::string shop_key(const GameContext& gc) {
    Json k = Json::parse(outcome_key(gc));
    k["gold"] = gc.gold;
    for (int i = 0; i < gc.potionCapacity; ++i) k["potions"].push_back(static_cast<int>(gc.potions[static_cast<std::size_t>(i)]));
    return k.dump();
}

std::vector<Option> shop_options(const GameContext& gc, std::vector<std::string>& keys) {
    using RA = sts::search::GameAction::RewardsActionType;
    const auto& shop = gc.info.shop;
    std::vector<Option> out;
    const auto add = [&](Json label, std::vector<int> actions, const GameContext& after_gc, Json after) {
        const auto k = shop_key(after_gc) + (label.contains("relic") ? label["relic"].dump() : "");
        if (std::find(keys.begin(), keys.end(), k) != keys.end()) return;
        keys.push_back(k);
        out.push_back({std::move(label), std::move(actions), std::move(after)});
    };
    add({{"action", "leave"}}, {static_cast<int>(sts::search::GameAction(RA::SKIP).bits)}, gc, stsrl::macro_sim::state_json(gc));
    for (int i = 0; i < 7; ++i) {
        const sts::search::GameAction a(RA::CARD, i);
        if (!a.isValidAction(gc)) continue;
        GameContext c = gc;
        a.execute(c);
        add({{"action", "card"}, {"card", lower(sts::cardEnumStrings[static_cast<int>(shop.cards[i].getId())])},
             {"upgraded", shop.cards[i].isUpgraded()}, {"price", shop.cardPrice(i)}},
            {static_cast<int>(a.bits)}, c, stsrl::macro_sim::state_json(c));
    }
    for (int i = 0; i < 3; ++i) {
        const sts::search::GameAction a(RA::POTION, i);
        if (!a.isValidAction(gc) || gc.potionCount >= gc.potionCapacity) continue;
        GameContext c = gc;
        a.execute(c);
        add({{"action", "potion"}, {"potion", lower(sts::potionEnumNames[static_cast<int>(shop.potions[i])])},
             {"price", shop.potionPrice(i)}}, {static_cast<int>(a.bits)}, c, stsrl::macro_sim::state_json(c));
    }
    for (int i = 0; i < 3; ++i) {
        const sts::search::GameAction a(RA::RELIC, i);
        if (!a.isValidAction(gc)) continue;
        auto after = stsrl::macro_sim::state_json(gc);
        const auto id = shop.relics[i];
        after["relics"].push_back({{"relic_id", static_cast<int>(id)}, {"data", 0},
                                   {"name", lower(sts::relicEnumNames[static_cast<int>(id)])}});
        after["gold"] = gc.gold - shop.relicPrice(i);
        GameContext keyed = gc;
        keyed.gold -= shop.relicPrice(i);
        add({{"action", "relic"}, {"relic", lower(sts::relicEnumNames[static_cast<int>(id)])}, {"price", shop.relicPrice(i)}},
            {static_cast<int>(a.bits)}, keyed, std::move(after));
    }
    const sts::search::GameAction remove(RA::CARD_REMOVE);
    if (remove.isValidAction(gc)) {
        GameContext c = gc;
        remove.execute(c);
        if (c.screenState == sts::ScreenState::CARD_SELECT)
            for (int j = 0; j < static_cast<int>(c.info.toSelectCards.size()); ++j) {
                GameContext c2 = c;
                sts::search::GameAction(j).execute(c2);
                const auto& card = c.info.toSelectCards[j].card;
                add({{"action", "remove"}, {"card", lower(sts::cardEnumStrings[static_cast<int>(card.id)])},
                     {"upgraded", card.getUpgraded()}, {"price", shop.removeCost}},
                    {static_cast<int>(remove.bits), static_cast<int>(sts::search::GameAction(j).bits)}, c2,
                    stsrl::macro_sim::state_json(c2));
            }
    }
    return out;
}

Json ask(const Json& message) {
    std::cout << message.dump() << '\n' << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) throw std::runtime_error{"stdin closed while waiting for a pick"};
    return Json::parse(line);
}

Json play_run(const Json& job) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto seed = job.at("seed").get<std::uint64_t>();
    const auto& sims = job.at("simulations");
    for (const auto* key : {"easy", "hard", "elite", "event", "boss"})
        if (sims.at(key).get<std::int64_t>() < 1) throw std::invalid_argument{"simulations must be positive"};
    const teacher::Leaf leaf{"guided_rollout", 0, 0};
    teacher::validate(leaf, false);

    std::set<std::string> decide;
    for (const auto& d : job.value("decide", Json::array())) {
        const auto name = d.get<std::string>();
        if (name != "rest" && name != "path" && name != "shop" && name != "neow") throw std::invalid_argument{"unknown decision: " + name};
        decide.insert(name);
    }
    GameContext gc{sts::CharacterClass::IRONCLAD, seed, job.at("ascension").get<int>()};
    sts::search::SimpleAgent agent;
    agent.curGameContext = &gc;
    Json steps = Json::array();
    steps.push_back({{"kind", "start"}, {"state", stsrl::macro_sim::state_json(gc)}});
    int fights = 0;
    bool boss_beaten = false;
    while (gc.outcome == sts::GameOutcome::UNDECIDED && gc.act == 1 && !boss_beaten) {
        if (gc.screenState != sts::ScreenState::BATTLE) {
            if (card_decision(gc)) {
                Json options = Json::array();
                for (const auto& c : last_reward(gc)) options.push_back(card_json(c));
                const int simple = simple_choice(gc);
                auto state = stsrl::macro_sim::state_json(gc);
                const auto reply = ask({{"type", "pick"}, {"state", state}, {"boss", encounter_name(gc.boss)},
                                        {"options", options}, {"simple", simple}});
                const int choice = reply.at("choice").get<int>();
                if (choice < 0 || choice > static_cast<int>(options.size())) throw std::invalid_argument{"bad choice"};
                take(gc, choice);
                steps.push_back({{"kind", "pick"}, {"state", std::move(state)}, {"options", options},
                                 {"choice", choice}, {"simple", simple}});
                continue;
            }
            const bool rest = gc.screenState == sts::ScreenState::REST_ROOM && decide.count("rest");
            const bool path = gc.screenState == sts::ScreenState::MAP_SCREEN && gc.act == 1 && decide.count("path");
            if (gc.screenState == sts::ScreenState::EVENT_SCREEN && gc.curEvent == sts::Event::NEOW && decide.count("neow")) {
                Json labels = Json::array();
                for (int i = 0; i < 4; ++i) {
                    const auto& o = gc.info.neowRewards[static_cast<std::size_t>(i)];
                    const std::string b = neow_bonus_names[static_cast<int>(o.r)], d = neow_drawback_names[static_cast<int>(o.d)];
                    labels.push_back({{"index", i}, {"bonus", b}, {"drawback", d}, {"arm", b + "|" + d}});
                }
                const auto reply = ask({{"type", "decide"}, {"decision", "neow"}, {"state", stsrl::macro_sim::state_json(gc)},
                                        {"boss", encounter_name(gc.boss)}, {"options", labels}, {"after", Json::array()},
                                        {"simple", 0}});
                const int choice = reply.at("choice").get<int>();
                if (choice < 0 || choice > 3) throw std::invalid_argument{"bad neow choice"};
                sts::search::GameAction(choice).execute(gc);
                steps.push_back({{"kind", "decide"}, {"decision", "neow"}, {"options", labels}, {"choice", choice},
                                 {"simple", 0}});
                continue;
            }
            const bool shop = gc.screenState == sts::ScreenState::SHOP_ROOM && decide.count("shop");
            if (shop) {
                // SimpleAgent's own next action(s) on a copy: one purchase, or removal + its card select.
                GameContext copy = gc;
                sts::search::SimpleAgent tmp = agent;
                tmp.curGameContext = &copy;
                const auto before = tmp.actionHistory.size();
                tmp.stepOutOfCombat(copy);
                if (copy.screenState == sts::ScreenState::CARD_SELECT) tmp.stepOutOfCombat(copy);
                const std::vector<int> simple_bits(tmp.actionHistory.begin() + static_cast<long>(before), tmp.actionHistory.end());
                std::vector<std::string> keys;
                auto options = shop_options(gc, keys);
                int simple = -1;
                for (int i = 0; i < static_cast<int>(options.size()); ++i)
                    if (options[static_cast<std::size_t>(i)].actions == simple_bits) simple = i;
                if (options.size() == 1) {  // only leave: nothing to decide
                    agent.stepOutOfCombat(gc);
                    continue;
                }
                Json labels = Json::array(), afters = Json::array();
                for (const auto& o : options) { labels.push_back(o.label); afters.push_back(o.after); }
                const auto reply = ask({{"type", "decide"}, {"decision", "shop"}, {"state", stsrl::macro_sim::state_json(gc)},
                                        {"boss", encounter_name(gc.boss)}, {"options", labels}, {"after", afters},
                                        {"simple", simple}});
                const int choice = reply.at("choice").get<int>();
                if (choice < -1 || choice >= static_cast<int>(options.size())) throw std::invalid_argument{"bad choice"};
                if (choice >= 0) {
                    for (int bits : options[static_cast<std::size_t>(choice)].actions) {
                        sts::search::GameAction{static_cast<std::uint32_t>(bits)}.execute(gc);
                    }
                    steps.push_back({{"kind", "decide"}, {"decision", "shop"}, {"options", labels}, {"choice", choice},
                                     {"simple", simple}, {"after", options[static_cast<std::size_t>(choice)].after}});
                    continue;
                }
                agent.stepOutOfCombat(gc);
                continue;
            }
            if (rest || path) {
                // SimpleAgent's own move, on a copy (the label and the "simple" policy). At the first map screen the
                // agent plans its whole route: keep that plan so SimpleAgent stays itself when it is followed.
                GameContext copy = gc;
                sts::search::SimpleAgent tmp = agent;
                tmp.curGameContext = &copy;
                tmp.stepOutOfCombat(copy);
                for (int k = 0; k < 3 && rest && copy.screenState == sts::ScreenState::CARD_SELECT; ++k) tmp.stepOutOfCombat(copy);
                if (path && gc.curMapNodeY < 0) {
                    agent = tmp;
                    agent.curGameContext = &gc;
                }
                std::vector<Option> options;
                int simple = -1;
                if (rest) {
                    std::vector<std::string> keys;
                    options = rest_options(gc, keys);
                    const auto k = outcome_key(copy);
                    for (int i = 0; i < static_cast<int>(keys.size()); ++i)
                        if (keys[i] == k) simple = i;
                } else {
                    const auto actions = sts::search::GameAction::getAllActionsInState(gc);
                    const auto state = stsrl::macro_sim::state_json(gc);
                    for (const auto& a : actions) {
                        const int x = a.getIdx1();
                        auto after = state;
                        Json kept = Json::array();
                        for (const auto& p : state["map"]["paths"])
                            for (const auto& fx : p["first_xs"])
                                if (fx.get<int>() == x) { kept.push_back(p); break; }
                        after["map"]["paths"] = kept;
                        std::string room = "?";
                        const int y = gc.curMapNodeY + 1;
                        if (y < 15) room = std::string(1, sts::getRoomSymbol(gc.map->getNode(x, y).room));
                        if (copy.curMapNodeX == x) simple = static_cast<int>(options.size());
                        options.push_back({{{"x", x}, {"room", room}}, {x}, std::move(after)});
                    }
                }
                if (options.size() >= 2 || (rest && !options.empty())) {
                    Json labels = Json::array(), afters = Json::array();
                    for (const auto& o : options) { labels.push_back(o.label); afters.push_back(o.after); }
                    const char* kind = rest ? "rest" : "path";
                    const auto reply = ask({{"type", "decide"}, {"decision", kind}, {"state", stsrl::macro_sim::state_json(gc)},
                                            {"boss", encounter_name(gc.boss)}, {"options", labels}, {"after", afters},
                                            {"simple", simple}});
                    const int choice = reply.at("choice").get<int>();
                    if (choice < -1 || choice >= static_cast<int>(options.size())) throw std::invalid_argument{"bad choice"};
                    if (choice >= 0) {
                        for (int a : options[static_cast<std::size_t>(choice)].actions) sts::search::GameAction(a).execute(gc);
                        steps.push_back({{"kind", "decide"}, {"decision", kind}, {"options", labels}, {"choice", choice},
                                         {"simple", simple}, {"after", options[static_cast<std::size_t>(choice)].after}});
                        continue;
                    }
                }
            }
            agent.stepOutOfCombat(gc);
            continue;
        }
        sts::BattleContext battle;
        battle.init(gc);
        const auto cat = category(gc, battle.encounter);
        const int hp_before = battle.player.curHp;
        const auto search = teacher::leaf_search(leaf, nullptr, {sims.at(cat).get<std::int64_t>(), teacher::particles});
        const auto end = teacher::play_fight(battle, search, false, teacher::particles, false);
        end.exitBattle(gc);
        ++fights;
        const bool won = gc.outcome != sts::GameOutcome::PLAYER_LOSS;
        boss_beaten = gc.curRoom == sts::Room::BOSS && won;
        steps.push_back({{"kind", "fight"}, {"state", stsrl::macro_sim::state_json(gc)},
                         {"encounter", encounter_name(battle.encounter)}, {"category", cat}, {"won", won},
                         {"hp_before", hp_before}});
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return {{"type", "done"}, {"seed", seed}, {"boss", encounter_name(gc.boss)},
            {"status", boss_beaten ? "act_complete" : "died"}, {"floor", gc.floorNum}, {"fights", fights},
            {"final_hp", gc.curHp}, {"seconds", seconds}, {"steps", steps}};
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        try {
            std::cout << play_run(Json::parse(line)).dump() << '\n' << std::flush;
        } catch (const std::exception& e) {
            std::cout << Json{{"type", "error"}, {"message", e.what()}}.dump() << '\n' << std::flush;
            return 1;
        }
    }
    return 0;
}
