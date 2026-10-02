// Real-run RL worker (docs/research/run-rl/README.md). Long-running; JSON lines on stdin/stdout.
// Plays real Ironclad runs through max_act (default 1): the MCTS teacher (guided-rollout leaves, no recording) plays every fight,
// SimpleAgent plays everything out of combat EXCEPT card rewards, which are asked of Python.
//
// Python -> worker, one line per run:
//   {"seed": S, "ascension": A, "max_act": 1 | 2 | 3, "simulations": {easy, hard, elite, event, boss}}
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
// Optional request keys "decide": ["rest", "path", "shop", "neow", "event", "boss_relic"], "lookahead_samples" (8),
// "lookahead_horizon" (0) (default none = SimpleAgent). Each listed decision is asked of Python
// as {"type": "decide", "decision", "state", "options": [label...], "after": [state...], "simple": i}: one AFTER-STATE
// per option, computed exactly on a copy of the game. Reply {"choice": i}; -1 = let SimpleAgent decide (only sent
// back when "simple" is -1, i.e. SimpleAgent did something outside the option list).
//   rest: options rest / smith <card> (one per distinct resulting deck) / lift (Girya). Recall, dig, toke: never offered.
//   shop: options leave / buy card i / buy potion i (free slot) / buy relic i / remove <card> (one per distinct deck).
//         Asked again after every purchase until leave. Card / potion / removal after-states are computed on a copy
//         (deterministic); a relic's is BUILT (relic added, gold paid), since some relics roll on pickup (no peeking).
//   neow: a floor-0 event, decided by the event lookahead below (labels add "arm" = "<bonus>|<drawback>").
//   event: (non-Neow event screens with >= 2 actions) SAMPLED LOOKAHEAD: for each option, `lookahead_samples` copies
//         of the game with fresh randomness (never the real rolls; same randomness per sample index across options),
//         option applied, played on with the current policy (greedy; fights by the fight model, MCTS) until
//         `lookahead_horizon` floors later and back on the map (0 = the event resolved), or the run ends. Asked as
//         {"type": "evaluate", "decision": "event", "options", "ends": [[{"state"} | {"terminal", "floor"}] per
//         sample] per option, "simple"}; Python scores the ends with V and replies {"choice": i}. Inside a sample,
//         event steps are one-step after-states ("event_inner", exact in that hypothetical world). Not decided (hidden
//         pre-rolled state a copy would reveal): Dead Adventurer, Match and Keep. Every message
//         carries "lookahead": true inside a sample (Python answers greedily and logs nothing).
//   rest_lookahead: optional job {samples: 4, sims_scale: 0.25}. Real rest decides with both kinds available
//         carry lookahead_pending=true. With non-null option-aligned reply values, compare rest vs the greedy
//         smith/lift via evaluate/rest_lookahead after fresh-random samples through the next fight and first map.
//         Pair indices: 0=rest, 1=non-rest; samples use scaled combat budgets (minimum 100), never recurse.
//   boss_relic: each offered relic + skip, with fresh-randomness after-states and SimpleAgent pickup follow-ups.
//   OBTAIN card selects (The Library) use pick with skip_allowed=false; ordinary/event REWARDS allow skip.
//   All asks carry the real seed, including inside lookahead samples. All logged steps carry their act's boss.
//   path: one option per next map node (only when there are >= 2). After-state = this state with only the remaining
//         routes that start at that node (macro_sim paths, first_xs); nothing is simulated.
#include "agents/combat/search/teacher_search.hpp"
#include "environments/combat/battle_snapshot.hpp"
#include "environments/overworld/game_state.hpp"
#include "environments/overworld/decisions.hpp"
#include "environments/overworld/macro_sim.hpp"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "sim/search/GameAction.h"
#include "sim/search/SimpleAgent.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <map>
#include <limits>
#include <memory>
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

// As environments/overworld/act1_run.cpp (budget category of a fight).
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
    stsrl::macro_sim::detach_map(copy);
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
using stsrl::overworld::Option;
using stsrl::overworld::neow_bonus_names;
using stsrl::overworld::neow_drawback_names;
using stsrl::overworld::outcome_key;
using stsrl::overworld::rest_options;
using stsrl::overworld::shop_options;

// ------------------------------------------------------------------ fight models
// A fight model resolves the fight on gc's BATTLE screen and leaves gc after the fight (as exitBattle would).
// mcts_fight_model plays the whole fight with the MCTS teacher. Another model (e.g. the learned combat outcome model
// via macro_sim::apply_battle_result) only has to return a FightModel.
struct FightRecord {
    std::string encounter, category;
    int hp_before = 0;
    bool won = true;
    std::string fight_id;
};
using FightModel = std::function<FightRecord(GameContext&, bool, int)>;

FightModel mcts_fight_model(const Json& sims, const Json& job, std::vector<Json>& records) {
    for (const auto* key : {"easy", "hard", "elite", "event", "boss"})
        if (sims.at(key).get<std::int64_t>() < 1) throw std::invalid_argument{"simulations must be positive"};
    const teacher::Leaf leaf{job.value("combat_leaf", std::string{"guided_rollout"}), 0, 0};
    std::shared_ptr<stsrl::ValueNet> net;
    if (job.contains("combat_weights")) net = std::make_shared<stsrl::ValueNet>(job.at("combat_weights").get<std::string>());
    teacher::validate(leaf, net != nullptr);
    const bool record = job.value("record_combat", false);
    const bool explore = job.value("combat_explore", false);
    const std::string collection = job.value("collection_id", std::string{"legacy"});
    return [sims, leaf, net, record, explore, collection, &records](GameContext& gc, bool sample, int index) {
        sts::BattleContext battle;
        battle.init(gc);
        FightRecord rec{encounter_name(battle.encounter), category(gc, battle.encounter), battle.player.curHp, true,
                        collection + ":" + std::to_string(gc.seed) + ":" + std::to_string(index)};
        const teacher::Leaf selected = rec.category == "easy" ? teacher::Leaf{"guided_rollout", 0, 0} : leaf;
        // Act 2+ "weak" hallway pools (Spheric Guardian, Shelled Parasite, Byrds, ...) are not easy: hard budget.
        const std::string budget = gc.act >= 2 && rec.category == "easy" ? "hard" : rec.category;
        const auto primary = teacher::leaf_search(selected, selected.kind == "guided_rollout" ? nullptr : net.get(), {sims.at(budget).get<std::int64_t>(), teacher::particles});
        const auto rescue = teacher::leaf_search({"guided_rollout", 0, 0}, nullptr, {5000, teacher::particles});
        bool fallback_used = false;
        const teacher::SearchFn search = [primary, rescue, &fallback_used, selected](auto& tree, std::size_t legal) {
            // Neural values can overvalue endless defensive play. A deterministic public-state safety valve,
            // not a timeout relabelled as a loss, preserves real outcomes and bounds wasted search.
            if (selected.kind != "guided_rollout" && !tree.particles.empty() && tree.particles.front().turn >= 30) {
                fallback_used = true;
                return rescue(tree, legal);
            }
            return primary(tree, legal);
        };
        const Json fight{{"run_seed", gc.seed}, {"episode_id", static_cast<std::int64_t>(gc.seed) * 100 + index},
                         {"fight_index", index}, {"act", gc.act}, {"floor", gc.floorNum},
                         {"encounter", rec.encounter}, {"category", rec.category}, {"ascension", gc.ascension},
                         {"starting_hp", battle.player.curHp}, {"starting_max_hp", battle.player.maxHp}};
        std::vector<Json> rows;
        const bool collecting = record && !sample;
        Json initial;
        std::string replay_error;
        if (collecting) {
            try { initial = stsrl::battle_snapshot(battle); }
            catch (const std::invalid_argument& e) { replay_error = e.what(); }
        }
        const auto end = collecting
            ? teacher::play_fight(battle, fight, rows, search, explore, false, teacher::particles, false, false)
            : teacher::play_fight(battle, search, false, teacher::particles, false);
        bool replay_verified = false;
        if (collecting && !initial.is_null()) {
            stsrl::CombatEnvironment replay{stsrl::battle_restore(initial)};
            for (const auto& row : rows) {
                if (row.at("row_kind") != "decision") continue;
                const auto n = replay.legal_action_count();
                const auto bits = row.at("executed_action_bits").get<std::uint32_t>();
                std::size_t chosen = n;
                for (std::size_t i = 0; i < n; ++i) if (replay.action_bits(i) == bits) { chosen = i; break; }
                if (chosen == n) throw std::runtime_error{"recorded action not legal during replay"};
                replay.step(chosen);
            }
            replay_verified = replay.done() && stsrl::battle_snapshot(replay.battle()) == stsrl::battle_snapshot(end);
            if (!replay_verified) throw std::runtime_error{"combat replay final snapshot mismatch"};
        }
        end.exitBattle(gc);
        if (collecting) {
            const auto outcome = rows.front();
            records.push_back({{"fight_id", rec.fight_id}, {"initial_state", initial}, {"rows", initial.is_null() ? std::vector<Json>{} : rows},
                {"result", {{"run_key", collection + ":" + std::to_string(gc.seed)}, {"run_seed", gc.seed},
                    {"fight_index", index}, {"category", rec.category}, {"encounter", rec.encounter},
                    {"status", "completed"}, {"won", outcome.at("won")},
                    {"battle_final_hp", outcome.at("final_hp")}, {"battle_potions", outcome.at("potions")},
                    {"post_state_json", stsrl::macro_sim::state_json(gc).dump()}, {"combat_leaf", selected.kind},
                    {"simulations", sims.at(budget)}, {"exploration_enabled", explore},
                    {"replay_verified", replay_verified}, {"replay_error", replay_error}, {"rollout_fallback_used", fallback_used}}}});
        }
        rec.won = gc.outcome != sts::GameOutcome::PLAYER_LOSS;
        return rec;
    };
}

// ------------------------------------------------------------------ sampled lookahead
// A sample = a copy of the game with FRESH randomness (every RNG stream and the seed the game derives per-floor
// streams from) and re-shuffled relic pools, so it never sees the real game's rolls. Not re-drawn: the pre-generated
// hallway / elite encounter lists (fine for events at horizon 0; must be re-drawn before searching across fights).
void fresh_randomness(GameContext& gc, std::uint64_t salt) {
    std::uint64_t x = salt;
    const auto next = [&x] {
        x += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };
    gc.seed = next() >> 24;  // < 2^40, like real run seeds
    for (auto* rng : {&gc.aiRng, &gc.cardRandomRng, &gc.cardRng, &gc.eventRng, &gc.mathUtilRng, &gc.merchantRng,
                      &gc.miscRng, &gc.monsterHpRng, &gc.monsterRng, &gc.neowRng, &gc.potionRng, &gc.relicRng,
                      &gc.shuffleRng, &gc.treasureRng})
        *rng = sts::Random(next());
    // Relic pools are shuffled once at game start: without this a sample would draw the REAL next relic.
    std::mt19937_64 shuffle{next()};
    for (auto* pool : {&gc.commonRelicPool, &gc.uncommonRelicPool, &gc.rareRelicPool, &gc.shopRelicPool, &gc.bossRelicPool})
        std::shuffle(pool->begin(), pool->end(), shuffle);
}

std::uint64_t mix(std::uint64_t a, std::uint64_t b) {
    std::uint64_t z = a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

Json slim_state(const GameContext& gc) {
    auto s = stsrl::macro_sim::state_json(gc);
    s["map"].erase("nodes");
    return s;
}

struct Ctx {
    std::set<std::string> decide;
    FightModel fight;
    FightModel rest_fight;  // scaled fight budgets, used only in rest lookahead samples
    int rest_samples = 0;   // absent rest_lookahead keeps the protocol and policy unchanged
    bool sample = false;  // inside a lookahead sample: nothing logged, asks are greedy, nested events one-step
    int max_act = 1;
    std::uint64_t seed = 0;  // real seed, also in hypothetical samples
    int samples = 8;      // lookahead samples per option (event)
    int horizon = 0;      // lookahead floors after the decision; 0 = until back on the map (event resolved)
    std::uint64_t nonce = 0;
};

Json ask(const Ctx& ctx, Json message) {
    message["lookahead"] = ctx.sample;
    message["seed"] = ctx.seed;
    std::cout << message.dump() << '\n' << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) throw std::runtime_error{"stdin closed while waiting for a reply"};
    return Json::parse(line);
}

// Plays a game (the real one, or a lookahead sample) one step at a time.
struct Player {
    GameContext& gc;
    sts::search::SimpleAgent agent;
    const Ctx& ctx;
    Json* steps;  // nullptr in a sample
    int fights = 0, events = 0;
    bool boss_beaten = false;
    int acts_cleared = 0, ticks = 0;
    std::vector<std::string> bosses;

    Player(GameContext& g, const sts::search::SimpleAgent& a, const Ctx& c, Json* s) : gc{g}, agent{a}, ctx{c}, steps{s} {
        agent.curGameContext = &gc;
        bosses.push_back(encounter_name(gc.boss));
    }
    bool done() const { return gc.outcome != sts::GameOutcome::UNDECIDED || gc.act > ctx.max_act || boss_beaten; }
    void log(Json step) {
        if (!step.contains("boss")) step["boss"] = encounter_name(gc.boss);
        if (steps) steps->push_back(std::move(step));
    }

    void step() {
        if (++ticks > 20000) throw std::runtime_error{screen_error("step limit exceeded")};
        if (bosses.size() < static_cast<std::size_t>(gc.act)) bosses.push_back(encounter_name(gc.boss));
        if (gc.screenState == sts::ScreenState::BATTLE) return fight();
        if (card_decision(gc)) return card_pick();
        if (gc.screenState == sts::ScreenState::CARD_SELECT &&
            gc.info.selectScreenType == sts::CardSelectScreenType::OBTAIN) return obtain_pick();
        if (gc.screenState == sts::ScreenState::BOSS_RELIC_REWARDS && ctx.decide.count("boss_relic"))
            return boss_relic();
        if (gc.screenState == sts::ScreenState::EVENT_SCREEN && gc.curEvent == sts::Event::NEOW) {
            if (ctx.decide.count("neow") && !ctx.sample && event()) return;  // a floor-0 event (sampled lookahead)
        } else if (gc.screenState == sts::ScreenState::EVENT_SCREEN && ctx.decide.count("event")) {
            if (event()) return;
        }
        if (gc.screenState == sts::ScreenState::SHOP_ROOM && ctx.decide.count("shop")) return shop();
        const bool rest = gc.screenState == sts::ScreenState::REST_ROOM && ctx.decide.count("rest");
        const bool path = gc.screenState == sts::ScreenState::MAP_SCREEN && ctx.decide.count("path");
        if ((rest || path) && rest_or_path(rest)) return;
        simple_step();
    }

    std::string screen_error(const std::string& reason) const {
        return reason + " (act=" + std::to_string(gc.act) + ", floor=" + std::to_string(gc.floorNum) +
            ", screen=" + std::to_string(static_cast<int>(gc.screenState)) +
            ", event=" + lower(sts::eventGameNames[static_cast<int>(gc.curEvent)]) + ")";
    }

    void simple_step() {
        if (gc.screenState == sts::ScreenState::CARD_SELECT && gc.info.toSelectCards.empty())
            throw std::runtime_error{screen_error("empty card selection")};
        const auto before = agent.actionHistory.size();
        agent.stepOutOfCombat(gc);
        if (agent.actionHistory.size() == before && !done())
            throw std::runtime_error{screen_error("SimpleAgent made no progress")};
    }

    void fight() {
        const auto boss = encounter_name(gc.boss);
        const int act = gc.act;
        const bool boss_fight = gc.curRoom == sts::Room::BOSS;
        // Ascension 20 act 3 has two consecutive bosses; the first win is not an act clear.
        const bool final_boss = !(act == 3 && gc.ascension >= 20 && gc.info.encounter == gc.boss);
        const auto rec = ctx.fight(gc, ctx.sample, fights);
        ++fights;
        if (boss_fight && final_boss && rec.won) acts_cleared = std::max(acts_cleared, act);
        boss_beaten = boss_fight && final_boss && act == ctx.max_act && rec.won;
        log({{"kind", "fight"}, {"boss", boss}, {"state", stsrl::macro_sim::state_json(gc)}, {"encounter", rec.encounter},
             {"category", rec.category}, {"fight_id", rec.fight_id}, {"won", rec.won}, {"hp_before", rec.hp_before}});
    }

    void card_pick() {
        Json options = Json::array();
        for (const auto& c : last_reward(gc)) options.push_back(card_json(c));
        const int simple = simple_choice(gc);
        auto state = stsrl::macro_sim::state_json(gc);
        const auto reply = ask(ctx, {{"type", "pick"}, {"state", state}, {"boss", encounter_name(gc.boss)},
                                     {"options", options}, {"simple", simple}});
        const int choice = reply.at("choice").get<int>();
        if (choice < 0 || choice > static_cast<int>(options.size())) throw std::invalid_argument{"bad choice"};
        take(gc, choice);
        log({{"kind", "pick"}, {"state", std::move(state)}, {"options", options}, {"choice", choice}, {"simple", simple}});
    }

    // OBTAIN (currently The Library) offers cards, not deck operations. No skip action is legal.
    void obtain_pick() {
        Json options = Json::array();
        for (const auto& c : gc.info.toSelectCards) options.push_back(card_json(c.card));
        if (options.empty()) throw std::runtime_error{screen_error("empty obtain selection")};
        GameContext copy = gc;
        stsrl::macro_sim::detach_map(copy);
        auto tmp = agent;
        tmp.curGameContext = &copy;
        const auto before = tmp.actionHistory.size();
        tmp.stepOutOfCombat(copy);
        if (tmp.actionHistory.size() == before) throw std::runtime_error{screen_error("no obtain baseline")};
        const int simple = sts::search::GameAction(tmp.actionHistory[before]).getIdx1();
        auto state = stsrl::macro_sim::state_json(gc);
        const int choice = ask(ctx, {{"type", "pick"}, {"state", state}, {"boss", encounter_name(gc.boss)},
            {"options", options}, {"simple", simple}, {"skip_allowed", false}}).at("choice").get<int>();
        if (choice < 0 || choice >= static_cast<int>(options.size())) throw std::invalid_argument{"bad obtain choice (skip not legal)"};
        sts::search::GameAction(choice).execute(gc);
        log({{"kind", "pick"}, {"state", state}, {"options", options}, {"choice", choice},
            {"simple", simple}, {"skip_allowed", false}});
    }

    void boss_relic() {
        GameContext copy = gc;
        stsrl::macro_sim::detach_map(copy);
        auto tmp = agent;
        tmp.curGameContext = &copy;
        const auto before = tmp.actionHistory.size();
        tmp.stepOutOfCombat(copy);
        const int simple = tmp.actionHistory.size() > before
            ? sts::search::GameAction(tmp.actionHistory[before]).getIdx1() : -1;
        Json labels = Json::array(), afters = Json::array();
        const auto salt = mix(ctx.seed, static_cast<std::uint64_t>(gc.floorNum));
        for (int i = 0; i < 4; ++i) {
            labels.push_back({{"index", i}, {"relic", i == 3 ? "skip" :
                lower(sts::relicEnumNames[static_cast<int>(gc.info.bossRelics[i])])}});
            GameContext c = gc;
            stsrl::macro_sim::detach_map(c);
            fresh_randomness(c, salt);
            sts::search::GameAction(i).execute(c);
            Ctx sub = ctx;
            sub.sample = true;
            Player p{c, agent, sub, nullptr};
            // Resolve pickup card selections/rewards, stopping before the first new map action.
            while (!p.done() && c.screenState != sts::ScreenState::MAP_SCREEN) p.simple_step_guarded();
            afters.push_back(stsrl::macro_sim::state_json(c));
        }
        const int choice = ask(ctx, {{"type", "decide"}, {"decision", "boss_relic"},
            {"state", stsrl::macro_sim::state_json(gc)}, {"boss", encounter_name(gc.boss)},
            {"options", labels}, {"after", afters}, {"simple", simple}}).at("choice").get<int>();
        if (choice < 0 || choice > 3) throw std::invalid_argument{"bad boss relic choice"};
        const auto boss = encounter_name(gc.boss);
        sts::search::GameAction(choice).execute(gc);
        log({{"kind", "decide"}, {"boss", boss}, {"decision", "boss_relic"}, {"options", labels},
            {"choice", choice}, {"simple", simple}, {"after", afters[choice]}});
    }

    void simple_step_guarded() {
        if (++ticks > 20000) throw std::runtime_error{screen_error("pickup step limit exceeded")};
        simple_step();
    }

    // Returns false when there is nothing to decide (SimpleAgent steps instead).
    bool event() {
        // Events whose setup pre-rolls state the player cannot see (a sample copies it, so lookahead would peek):
        // Dead Adventurer (reward order, which elite), Match and Keep (the face-down board). SimpleAgent decides.
        if (gc.curEvent == sts::Event::DEAD_ADVENTURER || gc.curEvent == sts::Event::MATCH_AND_KEEP) return false;
        const auto actions = sts::search::GameAction::getAllActionsInState(gc);
        if (actions.size() < 2) return false;
        GameContext copy = gc;
        stsrl::macro_sim::detach_map(copy);
        sts::search::SimpleAgent tmp = agent;
        tmp.curGameContext = &copy;
        const auto before = tmp.actionHistory.size();
        tmp.stepOutOfCombat(copy);
        const std::uint32_t simple_bits = tmp.actionHistory.size() > before ? tmp.actionHistory[before] : 0xFFFFFFFFu;
        const std::string name = lower(sts::eventGameNames[static_cast<int>(gc.curEvent)]);
        Json labels = Json::array();
        int simple = -1;
        for (int i = 0; i < static_cast<int>(actions.size()); ++i) {
            const int idx = actions[static_cast<std::size_t>(i)].getIdx1();
            labels.push_back({{"index", i}, {"event", name}, {"action", idx}});
            if (gc.curEvent == sts::Event::NEOW && idx >= 0 && idx < 4) {
                const auto& o = gc.info.neowRewards[static_cast<std::size_t>(idx)];
                labels.back()["arm"] = std::string(neow_bonus_names[static_cast<int>(o.r)]) + "|" +
                                       neow_drawback_names[static_cast<int>(o.d)];
            }
            if (actions[static_cast<std::size_t>(i)].bits == simple_bits) simple = i;
        }
        int choice;
        if (ctx.sample) {
            // Nested event step inside a sample: one-step after-states (exact in this hypothetical world).
            Json afters = Json::array();
            for (const auto& a : actions) {
                GameContext c = gc;
                stsrl::macro_sim::detach_map(c);
                a.execute(c);
                afters.push_back(slim_state(c));
            }
            choice = ask(ctx, {{"type", "decide"}, {"decision", "event_inner"}, {"state", slim_state(gc)},
                               {"boss", encounter_name(gc.boss)}, {"options", labels}, {"after", afters},
                               {"simple", simple}}).at("choice").get<int>();
            if (choice < 0) choice = simple >= 0 ? simple : 0;
        } else {
            // Real event step: sampled lookahead, ends[option][sample]; sample k uses the same randomness for every
            // option (common random numbers).
            Json ends = Json::array();
            const auto nonce = mix(mix(gc.seed, static_cast<std::uint64_t>(gc.floorNum)), static_cast<std::uint64_t>(++events));
            for (const auto& a : actions) {
                Json per = Json::array();
                for (int k = 0; k < ctx.samples; ++k) per.push_back(lookahead(a, mix(nonce, static_cast<std::uint64_t>(k))));
                ends.push_back(std::move(per));
            }
            choice = ask(ctx, {{"type", "evaluate"}, {"decision", "event"}, {"state", slim_state(gc)},
                               {"boss", encounter_name(gc.boss)}, {"options", labels}, {"ends", ends},
                               {"simple", simple}}).at("choice").get<int>();
        }
        if (choice < 0 || choice >= static_cast<int>(actions.size())) throw std::invalid_argument{"bad event choice"};
        actions[static_cast<std::size_t>(choice)].execute(gc);
        log({{"kind", "decide"}, {"decision", gc.floorNum == 0 && name == "neow" ? "neow" : "event"}, {"options", labels},
             {"choice", choice}, {"simple", simple}});
        return true;
    }

    // One lookahead sample of action `a`: fresh randomness, play on (current policy, greedy; fights by the fight
    // model) until `horizon` floors later and back on the map, or the run ends. Returns the end for Python to score:
    // {"state"} or {"terminal": died | cleared, "floor"}.
    Json lookahead(const sts::search::GameAction& a, std::uint64_t salt) {
        GameContext c = gc;
        stsrl::macro_sim::detach_map(c);
        fresh_randomness(c, salt);
        Ctx sub = ctx;
        sub.sample = true;
        Player p{c, agent, sub, nullptr};
        p.acts_cleared = acts_cleared;
        a.execute(c);
        const int stop_floor = gc.floorNum + ctx.horizon;
        while (!p.done()) {
            if (c.screenState == sts::ScreenState::MAP_SCREEN && c.floorNum >= stop_floor) break;
            p.step();
        }
        if (p.boss_beaten || c.act > ctx.max_act)
            return {{"terminal", "cleared"}, {"floor", c.floorNum}, {"act", c.act}, {"acts_cleared", p.acts_cleared}};
        if (c.outcome == sts::GameOutcome::PLAYER_LOSS)
            return {{"terminal", "died"}, {"floor", c.floorNum}, {"act", c.act}, {"acts_cleared", p.acts_cleared}};
        return {{"state", slim_state(c)}};
    }

    // Apply a campfire option, then resolve the next fight and its follow-up screens. A boss can
    // lead through rewards/relic pickup into the next act; stop at that first map, not at a floor cutoff.
    Json rest_sample(const Option& option, std::uint64_t salt) {
        GameContext c = gc;
        stsrl::macro_sim::detach_map(c);
        fresh_randomness(c, salt);
        Ctx sub = ctx;
        sub.sample = true;
        sub.fight = ctx.rest_fight;
        Player p{c, agent, sub, nullptr};
        p.acts_cleared = acts_cleared;
        for (int bits : option.actions) sts::search::GameAction(bits).execute(c);
        while (!p.done()) {
            if (p.fights > 0 && c.screenState == sts::ScreenState::MAP_SCREEN) break;
            p.step();
        }
        if (p.boss_beaten || c.act > ctx.max_act)
            return {{"terminal", "cleared"}, {"floor", c.floorNum}, {"act", c.act}, {"acts_cleared", p.acts_cleared}};
        if (c.outcome == sts::GameOutcome::PLAYER_LOSS)
            return {{"terminal", "died"}, {"floor", c.floorNum}, {"act", c.act}, {"acts_cleared", p.acts_cleared}};
        return {{"state", slim_state(c)}};
    }

    int refine_rest(const std::vector<Option>& options, int rest_idx, int choice, const Json& values, Json& trace) {
        // Exploration/simple replies have null values: do not override their chosen action.
        if (values.is_null()) return choice;
        if (!values.is_array() || values.size() != options.size())
            throw std::invalid_argument{"rest lookahead requires option-aligned values"};
        int non_rest = choice == rest_idx ? -1 : choice;
        if (non_rest < 0) {
            for (int i = 0; i < static_cast<int>(options.size()); ++i) {
                const auto action = options[i].label.value("action", std::string{});
                if (action != "smith" && action != "lift") continue;
                if (non_rest < 0 || values[i].get<double>() > values[non_rest].get<double>()) non_rest = i;
            }
        }
        if (non_rest < 0) return choice;
        const std::vector<int> pair{rest_idx, non_rest};
        Json labels = Json::array(), ends = Json::array();
        const auto nonce = mix(ctx.seed, static_cast<std::uint64_t>(gc.floorNum));
        for (int idx : pair) {
            labels.push_back(options[idx].label);
            Json per = Json::array();
            for (int k = 0; k < ctx.rest_samples; ++k)
                per.push_back(rest_sample(options[idx], mix(nonce, static_cast<std::uint64_t>(k))));
            ends.push_back(std::move(per));
        }
        const auto reply = ask(ctx, {{"type", "evaluate"}, {"decision", "rest_lookahead"},
            {"state", slim_state(gc)}, {"boss", encounter_name(gc.boss)}, {"options", labels},
            {"ends", ends}, {"simple", choice == rest_idx ? 0 : 1}});
        const int selected = reply.at("choice").get<int>();
        if (selected < 0 || selected > 1) throw std::invalid_argument{"bad rest lookahead choice"};
        trace = {{"options", pair}, {"values", reply.value("values", Json{})}};
        return pair[selected];
    }

    void shop() {
        // SimpleAgent's own next action(s) on a copy: one purchase, or removal + its card select.
        GameContext copy = gc;
        stsrl::macro_sim::detach_map(copy);
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
            return;
        }
        Json labels = Json::array(), afters = Json::array();
        for (const auto& o : options) { labels.push_back(o.label); afters.push_back(o.after); }
        const auto reply = ask(ctx, {{"type", "decide"}, {"decision", "shop"}, {"state", stsrl::macro_sim::state_json(gc)},
                                     {"boss", encounter_name(gc.boss)}, {"options", labels}, {"after", afters},
                                     {"simple", simple}});
        const int choice = reply.at("choice").get<int>();
        if (choice < -1 || choice >= static_cast<int>(options.size())) throw std::invalid_argument{"bad choice"};
        if (choice < 0) {
            agent.stepOutOfCombat(gc);
            return;
        }
        for (int bits : options[static_cast<std::size_t>(choice)].actions)
            sts::search::GameAction{static_cast<std::uint32_t>(bits)}.execute(gc);
        log({{"kind", "decide"}, {"decision", "shop"}, {"options", labels}, {"choice", choice}, {"simple", simple},
             {"after", options[static_cast<std::size_t>(choice)].after}});
    }

    // Returns false when there is nothing to decide (SimpleAgent steps instead).
    bool rest_or_path(bool rest) {
        const bool path = !rest;
        // SimpleAgent's own move, on a copy (the label and the "simple" policy). At the first map screen the
        // agent plans its whole route: keep that plan so SimpleAgent stays itself when it is followed.
        GameContext copy = gc;
        stsrl::macro_sim::detach_map(copy);
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
                if (keys[static_cast<std::size_t>(i)] == k) simple = i;
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
        if (!(options.size() >= 2 || (rest && !options.empty()))) return false;
        Json labels = Json::array(), afters = Json::array();
        for (const auto& o : options) { labels.push_back(o.label); afters.push_back(o.after); }
        const char* kind = rest ? "rest" : "path";
        int rest_idx = -1;
        bool non_rest = false;
        if (rest && !ctx.sample && ctx.rest_samples > 0) {
            for (int i = 0; i < static_cast<int>(options.size()); ++i) {
                const auto action = options[i].label.value("action", std::string{});
                if (action == "rest") rest_idx = i;
                if (action == "smith" || action == "lift") non_rest = true;
            }
        }
        Json message = {{"type", "decide"}, {"decision", kind}, {"state", stsrl::macro_sim::state_json(gc)},
                        {"boss", encounter_name(gc.boss)}, {"options", labels}, {"after", afters}, {"simple", simple}};
        const bool pending = rest_idx >= 0 && non_rest;
        if (pending) message["lookahead_pending"] = true;
        const auto reply = ask(ctx, std::move(message));
        int choice = reply.at("choice").get<int>();
        if (choice < -1 || choice >= static_cast<int>(options.size())) throw std::invalid_argument{"bad choice"};
        if (choice < 0) return false;
        Json trace;
        if (pending) choice = refine_rest(options, rest_idx, choice, reply.value("values", Json{}), trace);
        for (int a : options[static_cast<std::size_t>(choice)].actions) sts::search::GameAction(a).execute(gc);
        Json step = {{"kind", "decide"}, {"decision", kind}, {"options", labels}, {"choice", choice}, {"simple", simple},
                     {"after", options[static_cast<std::size_t>(choice)].after}};
        if (!trace.is_null()) step["lookahead"] = std::move(trace);
        log(std::move(step));
        return true;
    }
};

Json play_run(const Json& job) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto seed = job.at("seed").get<std::uint64_t>();
    Ctx ctx;
    ctx.seed = seed;
    ctx.max_act = job.value("max_act", 1);
    if (ctx.max_act < 1 || ctx.max_act > 3) throw std::invalid_argument{"max_act must be 1, 2 or 3"};
    for (const auto& d : job.value("decide", Json::array())) {
        const auto name = d.get<std::string>();
        if (name != "rest" && name != "path" && name != "shop" && name != "neow" && name != "event" && name != "boss_relic")
            throw std::invalid_argument{"unknown decision: " + name};
        ctx.decide.insert(name);
    }
    std::vector<Json> combat_records;
    ctx.fight = mcts_fight_model(job.at("simulations"), job, combat_records);
    if (job.contains("rest_lookahead")) {
        const auto& config = job.at("rest_lookahead");
        ctx.rest_samples = config.value("samples", 4);
        const double scale = config.value("sims_scale", 0.25);
        if (ctx.rest_samples < 1 || !std::isfinite(scale) || scale <= 0)
            throw std::invalid_argument{"rest_lookahead requires positive samples and finite positive sims_scale"};
        auto sims = job.at("simulations");
        for (auto& budget : sims.items()) {
            const double scaled = budget.value().get<std::int64_t>() * scale;
            if (!std::isfinite(scaled) || scaled >= static_cast<double>(std::numeric_limits<std::int64_t>::max()))
                throw std::invalid_argument{"rest_lookahead scaled simulations overflow"};
            budget.value() = std::max<std::int64_t>(100, static_cast<std::int64_t>(scaled));
        }
        ctx.rest_fight = mcts_fight_model(sims, job, combat_records);
    }
    ctx.samples = job.value("lookahead_samples", 8);
    ctx.horizon = job.value("lookahead_horizon", 0);
    GameContext gc{sts::CharacterClass::IRONCLAD, seed, job.at("ascension").get<int>()};
    Json steps = Json::array();
    steps.push_back({{"kind", "start"}, {"boss", encounter_name(gc.boss)}, {"state", stsrl::macro_sim::state_json(gc)}});
    Player player{gc, sts::search::SimpleAgent{}, ctx, &steps};
    while (!player.done()) player.step();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return {{"type", "done"}, {"run_key", job.value("collection_id", std::string{"legacy"}) + ":" + std::to_string(seed)}, {"seed", seed}, {"boss", encounter_name(gc.boss)},
            {"status", player.boss_beaten || gc.act > ctx.max_act ? "act_complete" : "died"},
            {"act", gc.act}, {"acts_cleared", player.acts_cleared}, {"bosses", player.bosses},
            {"floor", gc.floorNum}, {"fights", player.fights},
            {"final_hp", gc.curHp}, {"seconds", seconds}, {"steps", steps}, {"combat_records", combat_records}};
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
