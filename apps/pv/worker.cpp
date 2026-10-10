// pv_worker encode OUT.parquet: combat_v4 fights (+ search rows) on stdin (JSON lines, a pipe transport) → per-decision
//   training rows in one Parquet shard (apps/pv/shard.hpp).
// pv_worker turns: recorded fights on stdin → per-turn DFS measurement rows (pipe transport only).
// pv_worker evaluate MODEL: encoded input batches on stdin → predictions on stdout (parity/debug interface).
// pv_worker play MODEL SIMS [--explore] | teacher SIMS: {fight_id, start} lines on stdin → one result line per fight:
//   {status, fight (combat_v4 fights row; completed only), search: [combat_v4 search rows], stats: [telemetry per search]}.
#include "agents/combat/pv/search.hpp"
#include "agents/combat/pv/turn_search.hpp"
#include "agents/combat/pv/turn_pimc.hpp"
#include "agents/combat/pv/turn_targets.hpp"
#include "agents/combat/pv/real_turn.hpp"
#include "agents/combat/search/teacher_search.hpp"
#include "apps/pv/shard.hpp"
#include "apps/pv/turns.hpp"
#include "environments/combat/environment.hpp"
#include "environments/combat/record_v4.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;
namespace pv = stsrl::pv;
using Clock = std::chrono::steady_clock;

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// Root-halving (evaluation-only) searches have no approved training-target contract: refuse to encode them.
bool root_halving_marked(const Json& x) {
    return x.is_object() && (x.contains("root_halving") ||
        (x.contains("agent") && x.at("agent").is_string() && x.at("agent").get<std::string>().find("root_halving") != std::string::npos));
}

void encode(const Json& fight, pv::ShardWriter& writer) {
    if (root_halving_marked(fight) || (fight.contains("fight") && root_halving_marked(fight.at("fight"))))
        throw std::runtime_error{"PV: root_halving records are evaluation-only and cannot be encoded as training rows"};
    if (fight.contains("search")) for (const auto& row : fight.at("search"))
        if (root_halving_marked(row)) throw std::runtime_error{"PV: root_halving search rows cannot be encoded as training targets"};
    if (fight.value("status", std::string{"completed"}) != "completed") return;
    const auto actions = fight.at("actions").get<std::vector<std::uint32_t>>();
    // Reject a desynced fight before emitting any rows. This is the canonical replay implementation.
    const auto end = stsrl::combat_v4::replay(fight.at("start"), actions);
    const bool won = end.outcome == sts::Outcome::PLAYER_VICTORY;
    if (won != fight.at("won").get<bool>() || end.player.curHp != fight.at("final_hp").get<int>())
        throw std::runtime_error{"PV: recorded outcome did not replay"};
    std::map<std::size_t, const Json*> searched;
    for (const auto& row : fight.at("search"))
        if (!searched.emplace(row.at("step").get<std::size_t>(), &row).second)
            throw std::runtime_error{"PV: several search rows for one decision"};
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    const float value = pv::CombatObjective::terminal_value(end);
    for (std::size_t step = 0; step < actions.size(); ++step) {
        const auto moves = pv::legal_actions(state);
        std::vector<std::uint32_t> bits;
        for (const auto action : moves) bits.push_back(action.bits);
        std::vector<float> visits(moves.size(), 0);
        double total = 0, matched = 0;
        const Json* row = searched.contains(step) ? searched.at(step) : nullptr;
        if (row)
            for (const auto& child : row->at("children")) {
                const double count = child.at("visits");
                if (!std::isfinite(count) || count < 0) throw std::runtime_error{"PV: invalid recorded visit count"};
                total += count;
                for (std::size_t a = 0; a < bits.size(); ++a) if (bits[a] == child.at("action")) {
                    visits[a] += count; matched += count; break;
                }
            }
        if (matched != total) throw std::runtime_error{"PV: recorded search visits do not match legal actions"};
        if (total > 0) for (auto& v : visits) v /= total;
        const auto inputs = pv::encode(state, moves);
        std::optional<float> root;
        if (row && !row->at("root_value").is_null()) root = row->at("root_value").get<float>();
        writer.add({fight.at("fight_id").get<std::string>(), fight.at("start").at("seed").get<std::uint64_t>(),
                    fight.at("start").at("encounter").get<int>(), int(step), state.turn,
                    won, end.player.curHp, value, root, bits, visits, total > 0, &inputs});
        sts::search::Action{actions[step]}.execute(state);
    }
}

// Result line of one fight. Only a completed fight has a combat_v4 row; a capped one is not a loss.
Json result(const Json& fight, const std::vector<std::uint32_t>& actions, bool done, bool won, int hp,
            const std::string& agent, Json search, Json stats) {
    Json out{{"search", std::move(search)}, {"stats", std::move(stats)}, {"status", done ? "completed" : "capped"}};
    if (done)
        out["fight"] = {{"fight_id", fight.at("fight_id")}, {"version", stsrl::combat_v4::version}, {"start", fight.at("start")},
                        {"actions", actions}, {"explored", std::vector<bool>(actions.size(), false)}, {"won", won},
                        {"final_hp", hp}, {"agent", agent}};
    return out;
}

void check_replay(const Json& fight, const std::vector<std::uint32_t>& actions, const sts::BattleContext& end) {
    const auto rebuilt = stsrl::combat_v4::replay(fight.at("start"), actions);
    if (rebuilt.outcome != end.outcome || rebuilt.player.curHp != end.player.curHp)
        throw std::runtime_error{"PV: recorded actions do not rebuild the played fight"};
}

// PV plays directly on a BattleContext with its own legal-action list (the tree's root edges), so recorded bits replay.
Json play(const Json& fight, pv::Evaluator& evaluator, std::int64_t simulations, bool exploration, double c,
          double rollout_mix, bool oracle, bool policy_only, bool sample_turns, const std::string& agent,
          const pv::SearchSettings::RootHalving& root_halving = {}) {
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    pv::SearchSettings settings;
    settings.root_halving = root_halving;
    settings.noise_fraction = exploration ? 0.25 : 0;
    settings.rollout_mix = rollout_mix;
    settings.oracle = oracle;
    std::unique_ptr<pv::Search<>> tree;
    std::mt19937_64 move_random{fight.at("start").at("seed").get<std::uint64_t>()};
    Json search = Json::array(), stats = Json::array();
    std::vector<std::uint32_t> actions;
    while (state.outcome == sts::Outcome::UNDECIDED && state.turn < 50 && actions.size() < 512) {
        const auto moves = pv::legal_actions(state);
        auto chosen = moves.front();
        if (moves.size() > 1 && policy_only) {
            const auto t = Clock::now();
            const auto prediction = pv::evaluate_root(evaluator, state);
            chosen = moves[std::max_element(prediction.logits.begin(), prediction.logits.end()) - prediction.logits.begin()];
            stats.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"seconds", since(t)},
                             {"simulations", 0}, {"mean_depth", nullptr}, {"max_depth", nullptr},
                             {"mean_turns", nullptr}, {"max_turns", nullptr}, {"nodes", 0}});
        } else if (moves.size() > 1) {
            const auto t = Clock::now();
            // Evaluate the real root exactly once, then construct a fully initialized PV-only tree.
            if (!tree) {
                const auto root_prediction = pv::evaluate_root(evaluator, state);
                auto particles = oracle ? std::vector<sts::BattleContext>{state} : stsrl::teacher::public_particles(state, 8);
                tree = std::make_unique<pv::Search<>>(state, std::move(particles), root_prediction, settings, pv::PuctScore{c});
            }
            tree->run([&evaluator](std::span<const pv::Inputs> batch) { return evaluator.evaluate(batch); }, simulations);
            chosen = sample_turns && state.turn < 2 ? tree->sampled_action(move_random) : tree->selected_action();
            Json children = Json::array();
            double value_sum = 0, visits = 0;
            for (const auto& edge : tree->root().edges) if (edge.visits) {
                children.push_back({{"action", edge.action.bits}, {"visits", edge.visits}, {"value", edge.value_sum / edge.visits}});
                value_sum += edge.value_sum; visits += edge.visits;
            }
            const auto& d = tree->telemetry();
            const double n = tree->simulations();
            search.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"agent", agent},
                              {"root_value", value_sum / visits}, {"simulations", tree->simulations()}, {"children", children}});
            if (root_halving.enabled) {  // evaluation-only provenance; visits here are halving allocations, not targets
                Json improved = Json::array(); const auto pi = tree->improved_policy();
                for (std::size_t i = 0; i < pi.size(); ++i) improved.push_back({{"action", tree->root().edges[i].action.bits}, {"p", pi[i]}});
                search.back()["root_halving"] = true;
                search.back()["improved_policy_eval_only"] = std::move(improved);
                search.back()["phase_flushes"] = d.phase_flushes;
            }
            stats.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"seconds", since(t)},
                             {"simulations", tree->simulations()}, {"mean_depth", d.depth_sum / n}, {"max_depth", d.depth_max},
                             {"mean_turns", d.turns_sum / n}, {"max_turns", d.turns_max}, {"nodes", d.nodes}, {"turns_hist", d.turns_hist}});
        }
        actions.push_back(chosen.bits); chosen.execute(state);
        if (tree && (!oracle || state.outcome != sts::Outcome::UNDECIDED || !tree->advance(chosen, state))) tree.reset();
    }
    if (state.outcome == sts::Outcome::UNDECIDED && (state.turn >= 50 || actions.size() >= 512)) {
        state.outcome = sts::Outcome::PLAYER_LOSS;
        state.player.curHp = 0;
    }
    const bool done = state.outcome != sts::Outcome::UNDECIDED;
    if (done) check_replay(fight, actions, state);
    return result(fight, actions, done, state.outcome == sts::Outcome::PLAYER_VICTORY, state.player.curHp, agent, search, stats);
}

// Turn-search evaluation mode deliberately emits no invented per-action policy targets.
Json play_turn(const Json& fight, pv::Evaluator& evaluator, std::size_t budget, double c,
               pv::TurnSearchCaps caps, const std::string& agent, bool targets = false,
               bool exploration = false, bool sample_turns = false) {
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    std::mt19937_64 random{fight.at("start").at("seed").get<std::uint64_t>()};
    std::unique_ptr<pv::TurnSearch> tree;
    std::vector<std::uint32_t> actions;
    Json search = Json::array(), stats = Json::array(), turns = Json::array();
    while (state.outcome == sts::Outcome::UNDECIDED && state.turn < 50 && actions.size() < 512) {
        const auto started = Clock::now(); const auto step = actions.size(); const int turn = state.turn;
        if (!tree) tree = std::make_unique<pv::TurnSearch>(state, caps, c);
        const auto sequence = tree->decide(budget, [&](std::span<const pv::Inputs> batch) { return evaluator.evaluate(batch); },
            targets && exploration ? 0.25 : 0, targets ? &random : nullptr, targets && sample_turns && state.turn < 2);
        auto telemetry = tree->stats();
        if (!telemetry.fallback) {
            std::vector<std::uint32_t> prefix;
            for (auto bits : sequence) {
                if (actions.size() >= 512) break;
                const sts::search::Action chosen{bits};
                if (!chosen.isValidAction(state)) throw std::runtime_error{"turn search: live sequence illegal"};
                if (targets) {
                    const auto target = pv::turn_prefix_target(tree->root(), prefix, state);
                    Json children = Json::array();
                    for (const auto& child : target.children)
                        children.push_back({{"action", child.action}, {"visits", child.visits}, {"value", child.value}});
                    search.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"agent", agent},
                        {"root_value", target.value}, {"simulations", target.visits}, {"children", std::move(children)}});
                }
                actions.push_back(bits); chosen.execute(state); prefix.push_back(bits);
            }
            if (!tree->advance(state)) tree.reset();
        } else {
            tree.reset();
            pv::SearchSettings settings; settings.oracle = true;
            settings.noise_fraction = targets && exploration ? 0.25 : 0;
            std::unique_ptr<pv::Search<>> fallback;
            do {
                const auto moves = pv::legal_actions(state); auto chosen = moves.front();
                if (moves.size() > 1) {
                    const auto t = Clock::now();
                    if (!fallback) {
                        const auto prediction = pv::evaluate_root(evaluator, state);
                        ++telemetry.network_calls; ++telemetry.evaluated_states;
                        fallback = std::make_unique<pv::Search<>>(state, std::vector<sts::BattleContext>{state}, prediction, settings, pv::PuctScore{c});
                    }
                    fallback->run([&](std::span<const pv::Inputs> batch) {
                        ++telemetry.network_calls; telemetry.evaluated_states += batch.size();
                        return evaluator.evaluate(batch);
                    }, 800);
                    chosen = targets && sample_turns && state.turn < 2 ? fallback->sampled_action(random) : fallback->selected_action();
                    Json children = Json::array(); double value = 0, visits = 0;
                    for (const auto& edge : fallback->root().edges) if (edge.visits) {
                        children.push_back({{"action", edge.action.bits}, {"visits", edge.visits}, {"value", edge.value_sum / edge.visits}});
                        value += edge.value_sum; visits += edge.visits;
                    }
                    const auto& d = fallback->telemetry(); const double n = fallback->simulations();
                    search.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"agent", agent},
                        {"root_value", value / visits}, {"simulations", fallback->simulations()}, {"children", children}});
                    stats.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"seconds", since(t)},
                        {"simulations", fallback->simulations()}, {"mean_depth", d.depth_sum / n}, {"max_depth", d.depth_max},
                        {"mean_turns", d.turns_sum / n}, {"max_turns", d.turns_max}, {"nodes", d.nodes}, {"turns_hist", d.turns_hist}});
                }
                actions.push_back(chosen.bits); chosen.execute(state);
                if (fallback && (state.outcome != sts::Outcome::UNDECIDED || !fallback->advance(chosen, state))) fallback.reset();
            } while (state.outcome == sts::Outcome::UNDECIDED && state.turn < 50 && actions.size() < 512 &&
                     !(state.turn != turn && state.inputState == sts::InputState::PLAYER_NORMAL));
        }
        turns.push_back({{"fight_id", fight.at("fight_id")}, {"step", step}, {"turn", turn}, {"seconds", since(started)},
            {"expansions", telemetry.expansions}, {"network_calls", telemetry.network_calls}, {"evaluated_states", telemetry.evaluated_states},
            {"root_children", telemetry.root_children}, {"max_depth", telemetry.max_depth}, {"mean_leaf_depth", telemetry.mean_leaf_depth},
            {"fallback", telemetry.fallback}, {"reason", telemetry.reason}, {"time_overshoot", telemetry.time_overshoot}, {"reused", telemetry.reused}});
    }
    if (state.outcome == sts::Outcome::UNDECIDED && (state.turn >= 50 || actions.size() >= 512)) {
        state.outcome = sts::Outcome::PLAYER_LOSS;
        state.player.curHp = 0;
    }
    const bool done = state.outcome != sts::Outcome::UNDECIDED;
    if (done) check_replay(fight, actions, state);
    auto out = result(fight, actions, done, state.outcome == sts::Outcome::PLAYER_VICTORY, state.player.curHp, agent, search, stats);
    out["turn_stats"] = std::move(turns);
    return out;
}

// Real-play PIMC: only the first public action is played, then belief is rebuilt.
Json play_turn_real(const Json& fight, pv::Evaluator& evaluator, std::size_t budget, int particles,
                    double c, pv::TurnSearchCaps caps, const std::string& agent) {
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    std::vector<std::uint32_t> actions; Json decisions = Json::array();
    while (state.outcome == sts::Outcome::UNDECIDED && state.turn < 50 && actions.size() < 512) {
        const auto moves = pv::legal_actions(state); auto chosen = moves.front();
        if (moves.size() > 1) {
            const auto started = Clock::now();
            const auto belief = stsrl::teacher::public_particles(state, particles);
            const auto decision = pv::decide_turn_particles(state, belief, budget,
                [&](std::span<const pv::Inputs> batch) { return evaluator.evaluate(batch); }, caps, c);
            chosen = decision.action;
            decisions.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"turn", state.turn},
                {"particles", particles}, {"seconds", since(started)}, {"mean_depth", decision.mean_depth},
                {"fallback_count", decision.fallback_count}, {"time_fallback_count", decision.time_fallback_count},
                {"time_overshoot_count", decision.time_overshoot_count}, {"fallback_reasons", decision.fallback_reasons},
                {"network_calls", decision.network_calls},
                {"evaluated_states", decision.evaluated_states}});
        }
        if (!chosen.isValidAction(state)) throw std::runtime_error{"turn PIMC: real action illegal"};
        actions.push_back(chosen.bits); chosen.execute(state);
    }
    if (state.outcome == sts::Outcome::UNDECIDED && (state.turn >= 50 || actions.size() >= 512)) {
        state.outcome = sts::Outcome::PLAYER_LOSS;
        state.player.curHp = 0;
    }
    const bool done = state.outcome != sts::Outcome::UNDECIDED;
    if (done) check_replay(fight, actions, state);
    auto out = result(fight, actions, done, state.outcome == sts::Outcome::PLAYER_VICTORY, state.player.curHp,
                      agent, Json::array(), Json::array());
    out["particle_stats"] = std::move(decisions); return out;
}

// Shared current-turn plans only: stop before any particle-specific continuation.
Json play_real_turn(const Json& fight, pv::Evaluator& evaluator, int particles, double c,
                    pv::RealTurnCaps caps, const std::string& agent) {
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    std::vector<std::uint32_t> actions; Json search=Json::array(), stats=Json::array(), turns=Json::array();
    while(state.outcome==sts::Outcome::UNDECIDED && state.turn<50 && actions.size()<512) {
        const auto moves=pv::legal_actions(state); auto chosen=moves.front();
        if(moves.size()>1) {
            const auto started=Clock::now();
            auto belief=stsrl::teacher::public_particles(state,particles);
            auto remaining=caps;remaining.max_seconds-=since(started);pv::RealTurnResult decision;
            if(remaining.max_seconds<=0){decision.stats.fallback=true;decision.stats.reason="seconds";decision.stats.time_overshoot=true;}
            else decision=pv::decide_real_turn(state,belief,[&](std::span<const pv::Inputs> batch){return evaluator.evaluate(batch);},remaining);
            const auto& d=decision.stats;
            if(d.fallback) {
                const auto fallback_start=Clock::now();
                pv::Search<> fallback{state,stsrl::teacher::public_particles(state,8),pv::evaluate_root(evaluator,state),
                                      pv::SearchSettings{},pv::PuctScore{c}};
                fallback.run([&](std::span<const pv::Inputs> batch){return evaluator.evaluate(batch);},2000);
                chosen=fallback.selected_action(); Json children=Json::array(); double sum=0,visits=0;
                for(const auto& edge:fallback.root().edges)if(edge.visits){
                    children.push_back({{"action",edge.action.bits},{"visits",edge.visits},{"value",edge.value_sum/edge.visits}});
                    sum+=edge.value_sum;visits+=edge.visits;
                }
                search.push_back({{"fight_id",fight.at("fight_id")},{"step",actions.size()},{"agent",agent},
                    {"root_value",sum/visits},{"simulations",2000},{"children",children}});
                const auto& t=fallback.telemetry();
                stats.push_back({{"fight_id",fight.at("fight_id")},{"step",actions.size()},{"seconds",since(fallback_start)},
                    {"simulations",2000},{"mean_depth",t.depth_sum/2000.},{"max_depth",t.depth_max},
                    {"mean_turns",t.turns_sum/2000.},{"max_turns",t.turns_max},{"nodes",t.nodes},{"turns_hist",t.turns_hist}});
            } else chosen=decision.action;
            turns.push_back({{"fight_id",fight.at("fight_id")},{"step",actions.size()},{"turn",state.turn},
                {"particles",particles},{"seconds",since(started)},{"sequences",d.sequences},{"leaves",d.leaves},
                {"network_calls",d.network_calls},{"evaluated_states",d.evaluated_states},
                {"mean_leaf_depth",d.fallback?Json(nullptr):Json(d.mean_leaf_depth)},
                {"end_turn_fraction",d.fallback?Json(nullptr):Json(double(d.end_turn_leaves)/d.leaves)},
                {"reveal_fraction",d.fallback?Json(nullptr):Json(double(d.reveal_leaves)/d.leaves)},
                {"terminal_fraction",d.fallback?Json(nullptr):Json(double(d.terminal_leaves)/d.leaves)},
                {"end_turn_leaves",d.end_turn_leaves},{"reveal_leaves",d.reveal_leaves},{"terminal_leaves",d.terminal_leaves},
                {"fallback",d.fallback},{"reason",d.reason},{"time_overshoot",d.time_overshoot}});
        }
        if(!chosen.isValidAction(state))throw std::runtime_error{"real-turn: chosen real action illegal"};
        actions.push_back(chosen.bits);chosen.execute(state);
    }
    if (state.outcome == sts::Outcome::UNDECIDED && (state.turn >= 50 || actions.size() >= 512)) {
        state.outcome = sts::Outcome::PLAYER_LOSS;
        state.player.curHp = 0;
    }
    const bool done=state.outcome!=sts::Outcome::UNDECIDED;if(done)check_replay(fight,actions,state);
    auto out=result(fight,actions,done,state.outcome==sts::Outcome::PLAYER_VICTORY,state.player.curHp,agent,search,stats);
    out["real_turn_stats"]=std::move(turns);return out;
}

// The guided-rollout MCTS teacher (apps/combat_record/worker.cpp settings). Depth telemetry is PV-only: null here.
Json teach(const Json& fight, const stsrl::teacher::SearchFn& searcher, const std::string& agent) {
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    stsrl::CombatEnvironment env{std::move(state)};
    Json search = Json::array(), stats = Json::array();
    std::vector<std::uint32_t> actions;
    while (!env.done()) {
        const auto n = env.legal_action_count();
        if (n == 0) throw std::runtime_error{"no legal actions in an undecided fight"};
        std::size_t chosen = 0;
        if (n > 1) {
            const auto t = Clock::now();
            const auto d = stsrl::teacher::search_decision(env, n, searcher, false, stsrl::teacher::particles);
            const double seconds = since(t);
            chosen = d.chosen;
            Json children = Json::array();
            for (const auto& c : d.tried)
                children.push_back({{"action", env.action_bits(c.index)}, {"visits", c.visits}, {"value", c.value}});
            search.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"agent", agent},
                              {"root_value", d.value}, {"simulations", d.used}, {"children", children}});
            stats.push_back({{"fight_id", fight.at("fight_id")}, {"step", actions.size()}, {"seconds", seconds},
                             {"simulations", d.used}, {"mean_depth", nullptr}, {"max_depth", nullptr},
                             {"mean_turns", nullptr}, {"max_turns", nullptr}, {"nodes", nullptr}});
        }
        actions.push_back(env.action_bits(chosen)); env.step(chosen);
    }
    check_replay(fight, actions, env.battle());
    return result(fight, actions, true, env.won(), env.player_hp(), agent, search, stats);
}

int main(int argc, char** argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "";
        if (argc == 3 && command == "encode") {
            pv::ShardWriter writer{argv[2]};
            std::string line;
            while (std::getline(std::cin, line)) if (!line.empty()) encode(Json::parse(line), writer);
            writer.close();
        } else if (argc == 2 && command == "turns") {
            std::string line;
            while (std::getline(std::cin, line)) if (!line.empty()) pv::measure_turns(Json::parse(line));
        } else if (argc == 3 && command == "evaluate") {
            pv::Evaluator evaluator{argv[2]};
            std::string line;
            while (std::getline(std::cin, line)) {
                const auto states = Json::parse(line);
                std::vector<pv::Inputs> batch;
                for (const auto& state : states) {
                    pv::Inputs inputs;
                    for (std::size_t i = 0; i < pv::names.size(); ++i)
                        if (i == 0) inputs[i] = state.at(pv::names[i]).get<std::vector<float>>();
                        else for (const auto& row : state.at(pv::names[i])) {
                            const auto data = row.get<std::vector<float>>();
                            inputs[i].insert(inputs[i].end(), data.begin(), data.end());
                        }
                    batch.push_back(std::move(inputs));
                }
                Json out = Json::array();
                for (const auto& p : evaluator.evaluate(batch)) out.push_back({{"value", p.value}, {"logits", p.logits}});
                std::cout << out.dump() << std::endl;
            }
        } else if (argc >= 4 && command == "play") {
            bool exploration = false, oracle = false, policy_only = false, sample_turns = false, turn_search = false, turn_targets = false, real_turn = false;
            pv::TurnSearchCaps caps; pv::RealTurnCaps real_caps;
            int particles = 4; bool particles_set = false;
            double c = pv::PuctScore{}.exploration, rollout_mix = 0;
            pv::SearchSettings::RootHalving root_halving; bool halving_option = false;
            for (int i = 4; i < argc; ++i) {
                const std::string option = argv[i];
                if (option == "--explore") exploration = true;
                else if (option == "--root-halving") root_halving.enabled = true;
                else if (option == "--halving-m" && i + 1 < argc) { root_halving.max_considered = std::stoi(argv[++i]); halving_option = true; }
                else if (option == "--gumbel-scale" && i + 1 < argc) { root_halving.gumbel_scale = std::stod(argv[++i]); halving_option = true; }
                else if (option == "--oracle") oracle = true;
                else if (option == "--real-turn") real_turn = true;
                else if (option == "--real-turn-max-sequences" && i+1<argc) real_caps.max_sequences=std::stoull(argv[++i]);
                else if (option == "--real-turn-max-leaves" && i+1<argc) real_caps.max_leaves=std::stoull(argv[++i]);
                else if (option == "--real-turn-max-seconds" && i+1<argc) real_caps.max_seconds=std::stod(argv[++i]);
                else if (option == "--turn-search") turn_search = true;
                else if (option == "--turn-targets") turn_targets = true;
                else if (option == "--particles" && i + 1 < argc) { particles = std::stoi(argv[++i]); particles_set = true; }
                else if (option == "--turn-max-sequences" && i + 1 < argc) caps.max_sequences = std::stoull(argv[++i]);
                else if (option == "--turn-root-max-children" && i + 1 < argc) caps.root_max_children = std::stoull(argv[++i]);
                else if (option == "--turn-max-children" && i + 1 < argc) caps.max_children = std::stoull(argv[++i]);
                else if (option == "--turn-max-seconds" && i + 1 < argc) caps.max_seconds = std::stod(argv[++i]);
                else if (option == "--policy-only") policy_only = true;
                else if (option == "--sample-turns") sample_turns = true;
                else if (option == "--c" && i + 1 < argc) c = std::stod(argv[++i]);
                else if (option == "--rollout-mix" && i + 1 < argc) rollout_mix = std::stod(argv[++i]);
                else throw std::invalid_argument{"unknown play option " + option};
            }
            if (halving_option && !root_halving.enabled) throw std::invalid_argument{"--halving-m/--gumbel-scale require --root-halving"};
            if (root_halving.enabled && (exploration || sample_turns || oracle || policy_only || turn_search || turn_targets || real_turn))
                throw std::invalid_argument{"--root-halving is evaluation-only: no --explore/--sample-turns/--oracle/--policy-only/turn modes"};
            if(real_turn && (oracle || turn_search || turn_targets || exploration || sample_turns || policy_only || rollout_mix!=0))
                throw std::invalid_argument{"--real-turn requires real evaluation-only settings without --turn-search"};
            if(real_turn && !particles_set)particles=8;
            if (turn_targets && (!turn_search || !oracle || std::stoll(argv[3]) < 2))
                throw std::invalid_argument{"--turn-targets requires --oracle --turn-search and E >= 2"};
            if (turn_search && (policy_only || rollout_mix != 0 || ((exploration || sample_turns) && !turn_targets)))
                throw std::invalid_argument{"--turn-search requires evaluation-only settings"};
            if (particles_set && ((!turn_search && !real_turn) || oracle))
                throw std::invalid_argument{"--particles requires real --turn-search or --real-turn (without --oracle)"};
            if (particles < 1) throw std::invalid_argument{"--particles must be positive"};
            pv::Evaluator evaluator{argv[2]};
            const auto simulations = std::stoll(argv[3]);
            if (simulations < 1) throw std::invalid_argument{"PV needs at least one simulation"};
            std::string agent = "pv model=" + std::filesystem::path{argv[2]}.parent_path().filename().string() + "/" +
                std::filesystem::path{argv[2]}.filename().string() + " sims=" + std::to_string(simulations) +
                " explore=" + (exploration ? "0.25" : "0") + " c=" + std::to_string(c) + " q=minmax particles=" + (oracle ? "1" : (turn_search || real_turn) ? std::to_string(particles) : "8") + " batch=" +
                std::to_string(pv::SearchSettings{}.batch_size) +
                (rollout_mix > 0 ? " rollout_mix=" + std::to_string(rollout_mix) : "") +
                (oracle ? " oracle=1 reuse=1" : "") + (policy_only ? " policy_only=1" : "") +
                (sample_turns ? " sample_turns=2" : "") +
                (root_halving.enabled ? " root_halving=m" + std::to_string(root_halving.max_considered) + ",g" + std::to_string(root_halving.gumbel_scale) : "");
            if (turn_search) agent += " turn_search=1 E=" + std::to_string(simulations) +
                " caps=seq" + std::to_string(caps.max_sequences) + "/root" + std::to_string(caps.root_max_children) +
                "/node" + std::to_string(caps.max_children) +
                "/" + std::to_string(caps.max_seconds) + "s/512a/256MiB T=10 fallback_sims=800";
            if(real_turn)agent += " real_turn=1 reuse=0 fallback_sims=2000 caps=seq"+std::to_string(real_caps.max_sequences)+"/leaves"+std::to_string(real_caps.max_leaves)+"/"+std::to_string(real_caps.max_seconds)+"s/512a/256MiB";
            if (turn_targets) agent += " turn_targets=1";
            if (turn_search && !oracle) agent += " pimc=1 reuse=0";
            std::string line;
            while (std::getline(std::cin, line)) if (!line.empty()) {
                const auto fight = Json::parse(line);
                if(real_turn){std::cout<<play_real_turn(fight,evaluator,particles,c,real_caps,agent).dump()<<std::endl;continue;}
                if (turn_search && !oracle) {
                    std::cout << play_turn_real(fight, evaluator, simulations, particles, c, caps, agent).dump() << std::endl;
                    continue;
                }
                std::cout << (turn_search ? play_turn(fight, evaluator, simulations, c, caps, agent, turn_targets, exploration, sample_turns) :
                    play(fight, evaluator, simulations, exploration, c, rollout_mix, oracle, policy_only, sample_turns, agent, root_halving)).dump() << std::endl;
            }
        } else if (argc == 3 && command == "teacher") {
            const auto sims = std::stoll(argv[2]);
            const auto searcher = stsrl::teacher::leaf_search({"guided_rollout", 0, 0}, nullptr, {sims, stsrl::teacher::particles});
            const std::string agent = "mcts leaf=guided_rollout sims=" + std::to_string(sims) +
                                      " particles=" + std::to_string(stsrl::teacher::particles);
            std::string line;
            while (std::getline(std::cin, line)) if (!line.empty())
                std::cout << teach(Json::parse(line), searcher, agent).dump() << std::endl;
        } else {
            std::cerr << "usage: pv_worker encode OUT.parquet | turns | evaluate MODEL.onnx | play MODEL.onnx SIMS [--explore] [--oracle] [--turn-search] [--policy-only] [--sample-turns] [--c C] [--rollout-mix L] [--root-halving [--halving-m M] [--gumbel-scale S]] | teacher SIMS (JSONL stdin)\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
