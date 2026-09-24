#include "agents/teacher_search.hpp"

#include <random>
#include <stdexcept>
#include <utility>

namespace stsrl::teacher {
namespace {

using Json = nlohmann::json;
using sts::search::PublicBeliefCombatSearch;

}  // namespace

SearchDecision search_decision(const CombatEnvironment& env, std::size_t legal_count, const SearchFn& run) {
    auto search = make_search(env.battle());
    SearchDecision result;
    result.used = run(search, legal_count);
    result.chosen = legal_index(env, legal_count, search, search.selectedAction());
    double value_sum = 0;
    for (const auto& edge : search.root().edges) {
        value_sum += edge.valueSum;
        if (!edge.visits) continue;
        const auto index = legal_index(env, legal_count, search, edge.action);
        const auto mean = edge.valueSum / edge.visits;
        result.actions.push_back({{"action", index}, {"description", env.action_description(index)},
                                  {"visits", edge.visits}, {"mean_value", mean}});
        result.tried.push_back({index, static_cast<std::int64_t>(edge.visits), mean});
    }
    result.value = search.root().visits ? value_sum / search.root().visits : 0.0;
    return result;
}

namespace {

void record_children(std::vector<Json>& rows, const CombatEnvironment& env, const SearchDecision& decision,
                     const Json& fight, int index) {
    for (const auto& tried : decision.tried) {
        if (tried.index == decision.chosen || tried.visits < child_min_visits) continue;
        CombatEnvironment child{env.battle()};
        (void)child.decision();
        child.step(tried.index);
        if (child.done()) continue;
        Json row = child.decision().encoding;
        row.update(fight);
        row.update({{"decision_index", index}, {"turn", child.battle().turn}, {"actions", Json::array()},
                    {"chosen_action", -1}, {"was_random", false}, {"root_value", tried.value},
                    {"row_kind", "child"}, {"parent_action", tried.index}, {"simulations_used", 0}});
        rows.push_back(std::move(row));
    }
}

}  // namespace

Json outcome_columns(const CombatEnvironment& env, int max_hp) {
    const int hp = env.player_hp(), potions = env.battle().potionCount;
    const double terminal = env.won() ? (35.0 + hp + 4.0 * potions) / (55.0 + max_hp) : 0.0;
    return {{"won", env.won()}, {"final_hp", hp}, {"potions", potions}, {"terminal_value", terminal}};
}

Json settings(const Leaf& leaf) {
    auto result = search_settings(leaf);
    result["child_min_visits"] = child_min_visits;
    result["random_window"] = random_window;
    return result;
}

Json settings(const std::string& leaf) { return settings(Leaf{leaf}); }

sts::BattleContext play_fight(sts::BattleContext battle, const Json& fight, std::vector<Json>& rows,
                              const SearchFn& run, bool random_move) {
    CombatEnvironment env{std::move(battle)};
    const int max_hp = env.player_max_hp();
    std::mt19937_64 rng(fight.at("episode_id").get<std::uint64_t>() ^ 0xe9510ULL);
    const int random_at = random_move ? std::uniform_int_distribution<int>{0, random_window - 1}(rng) : -1;
    std::vector<Json> decisions, children;
    int index = 0;
    while (!env.done()) {
        auto state = env.decision();
        auto choice = search_decision(env, state.legal_actions.size(), run);
        const bool random = index == random_at;
        if (random) choice.chosen = std::uniform_int_distribution<std::size_t>{0, state.legal_actions.size() - 1}(rng);
        record_children(children, env, choice, fight, index);
        Json row = state.encoding;
        row.update(fight);
        row.update({{"decision_index", index}, {"turn", env.battle().turn}, {"actions", std::move(choice.actions)},
                    {"chosen_action", choice.chosen}, {"was_random", random}, {"root_value", choice.value},
                    {"row_kind", "decision"}, {"parent_action", -1}, {"simulations_used", choice.used}});
        decisions.push_back(std::move(row));
        env.step(choice.chosen);
        ++index;
    }
    const Json outcome = outcome_columns(env, max_hp);
    for (auto* part : {&decisions, &children})
        for (auto& row : *part) {
            row.update(outcome);
            rows.push_back(std::move(row));
        }
    return env.battle();
}

LearnerFight play_learner_fight(sts::BattleContext battle, const Json& fight, std::vector<Json>& rows,
                                const SearchFn& learner, const SearchFn& teacher, int max_decisions, int max_turns) {
    if (max_decisions < 1 || max_turns < 1) throw std::invalid_argument{"max_decisions and max_turns must be >= 1"};
    CombatEnvironment env{std::move(battle)};
    const int max_hp = env.player_max_hp();
    LearnerFight result;
    std::vector<Json> decisions;
    int index = 0;
    while (!env.done() && index < max_decisions && env.battle().turn < max_turns) {
        auto state = env.decision();
        const auto legal = state.legal_actions.size();
        // Two separate searches of the same unchanged state; the teacher's result never reaches the learner's.
        const auto label = search_decision(env, legal, teacher);
        const auto move = search_decision(env, legal, learner);
        Json row = state.encoding;
        row.update(fight);
        row.update({{"decision_index", index}, {"turn", env.battle().turn}, {"actions", label.actions},
                    {"chosen_action", move.chosen}, {"was_random", false}, {"root_value", label.value},
                    {"row_kind", "decision"}, {"parent_action", -1}, {"simulations_used", label.used}});
        if (index == 0) result.start = row;
        result.decisions.push_back({{"decision_index", index}, {"turn", env.battle().turn},
                                    {"learner_action", move.chosen}, {"teacher_action", label.chosen},
                                    {"learner_root_value", move.value}, {"teacher_root_value", label.value},
                                    {"learner_simulations", move.used}, {"teacher_simulations", label.used}});
        decisions.push_back(std::move(row));
        env.step(move.chosen);
        ++index;
    }
    result.completed = env.done();
    result.status = result.completed ? "completed" : index >= max_decisions ? "capped" : "turn_limit";
    if (result.completed) {
        const Json outcome = outcome_columns(env, max_hp);
        for (auto& row : decisions) {
            row.update(outcome);
            rows.push_back(std::move(row));
        }
    }
    result.battle = env.battle();
    return result;
}

}  // namespace stsrl::teacher
