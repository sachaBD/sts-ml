// The teacher playing and recording one fight. Search and leaf strategies (guided rollout, value net,
// hybrid): agents/teacher_leaves.hpp, re-exported here. Shared by apps/bootstrap (guided-rollout
// leaves) and apps/value_play. Row columns: runs/README.md.
#pragma once

#include "agents/teacher_leaves.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace stsrl::teacher {

constexpr std::int64_t child_min_visits = 50;
constexpr int random_window = 24;                 // the random move falls in decisions [0, 24)

struct TriedAction {
    std::size_t index;
    std::int64_t visits;
    double value;
};

// One search at env's current state. actions: {action, description, visits, mean_value} per visited root
// edge; chosen: the most-visited edge's legal index (oracle / max backup: the best-valued edge's); value: sum of edge value sums / root visits (the
// visit-weighted average of the explored edges, not max-Q); used: simulations run.
struct SearchDecision {
    nlohmann::json actions = nlohmann::json::array();
    std::vector<TriedAction> tried;
    std::size_t chosen;
    double value;
    std::int64_t used;
};

SearchDecision search_decision(const CombatEnvironment& env, std::size_t legal_count, const SearchFn& run,
                               bool oracle = false);

// combat_v3 outcome columns of a finished fight: won, final_hp, potions, terminal_value.
nlohmann::json outcome_columns(const CombatEnvironment& env, int max_hp);

// search_settings plus the recording settings (and `oracle`), as recorded in summary.json.
nlohmann::json settings(const Leaf& leaf, bool oracle = false);
// leaf = "guided_rollout" or "value_net".
nlohmann::json settings(const std::string& leaf, bool oracle = false);

// Teacher plays one fight. With `random_move`, one decision in [0, random_window), drawn from
// mt19937_64(episode_id ^ 0xe9510), plays a uniformly random legal move instead (was_random); without
// it every move is the search's. `oracle`: search the true state (make_search); every row gets
// oracle = true/false. Appends its decision rows, then its child rows, each = encoding +
// `fight` columns + search columns + outcome columns. Returns the finished battle.
sts::BattleContext play_fight(sts::BattleContext battle, const nlohmann::json& fight,
                              std::vector<nlohmann::json>& rows, const SearchFn& search, bool random_move = true,
                              bool oracle = false);

// A learner plays one fight while a teacher labels it (apps/dagger). At each decision, separate searches
// of the same pre-action state: `teacher` gives actions / root_value / simulations_used, `learner` picks
// chosen_action, and only the learner's move is played. No random move, no child rows.
//   status: completed (the fight ended), capped (max_decisions reached) or turn_limit (turn max_turns
//   reached: e.g. block-stacking stalls). Only a completed fight appends its decision
//   rows (with the outcome columns) to `rows`; the others append nothing (not a loss).
//   start: decision 0's row without outcome (identity check), decisions: per-decision diagnostics
//   {decision_index, turn, learner_action, teacher_action, learner/teacher_root_value, learner/teacher_simulations}.
struct LearnerFight {
    sts::BattleContext battle;
    bool completed = false;
    std::string status;
    nlohmann::json start;
    std::vector<nlohmann::json> decisions;
};

LearnerFight play_learner_fight(sts::BattleContext battle, const nlohmann::json& fight,
                                std::vector<nlohmann::json>& rows, const SearchFn& learner, const SearchFn& teacher,
                                int max_decisions, int max_turns = 50);

}  // namespace stsrl::teacher
