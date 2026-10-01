// The teacher playing and recording one fight. Search and leaf strategies (guided rollout, value net,
// hybrid): agents/combat/search/teacher_leaves.hpp, re-exported here. Shared by apps/bootstrap (guided-rollout
// leaves) and apps/value_play. Row columns: runs/README.md.
#pragma once

#include "agents/combat/search/teacher_leaves.hpp"

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
    std::int64_t retained = 0;  // root visits already in the tree before this search (tree reuse)
};

SearchDecision search_decision(const CombatEnvironment& env, std::size_t legal_count, const SearchFn& run,
                               bool oracle, int particles);
// Same, continuing `search` (rooted at env's state; e.g. rebased by rebase_search).
SearchDecision search_decision(const CombatEnvironment& env, std::size_t legal_count, const SearchFn& run,
                               sts::search::PublicBeliefCombatSearch& search);

// combat_v3 outcome columns of a finished fight: won, final_hp, potions, terminal_value.
nlohmann::json outcome_columns(const CombatEnvironment& env, int max_hp);

// search_settings plus the recording settings (and `oracle`), as recorded in summary.json.
// Under oracle, particles is reported as 1 (make_search ignores budget.particles).
nlohmann::json settings(const Leaf& leaf, bool oracle, const Budget& budget);

// Process-wide wall seconds play_fight has spent choosing moves (make_search + search_decision; not the
// state encoding or row recording). A cost measure for the search alone; workers are single-threaded.
double& search_seconds();

// Teacher plays one fight. With `random_move`, one decision in [0, random_window), drawn from
// mt19937_64(episode_id ^ 0xe9510), plays a uniformly random legal move instead (was_random); without
// it every move is the search's. `oracle`: search the true state (make_search); every row gets
// oracle = true/false. `particles`: make_search's belief particles. Appends its decision rows, then its child rows, each = encoding +
// `fight` columns + search columns + outcome columns. Returns the finished battle.
// `reuse` (or the tree_reuse tweak): keep the played move's subtree between decisions (rebase_search); its visits
// count toward the budget (topped_up). Decision rows get retained_visits (simulations_used counts new simulations
// only).
sts::BattleContext play_fight(sts::BattleContext battle, const nlohmann::json& fight,
                              std::vector<nlohmann::json>& rows, const SearchFn& search, bool random_move,
                              bool oracle, int particles, bool reuse);

// Teacher plays one fight and records nothing (for callers that keep only the outcome): no rows, no state
// encoding, and a forced move (one legal move) is played without a search (the recording player searches it
// only for its root_value label). Without `reuse` every move is the recording player's (no random move):
// each search is fresh and seeded by its public observation alone. With `reuse`, as play_fight's reuse,
// minus the forced moves' simulations.
sts::BattleContext play_fight(sts::BattleContext battle, const SearchFn& search, bool oracle, int particles,
                              bool reuse);

// A learner plays one fight while a teacher labels it (apps/dagger). At each decision, separate searches
// of the same pre-action state: `teacher` gives actions / root_value / simulations_used, `learner` picks
// chosen_action, and only the learner's move is played. Both search fair play (no oracle) with `particles`.
// No random move, no child rows.
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
                                int particles, int max_decisions, int max_turns);

}  // namespace stsrl::teacher
