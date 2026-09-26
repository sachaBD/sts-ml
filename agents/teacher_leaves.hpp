// The teacher's search and its leaf strategies: PublicBeliefCombatSearch (8 particles, rollout mode 2)
// with the teacher budget (forced moves, early stop). Leaves are scored by guided rollout (bootstrap),
// by the value net (immediate), or by a bounded guided rollout then the value net (hybrid).
// Recording a fight: agents/teacher_search.hpp.
#pragma once

#include "combat/environment.hpp"
#include "models/value_net.hpp"
#include "sim/search/PublicBeliefCombatSearch.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace stsrl::teacher {

constexpr std::int64_t simulations = 15'000;
constexpr int particles = 8;
constexpr int max_actions = 512;
constexpr std::int64_t forced_simulations = 500;  // one legal move: only for root_value / actions
constexpr std::int64_t chunk = 500;               // guided-rollout early-stop check interval
constexpr int value_net_batch = 64;               // value-net leaves per batch (and early-stop check)

// Search size: simulation cap per decision and belief particles (defaults: the teacher's 15k, 8).
// Forced moves still get min(forced_simulations, simulations).
struct Budget {
    std::int64_t simulations = teacher::simulations;
    int particles = teacher::particles;
};

// The teacher's search at `observed`: particles sampled from the public observation. With `oracle`,
// a single particle that is `observed` itself (true RNG state and draw order) with max backup (the
// move played is the best-valued root edge; no early stop): perfect-information search of the
// deterministic simulator, an upper-bound teacher, not a fair player. `particles`: the first n of the
// same deterministic particle stream (ignored under oracle).
sts::search::PublicBeliefCombatSearch make_search(const sts::BattleContext& observed, bool oracle = false,
                                                  int particles = teacher::particles);

// Tree reuse: after `played_bits` was played at `before` (the search's root) and `after` is observed, keep
// the played move's subtree as the new root (PublicBeliefCombatSearch::rebase) with `after`'s particles
// (as make_search). The budget still counts only new simulations, so each decision has at least a fresh
// search's evidence; kept visits only add to it (and can make early stop fire sooner).
void rebase_search(sts::search::PublicBeliefCombatSearch& search, const sts::BattleContext& before,
                   std::uint32_t played_bits, const sts::BattleContext& after, bool oracle = false,
                   int particles = teacher::particles);

// Guided-rollout search with the teacher budget; returns simulations used. Plays the same move as
// search.search(simulations) with less compute:
//   forced: one legal move -> only forced_simulations.
//   early stop: search in chunks; stop once N_best - N_second > simulations left, so the
//   most-visited root edge can no longer be caught.
std::int64_t run_teacher_search(sts::search::PublicBeliefCombatSearch& search, std::int64_t simulations,
                                std::size_t legal_moves, bool early_stop,
                                std::int64_t forced_simulations = teacher::forced_simulations,
                                std::int64_t chunk = teacher::chunk);

// Scores pending (non-terminal) leaf states: one value per leaf into `values`.
using LeafEvaluator =
    std::function<void(const std::vector<const sts::BattleContext*>& leaves, std::vector<float>& values)>;

// The value net on each leaf's encoding (same encoder as env.decision().encoding).
LeafEvaluator value_net_evaluator(const ValueNet& net);

// Same budget rules, but every non-terminal leaf is scored by `evaluate` (clamped to [0, 2]), in
// batches of value_net_batch; early stop is checked between batches. Before a leaf is left pending,
// a guided rollout runs from it for at most `rollout_turns` turn increments and `rollout_steps`
// actions (PublicBeliefCombatSearch::requestBatch); a rollout that ends the fight is backed up with
// its terminal value, not evaluated. 0, 0: the leaf is scored where it is expanded (immediate).
std::int64_t run_leaf_search(sts::search::PublicBeliefCombatSearch& search, const LeafEvaluator& evaluate,
                             std::int64_t simulations, std::size_t legal_moves, int rollout_turns = 0,
                             int rollout_steps = 0);

// run_leaf_search with the value net, immediate leaves.
std::int64_t run_value_net_search(sts::search::PublicBeliefCombatSearch& search, const ValueNet& net,
                                  std::int64_t simulations, std::size_t legal_moves);

// Index into env's legal actions of a search action.
std::size_t legal_index(const CombatEnvironment& env, std::size_t count,
                        const sts::search::PublicBeliefCombatSearch& search, sts::search::Action action);

// Runs the search on `search` with `legal_moves` legal moves; returns simulations used.
using SearchFn = std::function<std::int64_t(sts::search::PublicBeliefCombatSearch&, std::size_t legal_moves)>;

// Leaf strategy of the teacher search.
//   guided_rollout: no net, rollout bounds 0.
//   value_net: net, rollout bounds 0 (immediate).
//   hybrid: net, rollout_turns >= 1 and rollout_steps >= 1.
struct Leaf {
    std::string kind = "guided_rollout";
    int rollout_turns = 0;
    int rollout_steps = 0;
};

// Throws std::invalid_argument on an unknown kind, bounds that don't fit the kind, or a net given to
// (or missing from) the kind.
void validate(const Leaf& leaf, bool has_net);

// The search for `leaf` (validated) with budget.simulations; `net` must outlive the result.
SearchFn leaf_search(const Leaf& leaf, const ValueNet* net, const Budget& budget = {});

SearchFn guided_rollout_search(std::int64_t simulations = teacher::simulations);
SearchFn value_net_search(const ValueNet& net, std::int64_t simulations = teacher::simulations);
SearchFn hybrid_search(const ValueNet& net, int rollout_turns, int rollout_steps,
                       std::int64_t simulations = teacher::simulations);

// The search settings above for `leaf`: leaf, particles, simulations, early_stop, forced_simulations,
// max_actions, then chunk (guided_rollout) or batch (net leaves); hybrid adds rollout_turns /
// rollout_steps. particles / simulations come from `budget`. teacher::settings (teacher_search.hpp) adds
// the recording settings.
nlohmann::json search_settings(const Leaf& leaf, const Budget& budget = {});

}  // namespace stsrl::teacher
