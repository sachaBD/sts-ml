// The teacher's search and its leaf strategies: PublicBeliefCombatSearch (8 particles, rollout mode 2)
// with the teacher budget (forced moves, early stop). Leaves are scored by guided rollout (bootstrap),
// by the value net (immediate), or by a bounded guided rollout then the value net (hybrid).
// Recording a fight: agents/teacher_search.hpp.
#pragma once

#include "combat/environment.hpp"
#include "topology/value_net.hpp"
#include "sim/search/PublicBeliefCombatSearch.h"

#include <cstdint>
#include <functional>
#include <limits>
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

// Search size: simulation cap per decision and belief particles (the bootstrap teacher: simulations, particles
// above). Forced moves still get min(forced_simulations, simulations).
struct Budget {
    std::int64_t simulations;
    int particles;
};

// Opt-in search variants; defaults = the historical teacher. Process-wide: a worker sets them from its
// request's teacher settings (set_tweak) before searching (workers are single-threaded); settings()
// records the non-default ones. Measured in slop_docs/search_perf.md.
struct SearchTweaks {
    // Card plays / potions keyed by card (or potion) and target, not hand slot: identical cards in two
    // slots are one edge (PublicBeliefCombatSearch::mergeIdenticalCards).
    bool merge_identical_cards = false;
    // Early stop once N_best - N_second > stop_factor * simulations left. 1 = the exact rule (the most
    // visited move can no longer change); below 1 stops sooner, accepting that it rarely might have.
    double stop_factor = 1.0;
    // Leaf policy_net only (PublicBeliefCombatSearch::enablePolicyPriors): PUCT exploration constant, first-play
    // urgency reduction (values ~0..1), and the share of every node's prior spread uniformly over its moves
    // (priors = (1 - prior_floor) * softmax(policy) + prior_floor / moves: a zero-prior move is otherwise never
    // tried). Required settings of a policy_net request (no defaults); NaN = unset.
    double c_puct = std::numeric_limits<double>::quiet_NaN();
    double fpu_reduction = std::numeric_limits<double>::quiet_NaN();
    double prior_floor = std::numeric_limits<double>::quiet_NaN();
    // Mixed into every seed the search derives from the public observation (particle sampling and the search's
    // own RNG stream). 0 = the historical, unsalted seeds (bit-identical play); another value gives an equally
    // fair, independently seeded search of the same state (A/A noise floors, per-fight win probabilities).
    std::uint64_t search_salt = 0;
    // Tree reuse (play_fight only; the learner fight ignores it): keep the played move's subtree between
    // decisions (rebase_search), as play_fight's `reuse`. The budget counts the kept visits (see topped_up).
    bool tree_reuse = false;
};
SearchTweaks& tweaks();
// Applies teacher setting `key` if it is a tweak (merge_identical_cards: bool, stop_factor: number in
// (0, 1]); false for other keys. Throws on an invalid value.
bool set_tweak(const std::string& key, const nlohmann::json& value);

// The teacher's search at `observed`: particles sampled from the public observation. With `oracle`,
// a single particle that is `observed` itself (true RNG state and draw order) with max backup (the
// move played is the best-valued root edge; no early stop): perfect-information search of the
// deterministic simulator, an upper-bound teacher, not a fair player. `particles`: the first n of the
// same deterministic particle stream (ignored under oracle).
sts::search::PublicBeliefCombatSearch make_search(const sts::BattleContext& observed, bool oracle, int particles);

// Tree reuse: after `played_bits` was played at `before` (the search's root) and `after` is observed, keep
// the played move's subtree as the new root (PublicBeliefCombatSearch::rebase) with `after`'s particles
// (as make_search). The kept root visits count toward the next search's budget (topped_up).

// Top-up budget: the new simulations a search with cap `simulations` runs: the cap minus the root visits a
// rebased tree kept (search.retainedVisits; 0 for a fresh search, which runs the full cap), but at least a
// tenth of the cap. Forced moves keep min(forced_simulations, simulations). ~0.7x simulations on a boss
// fight (slop_docs/teacher_perf.md).
std::int64_t topped_up(const sts::search::PublicBeliefCombatSearch& search, std::int64_t simulations);
void rebase_search(sts::search::PublicBeliefCombatSearch& search, const sts::BattleContext& before,
                   std::uint32_t played_bits, const sts::BattleContext& after, bool oracle, int particles);

// Guided-rollout search with the teacher budget; returns simulations used. Plays the same move as
// search.search(simulations) with less compute:
//   forced: one legal move -> only forced_simulations.
//   early stop: search in chunks of `chunk`; stop once N_best - N_second > simulations left, so the
//   most-visited root edge can no longer be caught.
//   tree reuse: a rebased search runs topped_up(search, simulations) new simulations (a fresh one: all).
std::int64_t run_teacher_search(sts::search::PublicBeliefCombatSearch& search, std::int64_t simulations,
                                std::size_t legal_moves, bool early_stop);

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
                             std::int64_t simulations, std::size_t legal_moves, int rollout_turns,
                             int rollout_steps);

// Policy-prior search (leaf policy_net; the net must have a policy head): the search's policy-prior mode with
// objective 1 (35 + HP + 4 * potions on a win, 0 otherwise). Every leaf is evaluated once by the net: its
// value (converted to the search's normalization: value * (55 + leaf max HP) / (56 + objectiveMaxHp)) and
// priors over the leaf node's moves (tweaks c_puct / fpu_reduction / prior_floor). Same budget rules as
// run_leaf_search (forced moves, batches of value_net_batch, early stop between batches).
std::int64_t run_policy_net_search(sts::search::PublicBeliefCombatSearch& search, const ValueNet& net,
                                   std::int64_t simulations, std::size_t legal_moves);

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
//   policy_net: net with a policy head (deep_sets_v3), rollout bounds 0; run_policy_net_search.
struct Leaf {
    std::string kind;
    int rollout_turns = 0;
    int rollout_steps = 0;
};

// Throws std::invalid_argument on an unknown kind, bounds that don't fit the kind, or a net given to
// (or missing from) the kind.
void validate(const Leaf& leaf, bool has_net);

// The search for `leaf` (validated) with budget.simulations; `net` must outlive the result.
SearchFn leaf_search(const Leaf& leaf, const ValueNet* net, const Budget& budget);

SearchFn guided_rollout_search(std::int64_t simulations);
SearchFn value_net_search(const ValueNet& net, std::int64_t simulations);
SearchFn hybrid_search(const ValueNet& net, int rollout_turns, int rollout_steps, std::int64_t simulations);
SearchFn policy_net_search(const ValueNet& net, std::int64_t simulations);

// The search settings above for `leaf`: leaf, particles, simulations, early_stop, forced_simulations,
// max_actions, then chunk (guided_rollout) or batch (net leaves); hybrid adds rollout_turns /
// rollout_steps. particles / simulations come from `budget`. teacher::settings (teacher_search.hpp) adds
// the recording settings.
nlohmann::json search_settings(const Leaf& leaf, const Budget& budget);

}  // namespace stsrl::teacher
