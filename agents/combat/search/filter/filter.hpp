#pragma once
#include "agents/combat/search/search.hpp"

namespace stsrl::search {
enum class FilterMode { full, root_only, fixed_uncertainty };
struct FilterTrialTrace {
    int arm, world, spent;
    bool interior;
    double rate, candidate, reference, mean_before, mean_after, innovation_before, innovation_after;
    std::vector<double> scores;
    std::vector<std::string> line, reference_line;
    std::vector<std::uint64_t> keys, reference_keys;  // observation key of each evaluated position along each line
};
// One network evaluation used by the search, keyed by public observation.
struct FilterNodeTrace {
    std::uint64_t key;
    float value;
    std::vector<float> logits, thinking;
    std::vector<std::string> actions;  // labels, aligned with logits
    bool previous;                     // answered from the previous decision's cache
};
struct FilterTrace {
    int reserve = 0, spent = 0;
    double opening_rate = 0, last_rate = 0;
    Evaluation opening;
    std::vector<double> initial_scores;
    std::vector<FilterTrialTrace> trials;
    std::string stop;
    std::uint64_t root = 0;
    std::vector<FilterNodeTrace> nodes;  // every evaluation the search used, in arrival order
};
// Deterministic expected improvement per request. Cap counts fresh non-root evaluations.
// explore = 0 plays the leader; explore > 0 plays argmax(log policy + explore * Gumbel), seeded by explore_seed,
// so explore = 1 samples the move from the search policy.
std::unique_ptr<CombatSearch> filter_search(const sts::BattleContext&, int cap, int worlds, double price,
                                            FilterMode mode, std::uint64_t stream, FilterTrace* trace = nullptr,
                                            double explore = 0, std::uint64_t explore_seed = 0);
}
