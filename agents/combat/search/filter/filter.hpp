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
};
struct FilterTrace {
    int reserve = 0, spent = 0;
    double opening_rate = 0, last_rate = 0;
    Evaluation opening;
    std::vector<double> initial_scores;
    std::vector<FilterTrialTrace> trials;
    std::string stop;
};
// Deterministic expected improvement per request. Cap counts fresh non-root evaluations.
std::unique_ptr<CombatSearch> filter_search(const sts::BattleContext&, int cap, int worlds, double price,
                                            FilterMode mode, std::uint64_t stream, FilterTrace* trace = nullptr);
}
