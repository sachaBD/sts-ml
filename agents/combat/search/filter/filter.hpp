#pragma once
#include "agents/combat/search/search.hpp"

namespace stsrl::search {
enum class FilterMode { full, root_only, fixed_uncertainty };

// One node in the search tree (root = id 0; children created as edges expand or chance worlds sample).
struct FilterNodeTrace {
    int id = 0, parent = -1, parent_edge = -1, world = -1, born = 0, turn = 0;
    std::uint64_t key = 0;
    bool terminal = false, cycle = false, cached = false;
    float nn_value = 0;
    double prior_sd = 0;
    std::vector<float> logits, thinking;
    std::vector<std::string> actions;  // labels, aligned with logits
    std::vector<bool> edge_chance;     // per action: whether its transition was chance
};

// One purchase step in the search loop.
struct FilterStepTrace {
    int parent = 0, edge = 0, world = -1, child = 0, spent = 0;
    bool world_sample = false;  // true when sampling an additional world on an already-expanded chance edge
    double voi = 0, weight = 0, gap = 0, innovation = 0;
    std::vector<double> root_q;  // backed-up abs win rate Q(root, a) for each root move
};

struct FilterTrace {
    int spent = 0;
    double last_voi = 0;
    Evaluation opening;
    std::vector<double> initial_q;
    std::vector<FilterNodeTrace> nodes;
    std::vector<FilterStepTrace> steps;
    std::string stop;
};

// Recursive Kalman-VOI tree search: at every step, buy the single highest-VOI expansion or chance
// world sample if its expected root win-rate gain exceeds price. Cap counts fresh non-root evals.
// explore = 0 plays the leader; explore > 0 plays argmax(log policy + explore * Gumbel), seeded by explore_seed,
// so explore = 1 samples the move from the search policy.
std::unique_ptr<CombatSearch> filter_search(const sts::BattleContext&, int cap, int worlds, double price,
                                            FilterMode mode, std::uint64_t stream, FilterTrace* trace = nullptr,
                                            double explore = 0, std::uint64_t explore_seed = 0);
}
