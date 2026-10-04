#pragma once
// Evaluation-only oracle search over complete player turns. No changes to per-action PUCT.
#include "agents/combat/pv/evaluator.hpp"
#include <functional>
#include <memory>
#include <string>

namespace stsrl::pv {
struct TurnSearchCaps {
    std::size_t max_sequences = 20000, root_max_children = 2048, max_children = 512, max_actions = 512;
    std::size_t max_bytes = 256ULL << 20;
    double max_seconds = 1;
};
struct TurnSearchStats {
    double seconds = 0, mean_leaf_depth = 0;
    std::size_t expansions = 0, network_calls = 0, evaluated_states = 0, root_children = 0, max_depth = 0;
    bool fallback = false, time_overshoot = false, reused = false;
    std::string reason;
};
struct TurnChild;
struct TurnNode {
    std::vector<TurnChild> children;
    bool expanded = false, capped = false;
    std::size_t visits = 0;
    double value = 0;
    std::string reason;
};
struct TurnChild {
    std::vector<std::uint32_t> sequence;
    std::string key;
    double value = 0, prior = 0, sum = 0;
    std::size_t visits = 0;
    bool terminal = false, capped = false;
    std::unique_ptr<TurnNode> node;
    double q() const { return visits ? sum / visits : value; }
};
class TurnSearch {
public:
    using Evaluate = std::function<std::vector<Prediction>(std::span<const Inputs>)>;
    explicit TurnSearch(const sts::BattleContext&, TurnSearchCaps = {}, double c = 1.25, double temperature = 10);
    // Empty sequence only on an explicitly reported root fallback.
    std::vector<std::uint32_t> decide(std::size_t budget, const Evaluate&);
    bool advance(const sts::BattleContext& actual);
    const TurnNode& root() const { return *root_; }
    const TurnSearchStats& stats() const { return stats_; }
private:
    sts::BattleContext state_;
    TurnSearchCaps caps_;
    double c_, temperature_;
    std::unique_ptr<TurnNode> root_;
    TurnSearchStats stats_;
    std::size_t bytes_ = 0, selected_ = 0;
    bool reused_ = false;
    bool expand(TurnNode&, const sts::BattleContext&, const Evaluate&);
};
} // namespace stsrl::pv
