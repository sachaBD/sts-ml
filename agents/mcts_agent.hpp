#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace stsrl {

class CombatEnvironment;
using LeafEvaluator = std::function<double(CombatEnvironment&)>;

struct MctsActionStats { std::size_t execution_index{}; std::size_t visits{}; double q{}; };
struct MctsResult { std::size_t chosen_action{}; double root_value{}; std::size_t root_visits{}; std::vector<MctsActionStats> actions; };

struct MctsConfig {
    std::size_t simulations = 2'000;
    std::size_t rollout_limit = 512;
    double exploration = 1.4142135623730951;
};

class MctsAgent final {
public:
    explicit MctsAgent(std::uint64_t seed, MctsConfig config = {}, LeafEvaluator leaf_evaluator = {});
    ~MctsAgent();

    MctsAgent(MctsAgent&&) noexcept;
    MctsAgent& operator=(MctsAgent&&) noexcept;
    MctsAgent(const MctsAgent&) = delete;
    MctsAgent& operator=(const MctsAgent&) = delete;

    [[nodiscard]] MctsResult search(CombatEnvironment& environment);
    [[nodiscard]] std::size_t choose_action(CombatEnvironment& environment);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace stsrl
