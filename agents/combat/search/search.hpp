#pragma once
// A search never calls the network itself: it asks for evaluations. Whoever drives it decides how to answer, so one GPU can serve many searches at once (infra/workers), or a test can answer one search inline.
#include "combat/BattleContext.h"
#include "sim/search/Action.h"
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace stsrl::search {
using sts::search::Action;
using WinProbability = float;

// The search owns what a Query views until it is evaluated; logits answer its actions in the same order.
struct Query {
    const sts::BattleContext& state;
    std::span<const Action> actions;
};

struct Evaluation {
    WinProbability value;
    std::vector<float> logits;
};

struct Result {
    Action action;
    std::vector<Action> actions;
    std::vector<float> policy;
    WinProbability value;
};

class CombatSearch {
public:
    virtual ~CombatSearch() = default;
    virtual std::optional<Query> pending() const = 0;
    virtual void evaluated(const Evaluation&) = 0;
    virtual Result result() const = 0;
};

template<class F>
concept Evaluator = std::is_invocable_r_v<Evaluation, F&, const Query&>;

Result run(CombatSearch& search, Evaluator auto&& evaluate) {
    while (const auto query = search.pending()) search.evaluated(evaluate(*query));
    return search.result();
}
}  // namespace stsrl::search
