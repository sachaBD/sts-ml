#pragma once
// A search never calls the network itself: it asks for evaluations. Whoever drives it decides how to answer, so one GPU can serve many searches at once (infra/workers), or a test can answer one search inline.
#include "combat/BattleContext.h"
#include "sim/search/Action.h"
#include <functional>
#include <memory>
#include <random>
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
    // Queries that can be answered together; empty once the search is done.
    virtual std::span<const Query> pending() const = 0;
    virtual void evaluated(std::span<const Evaluation>) = 0;  // one per pending query, in order
    virtual Result result() const = 0;
};

// A fresh search for one decision. It draws all its randomness from `random`, so the caller decides reproducibility.
using SearchFactory = std::function<std::unique_ptr<CombatSearch>(const sts::BattleContext&, std::mt19937_64& random)>;

template<class F>
concept Evaluator = std::is_invocable_r_v<Evaluation, F&, const Query&>;

Result run(CombatSearch& search, Evaluator auto&& evaluate) {
    for (auto queries = search.pending(); !queries.empty(); queries = search.pending()) {
        std::vector<Evaluation> evaluations;
        for (const auto& query : queries) evaluations.push_back(evaluate(query));
        search.evaluated(evaluations);
    }
    return search.result();
}
}  // namespace stsrl::search
