#pragma once
// Gumbel MuZero (Danihelka et al. 2022; as mctx gumbel_muzero_policy) over the real simulator.
// Root: Gumbel top-k and Sequential Halving. Interior: deterministic improved-policy selection.
#include "agents/combat/search/search.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace stsrl::search {
class Gumbel final : public CombatSearch {
public:
    struct Settings {
        int simulations = 256;
        int considered_actions = 16;
        int wave = 8;  // late rounds join passes up to this many simulations; 16 cost ~6% tree depth
        double gumbel_scale = 0;  // 1 explores (self-play); 0 is deterministic (evaluation)
        int particles = 8;
        double value_scale = 0.1, maxvisit_init = 50;  // mctx qtransform_completed_by_mix_value
        std::uint64_t world_draw = 0;  // which particle worlds are sampled; a turn's searches share a draw
    };

    // Unvisited edges are unknown; a chance report cannot be downgraded by another particle.
    enum class TransitionKind { unknown, deterministic, chance };

    struct Graph {
        struct Child {
            std::uint64_t observation;
            int visits;
            bool bootstrap = false;  // closes a cycle, so counts the successor's evaluation_value
        };
        struct Edge {
            std::uint64_t action;
            double logit, prior, sigma, improved_policy;
            std::optional<double> q;  // visit-weighted mean of the outcomes' values; absent when unvisited
            int visits;
            std::vector<Child> children;
            TransitionKind transition = TransitionKind::unknown;
        };
        struct Node {
            std::uint64_t observation;
            bool terminal;
            double evaluation_value;
            double backed_value;  // the search's value of this node
            std::vector<Edge> edges;
        };
        std::uint64_t root;
        std::vector<Node> nodes;  // sorted by observation; children may have multiple parents
        std::vector<double> root_noise;  // root edge order
    };
    Graph graph() const;  // completed searches only; does not mutate the search

    // The root alone, as graph() reports it, plus each move's worlds: how often each world's simulations took the
    // move, and the world's own expectimax value of it (the best of the edges its simulations took, through the
    // links they took). Requires every pending query answered.
    struct WorldValue { int world, visits; double q; };
    struct Root {
        std::vector<Graph::Edge> edges;  // decision order
        std::vector<std::vector<WorldValue>> worlds;  // per edge, by world
        double value;  // the search's value of the root
    };
    Root root() const;

    // Steering: one wave of simulations, each down a chosen root move (an index into the decision's actions) in a
    // chosen world (an index into the draw's endless stream; the scheduled search samples the first `particles`).
    // Requires every pending query answered; pending() then holds the wave's queries.
    struct Pick { std::size_t edge; int world; };
    void simulate(std::span<const Pick>);

    Gumbel(const sts::BattleContext& observed, const Settings&, std::uint64_t seed);
    ~Gumbel() override;
    std::span<const Query> pending() const override;
    void evaluated(std::span<const Evaluation>) override;
    Result result() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace stsrl::search
