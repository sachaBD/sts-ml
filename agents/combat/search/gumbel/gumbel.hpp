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
        double gumbel_scale = 0;  // 1 explores (self-play); 0 is deterministic (evaluation)
        int particles = 8;
        double value_scale = 0.1, maxvisit_init = 50;  // mctx qtransform_completed_by_mix_value
    };

    struct Graph {
        struct Child { std::uint64_t observation; int visits; };
        struct Edge {
            std::uint64_t action;
            double logit, prior, sigma, improved_policy;
            std::optional<double> q;  // traversal-weighted mean of child means; absent when unvisited
            int visits;
            std::vector<Child> children;
        };
        struct Node {
            std::uint64_t observation;
            bool terminal;
            double evaluation_value, mean_value;
            int visits;  // includes the initial evaluation for nonterminal nodes
            std::vector<Edge> edges;
        };
        std::uint64_t root;
        std::vector<Node> nodes;  // sorted by observation; children may have multiple parents
        std::vector<double> root_noise;  // root edge order
    };
    Graph graph() const;  // completed searches only; does not mutate the search

    Gumbel(const sts::BattleContext& observed, const Settings&, std::uint64_t seed);
    ~Gumbel() override;
    std::optional<Query> pending() const override;
    void evaluated(const Evaluation&) override;
    Result result() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace stsrl::search
