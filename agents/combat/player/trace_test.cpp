#include "agents/combat/player/trace.hpp"
#include "agents/combat/search/config.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <random>

using namespace stsrl;

TEST(Trace, ReplayRebuildsNodeAndRootEdgeVisits) {
    const auto state = scenarios::jaw_worm(7);
    std::mt19937_64 random{1};
    auto created = search::SearchConfig{.sims = 64, .particles = 4, .gumbel_scale = 1, .rollout_mix = 0}.create(state, random);
    std::mt19937 answers{1};
    const auto result = search::run(*created.search, [&](const search::Query& q) {
        std::normal_distribution<float> logit;
        search::Evaluation e{std::uniform_real_distribution<float>{}(answers), {}};
        for (std::size_t i = 0; i < q.actions.size(); ++i) e.logits.push_back(logit(answers));
        return e;
    });
    const auto graph = created.gumbel.graph();
    const auto t = player::trace(graph, result, state);

    const std::size_t root = t.at("root");
    std::vector<int> visits(t.at("nodes").size());
    std::vector<int> root_edges(t.at("root_actions").size());
    for (const auto& path : t.at("backups")) {
        std::vector<std::size_t> sources{root};
        for (std::size_t i = 0; i + 1 < path.size(); i += 2) {
            if (sources.back() == root) ++root_edges[path[i].get<std::size_t>()];
            sources.push_back(path[i + 1]);
        }
        const auto end = sources.back();
        sources.pop_back();
        for (const auto node : sources) ++visits[node];
        if (std::ranges::find(sources, end) == sources.end()) ++visits[end];
    }
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) EXPECT_EQ(visits[i], graph.nodes[i].visits) << "node " << i;
    for (std::size_t e = 0; e < root_edges.size(); ++e) EXPECT_EQ(root_edges[e], graph.nodes[root].edges[e].visits);
}
