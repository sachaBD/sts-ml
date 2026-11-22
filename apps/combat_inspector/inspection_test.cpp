// Invariants owned by apps/combat_inspector/inspection: view attachment, per-world root Q, reaching filter, batching.
#include "apps/combat_inspector/inspection.hpp"
#include "constants/Cards.h"
#include "constants/MonsterEncounters.h"
#include "constants/Potions.h"
#include "constants/Rooms.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace stsrl;
using namespace stsrl::inspection;

namespace {
Json fight(std::uint64_t seed = 7, int hp = 80) {
    Json deck = Json::array();
    for (int i = 0; i < 5; ++i) deck.push_back({{"id", int(sts::CardId::STRIKE_RED)}, {"upgraded", false}, {"misc", 0}});
    for (int i = 0; i < 4; ++i) deck.push_back({{"id", int(sts::CardId::DEFEND_RED)}, {"upgraded", false}, {"misc", 0}});
    deck.push_back({{"id", int(sts::CardId::BASH)}, {"upgraded", false}, {"misc", 0}});
    const int empty = int(sts::Potion::EMPTY_POTION_SLOT);
    return {{"seed", seed}, {"ascension", 0}, {"act", 1}, {"floor", 1},
            {"encounter", int(sts::MonsterEncounter::JAW_WORM)},
            {"cur_room", int(sts::Room::MONSTER)}, {"last_room", int(sts::Room::NONE)},
            {"burning_elite_buff", -1}, {"hp", hp}, {"max_hp", 80}, {"gold", 99},
            {"misc_rng", {{"counter", 0}, {"seed0", seed ^ 1}, {"seed1", seed ^ 2}}},
            {"potion_rng", {{"counter", 0}, {"seed0", seed ^ 3}, {"seed1", seed ^ 4}}},
            {"potion_capacity", 2}, {"potions", {empty, empty}}, {"relics", Json::array()},
            {"deck", std::move(deck)}, {"bottled", {-1, -1, -1}}};
}

Predict evaluator(int* batches = nullptr) {
    return [batches](std::span<const infra::Tokens> requests) {
        if (batches) ++(*batches);
        std::vector<infra::Prediction> out;
        for (const auto& t : requests) {
            float sum = 0;
            for (const float v : t.data) sum += v;
            std::vector<float> logits(t.actions);
            for (int i = 0; i < t.actions; ++i) logits[i] = std::cos(sum + float(i));
            out.push_back({0.25f + 0.5f * std::abs(std::sin(sum)), std::move(logits)});
        }
        return out;
    };
}
}  // namespace

TEST(Inspection, GraphAttachesViewsAndEdgeDescriptionsToEveryNode) {
    Session session{nn::Encoding::inventory_v1, evaluator()};
    const auto opened = session.run({
        {"op", "open"},
        {"trees", {{{"start", fight(31, /*hp=*/1)}, {"moves", Json::array()},
                    {"search", {{"sims", 16}, {"particles", 2}, {"gumbel_scale", 0}, {"rollout_mix", 0}}},
                    {"include_graph", true}}}},
    });
    const auto& nodes = opened.at("trees").at(0).at("graph").at("nodes");
    bool saw_terminal = false;
    for (const auto& node : nodes) {
        ASSERT_TRUE(node.contains("view"));
        for (const auto& edge : node.at("edges")) EXPECT_TRUE(edge.contains("description"));
        if (node.at("terminal").get<bool>()) {
            saw_terminal = true;
            EXPECT_EQ(node.at("view").at("kind"), "done");
        }
    }
    EXPECT_TRUE(saw_terminal);
}

TEST(Inspection, RootWorldExpectimaxMatchesSingleWorldSearchQ) {
    Session session{nn::Encoding::inventory_v1, evaluator()};
    const auto opened = session.run({
        {"op", "open"}, {"trees", {{{"start", fight(7)}, {"moves", Json::array()}, {"draw", 0}}}},
    });
    const int id = opened.at("trees").at(0).at("id").get<int>();
    Json wave = Json::array();
    for (std::size_t m = 0; m < opened.at("trees").at(0).at("moves").size(); ++m) wave.push_back(Json::array({m, 0, true}));
    Json last;
    for (int i = 0; i < 6; ++i) last = session.run({{"op", "simulate"}, {"picks", {{std::to_string(id), wave}}}});
    for (const auto& move : last.at("roots").at(std::to_string(id)).at("moves")) {
        ASSERT_EQ(move.at("worlds").size(), 1u);
        EXPECT_DOUBLE_EQ(move.at("worlds").at(0).at("q").get<double>(), move.at("q").get<double>());
    }
}

TEST(Inspection, StateReachingMatchesPerWorldQueries) {
    Session session{nn::Encoding::inventory_v1, evaluator()};
    const auto opened = session.run({
        {"op", "open"}, {"trees", {{{"start", fight(19)}, {"moves", Json::array()}, {"draw", 0}}}},
    });
    const int id = opened.at("trees").at(0).at("id").get<int>();
    session.run({{"op", "simulate"},
                 {"picks", {{std::to_string(id), Json::array({Json::array({Json::array({0}), 0, false})})}}}});
    const auto st = session.run({{"op", "state"}, {"tree", id}, {"path", {0}}, {"world", 0}, {"worlds", 4}});
    std::vector<int> expected;
    for (int w = 0; w < 4; ++w) {
        try {
            if (session.run({{"op", "state"}, {"tree", id}, {"path", {0}}, {"world", w}}).at("observation") ==
                st.at("observation"))
                expected.push_back(w);
        } catch (const std::invalid_argument&) {
        }
    }
    EXPECT_EQ(st.at("reaching").get<std::vector<int>>(), expected);
}

TEST(Inspection, SessionBatchesAcrossOpenTrees) {
    int batches = 0;
    Session session{nn::Encoding::inventory_v1, evaluator(&batches)};
    const auto opened = session.run({
        {"op", "open"},
        {"trees", Json::array({{{"start", fight(1)}, {"moves", Json::array()}, {"draw", 0}},
                               {{"start", fight(2)}, {"moves", Json::array()}, {"draw", 0}}})},
    });
    EXPECT_EQ(batches, 1);
    const int a = opened.at("trees").at(0).at("id").get<int>();
    const int b = opened.at("trees").at(1).at("id").get<int>();
    session.run({{"op", "simulate"},
                 {"picks", {{std::to_string(a), Json::array({Json::array({Json::array({0}), 0, false})})},
                            {std::to_string(b), Json::array({Json::array({Json::array({0}), 0, false})})}}}});
    EXPECT_EQ(batches, 2);
}
