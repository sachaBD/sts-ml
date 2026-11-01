#include "agents/combat/nn/explanation.hpp"
#include "agents/combat/nn/fake_network.hpp"
#include "environments/combat/record_v4.hpp"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include <gtest/gtest.h>
#include <cmath>

using Json = nlohmann::json;
using stsrl::nn::explain;
using stsrl::nn::fake_network;

namespace {
Json request(std::uint64_t seed, double rollout_mix) {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, 1};
    game.floorNum = 1;
    game.curRoom = sts::Room::MONSTER;
    game.info.encounter = sts::MonsterEncounter::JAW_WORM;
    return {{"start", stsrl::combat_v4::start_json(game)}, {"moves", Json::array()},
            {"search", {{"sims", 8}, {"particles", 2}, {"gumbel_scale", 1}, {"rollout_mix", rollout_mix}}}};
}

// Every explanation along a fight that follows the agent's own decisions.
std::vector<Json> explanations(Json r) {
    std::vector<Json> out;
    for (out.push_back(explain(r, fake_network)); !out.back()["over"]; out.push_back(explain(r, fake_network)))
        r["moves"].push_back(out.back()["decision"]["action"]);
    return out;
}
}  // namespace

TEST(Explain, IsAFunctionOfTheRequest) {
    for (const std::uint64_t seed : {1, 2}) {
        auto r = request(seed, 0.5);
        const auto first = explanations(r);
        ASSERT_GT(first.size(), 2u);
        EXPECT_EQ(explanations(r), first);
    }
}

TEST(Explain, LeafValuesAreTheNetworkBlendedWithAWonOrLostRollout) {
    for (const double mix : {0.0, 0.3, 0.5, 1.0})
        for (const auto& e : explanations(request(1, mix))) {
            if (e["over"]) continue;
            for (const auto& node : e["tree"]["nodes"]) {
                const double value = node["value"], network = node["network"];
                if (mix == 0) {
                    EXPECT_TRUE(node["rollout"].is_null());
                    EXPECT_EQ(value, network);
                    continue;
                }
                const double rollout = (value - (1 - mix) * network) / mix;
                EXPECT_NEAR(rollout, std::round(rollout), 1e-4) << "mix " << mix;
                EXPECT_EQ(node["rollout"], std::round(rollout));
                EXPECT_TRUE(node["rollout"] == 0 || node["rollout"] == 1);
            }
        }
}
