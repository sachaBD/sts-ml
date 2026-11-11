#include "agents/combat/player/explanation.hpp"
#include "agents/combat/player/decision.hpp"
#include "agents/combat/player/fake_network.hpp"
#include "agents/combat/search/config.hpp"
#include "environments/combat/fight_start.hpp"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "constants/Cards.h"
#include "constants/Potions.h"
#include "game/Random.h"
#include <gtest/gtest.h>

using namespace stsrl;
using Json = nlohmann::json;

namespace {
Json request(std::uint64_t seed, double mix) {
    const sts::Random rng{seed};
    const Json random{{"counter", rng.counter}, {"seed0", rng.seed0}, {"seed1", rng.seed1}};
    Json deck = Json::array();
    for (int i = 0; i < 10; ++i)
        deck.push_back({{"id", int(i < 5 ? sts::CardId::STRIKE_RED : sts::CardId::DEFEND_RED)},
                        {"upgraded", false}, {"misc", 0}});
    return {{"start", {{"seed", seed}, {"ascension", 1}, {"act", 1}, {"floor", 1},
                       {"encounter", int(sts::MonsterEncounter::JAW_WORM)}, {"cur_room", int(sts::Room::MONSTER)},
                       {"last_room", int(sts::Room::NONE)}, {"burning_elite_buff", -1},
                       {"hp", 80}, {"max_hp", 80}, {"gold", 99}, {"misc_rng", random}, {"potion_rng", random},
                       {"potion_capacity", 3}, {"potions", {int(sts::Potion::EMPTY_POTION_SLOT),
                       int(sts::Potion::EMPTY_POTION_SLOT), int(sts::Potion::EMPTY_POTION_SLOT)}},
                       {"relics", Json::array()}, {"deck", deck}, {"bottled", {-1, -1, -1}}}},
            {"moves", Json::array()}, {"search", {{"sims", 16}, {"particles", 2},
            {"gumbel_scale", 1}, {"rollout_mix", mix}}}};
}
}

TEST(Explanation, ReplaysTheSameDecisionsAsThePlayer) {
    for (const auto seed : {1u, 2u}) for (const double mix : {0.0, 0.5}) {
        auto r = request(seed, mix);
        player::Fight fight{combat::start_battle(r["start"]), seed};
        const nn::NetworkEvaluator evaluator{nn::Encoding::entity};
        while (!fight.over()) {
            const auto explanation = player::explain(r, nn::Encoding::entity, player::fake_network);
            auto random = fight.decision_random();
            const search::SearchConfig config{16, 2, 1, mix};
            player::Decision decision{config.factory()(fight.state(), random), evaluator};
            while (const auto tokens = decision.pending()) decision.evaluated(player::fake_network(*tokens));
            const auto expected = decision.result();
            EXPECT_EQ(explanation["decision"]["action"], expected.action.bits);
            EXPECT_FLOAT_EQ(explanation["decision"]["value"].get<float>(), expected.value);
            EXPECT_EQ(explanation, player::explain(r, nn::Encoding::entity, player::fake_network));
            r["moves"].push_back(expected.action.bits);
            fight.play(expected.action);
        }
        EXPECT_TRUE(player::explain(r, nn::Encoding::entity, player::fake_network)["over"]);
    }
}

TEST(Explanation, RejectsInvalidSettingsAndMoves) {
    auto r = request(1, 0);
    r["search"]["particles"] = 0;
    EXPECT_THROW(player::explain(r, nn::Encoding::entity, player::fake_network), std::invalid_argument);
    r = request(1, 0);
    r["moves"].push_back(-1);
    EXPECT_THROW(player::explain(r, nn::Encoding::entity, player::fake_network), std::invalid_argument);
    r["moves"][0] = UINT32_MAX;
    EXPECT_THROW(player::explain(r, nn::Encoding::entity, player::fake_network), std::invalid_argument);
}
