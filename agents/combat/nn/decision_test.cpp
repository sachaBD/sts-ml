#include "agents/combat/nn/decision.hpp"
#include "agents/combat/nn/fake_network.hpp"
#include "agents/combat/nn/tokens.hpp"
#include "agents/combat/search/gumbel/gumbel.hpp"
#include "environments/combat/public.hpp"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include <gtest/gtest.h>

using namespace stsrl::nn;
using stsrl::combat::legal_actions;
using stsrl::search::Gumbel;

namespace {
sts::BattleContext jaw_worm(std::uint64_t seed) {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, 1};
    game.floorNum = 1;
    game.curRoom = sts::Room::MONSTER;
    sts::BattleContext battle;
    battle.init(game, sts::MonsterEncounter::JAW_WORM);
    return battle;
}

Gumbel::Settings gumbel() {
    Gumbel::Settings s;
    s.simulations = 8;
    s.particles = 2;
    s.gumbel_scale = 1;
    return s;
}

// As nn_play decides.
stsrl::search::Result decide(const Fight& fight, double rollout_mix) {
    auto random = fight.decision_random();
    auto search = std::make_unique<Gumbel>(fight.state(), gumbel(), random());
    Decision decision{fight.state(), std::move(search), rollout_mix, random()};
    while (const auto query = decision.pending()) decision.evaluated(fake_network(*query));
    return decision.result();
}

const std::vector<std::uint64_t> seeds{1, 2, 3};
}  // namespace

TEST(Fight, StopsOnlyAtAChoiceOrTheEnd) {
    for (const auto seed : seeds) {
        auto start = jaw_worm(seed);
        if (seed == 1) start.player.energy = 0;  // begins with a forced End Turn
        Fight fight{start, seed};
        std::mt19937_64 random{seed};
        for (;;) {
            ASSERT_TRUE(fight.over() || legal_actions(fight.state()).size() > 1);
            if (fight.over()) break;
            const auto actions = legal_actions(fight.state());
            fight.play(actions[random() % actions.size()]);
        }
    }
}

TEST(Decision, ReplaysFromAFreshFightAndTheMovesAlone) {
    for (const auto seed : seeds) {
        Fight played{jaw_worm(seed), seed};
        std::vector<sts::search::Action> moves;
        std::vector<stsrl::search::Result> results;
        while (!played.over()) {
            results.push_back(decide(played, 0.5));
            moves.push_back(results.back().action);
            played.play(moves.back());
        }
        for (std::size_t k = 0; k < results.size(); ++k) {
            Fight replay{jaw_worm(seed), seed};
            for (std::size_t i = 0; i < k; ++i) replay.play(moves[i]);
            const auto r = decide(replay, 0.5);
            EXPECT_EQ(r.action, results[k].action) << "seed " << seed << " decision " << k;
            EXPECT_EQ(r.policy, results[k].policy) << "seed " << seed << " decision " << k;
            EXPECT_EQ(r.value, results[k].value) << "seed " << seed << " decision " << k;
        }
    }
}

TEST(Decision, WithoutRolloutsIsTheSearchOnTheNetwork) {
    for (const auto seed : seeds) {
        Fight fight{jaw_worm(seed), seed};
        while (!fight.over()) {
            auto random = fight.decision_random();
            Gumbel bare{fight.state(), gumbel(), random()};
            const auto expected = stsrl::search::run(bare, [](const stsrl::search::Query& q) {
                const auto p = fake_network(encode(q.state, q.actions));
                return stsrl::search::Evaluation{p.value, p.logits};
            });
            const auto r = decide(fight, 0);
            EXPECT_EQ(r.action, expected.action);
            EXPECT_EQ(r.policy, expected.policy);
            EXPECT_EQ(r.value, expected.value);
            fight.play(r.action);
        }
    }
}

TEST(Tokens, TargetReferencesIdentifyInstancesAndLeaveLegacyInputsUnchanged) {
    auto state = jaw_worm(1);
    // Two otherwise identical instances must still have separate references.
    state.monsters.arr[1] = state.monsters.arr[0];
    state.monsters.arr[1].idx = 1;
    state.cards.cardsInHand = 1;
    state.cards.hand[0] = sts::CardInstance{sts::CardId::TRIP};
    state.obtainPotion(sts::Potion::FIRE_POTION);
    using A = sts::search::Action;
    using T = sts::search::ActionType;
    const std::vector<A> actions{{T::CARD, 0, 0}, {T::CARD, 0, 1},
                                 {T::POTION, 0, 0}, {T::POTION, 0, 1}, {T::END_TURN}};
    auto linked = encode(state, actions, true);
    const auto legacy = encode(state, actions);
    const int first = linked.count - linked.actions;
    for (int i = 0; i < 4; ++i) {
        const auto column = (first + i) * width + width - 1;
        const int target = int(linked.data[column]);
        ASSERT_GT(target, 0);
        ASSERT_LT(target, first);
        EXPECT_EQ(linked.data[target * width], 3);  // monster
        EXPECT_EQ(linked.data[column], linked.data[(first + i % 2) * width + width - 1]);
        EXPECT_EQ(legacy.data[column], 0);
    }
    EXPECT_NE(linked.data[first * width + width - 1], linked.data[(first + 1) * width + width - 1]);
    for (int i = 0; i < 4; ++i) linked.data[(first + i) * width + width - 1] = 0;
    EXPECT_EQ(linked.data, legacy.data);

    state.monsters.arr[0].curHp /= 2;
    for (int ordering = 0; ordering < 2; ++ordering) {
        const auto encoded = encode(state, actions, true);
        for (int i = 0; i < 4; ++i) {
            const int target = int(encoded.data[(first + i) * width + width - 1]);
            EXPECT_FLOAT_EQ(encoded.data[target * width + 4], state.monsters.arr[i % 2].curHp / 100.f);
        }
        std::swap(state.monsters.arr[0], state.monsters.arr[1]);
        state.monsters.arr[0].idx = 0;
        state.monsters.arr[1].idx = 1;
    }
}
