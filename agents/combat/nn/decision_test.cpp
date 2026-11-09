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
    while (const auto query = decision.next_query()) decision.evaluated(query->id, fake_network(query->tokens));
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
TEST(Decision, ClaimedEncodingsRemainUsableDuringOutOfOrderCompletions) {
    Fight fight{jaw_worm(1), 1};
    auto random = fight.decision_random();
    Decision decision{fight.state(), std::make_unique<Gumbel>(fight.state(), gumbel(), random()), 0.5, random()};
    std::map<std::uint64_t, std::pair<stsrl::infra::Tokens, std::vector<float>>> pending;
    int peak = 0;
    while (!decision.done()) {
        while (auto q = decision.next_query()) {
            auto expected = q->tokens.data;
            pending.emplace(q->id, std::pair{std::move(q->tokens), std::move(expected)});
        }
        peak = std::max(peak, int(pending.size()));
        for (const auto& [id, tokens] : pending) EXPECT_EQ(tokens.first.data, tokens.second);
        if (pending.empty()) { ASSERT_TRUE(decision.done()); break; }
        const auto it = std::prev(pending.end());
        decision.evaluated(it->first, fake_network(it->second.first));
        pending.erase(it);
    }
    EXPECT_GT(peak, 1);
    EXPECT_TRUE(pending.empty());
    EXPECT_FALSE(decision.result().policy.empty());
}

TEST(Decision, ClaimTransfersTokensAndCannotRepeatBeforeEvaluation) {
    Fight fight{jaw_worm(1), 1};
    auto random = fight.decision_random();
    Decision decision{fight.state(), std::make_unique<Gumbel>(fight.state(), gumbel(), random()), 0, random()};
    const auto query = decision.next_query();
    ASSERT_TRUE(query);
    EXPECT_FALSE(decision.next_query());
    decision.evaluated(query->id, fake_network(query->tokens));
    EXPECT_TRUE(decision.next_query());
}
