#include "agents/combat/player/decision.hpp"
#include "agents/combat/player/probe_grader.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <unordered_set>

using namespace stsrl::player;
using namespace stsrl::search;
using stsrl::combat::action_key;
using stsrl::combat::legal_actions;
using stsrl::combat::observation_key;

namespace {
template<class F>
std::size_t run(ProbeGrader& grader, F&& evaluate) {
    std::size_t queries = 0;
    for (auto batch = grader.pending(); !batch.empty(); batch = grader.pending()) {
        std::vector<Evaluation> out;
        out.reserve(batch.size());
        for (const auto& q : batch) out.push_back(evaluate(q));
        queries += batch.size();
        grader.evaluated(out);
    }
    return queries;
}

Evaluation hash_eval(const Query& q) {
    const auto key = observation_key(q.state);
    std::vector<float> logits;
    for (const auto a : q.actions) logits.push_back(float(std::sin(double((action_key(q.state, a) ^ key) & 0xFFFFu))));
    return {float(0.1 + 0.8 * double(key & 0xFFFFu) / 65535.0), std::move(logits)};
}

std::optional<Action> find_card(const sts::BattleContext& s, sts::CardId id) {
    for (const auto a : legal_actions(s))
        if (a.getActionType() == sts::search::ActionType::CARD && s.cards.hand[a.getSourceIdx()].id == id) return a;
    return std::nullopt;
}
}  // namespace

TEST(ProbeGrader, AgreementUsesZeroEvalsAndReturnsExactZero) {
    const auto root = stsrl::scenarios::jaw_worm(7);
    const auto prior = legal_actions(root).front();
    const std::array repeated{prior, prior};
    for (const auto cands : {std::span<const Action>{}, std::span<const Action>{repeated}}) {
        ProbeGrader grader{root, prior, cands, 12345};
        EXPECT_TRUE(grader.pending().empty());
        EXPECT_EQ(grader.gain(prior), 0.0f);
    }
}

TEST(ProbeGrader, SharedParticlesCancelDrawVarianceAndPreserveSign) {
    const auto root = stsrl::scenarios::jaw_worm(7);
    const auto strike = find_card(root, sts::CardId::STRIKE_RED);
    const auto defend = find_card(root, sts::CardId::DEFEND_RED);
    ASSERT_TRUE(strike && defend);

    // Commuting cards reach the same next-turn state on each shared particle, so draw variance cancels to 0.
    auto commute_eval = [](const Query& q) -> Evaluation {
        std::vector<float> logits(q.actions.size(), -10.f);
        for (std::size_t i = 0; i < q.actions.size(); ++i) {
            const auto a = q.actions[i];
            if (a.getActionType() == sts::search::ActionType::CARD) {
                const auto id = q.state.cards.hand[a.getSourceIdx()].id;
                logits[i] = id == sts::CardId::STRIKE_RED ? 2.f : id == sts::CardId::DEFEND_RED ? 1.f : -10.f;
            } else if (a.getActionType() == sts::search::ActionType::END_TURN) {
                logits[i] = 0.f;
            }
        }
        return {float((observation_key(q.state) % 1000) / 1000.0), std::move(logits)};
    };
    const std::array cands{*defend};
    ProbeGrader commuting{root, *strike, cands, 99999};
    run(commuting, commute_eval);
    EXPECT_EQ(commuting.gain(*defend), 0.0f);

    // Swapping prior and candidate on the same stream negates the signed gain identically.
    const auto actions = legal_actions(root);
    const std::array fwd_cands{actions[1]}, rev_cands{actions[0]};
    ProbeGrader fwd{root, actions[0], fwd_cands, 4242};
    ProbeGrader rev{root, actions[1], rev_cands, 4242};
    run(fwd, hash_eval);
    run(rev, hash_eval);
    EXPECT_EQ(*rev.gain(actions[0]), -*fwd.gain(actions[1]));
}

TEST(ProbeGrader, DeduplicatesObservationKeysAndDuplicateCandidates) {
    const auto root = stsrl::scenarios::jaw_worm(7);
    const auto a = legal_actions(root);
    ASSERT_GE(a.size(), 3u);
    const std::array uniq{a[1], a[2]};
    const std::array dups{a[1], a[0], a[2], a[1]};
    ProbeGrader g1{root, a[0], uniq, 4242}, g2{root, a[0], dups, 4242};

    std::unordered_set<std::uint64_t> seen;
    const auto q1 = run(g1, [&](const Query& q) {
        EXPECT_TRUE(seen.insert(observation_key(q.state)).second);
        return hash_eval(q);
    });
    EXPECT_EQ(q1, run(g2, hash_eval));
    EXPECT_EQ(g1.gain(a[1]), g2.gain(a[1]));
    EXPECT_EQ(g1.gain(a[2]), g2.gain(a[2]));
}

TEST(ProbeGrader, TerminalVictoryScoresOneAndPreservesNegativeGain) {
    auto root = stsrl::scenarios::jaw_worm(7);
    root.monsters.arr[0].curHp = 5;  // Strike is lethal on step 0
    const auto strike = find_card(root, sts::CardId::STRIKE_RED);
    Action end_turn{};
    for (const auto a : legal_actions(root))
        if (a.getActionType() == sts::search::ActionType::END_TURN) end_turn = a;
    ASSERT_TRUE(strike);

    auto low = [](const Query& q) -> Evaluation {
        EXPECT_EQ(q.state.outcome, sts::Outcome::UNDECIDED);
        return {0.25f, std::vector<float>(q.actions.size(), 0.f)};
    };
    const std::array pos_cands{*strike}, neg_cands{end_turn};
    ProbeGrader pos{root, end_turn, pos_cands, 777}, neg{root, *strike, neg_cands, 777};
    run(pos, low);
    run(neg, low);
    EXPECT_FLOAT_EQ(*pos.gain(*strike), 0.75f);
    EXPECT_FLOAT_EQ(*neg.gain(end_turn), -0.75f);
}

TEST(ProbeGrader, GradingStreamIsDisjointFromProbeAndPlayStreams) {
    for (const std::uint64_t seed : {1ull, 7ull, 42ull}) {
        const auto root = stsrl::scenarios::jaw_worm(seed);
        const Fight fight{root, seed};
        const auto grade_stream = fight.decision_random(Fight::RandomStream::thinking_grade)();
        const auto probe_stream = root.seed ^ (std::uint64_t(root.turn) << 32);
        EXPECT_NE(grade_stream, fight.decision_random(Fight::RandomStream::play)());
        EXPECT_NE(grade_stream, fight.decision_random(Fight::RandomStream::thinking_probe)());
        EXPECT_NE(grade_stream, probe_stream);
        const auto pw = stsrl::combat::particles(root, 8, probe_stream);
        const auto gw = stsrl::combat::particles(root, 8, grade_stream);
        for (int p = 0; p < 8; ++p) EXPECT_NE(pw[p].seed, gw[p].seed);
    }
}
