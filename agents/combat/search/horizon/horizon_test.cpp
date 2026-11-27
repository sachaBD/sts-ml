// Invariants owned by unified Horizon search:
// - Trivial/decided root exits in a single evaluation with a valid normalized policy.
// - No public observation is ever queried twice within a search.
// - Higher compute price (lambda) stops strictly earlier than fixed-budget search.
// - Unified Max-VOI widens beyond initial contenders when leaders underperform root value.
// - World-adjusted root Q eliminates Simpson's paradox on unequal world subsets.
// - Paired waves are atomic at the simulation cap and execution is deterministic and bounded.
// - Multi-turn boundary blending keeps per-world hidden continuations isolated.
#include "agents/combat/search/horizon/horizon.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

using namespace stsrl;
using namespace stsrl::search;

namespace {
Evaluation deterministic_eval(const Query& q) {
    const auto obs = combat::observation_key(q.state);
    std::vector<float> logits(q.actions.size());
    for (std::size_t i = 0; i < q.actions.size(); ++i)
        logits[i] = float(std::cos(double((obs + i) % 97)));
    const float v = 0.25f + 0.5f * float((obs % 100) / 100.0);
    return {v, std::move(logits)};
}

void expect_valid_result(const Result& r) {
    ASSERT_FALSE(r.actions.empty());
    ASSERT_EQ(r.policy.size(), r.actions.size());
    EXPECT_TRUE(std::isfinite(r.value));
    EXPECT_GE(r.value, 0.0f);
    EXPECT_LE(r.value, 1.0f);
    double sum = 0.0;
    for (const float p : r.policy) {
        EXPECT_TRUE(std::isfinite(p));
        EXPECT_GE(p, 0.0f);
        sum += double(p);
    }
    EXPECT_NEAR(sum, 1.0, 1e-5);
    EXPECT_NE(std::ranges::find(r.actions, r.action), r.actions.end());
}
}  // namespace

TEST(Horizon, SkipsDecidedRootWithSingleCall) {
    const auto start = scenarios::jaw_worm(7);
    Horizon::Settings settings;
    Horizon search{start, settings, 1};
    int calls = 0;
    const auto r = run(search, [&](const Query& q) {
        ++calls;
        return Evaluation{0.9999f, std::vector<float>(q.actions.size(), 0.0f)};
    });
    EXPECT_EQ(calls, 1);
    expect_valid_result(r);
}

TEST(Horizon, NeverEvaluatesSamePublicObservationTwice) {
    const auto start = scenarios::jaw_worm(11);
    Horizon::Settings settings;
    settings.simulations = 48;
    settings.particles = 8;
    settings.lambda = 0.0;
    Horizon search{start, settings, 42};
    std::unordered_set<std::uint64_t> seen;
    const auto r = run(search, [&](const Query& q) {
        const auto obs = combat::observation_key(q.state);
        EXPECT_TRUE(seen.insert(obs).second) << "duplicate query for same public observation";
        return deterministic_eval(q);
    });
    EXPECT_GT(seen.size(), 1u);
    expect_valid_result(r);
}

TEST(Horizon, HighLambdaStopsEarlierThanZeroLambda) {
    const auto start = scenarios::jaw_worm(19);
    Horizon::Settings fixed;
    fixed.simulations = 64;
    fixed.particles = 8;
    fixed.lambda = 0.0;
    Horizon::Settings adaptive = fixed;
    adaptive.lambda = 0.02;

    int fixed_calls = 0, adaptive_calls = 0;
    Horizon s_fixed{start, fixed, 7};
    Horizon s_adapt{start, adaptive, 7};
    const auto r1 = run(s_fixed, [&](const Query& q) { ++fixed_calls; return deterministic_eval(q); });
    const auto r2 = run(s_adapt, [&](const Query& q) { ++adaptive_calls; return deterministic_eval(q); });
    expect_valid_result(r1);
    expect_valid_result(r2);
    EXPECT_LT(adaptive_calls, fixed_calls);
}

TEST(Horizon, WidensBeyondInitialContendersOnTrapRoot) {
    const auto start = scenarios::jaw_worm(23);
    const auto root_obs = combat::observation_key(start);
    const auto root_actions = combat::legal_actions(start);
    ASSERT_GE(root_actions.size(), 4u);

    // Trap root: root value is high (0.80), top-3 actions return poor Turn+1 values (0.20),
    // while rank-4 action (End Turn) returns 0.95.
    const std::size_t rank4 = root_actions.size() - 1;
    std::vector<float> root_logits(root_actions.size(), -5.0f);
    root_logits[0] = 0.0f;
    root_logits[1] = -0.3f;
    root_logits[2] = -0.6f;
    root_logits[rank4] = -1.2f;

    std::unordered_set<std::uint64_t> good_subtrees;
    const auto worlds = combat::particles(start, 8, start.seed ^ (std::uint64_t(start.turn) << 32));
    for (int w = 0; w < 8; ++w) {
        auto st = worlds[w];
        root_actions[rank4].execute(st);
        good_subtrees.insert(combat::observation_key(st));
    }

    Horizon::Settings s;
    s.simulations = 64;
    s.particles = 8;
    s.init_contenders = 3;
    s.init_worlds = 2;
    s.wave_worlds = 2;
    s.horizon_turns = 1;
    s.lambda = 1e-4;

    Horizon search{start, s, 9};
    const auto r = run(search, [&](const Query& q) {
        const auto obs = combat::observation_key(q.state);
        if (obs == root_obs) return Evaluation{0.80f, root_logits};
        const float v = good_subtrees.contains(obs) ? 0.95f : (q.state.turn > start.turn ? 0.20f : 0.25f);
        std::vector<float> logits(q.actions.size(), 0.0f);
        if (!logits.empty()) logits.back() = 2.0f;
        return Evaluation{v, std::move(logits)};
    });
    expect_valid_result(r);
    EXPECT_EQ(combat::action_key(start, r.action), combat::action_key(start, root_actions[rank4]));
}

TEST(Horizon, DeterministicAndBoundedAtTightSimulationCaps) {
    const auto start = scenarios::jaw_worm(31);
    for (const int sim_cap : {0, 1, 3, 7, 13}) {
        Horizon::Settings s;
        s.simulations = sim_cap;
        s.particles = 8;
        s.init_contenders = 3;
        s.init_worlds = 2;
        s.wave_worlds = 4;
        s.lambda = 0.0;

        int calls_a = 0, calls_b = 0;
        Horizon a{start, s, 1};
        Horizon b{start, s, 1};
        const auto ra = run(a, [&](const Query& q) { ++calls_a; return deterministic_eval(q); });
        const auto rb = run(b, [&](const Query& q) { ++calls_b; return deterministic_eval(q); });
        expect_valid_result(ra);
        expect_valid_result(rb);
        EXPECT_EQ(calls_a, calls_b);
        EXPECT_EQ(ra.action, rb.action);
        EXPECT_FLOAT_EQ(ra.value, rb.value);
    }
}

TEST(Horizon, WorldAdjustedRootQResistsUnequalWorldSubsetSkew) {
    // Construct a root with 3 near-equal-logit actions where worlds {0,1} are lucky (+0.20)
    // and worlds {2,3,4,5} are unlucky (-0.10), while action 0 strictly beats actions 1 and 2
    // on every single world. Even if action 2 is only sampled on lucky worlds {0,1}, action 0 must win.
    const auto start = scenarios::jaw_worm(41);
    const auto root_obs = combat::observation_key(start);
    const auto root_actions = combat::legal_actions(start);
    ASSERT_GE(root_actions.size(), 3u);

    std::vector<float> root_logits(root_actions.size(), -8.0f);
    root_logits[0] = 0.0f;
    root_logits[1] = -0.05f;
    root_logits[2] = -0.10f;

    const auto worlds = combat::particles(start, 8, start.seed ^ (std::uint64_t(start.turn) << 32));
    // Map each world's Turn+1 obs (after action + End Turn) to world_shift + action_bonus.
    std::unordered_map<std::uint64_t, float> obs_val;
    const float bonus[3] = {0.06f, 0.05f, 0.00f};
    for (int w = 0; w < 8; ++w) {
        const float w_shift = (w < 2) ? 0.75f : 0.45f;
        for (int a = 0; a < 3; ++a) {
            auto st = worlds[w];
            root_actions[a].execute(st);
            if (st.outcome == sts::Outcome::UNDECIDED && st.turn == start.turn) {
                const auto leg = combat::legal_actions(st);
                if (!leg.empty()) leg.back().execute(st);
            }
            if (st.outcome == sts::Outcome::UNDECIDED)
                obs_val.emplace(combat::observation_key(st), w_shift + bonus[a]);
        }
    }

    Horizon::Settings s;
    s.simulations = 14;  // Wave 1: 3x2 = 6 sims ({0,1}); Wave 2: 2x4 = 8 sims ({2,3,4,5} on actions 0 & 1 only)
    s.particles = 8;
    s.init_contenders = 3;
    s.init_worlds = 2;
    s.wave_worlds = 4;
    s.horizon_turns = 1;
    s.lambda = 0.0;

    Horizon search{start, s, 5};
    const auto r = run(search, [&](const Query& q) {
        const auto obs = combat::observation_key(q.state);
        if (obs == root_obs) return Evaluation{0.55f, root_logits};
        std::vector<float> logits(q.actions.size(), 0.0f);
        if (!logits.empty()) logits.back() = 4.0f;  // End Turn greedy
        const float v = obs_val.contains(obs) ? obs_val.at(obs) : 0.50f;
        return Evaluation{v, std::move(logits)};
    });
    expect_valid_result(r);
    EXPECT_EQ(combat::action_key(start, r.action), combat::action_key(start, root_actions[0]));
}

TEST(Horizon, ProgressiveTwoTurnHorizonExecutesCleanly) {
    const auto start = scenarios::jaw_worm(53);
    Horizon::Settings s;
    s.simulations = 32;
    s.particles = 8;
    s.horizon_turns = 2;
    s.lambda = 6e-5;

    std::unordered_set<std::uint64_t> seen;
    Horizon search{start, s, 11};
    const auto r = run(search, [&](const Query& q) {
        EXPECT_TRUE(seen.insert(combat::observation_key(q.state)).second);
        return deterministic_eval(q);
    });
    expect_valid_result(r);
    EXPECT_GT(seen.size(), 1u);
}

TEST(Horizon, CheckpointsCountOnlyAnsweredNonRootQueries) {
    auto s = Horizon::Settings{};
    s.simulations = 48;
    s.lambda = 0;
    int answered = 0;
    std::optional<int> last_count;
    Horizon search{scenarios::jaw_worm(53), s, 11, [&](const Horizon::Wave& wave) {
        last_count = wave.evals;
        EXPECT_GE(answered, 1);
        EXPECT_EQ(wave.evals, answered - 1);
    }};
    run(search, [&](const Query& q) { ++answered; return deterministic_eval(q); });
    EXPECT_GT(answered, 1) << "exercise non-root accounting";
    EXPECT_EQ(last_count, answered - 1);
}

TEST(Horizon, LastCheckpointRecommendsTheReturnedMove) {
    for (const int budget : {0, 48}) {
        SCOPED_TRACE(budget);
        auto s = Horizon::Settings{};
        s.simulations = budget;
        std::optional<Horizon::Wave> last;
        Horizon search{scenarios::jaw_worm(53), s, 11, [&](const Horizon::Wave& wave) { last = wave; }};
        const auto result = run(search, deterministic_eval);
        ASSERT_TRUE(last);
        EXPECT_EQ(last->selected, result.action);
    }
}

TEST(Horizon, ComputePriceCanStopSearchButCannotChangeEarlierCheckpoints) {
    const auto start = scenarios::jaw_worm(53);
    auto s = Horizon::Settings{};
    s.simulations = 160;
    const auto trace = [&](double price) {
        s.lambda = price;
        std::vector<Horizon::Wave> checkpoints;
        Horizon search{start, s, 11, [&](const Horizon::Wave& wave) { checkpoints.push_back(wave); }};
        run(search, deterministic_eval);
        return checkpoints;
    };
    const auto full = trace(0), priced = trace(1);
    ASSERT_FALSE(priced.empty());
    ASSERT_LT(priced.size(), full.size()) << "exercise early stopping";
    for (std::size_t i = 0; i < priced.size(); ++i) {
        SCOPED_TRACE(i);
        const auto& a = priced[i];
        const auto& b = full[i];
        EXPECT_EQ(a.simulations, b.simulations);
        EXPECT_EQ(a.evals, b.evals);
        EXPECT_EQ(a.prior, b.prior);
        EXPECT_EQ(a.selected, b.selected);
        EXPECT_EQ(a.prior_q, b.prior_q);
        EXPECT_EQ(a.selected_q, b.selected_q);
    }
}
