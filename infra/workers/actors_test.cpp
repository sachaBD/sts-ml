// run_actors: work is completed, routed and owned correctly, and every failure surfaces as an error.
// No GPU or ONNX Runtime needed. Fakes run on Actor/server threads, so they report violations by throwing,
// which run_actors propagates to the test.
#include "infra/workers/actors.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <limits>
#include <map>
#include <mutex>
#include <thread>

using namespace stsrl::infra;
using namespace std::chrono_literals;

namespace {
constexpr int token_width = 8;

void require(bool condition, const char* violation) { if (!condition) throw std::runtime_error{violation}; }

// What the fakes observed, shared across threads.
struct Log {
    std::atomic<int> started{0}, finished{0}, evaluated{0}, alive{0}, most_alive{0};
    std::mutex mutex;
    std::map<std::thread::id, int> alive_per_thread;
    int most_alive_per_thread = 0;
};

// Requests `steps` predictions. Each must come back to it, on the thread that created it, exactly once,
// carrying its own id and step number (the backend echoes them).
class FakeEpisode final : public Episode {
    Log& log_;
    const std::thread::id owner_ = std::this_thread::get_id();
    const int id_;
    int left_, step_ = 0;
    bool waiting_ = false, finished_ = false;
    Tokens tokens_{std::vector<float>(2 * token_width), 2, 1};

    void on_owner() const { require(std::this_thread::get_id() == owner_, "episode used off its Actor thread"); }
public:
    FakeEpisode(Log& log, int id, int steps) : log_{log}, id_{id}, left_{steps} {
        tokens_.data[0] = float(id + 1) / 1000;
        const int alive = ++log_.alive;
        for (int seen = log_.most_alive; seen < alive && !log_.most_alive.compare_exchange_weak(seen, alive);) {}
        std::lock_guard lock{log_.mutex};
        log_.most_alive_per_thread = std::max(log_.most_alive_per_thread, ++log_.alive_per_thread[owner_]);
    }
    ~FakeEpisode() override {
        --log_.alive;
        std::lock_guard lock{log_.mutex};
        --log_.alive_per_thread[owner_];
    }
    std::optional<EvaluationRequest> next_request() override {
        on_owner();
        require(!finished_, "advanced while finished");
        if (waiting_) return std::nullopt;
        if (!left_) return std::nullopt;
        waiting_ = true;
        tokens_.data[1] = float(step_);
        return EvaluationRequest{std::uint64_t(step_), &tokens_};
    }
    void completed(std::uint64_t id, const Prediction& p) override {
        on_owner();
        require(id == std::uint64_t(step_), "wrong request id");
        require(waiting_, "prediction without a request");
        require(p.value == float(id_ + 1) / 1000 && p.logits == std::vector<float>{float(step_)}, "prediction misrouted");
        waiting_ = false; --left_; ++step_; ++log_.evaluated;
    }
    bool done() const override { on_owner(); return !left_ && !waiting_; }
    void finish() override {
        on_owner();
        require(!waiting_ && !left_ && !finished_, "finished early or twice");
        finished_ = true; ++log_.finished;
    }
    std::string label() const override { return "fake-" + std::to_string(id_); }
};

// `count` episodes; episode i makes steps(i) requests.
NextEpisode episodes(Log& log, int count, std::function<int(int)> steps = [](int i) { return i % 19; }) {
    return [&log, count, steps]() -> std::unique_ptr<Episode> {
        const int id = log.started++;
        return id < count ? std::make_unique<FakeEpisode>(log, id, steps(id)) : nullptr;
    };
}
const auto three_steps = [](int) { return 3; };  // every episode needs inference before it can finish
int total_steps(int count) { int n = 0; for (int i = 0; i < count; ++i) n += i % 19; return n; }

std::vector<Prediction> echo(std::span<const Input> inputs) {
    std::vector<Prediction> out;
    for (const auto& input : inputs) out.push_back({input.tokens->data[0], {input.tokens->data[1]}});
    return out;
}

class FakeBackend final : public Backend {
public:
    using Evaluate = std::function<std::vector<Prediction>(std::span<const Input>)>;
    using Prepare = std::function<void(const Tokens&, std::vector<std::uint16_t>&)>;
    explicit FakeBackend(Evaluate evaluate = echo, Prepare prepare = {}, int lanes = 1,
                         std::function<bool(const Tokens&)> accepts = {}, int batch = 64)
        : evaluate_{std::move(evaluate)}, prepare_{std::move(prepare)}, lanes_{lanes}, accepts_{std::move(accepts)}, batch_{batch} {}
    int width() const override { return token_width; }
    int max_batch() const override { return batch_; }
    bool accepts(const Tokens& t) const override { return !accepts_ || accepts_(t); }
    int lanes() const override { return lanes_; }
    void prepare(const Tokens& t, std::vector<std::uint16_t>& packed) const override { if (prepare_) prepare_(t, packed); else packed.clear(); }
    std::vector<Prediction> evaluate(std::span<const Input> inputs) override { return evaluate_(inputs); }
private:
    Evaluate evaluate_;
    Prepare prepare_;
    int lanes_;
    std::function<bool(const Tokens&)> accepts_;
    int batch_;
};

ActorConfig small(int slots = 3, int batch = 7) {
    ActorConfig c;
    c.actors = 4; c.slots = slots; c.batch = batch; c.timeout = 2s; c.shutdown_timeout = 1s;
    return c;
}

// run_actors must fail with `error`, and no episode may outlive it.
void expect_error(const ActorConfig& config, Log& log, NextEpisode next, FakeBackend backend, const std::string& error) {
    try {
        run_actors(config, std::move(next), backend);
        FAIL() << "expected an error containing: " << error;
    } catch (const std::exception& e) {
        EXPECT_NE(std::string{e.what()}.find(error), std::string::npos) << e.what();
    }
    EXPECT_EQ(log.alive, 0) << "episodes outlived run_actors";
}
}  // namespace

// --- Correctness -------------------------------------------------------------------------------------------

class CompletesAllWork : public testing::TestWithParam<std::tuple<int, int, int>> {};  // (slots, batch, lanes)

TEST_P(CompletesAllWork, EveryRequestAnsweredOnceAndCapacityRespected) {
    const auto [slots, batch, lanes] = GetParam();
    Log log;
    FakeBackend backend{echo, {}, lanes};
    run_actors(small(slots, batch), episodes(log, 100), backend);
    EXPECT_EQ(log.finished, 100);
    EXPECT_EQ(log.evaluated, total_steps(100));
    EXPECT_EQ(log.alive, 0);
    EXPECT_LE(log.most_alive, 4 * slots);
    EXPECT_LE(log.most_alive_per_thread, slots);
}
// Slots fewer/more than a batch; slots are reused as episodes finish; one or two server lanes.
INSTANTIATE_TEST_SUITE_P(SlotsAndBatches, CompletesAllWork,
                         testing::Combine(testing::Values(1, 2, 3), testing::Values(2, 17), testing::Values(1, 2)));

TEST(Actors, LanesEvaluateConcurrentlyUpToTheirCount) {
    for (int lanes : {1, 2, 3}) {
        Log log;
        std::atomic<int> running{0}, most{0};
        FakeBackend backend{[&](std::span<const Input> inputs) {
            const int now = ++running;
            for (int seen = most; seen < now && !most.compare_exchange_weak(seen, now);) {}
            std::this_thread::sleep_for(1ms);  // long enough for another lane to gather a batch
            --running;
            return echo(inputs);
        }, {}, lanes};
        run_actors(small(3, 2), episodes(log, 100, three_steps), backend);
        EXPECT_EQ(log.finished, 100);
        EXPECT_EQ(most, lanes) << "lanes must overlap, and never exceed the backend's count";
    }
}

TEST(Actors, PrepareRunsOnOwnerAndPackedInputStaysStable) {
    Log log;
    FakeBackend backend{[](std::span<const Input> inputs) {
        std::this_thread::sleep_for(1ms);  // Actors must not touch pending buffers meanwhile
        for (const auto& input : inputs)
            require(input.packed.size() == 2 && input.packed[0] == std::uint16_t(input.tokens->data[0] * 1000 + .5f) &&
                    input.packed[1] == std::uint16_t(input.tokens->data[1]), "packed input changed while pending");
        return echo(inputs);
    }, [&log](const Tokens& tokens, std::vector<std::uint16_t>& packed) {
        std::lock_guard lock{log.mutex};
        require(log.alive_per_thread[std::this_thread::get_id()] > 0, "prepare ran off the owning Actor");
        packed = {std::uint16_t(tokens.data[0] * 1000 + .5f), std::uint16_t(tokens.data[1])};
    }};
    run_actors(small(), episodes(log, 40), backend);
    EXPECT_EQ(log.finished, 40);
}

TEST(Actors, CountersCountEveryRequestOnceAndTimeWithinWallTime) {
    Log log;
    const auto slow = [](std::span<const Input> inputs) { std::this_thread::sleep_for(1ms); return echo(inputs); };
    FakeBackend odd{slow, {}, 1, [](const Tokens& t) { return int(t.data[0] * 1000 + .5f) % 2 == 1; }}, rest{slow};
    Counters last;
    auto config = small();
    config.on_counters = [&last](const Counters& counters) { last = counters; };
    const auto started = std::chrono::steady_clock::now();
    run_actors(config, episodes(log, 100), std::vector<Backend*>{&odd, &rest});
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;

    ASSERT_EQ(last.actors.size(), std::size_t(config.actors));
    long requests = 0;
    for (const auto& a : last.actors) { requests += a.requests; EXPECT_LE(a.blocked, elapsed); }
    EXPECT_EQ(requests, log.evaluated) << "the final counters count every request exactly once";

    ASSERT_EQ(last.queues.size(), 2u);
    requests = 0;
    for (const auto& q : last.queues) {
        requests += q.requests;
        EXPECT_LE(q.waited, q.requests * elapsed);
        EXPECT_EQ(q.depth, 0u);
        EXPECT_GT(q.peak_depth, 0u);
        EXPECT_LE(q.peak_depth, std::size_t(config.actors * config.slots));
    }
    EXPECT_EQ(requests, log.evaluated) << "every request leaves exactly one queue";

    ASSERT_EQ(last.lanes.size(), 2u);
    requests = 0;
    for (const auto& l : last.lanes) { requests += l.requests; EXPECT_LE(l.requests, l.capacity); EXPECT_LE(l.busy, elapsed); }
    EXPECT_EQ(requests, log.evaluated) << "every request is evaluated by exactly one lane";
}

TEST(Actors, QueueDepthExcludesEvaluatingBatchAndPeakSurvivesDrain) {
    Log log;
    std::atomic<bool> sampled{false};
    FakeBackend backend{[&](std::span<const Input> inputs) {
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        while (!sampled && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(1ms);
        require(sampled, "no counters published while evaluation was blocked");
        return echo(inputs);
    }};
    auto config = small(4, 1);
    config.actors = 1;
    config.timeout = 5s;
    std::vector<Counters> snapshots;
    config.on_counters = [&](const Counters& counters) {
        snapshots.push_back(counters);
        sampled = true;
    };
    run_actors(config, episodes(log, config.slots, [](int) { return 1; }), backend);

    ASSERT_GE(snapshots.size(), 2u);
    const auto& waiting = snapshots.front().queues.at(0);
    const auto& drained = snapshots.back().queues.at(0);
    EXPECT_EQ(waiting.depth, std::size_t(config.slots - 1)) << "the evaluating request is no longer queued";
    EXPECT_GE(waiting.peak_depth, waiting.depth);
    EXPECT_LE(waiting.peak_depth, std::size_t(config.slots));
    EXPECT_EQ(drained.depth, 0u);
    EXPECT_EQ(drained.peak_depth, waiting.peak_depth) << "draining must not reset the high-water mark";
}

TEST(Actors, NoEpisodes) {
    Log log;
    FakeBackend backend;
    run_actors(small(), episodes(log, 0), backend);
    EXPECT_EQ(log.finished, 0);
}

TEST(Actors, FewerEpisodesThanActorsStillExits) {
    Log log;
    FakeBackend backend;
    run_actors(small(), episodes(log, 2), backend);
    EXPECT_EQ(log.finished, 2);
}

// --- Failures surface as errors, and nothing is recorded as finished ---------------------------------------

TEST(ActorFailures, BackendThrows) {
    Log log;
    expect_error(small(), log, episodes(log, 20, three_steps), FakeBackend{[](auto) -> std::vector<Prediction> {
        throw std::runtime_error{"backend failure"};
    }}, "backend failure");
    EXPECT_EQ(log.finished, 0);
}

TEST(ActorFailures, OneLaneFailsWhileAnotherStillOwnsInputs) {
    Log log;
    std::atomic<int> calls{0};
    std::atomic<bool> failed{false};
    FakeBackend backend{[&](auto inputs) {
        if (calls++ != 0) {
            failed = true;
            throw std::runtime_error{"lane failed"};
        }
        while (!failed) std::this_thread::sleep_for(1ms);
        std::this_thread::sleep_for(10ms);  // cancellation must not destroy the first lane's borrowed tokens
        require(log.alive > 0, "episodes freed before both lanes returned");
        return echo(inputs);
    }, {}, 2};
    expect_error(small(3, 2), log, episodes(log, 40, three_steps), std::move(backend), "lane failed");
}

TEST(ActorFailures, WrongPredictionCount) {
    Log log;
    expect_error(small(), log, episodes(log, 20), FakeBackend{[](auto) { return std::vector<Prediction>{}; }},
                 "prediction count mismatch");
}

TEST(ActorFailures, NonFinitePrediction) {
    Log log;
    expect_error(small(), log, episodes(log, 20), FakeBackend{[](auto inputs) {
        auto out = echo(inputs);
        out[0].value = std::numeric_limits<float>::quiet_NaN();
        return out;
    }}, "invalid prediction");
}

TEST(ActorFailures, EpisodeSourceThrows) {
    Log log;
    auto next = episodes(log, 20);
    expect_error(small(), log, [&]() -> std::unique_ptr<Episode> {
        if (log.started >= 3) throw std::runtime_error{"start failure"};
        return next();
    }, FakeBackend{}, "start failure");
}

TEST(ActorFailures, SlowInferenceTimesOutWithoutFreeingPendingEpisodes) {
    Log log;
    auto config = small();
    config.timeout = 30ms;
    expect_error(config, log, episodes(log, 20, three_steps), FakeBackend{[&log](auto inputs) {
        std::this_thread::sleep_for(100ms);
        require(log.alive > 0, "episodes freed while their request was pending");
        return echo(inputs);
    }}, "inference timeout");
    EXPECT_EQ(log.finished, 0);
}

TEST(ActorFailures, StalledEpisodeTimesOut) {
    struct Stalls final : Episode {
        Log& log;
        explicit Stalls(Log& l) : log{l} {}
        std::optional<EvaluationRequest> next_request() override { std::this_thread::sleep_for(100ms); return std::nullopt; }
        void completed(std::uint64_t, const Prediction&) override {}
        bool done() const override { return false; }
        void finish() override { ++log.finished; }
        std::string label() const override { return "stalls"; }
    };
    Log log;
    auto config = small();
    config.timeout = 30ms;
    expect_error(config, log, [&]() -> std::unique_ptr<Episode> {
        return log.started++ == 0 ? std::make_unique<Stalls>(log) : nullptr;
    }, FakeBackend{}, "actor timeout");
    EXPECT_EQ(log.finished, 0);
}

TEST(ActorFailures, InvalidConfig) {
    Log log;
    expect_error(small(0), log, episodes(log, 1), FakeBackend{}, "invalid actor configuration");
    expect_error(small(), log, episodes(log, 1), FakeBackend{echo, {}, 0}, "invalid actor configuration");  // no lanes
}

// A backend call that never returns cannot be interrupted: the process must exit rather than hang.
TEST(ActorsDeathTest, HungBackendExitsProcess) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT({
        Log log;
        auto config = small();
        config.timeout = 30ms; config.shutdown_timeout = 60ms;
        FakeBackend backend{[](auto) -> std::vector<Prediction> { for (;;) std::this_thread::sleep_for(1s); }};
        run_actors(config, episodes(log, 20, three_steps), backend);
    }, testing::ExitedWithCode(1), "shutdown timeout");
}

TEST(Actors, MixedBackendsPreserveRepliesWithoutWideWorkBlockingNarrow) {
    for (bool wide_accepts : {true, false}) {
        Log log;
        std::atomic<int> fast_calls{0}, fast_rows{0}, wide_rows{0};
        const auto fits = [](const Tokens& t) { return (int(t.data[0] * 1000 + .5f) + int(t.data[1])) % 2 == 0; };
        FakeBackend narrow{[&](auto inputs) {
            require(inputs.size() <= 7, "narrow batch too large");
            for (const auto& i : inputs) require(fits(*i.tokens) && i.packed.size() == 1, "wrong narrow request");
            ++fast_calls; fast_rows += inputs.size();
            return echo(inputs);
        }, [](const Tokens&, auto& packed) { packed.assign(1, 1); }, 2, fits, 7};
        FakeBackend wide{[&](auto inputs) {
            require(inputs.size() <= 2, "wide batch too large");
            for (const auto& i : inputs) require(!fits(*i.tokens) && i.packed.size() == 2, "wrong wide request");
            const auto deadline = std::chrono::steady_clock::now() + 1s;
            while (fast_calls < 2 && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(1ms);
            require(fast_calls >= 2, "wide work blocked narrow lanes");
            wide_rows += inputs.size();
            return echo(inputs);
        }, [](const Tokens&, auto& packed) { packed.assign(2, 1); }, 1,
           [=](const Tokens&) { return wide_accepts; }, 2};
        run_actors(small(7, 7), episodes(log, 100), {&narrow, &wide});
        EXPECT_EQ(log.finished, 100);
        EXPECT_EQ(log.evaluated, total_steps(100));
        EXPECT_EQ(fast_rows + wide_rows, total_steps(100));
        EXPECT_GT(fast_rows, 0);
        EXPECT_GT(wide_rows, 0);
        EXPECT_EQ(log.alive, 0);
    }
}

TEST(Actors, RowsNoBackendAcceptsReachOnlyTheLastWithoutBlockingTheOthers) {
    Log log;
    std::atomic<int> wide_calls{0}, rows{0};
    const auto kind = [](const Tokens& t) { return (int(t.data[0] * 1000 + .5f) + int(t.data[1])) % 3; };
    const auto only = [&](int k, const char* name) {
        return [&, k, name](std::span<const Input> inputs) {
            for (const auto& i : inputs) require(kind(*i.tokens) == k, name);
            rows += int(inputs.size());
            return echo(inputs);
        };
    };
    FakeBackend narrow{only(0, "narrow got a row it does not accept"), {}, 1, [&](const Tokens& t) { return kind(t) == 0; }, 7};
    FakeBackend wide{[&, check = only(1, "wide got a row it does not accept")](auto inputs) {
        ++wide_calls;
        return check(inputs);
    }, {}, 1, [&](const Tokens& t) { return kind(t) == 1; }, 7};
    FakeBackend overflow{[&, check = only(2, "overflow got an accepted row")](auto inputs) {
        const auto deadline = std::chrono::steady_clock::now() + 1s;
        while (wide_calls < 2 && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(1ms);
        require(wide_calls >= 2, "overflow work blocked the wide lane");
        return check(inputs);
    }, {}, 1, [](const Tokens&) { return false; }, 7};
    run_actors(small(7, 7), episodes(log, 100), {&narrow, &wide, &overflow});
    EXPECT_EQ(log.finished, 100);
    EXPECT_EQ(rows, total_steps(100));
    EXPECT_EQ(log.alive, 0);
}

TEST(Actors, OneEpisodePipelinesAcrossBackendsWithoutWaitingForTheSlowReply) {
    struct Pipeline final : Episode {
        std::map<std::uint64_t, Tokens> pending;
        int issued = 0;
        std::vector<std::uint64_t>& completions;
        bool& finished;
        Pipeline(std::vector<std::uint64_t>& c, bool& f) : completions{c}, finished{f} {}
        std::optional<EvaluationRequest> next_request() override {
            if (issued == 3 || pending.size() == 2) return std::nullopt;
            auto [it, inserted] = pending.emplace(issued, Tokens{std::vector<float>(2 * token_width), 2, 1});
            it->second.data[0] = 0.5f;
            it->second.data[1] = float(issued);
            return EvaluationRequest{std::uint64_t(issued++), &it->second};
        }
        void completed(std::uint64_t id, const Prediction& p) override {
            require(p.value == 0.5f && p.logits == std::vector<float>{float(id)}, "pipeline prediction misrouted");
            require(pending.erase(id) == 1, "duplicate completion");
            completions.push_back(id);
        }
        bool done() const override { return issued == 3 && pending.empty(); }
        void finish() override {
            require(issued == 3 && pending.empty() && completions.size() == 3, "pipeline finished early");
            finished = true;
        }
        std::string label() const override { return "pipeline"; }
    };
    std::atomic<bool> continued{false};
    auto prepare = [](const Tokens& tokens, std::vector<std::uint16_t>& packed) {
        packed = {std::uint16_t(tokens.data[1])};
    };
    FakeBackend slow{[&](std::span<const Input> inputs) {
        const auto deadline = std::chrono::steady_clock::now() + 1s;
        while (!continued && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(1ms);
        require(continued, "ready branch could not continue while slow request was outstanding");
        require(inputs.front().tokens->data[1] == 0 && inputs.front().packed[0] == 0,
                "pending token or packed storage was invalidated");
        return echo(inputs);
    }, prepare, 1, [](const Tokens& tokens) { return tokens.data[1] == 0; }, 1};
    FakeBackend fast{[&](std::span<const Input> inputs) {
        for (const auto& input : inputs) {
            require(input.packed[0] == std::uint16_t(input.tokens->data[1]), "packed request misrouted");
            if (input.tokens->data[1] == 2) continued = true;
        }
        return echo(inputs);
    }, prepare, 1, {}, 1};
    std::vector<std::uint64_t> completed;
    bool started = false, finished = false;
    auto config = small(1, 1);
    config.actors = 1;
    run_actors(config, [&]() -> std::unique_ptr<Episode> {
        if (started) return nullptr;
        started = true;
        return std::make_unique<Pipeline>(completed, finished);
    }, {&slow, &fast});
    EXPECT_TRUE(finished);
    ASSERT_EQ(completed.size(), 3);
    EXPECT_EQ(completed.front(), 1);
    std::sort(completed.begin(), completed.end());
    EXPECT_EQ(completed, (std::vector<std::uint64_t>{0, 1, 2}));
}

TEST(ActorFailures, InvalidRequestLifecyclesAreRejectedRatherThanFinishedOrStalled) {
    enum class Violation { blocked_without_requests, done_with_requests, request_when_done, duplicate_id };
    struct Invalid final : Episode {
        Violation violation;
        std::atomic<bool>& second_claim;
        int claims = 0;
        Tokens tokens{std::vector<float>(2 * token_width, 0.f), 2, 1};
        Invalid(Violation v, std::atomic<bool>& s) : violation{v}, second_claim{s} {}
        std::optional<EvaluationRequest> next_request() override {
            if (violation == Violation::blocked_without_requests) return std::nullopt;
            if (++claims == 2) {
                second_claim = true;
                if (violation == Violation::done_with_requests) return std::nullopt;
            }
            return EvaluationRequest{0, &tokens};
        }
        void completed(std::uint64_t, const Prediction&) override {}
        bool done() const override {
            return violation == Violation::request_when_done ||
                   (violation == Violation::done_with_requests && claims == 2);
        }
        void finish() override { throw std::runtime_error{"invalid episode was finished"}; }
        std::string label() const override { return "invalid-lifecycle"; }
    };
    for (const auto& [violation, error] : {
            std::pair{Violation::blocked_without_requests, "episode blocked without outstanding requests"},
            std::pair{Violation::done_with_requests, "episode done with outstanding requests"},
            std::pair{Violation::request_when_done, "done episode produced a request"},
            std::pair{Violation::duplicate_id, "duplicate episode request id"}}) {
        SCOPED_TRACE(error);
        std::atomic<bool> second_claim{false};
        bool started = false;
        Log log;
        auto config = small(1, 1);
        config.actors = 1;
        FakeBackend backend{[&](std::span<const Input> inputs) {
            const auto deadline = std::chrono::steady_clock::now() + 1s;
            while (!second_claim && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(1ms);
            require(second_claim, "actor did not attempt to claim available work");
            return echo(inputs);
        }};
        expect_error(config, log, [&]() -> std::unique_ptr<Episode> {
            if (started) return nullptr;
            started = true;
            return std::make_unique<Invalid>(violation, second_claim);
        }, std::move(backend), error);
    }
}
