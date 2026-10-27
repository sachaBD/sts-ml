#include "agents/combat/nn/workers.hpp"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#if defined(__unix__)
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace stsrl::nn;
using namespace std::chrono_literals;
namespace {
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error{message}; }
struct Counts {
    std::atomic<int> next{0}, finished{0}, evaluated{0}, alive{0}, maximum{0};
    std::mutex mutex;
    std::map<std::thread::id, int> local;
    int max_local = 0;
};
class FakeGame final : public Simulation {
    Counts& counts_;
    std::thread::id owner_ = std::this_thread::get_id();
    int id_, remaining_, step_ = 0;
    bool waiting_ = false, finished_ = false;
    Tokens tokens_;
    void owner() const { check(std::this_thread::get_id() == owner_, "simulation migrated threads"); }
public:
    FakeGame(Counts& counts, int id, int steps) : counts_{counts}, id_{id}, remaining_{steps} {
        tokens_.count = 2; tokens_.actions = 1; tokens_.data.resize(2 * width);
        tokens_.data[0] = 1; tokens_.data[width] = 6;
        tokens_.data[3] = float(id_ + 1) / 1000;
        const int alive = ++counts_.alive;
        for (int previous = counts_.maximum; previous < alive && !counts_.maximum.compare_exchange_weak(previous, alive);) {}
        std::lock_guard lock{counts_.mutex};
        counts_.max_local = std::max(counts_.max_local, ++counts_.local[owner_]);
    }
    ~FakeGame() override {
        --counts_.alive;
        std::lock_guard lock{counts_.mutex}; --counts_.local[owner_];
    }
    const Tokens* advance() override {
        owner(); check(!waiting_ && !finished_, "advanced waiting/finished game");
        if (!remaining_) return nullptr;
        waiting_ = true; tokens_.data[4] = float(step_);
        return &tokens_;
    }
    void evaluated(const Prediction& p) override {
        owner(); check(waiting_, "duplicate/unexpected prediction");
        check(p.value == float(id_ + 1) / 1000 && p.logits == std::vector<float>{float(step_)}, "misrouted prediction");
        waiting_ = false; --remaining_; ++step_; ++counts_.evaluated;
    }
    void finish() override {
        owner(); check(!waiting_ && !remaining_ && !finished_, "incorrect finish");
        finished_ = true; ++counts_.finished;
    }
    std::string label() const override { return "fake-" + std::to_string(id_); }
};
std::vector<Prediction> evaluate(std::vector<const Tokens*> tokens) {
    std::vector<Prediction> result;
    for (auto t : tokens) result.push_back({t->data[3], {t->data[4]}});
    return result;
}
NextSimulation games(Counts& counts, int count, bool varied = true) {
    return [&, count, varied]() -> std::unique_ptr<Simulation> {
        const int id = counts.next++;
        if (id >= count) return nullptr;
        return std::make_unique<FakeGame>(counts, id, varied ? id % 19 : 3);
    };
}
WorkerConfig config() {
    WorkerConfig c; c.workers = 4; c.slots = 3; c.batch = 7;
    c.timeout = 2s; c.shutdown_timeout = 1s;
    return c;
}
void expect_failure(const WorkerConfig& c, Counts& counts, NextSimulation next, Evaluate eval, const std::string& text) {
    bool failed = false;
    try { run_workers(c, std::move(next), std::move(eval)); }
    catch (const std::exception& e) { failed = std::string{e.what()}.find(text) != std::string::npos; }
    check(failed, "expected error not propagated");
    check(counts.alive == 0, "games survived completed shutdown");
}
void tests() {
    for (int repeat = 0; repeat < 8; ++repeat) {
        Counts counts;
        auto c = config();
        c.slots = repeat % 3 + 1;
        c.batch = repeat % 2 ? 2 : 17; // queues smaller/larger than a batch, repeated slot reuse
        run_workers(c, games(counts, 100), evaluate);
        int expected = 0; for (int i = 0; i < 100; ++i) expected += i % 19;
        check(counts.finished == 100 && counts.evaluated == expected, "lost/duplicated work");
        check(counts.alive == 0 && counts.maximum <= c.workers * c.slots && counts.max_local <= c.slots, "capacity exceeded");
    }
    {
        Counts counts;
        run_workers(config(), games(counts, 40), [&](std::vector<InferenceInput> inputs) {
            std::this_thread::sleep_for(1ms); // owners must retain buffers while inference is pending
            std::vector<const Tokens*> tokens;
            for (const auto& input : inputs) {
                check(input.packed.size() == 2 && input.packed[0] == std::uint16_t(input.tokens->data[3]*1000+.5f) &&
                      input.packed[1] == std::uint16_t(input.tokens->data[4]), "stale prepared input");
                tokens.push_back(input.tokens);
            }
            return evaluate(std::move(tokens));
        }, [&](const Tokens& tokens, std::vector<std::uint16_t>& packed) {
            std::lock_guard lock{counts.mutex};
            check(counts.local[std::this_thread::get_id()] > 0, "packing not on owner worker");
            packed.resize(2);
            packed[0] = std::uint16_t(tokens.data[3]*1000+.5f);
            packed[1] = std::uint16_t(tokens.data[4]);
        });
        check(counts.finished == 40 && counts.alive == 0, "prepared games lost");
    }
    { Counts counts; run_workers(config(), games(counts, 0), evaluate); check(counts.finished == 0, "empty job evaluated"); }
    { Counts counts; run_workers(config(), games(counts, 2), evaluate); check(counts.finished == 2, "idle workers failed to exit"); }
    {
        Counts counts;
        expect_failure(config(), counts, games(counts, 20, false), [](auto) -> std::vector<Prediction> {
            throw std::runtime_error{"backend failure"};
        }, "backend failure");
        check(counts.finished == 0, "failed inference recorded a game");
    }
    {
        Counts counts;
        expect_failure(config(), counts, games(counts, 20), [](auto) { return std::vector<Prediction>{}; }, "count mismatch");
    }
    {
        Counts counts;
        expect_failure(config(), counts, games(counts, 20), [](auto tokens) {
            auto result = evaluate(tokens); result[0].value = std::numeric_limits<float>::quiet_NaN(); return result;
        }, "invalid inference");
    }
    {
        Counts counts;
        auto next = games(counts, 20);
        expect_failure(config(), counts, [&]() -> std::unique_ptr<Simulation> {
            if (counts.next >= 3) throw std::runtime_error{"start failure"};
            return next();
        }, evaluate, "start failure");
    }
    {
        Counts counts;
        auto c = config(); c.timeout = 30ms;
        expect_failure(c, counts, games(counts, 20, false), [&](auto tokens) {
            std::this_thread::sleep_for(100ms);
            // Failed workers must not destroy their slots while inference still uses the views.
            check(counts.alive > 0, "cancelled games freed during inference");
            return evaluate(tokens);
        }, "timeout");
        check(counts.finished == 0, "timeout recorded as a game result");
    }
    {
        Counts counts;
        class SlowFinish final : public Simulation {
            Counts& counts_;
        public:
            explicit SlowFinish(Counts& counts) : counts_{counts} {}
            const Tokens* advance() override { std::this_thread::sleep_for(100ms); return nullptr; }
            void evaluated(const Prediction&) override { throw std::runtime_error{"unexpected prediction"}; }
            void finish() override { ++counts_.finished; }
            std::string label() const override { return "slow-simulation"; }
        };
        auto c = config(); c.timeout = 30ms;
        expect_failure(c, counts, [&]() -> std::unique_ptr<Simulation> {
            return counts.next++ == 0 ? std::make_unique<SlowFinish>(counts) : nullptr;
        }, evaluate, "simulation timeout");
        check(counts.finished == 0, "timed-out simulation recorded as finished");
    }
    {
        auto c = config(); c.slots = 0;
        Counts counts; expect_failure(c, counts, games(counts, 1), evaluate, "invalid simulation");
    }
}
}
int main(int argc, char** argv) try {
    if (argc == 2 && std::string{argv[1]} == "--hang") {
        Counts counts; auto c = config(); c.timeout = 30ms; c.shutdown_timeout = 60ms;
        run_workers(c, games(counts, 20, false), [](auto) -> std::vector<Prediction> {
            for (;;) std::this_thread::sleep_for(1s);
        });
        return 2;
    }
    tests();
#if defined(__unix__)
    const auto pid = fork();
    if (pid == 0) { execl(argv[0], argv[0], "--hang", static_cast<char*>(nullptr)); _exit(3); }
    check(pid > 0, "fork failed");
    int status = 0;
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (waitpid(pid, &status, WNOHANG) == 0) {
        if (std::chrono::steady_clock::now() >= deadline) { kill(pid, SIGKILL); waitpid(pid, &status, 0); throw std::runtime_error{"watchdog failed to exit"}; }
        std::this_thread::sleep_for(10ms);
    }
    check(WIFEXITED(status) && WEXITSTATUS(status) == 1, "hung backend did not fail nonzero");
#endif
    std::cout << "worker tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
}
