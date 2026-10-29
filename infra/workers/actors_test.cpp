// Scheduling, ownership and failure tests for run_actors. No GPU or ONNX Runtime needed.
#include "infra/workers/actors.hpp"
#include <algorithm>
#include <atomic>
#include <csignal>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using namespace stsrl::infra;
using namespace std::chrono_literals;

namespace {
constexpr int token_width = 8;
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error{message}; }

struct Counts {
    std::atomic<int> next{0}, finished{0}, evaluated{0}, alive{0}, maximum{0};
    std::mutex mutex;
    std::map<std::thread::id, int> local;  // live episodes per Actor thread
    int max_local = 0;
};

// Asks for `steps` predictions, each of which must come back to it, on its own thread, exactly once.
class FakeEpisode final : public Episode {
    Counts& counts_;
    std::thread::id owner_ = std::this_thread::get_id();
    int id_, remaining_, step_ = 0;
    bool waiting_ = false, finished_ = false;
    Tokens tokens_;
    void owned() const { check(std::this_thread::get_id() == owner_, "episode migrated threads"); }
public:
    FakeEpisode(Counts& counts, int id, int steps) : counts_{counts}, id_{id}, remaining_{steps} {
        tokens_.count = 2; tokens_.actions = 1; tokens_.data.resize(2 * token_width);
        tokens_.data[0] = float(id_ + 1) / 1000;
        const int alive = ++counts_.alive;
        for (int seen = counts_.maximum; seen < alive && !counts_.maximum.compare_exchange_weak(seen, alive);) {}
        std::lock_guard lock{counts_.mutex};
        counts_.max_local = std::max(counts_.max_local, ++counts_.local[owner_]);
    }
    ~FakeEpisode() override {
        --counts_.alive;
        std::lock_guard lock{counts_.mutex};
        --counts_.local[owner_];
    }
    const Tokens* advance() override {
        owned(); check(!waiting_ && !finished_, "advanced a waiting/finished episode");
        if (!remaining_) return nullptr;
        waiting_ = true; tokens_.data[1] = float(step_);
        return &tokens_;
    }
    void evaluated(const Prediction& p) override {
        owned(); check(waiting_, "unexpected prediction");
        check(p.value == float(id_ + 1) / 1000 && p.logits == std::vector<float>{float(step_)}, "misrouted prediction");
        waiting_ = false; --remaining_; ++step_; ++counts_.evaluated;
    }
    void finish() override {
        owned(); check(!waiting_ && !remaining_ && !finished_, "incorrect finish");
        finished_ = true; ++counts_.finished;
    }
    std::string label() const override { return "fake-" + std::to_string(id_); }
};

// Echoes each episode's identity back so misrouting is detectable.
std::vector<Prediction> echo(std::span<const Input> inputs) {
    std::vector<Prediction> out;
    for (const auto& input : inputs) out.push_back({input.tokens->data[0], {input.tokens->data[1]}});
    return out;
}

using Evaluate = std::function<std::vector<Prediction>(std::span<const Input>)>;
using Prepare = std::function<void(const Tokens&, std::vector<std::uint16_t>&)>;
class FakeBackend final : public Backend {
    Evaluate evaluate_;
    Prepare prepare_;
public:
    explicit FakeBackend(Evaluate e = echo, Prepare p = {}) : evaluate_{std::move(e)}, prepare_{std::move(p)} {}
    int width() const override { return token_width; }
    int max_batch() const override { return 64; }
    void prepare(const Tokens& t, std::vector<std::uint16_t>& packed) const override { if (prepare_) prepare_(t, packed); else packed.clear(); }
    std::vector<Prediction> evaluate(std::span<const Input> inputs) override { return evaluate_(inputs); }
};

NextEpisode episodes(Counts& counts, int count, bool varied = true) {
    return [&, count, varied]() -> std::unique_ptr<Episode> {
        const int id = counts.next++;
        if (id >= count) return nullptr;
        return std::make_unique<FakeEpisode>(counts, id, varied ? id % 19 : 3);
    };
}

ActorConfig config() {
    ActorConfig c; c.actors = 4; c.slots = 3; c.batch = 7; c.timeout = 2s; c.shutdown_timeout = 1s;
    return c;
}

void expect_failure(const ActorConfig& c, Counts& counts, NextEpisode next, Backend&& backend, const std::string& text) {
    bool failed = false;
    try { run_actors(c, std::move(next), backend); }
    catch (const std::exception& e) { failed = std::string{e.what()}.find(text) != std::string::npos; }
    check(failed, "expected error not propagated");
    check(counts.alive == 0, "episodes survived shutdown");
}

void tests() {
    for (int repeat = 0; repeat < 8; ++repeat) {  // queues smaller/larger than a batch, repeated slot reuse
        Counts counts;
        auto c = config();
        c.slots = repeat % 3 + 1;
        c.batch = repeat % 2 ? 2 : 17;
        FakeBackend backend;
        run_actors(c, episodes(counts, 100), backend);
        int expected = 0;
        for (int i = 0; i < 100; ++i) expected += i % 19;
        check(counts.finished == 100 && counts.evaluated == expected, "lost or duplicated work");
        check(counts.alive == 0 && counts.maximum <= c.actors * c.slots && counts.max_local <= c.slots, "capacity exceeded");
    }
    {
        Counts counts;
        FakeBackend backend{[](std::span<const Input> inputs) {
            std::this_thread::sleep_for(1ms);  // owners must keep their buffers while inference is pending
            for (const auto& input : inputs)
                check(input.packed.size() == 2 && input.packed[0] == std::uint16_t(input.tokens->data[0] * 1000 + .5f) &&
                      input.packed[1] == std::uint16_t(input.tokens->data[1]), "stale prepared input");
            return echo(inputs);
        }, [&](const Tokens& tokens, std::vector<std::uint16_t>& packed) {
            std::lock_guard lock{counts.mutex};
            check(counts.local[std::this_thread::get_id()] > 0, "prepare not on the owning Actor");
            packed = {std::uint16_t(tokens.data[0] * 1000 + .5f), std::uint16_t(tokens.data[1])};
        }};
        run_actors(config(), episodes(counts, 40), backend);
        check(counts.finished == 40 && counts.alive == 0, "prepared episodes lost");
    }
    { Counts counts; FakeBackend b; run_actors(config(), episodes(counts, 0), b); check(counts.finished == 0, "empty job"); }
    { Counts counts; FakeBackend b; run_actors(config(), episodes(counts, 2), b); check(counts.finished == 2, "idle actors did not exit"); }
    {
        Counts counts;
        expect_failure(config(), counts, episodes(counts, 20, false),
                       FakeBackend{[](auto) -> std::vector<Prediction> { throw std::runtime_error{"backend failure"}; }}, "backend failure");
        check(counts.finished == 0, "failed inference recorded an episode");
    }
    {
        Counts counts;
        expect_failure(config(), counts, episodes(counts, 20), FakeBackend{[](auto) { return std::vector<Prediction>{}; }}, "count mismatch");
    }
    {
        Counts counts;
        expect_failure(config(), counts, episodes(counts, 20), FakeBackend{[](auto inputs) {
            auto out = echo(inputs); out[0].value = std::numeric_limits<float>::quiet_NaN(); return out;
        }}, "invalid prediction");
    }
    {
        Counts counts;
        auto next = episodes(counts, 20);
        expect_failure(config(), counts, [&]() -> std::unique_ptr<Episode> {
            if (counts.next >= 3) throw std::runtime_error{"start failure"};
            return next();
        }, FakeBackend{}, "start failure");
    }
    {
        Counts counts;
        auto c = config(); c.timeout = 30ms;
        expect_failure(c, counts, episodes(counts, 20, false), FakeBackend{[&](auto inputs) {
            std::this_thread::sleep_for(100ms);
            check(counts.alive > 0, "cancelled episodes freed during inference");  // views must outlive the call
            return echo(inputs);
        }}, "timeout");
        check(counts.finished == 0, "timeout recorded as a result");
    }
    {
        Counts counts;
        class Slow final : public Episode {
            Counts& counts_;
        public:
            explicit Slow(Counts& counts) : counts_{counts} {}
            const Tokens* advance() override { std::this_thread::sleep_for(100ms); return nullptr; }
            void evaluated(const Prediction&) override { throw std::runtime_error{"unexpected prediction"}; }
            void finish() override { ++counts_.finished; }
            std::string label() const override { return "slow"; }
        };
        auto c = config(); c.timeout = 30ms;
        expect_failure(c, counts, [&]() -> std::unique_ptr<Episode> {
            return counts.next++ == 0 ? std::make_unique<Slow>(counts) : nullptr;
        }, FakeBackend{}, "actor timeout");
        check(counts.finished == 0, "timed-out episode recorded as finished");
    }
    { auto c = config(); c.slots = 0; Counts counts; expect_failure(c, counts, episodes(counts, 1), FakeBackend{}, "invalid actor"); }
    { auto c = config(); c.batch = 65; Counts counts; expect_failure(c, counts, episodes(counts, 1), FakeBackend{}, "invalid actor"); }
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc == 2 && std::string{argv[1]} == "--hang") {  // child: a backend call that never returns
        Counts counts; auto c = config(); c.timeout = 30ms; c.shutdown_timeout = 60ms;
        FakeBackend backend{[](auto) -> std::vector<Prediction> { for (;;) std::this_thread::sleep_for(1s); }};
        run_actors(c, episodes(counts, 20, false), backend);
        return 2;
    }
    tests();
    const auto pid = fork();
    if (pid == 0) { execl(argv[0], argv[0], "--hang", static_cast<char*>(nullptr)); _exit(3); }
    check(pid > 0, "fork failed");
    int status = 0;
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (waitpid(pid, &status, WNOHANG) == 0) {
        if (std::chrono::steady_clock::now() >= deadline) { kill(pid, SIGKILL); waitpid(pid, &status, 0); throw std::runtime_error{"hung backend was not killed"}; }
        std::this_thread::sleep_for(10ms);
    }
    check(WIFEXITED(status) && WEXITSTATUS(status) == 1, "hung backend did not exit nonzero");
    std::cout << "actors tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
