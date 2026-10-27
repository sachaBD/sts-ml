#include "agents/combat/nn/workers.hpp"
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

namespace stsrl::nn {
namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
long long now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count(); }

struct Request {
    std::size_t slot;
    std::uint64_t generation;
    const Tokens* tokens;
};
struct Completion {
    std::size_t slot;
    std::uint64_t generation;
    Prediction prediction;
};

// One mutex per worker, deliberately not a lock-free queue. Capacity of EACH queue is slots.
struct Mailbox {
    std::mutex mutex;
    std::condition_variable completed;
    std::deque<Request> requests;
    std::deque<Completion> completions;
    std::atomic<long long> heartbeat{now_ms()};
    std::atomic<int> current_slot{-1};
    std::atomic<bool> done{false};
};

struct Runtime {
    const WorkerConfig config;
    NextSimulation next;
    Evaluate evaluate;
    std::vector<std::unique_ptr<Mailbox>> mailboxes;
    std::atomic<bool> stopped{false}, coordinator_done{false};
    std::atomic<int> workers_done{0}, queued{0};
    std::atomic<long long> coordinator_heartbeat{now_ms()};
    std::mutex wake_mutex, error_mutex;
    std::condition_variable requested;
    std::exception_ptr error;

    Runtime(WorkerConfig c, NextSimulation n, Evaluate e) : config{c}, next{std::move(n)}, evaluate{std::move(e)} {
        for (int i = 0; i < config.workers; ++i) mailboxes.push_back(std::make_unique<Mailbox>());
    }
    void notify() {
        // Pair notifications with the wait mutex so an empty-to-nonempty transition cannot be lost.
        std::lock_guard lock{wake_mutex};
        requested.notify_one();
    }
    void fail(std::exception_ptr failure) {
        { std::lock_guard lock{error_mutex}; if (!error) error = failure; }
        stopped = true;
        notify(); // workers also wake at least every 10 ms to check cancellation/timeouts
    }
    bool all_done() const { return coordinator_done && workers_done == config.workers; }
};

class SimWorker {
    enum class Phase { ready, waiting_for_nn, finished };
    struct Slot {
        std::unique_ptr<Simulation> game;
        Phase phase = Phase::finished;
        std::uint64_t generation = 0; // incremented for EVERY request, including after slot reuse
        Clock::time_point submitted;
    };
    Runtime& runtime_;
    const int id_;
    Mailbox& mailbox_;
    std::vector<Slot> slots_;
    std::deque<std::size_t> ready_;
    int active_ = 0;

    void touch(std::size_t slot) {
        mailbox_.current_slot = int(slot);
        mailbox_.heartbeat = now_ms();
    }
    void refill(std::size_t index) {
        auto& slot = slots_[index];
        slot.game = runtime_.next(); // shared start queue; construction happens on the owning worker
        if (slot.game) { slot.phase = Phase::ready; ready_.push_back(index); ++active_; }
        else slot.phase = Phase::finished;
    }
    void receive_completions() {
        std::deque<Completion> completions;
        { std::lock_guard lock{mailbox_.mutex}; completions.swap(mailbox_.completions); }
        for (auto& completion : completions) {
            if (completion.slot >= slots_.size()) throw std::runtime_error{"completion has invalid slot"};
            auto& slot = slots_[completion.slot];
            if (slot.phase != Phase::waiting_for_nn || slot.generation != completion.generation)
                throw std::runtime_error{"stale or duplicate completion"};
            touch(completion.slot);
            slot.game->evaluated(completion.prediction); // only the owner touches simulator/search state
            slot.phase = Phase::ready;
            ready_.push_back(completion.slot);
        }
    }
    void check_timeouts() const {
        const auto now = Clock::now();
        for (std::size_t i = 0; i < slots_.size(); ++i) {
            const auto& slot = slots_[i];
            if (slot.phase == Phase::waiting_for_nn && now - slot.submitted >= runtime_.config.timeout)
                throw std::runtime_error{"NN timeout: worker " + std::to_string(id_) + " slot " + std::to_string(i) +
                    " request " + std::to_string(slot.generation) + " fight " + slot.game->label()};
        }
    }
    void advance(std::size_t index) {
        auto& slot = slots_[index];
        if (slot.phase != Phase::ready) throw std::runtime_error{"ready queue state mismatch"};
        touch(index);
        const Tokens* tokens = slot.game->advance();
        if (runtime_.stopped) return; // a timed-out native simulation must not be recorded as finished
        if (tokens) {
            if (tokens->count <= 0 || tokens->actions <= 0 || tokens->actions > tokens->count ||
                tokens->data.size() != std::size_t(tokens->count) * width)
                throw std::runtime_error{"invalid pending tokens"};
            slot.phase = Phase::waiting_for_nn;
            slot.submitted = Clock::now();
            const Request request{index, ++slot.generation, tokens};
            {
                std::lock_guard lock{mailbox_.mutex};
                // One outstanding request per slot makes overflow a bug, never a blocking push.
                if (mailbox_.requests.size() >= slots_.size()) throw std::runtime_error{"request queue overflow"};
                mailbox_.requests.push_back(request);
                ++runtime_.queued;
            }
            runtime_.notify();
        } else {
            slot.game->finish(); // output belongs to workers, not the inference coordinator
            slot.game.reset();
            slot.phase = Phase::finished;
            --active_;
            refill(index);
        }
    }
public:
    SimWorker(Runtime& runtime, int id) : runtime_{runtime}, id_{id}, mailbox_{*runtime.mailboxes[id]},
                                         slots_(runtime.config.slots) {}
    void run() {
        for (std::size_t i = 0; i < slots_.size() && !runtime_.stopped; ++i) { touch(i); refill(i); }
        auto check_at = Clock::now();
        while (!runtime_.stopped && active_ != 0) {
            mailbox_.heartbeat = now_ms();
            receive_completions();
            if (Clock::now() >= check_at) { check_timeouts(); check_at = Clock::now() + 10ms; }
            if (!ready_.empty()) {
                const auto index = ready_.front(); ready_.pop_front();
                advance(index);
            } else {
                std::unique_lock lock{mailbox_.mutex};
                mailbox_.completed.wait_for(lock, 10ms, [&] { return runtime_.stopped || !mailbox_.completions.empty(); });
            }
        }
        // Slots deliberately remain owned by this object until the coordinator has joined, even on failure.
    }
};

class Coordinator {
    struct RoutedRequest { int worker; Request request; };
    Runtime& runtime_;
    int cursor_ = 0;
    long batches_ = 0, leaves_ = 0;

    bool take_one(std::vector<RoutedRequest>& batch) {
        for (int i = 0; i < runtime_.config.workers; ++i) {
            const int worker = cursor_;
            cursor_ = (cursor_ + 1) % runtime_.config.workers;
            auto& mailbox = *runtime_.mailboxes[worker];
            std::lock_guard lock{mailbox.mutex};
            if (!mailbox.requests.empty()) {
                batch.push_back({worker, mailbox.requests.front()});
                mailbox.requests.pop_front();
                --runtime_.queued;
                return true;
            }
        }
        return false;
    }
    std::vector<RoutedRequest> gather() {
        std::vector<RoutedRequest> batch;
        batch.reserve(runtime_.config.batch);
        const auto deadline = Clock::now() + runtime_.config.batch_wait;
        while (!runtime_.stopped && int(batch.size()) < runtime_.config.batch) {
            if (take_one(batch)) continue;
            if (Clock::now() >= deadline) break;
            std::unique_lock lock{runtime_.wake_mutex};
            runtime_.requested.wait_until(lock, deadline, [&] { return runtime_.stopped || runtime_.queued != 0; });
        }
        return batch;
    }
    void dispatch(const std::vector<RoutedRequest>& batch) {
        std::vector<const Tokens*> tokens;
        for (const auto& item : batch) tokens.push_back(item.request.tokens);
        auto predictions = runtime_.evaluate(std::move(tokens)); // one synchronous backend, on THIS thread
        if (runtime_.stopped) return;
        if (predictions.size() != batch.size()) throw std::runtime_error{"inference result count mismatch"};
        // Validate before publishing anything: publication allows owners to mutate the pending tokens again.
        for (std::size_t i = 0; i < batch.size(); ++i) {
            const auto& p = predictions[i];
            if (p.logits.size() != std::size_t(batch[i].request.tokens->actions) ||
                !std::isfinite(p.value) || p.value < 0 || p.value > 1)
                throw std::runtime_error{"invalid inference prediction"};
            for (float logit : p.logits) if (!std::isfinite(logit)) throw std::runtime_error{"nonfinite policy logit"};
        }
        std::vector<std::deque<Completion>> replies(runtime_.config.workers);
        for (std::size_t i = 0; i < batch.size(); ++i) {
            const auto& item = batch[i];
            replies[item.worker].push_back({item.request.slot, item.request.generation, std::move(predictions[i])});
        }
        for (int worker = 0; worker < runtime_.config.workers; ++worker) {
            if (replies[worker].empty()) continue;
            auto& mailbox = *runtime_.mailboxes[worker];
            {
                std::lock_guard lock{mailbox.mutex};
                if (mailbox.completions.size() + replies[worker].size() > std::size_t(runtime_.config.slots))
                    throw std::runtime_error{"completion queue overflow"};
                for (auto& reply : replies[worker]) mailbox.completions.push_back(std::move(reply));
            }
            mailbox.completed.notify_one();
        }
        ++batches_; leaves_ += batch.size();
    }
public:
    explicit Coordinator(Runtime& runtime) : runtime_{runtime} {}
    void run() {
        while (!runtime_.stopped) {
            runtime_.coordinator_heartbeat = now_ms();
            {
                std::unique_lock lock{runtime_.wake_mutex};
                runtime_.requested.wait_for(lock, 10ms, [&] {
                    return runtime_.stopped || runtime_.queued != 0 || runtime_.workers_done == runtime_.config.workers;
                });
            }
            if (runtime_.queued == 0 && runtime_.workers_done == runtime_.config.workers) return;
            const auto batch = gather();
            if (!batch.empty() && !runtime_.stopped) dispatch(batch);
        }
    }
    void report() const {
        std::fprintf(stderr, "coordinator: %ld batches, %ld leaves, %d sim workers, %d slots/worker\n",
                     batches_, leaves_, runtime_.config.workers, runtime_.config.slots);
    }
};

std::string error_message(std::exception_ptr error) {
    try { std::rethrow_exception(error); }
    catch (const std::exception& e) { return e.what(); }
    catch (...) { return "unknown worker error"; }
}
} // namespace

void run_workers(const WorkerConfig& config, NextSimulation next, Evaluate evaluate) {
    if (config.workers < 1 || config.slots < 1 || config.batch < 1 || config.batch_wait.count() < 0 ||
        config.timeout.count() <= 0 || config.shutdown_timeout.count() <= 0 || !next || !evaluate)
        throw std::invalid_argument{"invalid simulation worker configuration"};
    Runtime runtime{config, std::move(next), std::move(evaluate)};
    std::vector<std::unique_ptr<SimWorker>> workers;
    for (int i = 0; i < config.workers; ++i) workers.push_back(std::make_unique<SimWorker>(runtime, i));
    Coordinator coordinator{runtime};
    std::jthread coordinator_thread;
    std::vector<std::jthread> threads;
    try {
        coordinator_thread = std::jthread{[&] {
            try { coordinator.run(); } catch (...) { runtime.fail(std::current_exception()); }
            runtime.coordinator_done = true;
        }};
        for (int i = 0; i < config.workers; ++i) threads.emplace_back([&, i] {
            try { workers[i]->run(); } catch (...) { runtime.fail(std::current_exception()); }
            runtime.mailboxes[i]->done = true;
            ++runtime.workers_done;
            runtime.notify();
        });
    } catch (...) {
        runtime.fail(std::current_exception());
        for (std::size_t i = threads.size(); i < workers.size(); ++i) { runtime.mailboxes[i]->done = true; ++runtime.workers_done; }
        if (!coordinator_thread.joinable()) runtime.coordinator_done = true;
    }
    std::optional<Clock::time_point> cancelling;
    while (!runtime.all_done()) {
        if (!runtime.stopped) {
            const auto now = now_ms();
            for (int i = 0; i < config.workers; ++i) {
                auto& mailbox = *runtime.mailboxes[i];
                if (!mailbox.done && now - mailbox.heartbeat >= config.timeout.count())
                    runtime.fail(std::make_exception_ptr(std::runtime_error{"simulation timeout: worker " + std::to_string(i) +
                        " slot " + std::to_string(mailbox.current_slot.load())}));
            }
            if (!runtime.coordinator_done && now - runtime.coordinator_heartbeat >= config.timeout.count())
                runtime.fail(std::make_exception_ptr(std::runtime_error{"coordinator/inference timeout"}));
        }
        if (runtime.stopped) {
            if (!cancelling) {
                cancelling = Clock::now();
                std::lock_guard lock{runtime.error_mutex};
                std::fprintf(stderr, "workers: cancelling: %s\n", error_message(runtime.error).c_str());
            } else if (Clock::now() - *cancelling >= config.shutdown_timeout) {
                std::fprintf(stderr, "workers: shutdown timeout; native call did not return; exiting without freeing in-use games\n");
                std::fflush(stderr);
                std::_Exit(1);
            }
        }
        std::this_thread::sleep_for(10ms); // supervision, not a busy worker or extra watchdog thread
    }
    if (coordinator_thread.joinable()) coordinator_thread.join();
    threads.clear();
    coordinator.report();
    if (runtime.error) std::rethrow_exception(runtime.error);
}
} // namespace stsrl::nn
