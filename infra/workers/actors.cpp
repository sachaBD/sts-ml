#include "infra/workers/actors.hpp"
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <time.h>

namespace stsrl::infra {
namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

long long now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count(); }
double thread_cpu_seconds() {
    timespec t{};
    return ::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) == 0 ? t.tv_sec + t.tv_nsec / 1e9 : std::nan("");
}

struct Request {
    std::size_t slot;
    std::uint64_t generation;
    Input input;
    const std::string* label;  // owned by the slot, unchanged until the reply
};
struct Reply {
    std::size_t slot;
    std::uint64_t generation;
    Prediction prediction;
};

// One per Actor: the only state the Actor shares with the InferenceServer. Each deque holds at most `slots`
// entries because a slot has at most one request outstanding, so overflow is a bug, never back-pressure.
struct Mailbox {
    std::mutex mutex;
    std::condition_variable replied;
    std::deque<Request> requests;
    std::deque<Reply> replies;
    std::atomic<long long> heartbeat{now_ms()};
    std::atomic<int> current_slot{-1};
    std::atomic<bool> done{false};
    double cpu_seconds = std::nan("");  // written by the Actor, read after join
};

struct Shared {
    const ActorConfig config;
    NextEpisode next;
    Backend& backend;
    std::vector<std::unique_ptr<Mailbox>> mailboxes;
    std::atomic<bool> stopped{false}, server_done{false};
    std::atomic<int> actors_done{0}, queued{0};
    std::atomic<long long> server_heartbeat{now_ms()};
    std::mutex wake_mutex, error_mutex;
    std::condition_variable requested;
    std::exception_ptr error;
    double server_cpu_seconds = std::nan("");

    Shared(const ActorConfig& c, NextEpisode n, Backend& b) : config{c}, next{std::move(n)}, backend{b} {
        for (int i = 0; i < config.actors; ++i) mailboxes.push_back(std::make_unique<Mailbox>());
    }
    void notify() {
        std::lock_guard lock{wake_mutex};  // a notify outside the wait mutex can be lost
        requested.notify_one();
    }
    void fail(std::exception_ptr failure) {
        { std::lock_guard lock{error_mutex}; if (!error) error = failure; }
        stopped = true;
        notify();  // Actors also wake every 10 ms
    }
    bool all_done() const { return server_done && actors_done == config.actors; }
};

class Actor {
    enum class Phase { ready, waiting, empty };
    struct Slot {
        std::unique_ptr<Episode> episode;
        Phase phase = Phase::empty;
        std::uint64_t generation = 0;  // bumped on EVERY request, so a stale reply can never match
        Clock::time_point submitted;
        std::vector<std::uint16_t> packed;
        std::string label;
    };
    Shared& shared_;
    const int id_;
    Mailbox& mailbox_;
    std::vector<Slot> slots_;
    std::deque<std::size_t> ready_;
    int active_ = 0;

    void touch(std::size_t slot) { mailbox_.current_slot = int(slot); mailbox_.heartbeat = now_ms(); }

    void refill(std::size_t i) {
        auto& slot = slots_[i];
        slot.episode = shared_.next();
        slot.phase = slot.episode ? Phase::ready : Phase::empty;
        slot.label = slot.episode ? slot.episode->label() : std::string{};
        if (slot.episode) { ready_.push_back(i); ++active_; }
    }

    void receive() {
        std::deque<Reply> replies;
        { std::lock_guard lock{mailbox_.mutex}; replies.swap(mailbox_.replies); }
        for (auto& reply : replies) {
            auto& slot = slots_.at(reply.slot);
            if (slot.phase != Phase::waiting || slot.generation != reply.generation)
                throw std::runtime_error{"stale or duplicate reply"};
            touch(reply.slot);
            slot.episode->evaluated(reply.prediction);
            slot.phase = Phase::ready;
            ready_.push_back(reply.slot);
        }
    }

    void check_timeouts() const {
        const auto now = Clock::now();
        for (std::size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].phase == Phase::waiting && now - slots_[i].submitted >= shared_.config.timeout)
                throw std::runtime_error{"inference timeout: actor " + std::to_string(id_) + " slot " +
                                         std::to_string(i) + " episode " + slots_[i].label};
    }

    void advance(std::size_t i) {
        auto& slot = slots_[i];
        touch(i);
        const Tokens* tokens = slot.episode->advance();
        if (shared_.stopped) return;  // a cancelled episode must not be recorded as finished
        if (!tokens) {
            slot.episode->finish();
            slot.episode.reset();
            --active_;
            refill(i);
            return;
        }
        const int width = shared_.backend.width();
        if (tokens->count <= 0 || tokens->actions <= 0 || tokens->actions > tokens->count ||
            tokens->data.size() != std::size_t(tokens->count) * width)
            throw std::runtime_error{"invalid tokens from episode " + slot.label};
        shared_.backend.prepare(*tokens, slot.packed);
        slot.phase = Phase::waiting;
        slot.submitted = Clock::now();
        {
            std::lock_guard lock{mailbox_.mutex};
            if (mailbox_.requests.size() >= slots_.size()) throw std::runtime_error{"request queue overflow"};
            mailbox_.requests.push_back({i, ++slot.generation, {tokens, slot.packed}, &slot.label});
            ++shared_.queued;
        }
        shared_.notify();
    }

public:
    Actor(Shared& shared, int id) : shared_{shared}, id_{id}, mailbox_{*shared.mailboxes[id]}, slots_(shared.config.slots) {}

    void run() {
        for (std::size_t i = 0; i < slots_.size() && !shared_.stopped; ++i) { touch(i); refill(i); }
        auto check_at = Clock::now();
        while (!shared_.stopped && active_ != 0) {
            mailbox_.heartbeat = now_ms();
            receive();
            if (Clock::now() >= check_at) { check_timeouts(); check_at = Clock::now() + 10ms; }
            if (!ready_.empty()) {
                const auto i = ready_.front();
                ready_.pop_front();
                advance(i);
            } else {
                std::unique_lock lock{mailbox_.mutex};
                mailbox_.replied.wait_for(lock, 10ms, [&] { return shared_.stopped || !mailbox_.replies.empty(); });
            }
        }
        // Slots stay alive (owned by this object) until the server has joined, even on failure.
    }
};

class InferenceServer {
    struct Routed { int actor; Request request; };
    Shared& shared_;
    int cursor_ = 0;

    // Round-robin over Actors so none is starved.
    bool take_one(std::vector<Routed>& batch) {
        for (int n = 0; n < shared_.config.actors; ++n) {
            const int actor = cursor_;
            cursor_ = (cursor_ + 1) % shared_.config.actors;
            auto& mailbox = *shared_.mailboxes[actor];
            std::lock_guard lock{mailbox.mutex};
            if (mailbox.requests.empty()) continue;
            batch.push_back({actor, mailbox.requests.front()});
            mailbox.requests.pop_front();
            --shared_.queued;
            return true;
        }
        return false;
    }

    std::vector<Routed> gather() {
        std::vector<Routed> batch;
        batch.reserve(shared_.config.batch);
        const auto deadline = Clock::now() + shared_.config.batch_wait;
        while (!shared_.stopped && int(batch.size()) < shared_.config.batch) {
            if (take_one(batch)) continue;
            if (Clock::now() >= deadline) break;
            std::unique_lock lock{shared_.wake_mutex};
            shared_.requested.wait_until(lock, deadline, [&] { return shared_.stopped || shared_.queued != 0; });
        }
        return batch;
    }

    static void validate(const Routed& item, const Prediction& p) {
        const auto& tokens = *item.request.input.tokens;
        bool finite = std::isfinite(p.value) && p.value >= 0 && p.value <= 1;
        for (float x : p.logits) finite = finite && std::isfinite(x);
        if (finite && p.logits.size() == std::size_t(tokens.actions)) return;
        std::ostringstream message;
        message << "invalid prediction: episode=" << *item.request.label << " value=" << p.value
                << " logits=" << p.logits.size() << " actions=" << tokens.actions << " tokens=" << tokens.count;
        throw std::runtime_error{message.str()};
    }

    void dispatch(const std::vector<Routed>& batch) {
        std::vector<Input> inputs;
        inputs.reserve(batch.size());
        for (const auto& item : batch) inputs.push_back(item.request.input);
        auto predictions = shared_.backend.evaluate(inputs);
        if (shared_.stopped) return;
        if (predictions.size() != batch.size()) throw std::runtime_error{"prediction count mismatch"};
        // Validate everything before replying: a reply lets its Actor mutate the tokens again.
        for (std::size_t i = 0; i < batch.size(); ++i) validate(batch[i], predictions[i]);
        std::vector<std::deque<Reply>> replies(shared_.config.actors);
        for (std::size_t i = 0; i < batch.size(); ++i)
            replies[batch[i].actor].push_back({batch[i].request.slot, batch[i].request.generation, std::move(predictions[i])});
        for (int actor = 0; actor < shared_.config.actors; ++actor) {
            if (replies[actor].empty()) continue;
            auto& mailbox = *shared_.mailboxes[actor];
            {
                std::lock_guard lock{mailbox.mutex};
                if (mailbox.replies.size() + replies[actor].size() > std::size_t(shared_.config.slots))
                    throw std::runtime_error{"reply queue overflow"};
                for (auto& reply : replies[actor]) mailbox.replies.push_back(std::move(reply));
            }
            mailbox.replied.notify_one();
        }
        ++batches; requests += long(batch.size());
    }

public:
    long batches = 0, requests = 0;
    explicit InferenceServer(Shared& shared) : shared_{shared} {}

    void run() {
        while (!shared_.stopped) {
            shared_.server_heartbeat = now_ms();
            {
                std::unique_lock lock{shared_.wake_mutex};
                shared_.requested.wait_for(lock, 10ms, [&] {
                    return shared_.stopped || shared_.queued != 0 || shared_.actors_done == shared_.config.actors;
                });
            }
            if (shared_.queued == 0 && shared_.actors_done == shared_.config.actors) return;
            const auto batch = gather();
            if (!batch.empty() && !shared_.stopped) dispatch(batch);
        }
    }
};

std::string describe(std::exception_ptr error) {
    try { std::rethrow_exception(error); }
    catch (const std::exception& e) { return e.what(); }
    catch (...) { return "unknown error"; }
}

// The calling thread: heartbeats, cancellation, and the hard exit if native code will not return.
void supervise(Shared& shared) {
    std::optional<Clock::time_point> cancelling;
    while (!shared.all_done()) {
        if (!shared.stopped) {
            const long long now = now_ms(), limit = shared.config.timeout.count();
            for (int i = 0; i < shared.config.actors; ++i) {
                const auto& mailbox = *shared.mailboxes[i];
                if (!mailbox.done && now - mailbox.heartbeat >= limit)
                    shared.fail(std::make_exception_ptr(std::runtime_error{"actor timeout: actor " + std::to_string(i) +
                        " slot " + std::to_string(mailbox.current_slot.load())}));
            }
            if (!shared.server_done && now - shared.server_heartbeat >= limit)
                shared.fail(std::make_exception_ptr(std::runtime_error{"inference server timeout"}));
        }
        if (shared.stopped && !cancelling) {
            cancelling = Clock::now();
            std::lock_guard lock{shared.error_mutex};
            std::fprintf(stderr, "actors: cancelling: %s\n", describe(shared.error).c_str());
        } else if (cancelling && Clock::now() - *cancelling >= shared.config.shutdown_timeout) {
            std::fprintf(stderr, "actors: shutdown timeout; a native call did not return; exiting without freeing episodes\n");
            std::fflush(stderr);
            std::_Exit(1);
        }
        std::this_thread::sleep_for(10ms);
    }
}
}  // namespace

void run_actors(const ActorConfig& config, NextEpisode next, Backend& backend) {
    if (config.actors < 1 || config.slots < 1 || config.batch < 1 || config.batch > backend.max_batch() ||
        config.batch_wait.count() < 0 || config.timeout.count() <= 0 || config.shutdown_timeout.count() <= 0 || !next)
        throw std::invalid_argument{"invalid actor configuration"};
    const auto started = Clock::now();
    Shared shared{config, std::move(next), backend};
    std::vector<std::unique_ptr<Actor>> actors;
    for (int i = 0; i < config.actors; ++i) actors.push_back(std::make_unique<Actor>(shared, i));
    InferenceServer server{shared};
    std::jthread server_thread;
    std::vector<std::jthread> actor_threads;
    try {
        server_thread = std::jthread{[&] {
            const double cpu = thread_cpu_seconds();
            try { server.run(); } catch (...) { shared.fail(std::current_exception()); }
            shared.server_cpu_seconds = thread_cpu_seconds() - cpu;
            shared.server_done = true;
        }};
        for (int i = 0; i < config.actors; ++i) actor_threads.emplace_back([&, i] {
            const double cpu = thread_cpu_seconds();
            try { actors[i]->run(); } catch (...) { shared.fail(std::current_exception()); }
            shared.mailboxes[i]->cpu_seconds = thread_cpu_seconds() - cpu;
            shared.mailboxes[i]->done = true;
            ++shared.actors_done;
            shared.notify();
        });
    } catch (...) {  // thread creation failed: count the never-started threads as done
        shared.fail(std::current_exception());
        for (std::size_t i = actor_threads.size(); i < actors.size(); ++i) { shared.mailboxes[i]->done = true; ++shared.actors_done; }
        if (!server_thread.joinable()) shared.server_done = true;
    }
    supervise(shared);
    if (server_thread.joinable()) server_thread.join();
    actor_threads.clear();

    const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
    double actor_cpu = 0;
    for (const auto& mailbox : shared.mailboxes) actor_cpu += mailbox->cpu_seconds;
    std::fprintf(stderr, "actors_stats: {\"seconds\":%.3f,\"actors\":%d,\"slots\":%d,\"batches\":%ld,\"requests\":%ld,"
                 "\"mean_batch\":%.1f,\"actor_cpu_pct\":%.1f,\"server_cpu_pct\":%.1f}\n",
                 seconds, config.actors, config.slots, server.batches, server.requests,
                 server.batches ? double(server.requests) / server.batches : 0.,
                 100 * actor_cpu / (config.actors * seconds), 100 * shared.server_cpu_seconds / seconds);
    if (shared.error) std::rethrow_exception(shared.error);
}
}  // namespace stsrl::infra
