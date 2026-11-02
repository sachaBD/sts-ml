#include "infra/workers/actors.hpp"
#include <algorithm>
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
    int actor;
    std::size_t slot;
    std::uint64_t generation;
    Input input;
    Clock::time_point enqueued;
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
    std::deque<Reply> replies;
    std::atomic<long long> heartbeat{now_ms()};
    std::atomic<int> current_slot{-1};
    std::atomic<bool> done{false};
    std::mutex stats_mutex;
    ActorSnapshot stats;
    std::array<double, 3> slot_counts;
    TimeIntegral<3> slots;
    TimeIntegral<1> working;
    explicit Mailbox(int id, int capacity) : slot_counts{0, 0, double(capacity)}, slots{slot_counts} {
        stats.id = id; stats.slots = capacity;
    }
    void move_slot(int from, int to) {
        std::lock_guard lock{stats_mutex};
        --slot_counts[from]; ++slot_counts[to]; slots.set(slot_counts);
        if (to == 1) ++stats.requests;
        if (from == 0 && to == 2) ++stats.completed;
    }
    void set_working(bool active) {
        std::lock_guard lock{stats_mutex}; working.set({active ? 1. : 0.});
    }
    double cpu_seconds = std::nan("");  // written by the Actor, read after join
};

class InferencePool {
public:
    struct Queue {
        Backend& backend;
        const int id, batch;
        const bool observing;
        TimeIntegral<1> depth_integral;
        int peak_depth = 0;
        QueueWaitCounts wait_counts{};
        std::mutex mutex;
        std::condition_variable requested;
        std::deque<Request> pending;
        Queue(Backend& b, int limit, int id, bool observing)
            : backend{b}, id{id}, batch{std::min(limit, b.max_batch())}, observing{observing} {}
    };

    InferencePool(const std::vector<Backend*>& backends, int batch, bool observing) {
        if (backends.empty() || !backends.front()) throw std::invalid_argument{"invalid actor configuration"};
        for (auto* backend : backends) {
            if (!backend || backend->lanes() < 1 || backend->max_batch() < 1 || backend->width() != backends.front()->width())
                throw std::invalid_argument{"invalid actor configuration"};
            queues_.push_back(std::make_unique<Queue>(*backend, batch, int(queues_.size()), observing));
            for (int i = 0; i < backend->lanes(); ++i) lanes_.push_back(queues_.back().get());
        }
    }

    int width() const { return queues_.front()->backend.width(); }
    const auto& lanes() const { return lanes_; }
    const auto& queues() const { return queues_; }

    void submit(Request request, std::vector<std::uint16_t>& packed) {
        std::size_t i = 0;
        while (i + 1 < queues_.size() && !queues_[i]->backend.accepts(*request.input.tokens)) ++i;
        auto& queue = *queues_[i];
        queue.backend.prepare(*request.input.tokens, packed);
        request.input.packed = packed;
        {
            std::lock_guard lock{queue.mutex};
            if (queue.observing) request.enqueued = Clock::now();
            queue.pending.push_back(request);
            if (queue.observing) {
                queue.depth_integral.set({double(queue.pending.size())});
                queue.peak_depth = std::max(queue.peak_depth, int(queue.pending.size()));
            }
        }
        queue.requested.notify_one();
    }

    void notify() {
        for (auto& queue : queues_) {
            std::lock_guard lock{queue->mutex};
            queue->requested.notify_all();
        }
    }

private:
    std::vector<std::unique_ptr<Queue>> queues_;
    std::vector<Queue*> lanes_;
};

struct Shared {
    const ActorConfig config;
    NextEpisode next;
    InferencePool inference;
    std::vector<std::unique_ptr<Mailbox>> mailboxes;
    const int servers;
    const bool observing;
    std::atomic<bool> stopped{false};
    std::atomic<int> actors_done{0}, servers_done{0};
    std::mutex error_mutex;
    std::exception_ptr error;

    Shared(const ActorConfig& c, NextEpisode n, const std::vector<Backend*>& b, bool observing)
        : config{c}, next{std::move(n)}, inference{b, c.batch, observing}, servers{int(inference.lanes().size())}, observing{observing} {
        for (int i = 0; i < config.actors; ++i) mailboxes.push_back(std::make_unique<Mailbox>(i, c.slots));
    }
    void notify() { inference.notify(); }
    void fail(std::exception_ptr failure) {
        { std::lock_guard lock{error_mutex}; if (!error) error = failure; }
        stopped = true;
        notify();  // Actors also wake every 10 ms
    }
    bool all_done() const { return servers_done == servers && actors_done == config.actors; }
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
        if (slot.episode) {
            ready_.push_back(i); ++active_;
            if (shared_.observing) mailbox_.move_slot(2, 0);
        }
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
            if (shared_.observing) mailbox_.move_slot(1, 0);
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
            if (shared_.observing) mailbox_.move_slot(0, 2);
            slot.episode.reset();
            --active_;
            refill(i);
            return;
        }
        const int width = shared_.inference.width();
        if (tokens->count <= 0 || tokens->actions <= 0 || tokens->actions > tokens->count ||
            tokens->data.size() != std::size_t(tokens->count) * width || tokens->attended < 0 || tokens->attended > tokens->count)
            throw std::runtime_error{"invalid tokens from episode " + slot.label};
        slot.phase = Phase::waiting;
        if (shared_.observing) mailbox_.move_slot(0, 1);
        slot.submitted = Clock::now();
        shared_.inference.submit({id_, i, ++slot.generation, {tokens, {}}, {}, &slot.label}, slot.packed);
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
                if (shared_.observing) mailbox_.set_working(false);
                mailbox_.replied.wait_for(lock, 10ms, [&] { return shared_.stopped || !mailbox_.replies.empty(); });
                if (shared_.observing) mailbox_.set_working(true);
            }
        }
        // Slots stay alive (owned by this object) until the servers have joined, even on failure.
    }
};

class InferenceServer {
    Shared& shared_;
    InferencePool::Queue& queue_;
    mutable std::mutex stats_mutex_;
    DispatcherSnapshot stats_;

    std::vector<Request> gather() {
        std::vector<Request> batch;
        batch.reserve(queue_.batch);
        std::unique_lock lock{queue_.mutex};
        const auto ready = [&] { return shared_.stopped || !queue_.pending.empty() || shared_.actors_done == shared_.config.actors; };
        queue_.requested.wait_for(lock, 10ms, ready);
        const auto deadline = Clock::now() + shared_.config.batch_wait;
        while (!shared_.stopped && int(batch.size()) < queue_.batch) {
            if (queue_.pending.empty()) {
                if (shared_.actors_done == shared_.config.actors || !queue_.requested.wait_until(lock, deadline, ready)) break;
                continue;
            }
            batch.push_back(queue_.pending.front());
            queue_.pending.pop_front();
            if (queue_.observing) {
                const auto now = Clock::now();
                queue_.depth_integral.set({double(queue_.pending.size())}, now);
                record_queue_wait(queue_.wait_counts, std::chrono::duration<double>(now - batch.back().enqueued).count());
            }
        }
        return batch;
    }

    static void validate(const Request& item, const Prediction& p) {
        const auto& tokens = *item.input.tokens;
        bool finite = std::isfinite(p.value) && p.value >= 0 && p.value <= 1;
        for (float x : p.logits) finite = finite && std::isfinite(x);
        if (finite && p.logits.size() == std::size_t(tokens.actions)) return;
        std::ostringstream message;
        message << "invalid prediction: episode=" << *item.label << " value=" << p.value
                << " logits=" << p.logits.size() << " actions=" << tokens.actions << " tokens=" << tokens.count;
        throw std::runtime_error{message.str()};
    }

    void dispatch(const std::vector<Request>& batch) {
        std::vector<Input> inputs;
        inputs.reserve(batch.size());
        for (const auto& item : batch) inputs.push_back(item.input);
        auto predictions = queue_.backend.evaluate(inputs);
        if (shared_.stopped) return;
        if (predictions.size() != batch.size()) throw std::runtime_error{"prediction count mismatch"};
        // Validate everything before replying: a reply lets its Actor mutate the tokens again.
        for (std::size_t i = 0; i < batch.size(); ++i) validate(batch[i], predictions[i]);
        std::vector<std::deque<Reply>> replies(shared_.config.actors);
        for (std::size_t i = 0; i < batch.size(); ++i)
            replies[batch[i].actor].push_back({batch[i].slot, batch[i].generation, std::move(predictions[i])});
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
        if (shared_.observing) {
            std::lock_guard lock{stats_mutex_};
            stats_.batches = batches; stats_.requests = requests; stats_.capacity_rows += queue_.batch;
        }
    }

public:
    long batches = 0, requests = 0;
    std::atomic<long long> heartbeat{now_ms()};
    std::atomic<bool> done{false};
    double cpu_seconds = std::nan("");
    InferenceServer(Shared& shared, InferencePool::Queue& queue, int id) : shared_{shared}, queue_{queue} {
        stats_.id = id; stats_.backend_id = queue.id;
    }
    DispatcherSnapshot telemetry() const {
        std::lock_guard lock{stats_mutex_}; return stats_;
    }

    void run() {
        while (!shared_.stopped) {
            heartbeat = now_ms();
            const auto batch = gather();
            if (batch.empty() && shared_.actors_done == shared_.config.actors) return;
            if (!batch.empty() && !shared_.stopped) dispatch(batch);
        }
    }
};

std::string describe(std::exception_ptr error) {
    try { std::rethrow_exception(error); }
    catch (const std::exception& e) { return e.what(); }
    catch (...) { return "unknown error"; }
}

RuntimeSnapshot telemetry(Shared& shared, const std::vector<std::unique_ptr<InferenceServer>>& servers, Clock::time_point started, std::uint64_t sequence) {
    RuntimeSnapshot result;
    result.sequence = sequence;
    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    result.captured_unix_seconds = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (const auto& mailbox : shared.mailboxes) {
        std::lock_guard lock{mailbox->stats_mutex};
        auto s = mailbox->stats;
        const auto now = Clock::now();
        s.slot_seconds = mailbox->slots.snapshot(now); s.working_seconds = mailbox->working.snapshot(now)[0];
        result.actors.push_back(s);
    }
    for (const auto& queue : shared.inference.queues()) {
        QueueSnapshot s;
        {
            std::lock_guard lock{queue->mutex};
            const auto now = Clock::now();
            s.id = queue->id; s.depth = int(queue->pending.size()); s.peak_depth = queue->peak_depth;
            s.depth_seconds = queue->depth_integral.snapshot(now)[0]; s.wait_counts = queue->wait_counts;
            if (!queue->pending.empty()) s.oldest_seconds = std::chrono::duration<double>(now - queue->pending.front().enqueued).count();
            queue->peak_depth = s.depth; // next reporting interval starts with the current depth
        }
        result.queues.push_back(s);
        auto backend = queue->backend.telemetry(); backend.id = queue->id; backend.max_batch = queue->backend.max_batch();
        result.backends.push_back(std::move(backend));
    }
    for (const auto& server : servers) result.dispatchers.push_back(server->telemetry());
    return result;
}

// The calling thread: heartbeats, cancellation, and the hard exit if native code will not return.
void supervise(Shared& shared, const std::vector<std::unique_ptr<InferenceServer>>& servers, Clock::time_point started, RuntimeObserver observer) {
    std::optional<Clock::time_point> cancelling;
    auto report_at = Clock::now();
    std::uint64_t sequence = 0;
    const auto report = [&] {
        if (!observer) return;
        try { observer(telemetry(shared, servers, started, sequence++)); }
        catch (...) { std::fprintf(stderr, "runtime telemetry: observer failed; disabled\n"); observer = {}; }
    };
    while (!shared.all_done()) {
        if (observer && Clock::now() >= report_at) { report(); report_at = Clock::now() + 1s; }
        if (!shared.stopped) {
            const long long now = now_ms(), limit = shared.config.timeout.count();
            for (int i = 0; i < shared.config.actors; ++i) {
                const auto& mailbox = *shared.mailboxes[i];
                if (!mailbox.done && now - mailbox.heartbeat >= limit)
                    shared.fail(std::make_exception_ptr(std::runtime_error{"actor timeout: actor " + std::to_string(i) +
                        " slot " + std::to_string(mailbox.current_slot.load())}));
            }
            for (const auto& server : servers)
                if (!server->done && now - server->heartbeat >= limit)
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
    report();
}
}  // namespace

void run_actors(const ActorConfig& config, NextEpisode next, Backend& backend, RuntimeObserver observer) {
    run_actors(config, std::move(next), std::vector<Backend*>{&backend}, std::move(observer));
}

void run_actors(const ActorConfig& config, NextEpisode next, const std::vector<Backend*>& backends, RuntimeObserver observer) {
    if (config.actors < 1 || config.slots < 1 || config.batch < 1 ||
        config.batch_wait.count() < 0 || config.timeout.count() <= 0 || config.shutdown_timeout.count() <= 0 || !next)
        throw std::invalid_argument{"invalid actor configuration"};
    const auto started = Clock::now();
    for (auto* backend : backends) if (backend) backend->enable_telemetry(bool(observer));
    Shared shared{config, std::move(next), backends, bool(observer)};
    std::vector<std::unique_ptr<Actor>> actors;
    for (int i = 0; i < config.actors; ++i) actors.push_back(std::make_unique<Actor>(shared, i));
    std::vector<std::unique_ptr<InferenceServer>> servers;
    for (auto* queue : shared.inference.lanes()) servers.push_back(std::make_unique<InferenceServer>(shared, *queue, int(servers.size())));
    std::vector<std::jthread> server_threads, actor_threads;
    try {
        for (auto& server : servers) server_threads.emplace_back([&shared, &server = *server] {
            const double cpu = thread_cpu_seconds();
            try { server.run(); } catch (...) { shared.fail(std::current_exception()); }
            server.cpu_seconds = thread_cpu_seconds() - cpu;
            server.done = true;
            ++shared.servers_done;
        });
        for (int i = 0; i < config.actors; ++i) actor_threads.emplace_back([&, i] {
            const double cpu = thread_cpu_seconds();
            if (shared.observing) shared.mailboxes[i]->set_working(true);
            try { actors[i]->run(); } catch (...) { shared.fail(std::current_exception()); }
            if (shared.observing) shared.mailboxes[i]->set_working(false);
            shared.mailboxes[i]->cpu_seconds = thread_cpu_seconds() - cpu;
            shared.mailboxes[i]->done = true;
            ++shared.actors_done;
            shared.notify();
        });
    } catch (...) {  // thread creation failed: count the never-started threads as done
        shared.fail(std::current_exception());
        for (std::size_t i = actor_threads.size(); i < actors.size(); ++i) { shared.mailboxes[i]->done = true; ++shared.actors_done; }
        for (std::size_t i = server_threads.size(); i < servers.size(); ++i) { servers[i]->done = true; ++shared.servers_done; }
    }
    supervise(shared, servers, started, std::move(observer));
    server_threads.clear();
    actor_threads.clear();

    const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
    double actor_cpu = 0;
    for (const auto& mailbox : shared.mailboxes) actor_cpu += mailbox->cpu_seconds;
    long batches = 0, requests = 0;
    double server_cpu = 0;
    for (const auto& server : servers) { batches += server->batches; requests += server->requests; server_cpu += server->cpu_seconds; }
    std::fprintf(stderr, "actors_stats: {\"seconds\":%.3f,\"actors\":%d,\"slots\":%d,\"servers\":%d,\"batches\":%ld,\"requests\":%ld,"
                 "\"mean_batch\":%.1f,\"actor_cpu_pct\":%.1f,\"server_cpu_pct\":%.1f}\n",
                 seconds, config.actors, config.slots, shared.servers, batches, requests,
                 batches ? double(requests) / batches : 0.,
                 100 * actor_cpu / (config.actors * seconds), 100 * server_cpu / (shared.servers * seconds));
    if (shared.error) std::rethrow_exception(shared.error);
}
}  // namespace stsrl::infra
