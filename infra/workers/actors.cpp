#include "infra/workers/actors.hpp"
#include "infra/workers/thread_time.hpp"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

namespace stsrl::infra {
namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

struct ActorStatus {
    std::atomic<long long> heartbeat{now_ms()};
    std::atomic<int> current_slot{-1};
    std::atomic<bool> done{false};
    double cpu_seconds = std::nan("");
    std::atomic<long> requests{0};
    std::atomic<Clock::rep> blocked{0};
};

struct Shared {
    const ActorConfig config;
    NextEpisode next;
    InferenceService inference;
    std::vector<std::unique_ptr<ActorStatus>> actors;
    const int servers;
    std::atomic<bool> stopped{false};
    std::atomic<int> actors_done{0}, servers_done{0};
    std::mutex error_mutex;
    std::exception_ptr error;

    Shared(const ActorConfig& c, NextEpisode n, const std::vector<Backend*>& b)
        : config{c}, next{std::move(n)}, inference{b, c.batch, c.batch_wait}, servers{inference.lanes()} {
        for (int i = 0; i < config.actors; ++i) actors.push_back(std::make_unique<ActorStatus>());
    }
    void fail(std::exception_ptr failure) {
        { std::lock_guard lock{error_mutex}; if (!error) error = failure; }
        stopped = true;
        inference.stop();  // Actors also wake every 10 ms
    }
    bool all_done() const { return servers_done == servers && actors_done == config.actors; }
};

class Actor {
    struct Slot {
        std::unique_ptr<Episode> episode;
        bool ready = false;
        std::optional<std::uint64_t> last_id;
        std::string label;
    };
    Shared& shared_;
    const int id_;
    ActorStatus& status_;
    InferenceService::Connection connection_;
    std::vector<Slot> slots_;
    std::deque<std::size_t> ready_;
    int active_ = 0;

    void touch(std::size_t slot) { status_.current_slot = int(slot); status_.heartbeat = now_ms(); }

    void ready(std::size_t i) {
        if (!slots_[i].ready) { slots_[i].ready = true; ready_.push_back(i); }
    }

    void refill(std::size_t i) {
        auto& slot = slots_[i];
        slot.episode = shared_.next();
        slot.last_id.reset();
        slot.label = slot.episode ? slot.episode->label() : std::string{};
        if (slot.episode) { ready(i); ++active_; }
    }

    void receive() {
        for (auto& completion : connection_.receive()) {
            const auto i = completion.slot.value;
            touch(i);
            slots_.at(i).episode->completed(completion.id, completion.prediction);
            ready(i);
        }
    }

    void check_timeouts() const {
        if (const auto slot = connection_.timed_out(shared_.config.timeout))
            throw std::runtime_error{"inference timeout: actor " + std::to_string(id_) + " slot " +
                                     std::to_string(slot->value) + " episode " + slots_[slot->value].label};
    }

    void advance(std::size_t i) {
        auto& slot = slots_[i];
        touch(i);
        auto request = slot.episode->next_request();
        if (shared_.stopped) return;  // a cancelled episode must not be recorded as finished
        if (!request) {
            if (!slot.episode->done()) {
                if (!connection_.pending(SlotId{i})) throw std::runtime_error{"episode blocked without outstanding requests"};
                return;
            }
            if (connection_.pending(SlotId{i})) throw std::runtime_error{"episode done with outstanding requests"};
            slot.episode->finish();
            slot.episode.reset();
            --active_;
            refill(i);
            return;
        }
        if (slot.episode->done()) throw std::runtime_error{"done episode produced a request"};
        if (slot.last_id && request->id <= *slot.last_id)
            throw std::runtime_error{"episode request IDs must strictly increase"};
        slot.last_id = request->id;
        connection_.submit(SlotId{i}, std::move(*request), slot.label);
        ++status_.requests;
        ready(i);
    }

public:
    Actor(Shared& shared, int id) : shared_{shared}, id_{id}, status_{*shared.actors[id]}, connection_{shared.inference.connect()}, slots_(shared.config.slots) {}

    void run() {
        for (std::size_t i = 0; i < slots_.size() && !shared_.stopped; ++i) { touch(i); refill(i); }
        auto check_at = Clock::now();
        while (!shared_.stopped && active_ != 0) {
            status_.heartbeat = now_ms();
            receive();
            if (Clock::now() >= check_at) { check_timeouts(); check_at = Clock::now() + 10ms; }
            if (!ready_.empty()) {
                const auto i = ready_.front();
                ready_.pop_front();
                slots_[i].ready = false;
                advance(i);
            } else {
                const auto blocked = Clock::now();
                connection_.wait(10ms);
                status_.blocked += (Clock::now() - blocked).count();
            }
        }
        // Slots stay alive (owned by this object) until the servers have joined, even on failure.
    }
};

std::string describe(std::exception_ptr error) {
    try { std::rethrow_exception(error); }
    catch (const std::exception& e) { return e.what(); }
    catch (...) { return "unknown error"; }
}

void publish(const Shared& shared) {
    if (!shared.config.on_counters) return;
    const auto inference = shared.inference.counters();
    Counters counters{std::chrono::system_clock::now(), {}, inference.queues, inference.lanes};
    for (const auto& actor : shared.actors) counters.actors.push_back({actor->requests, Clock::duration{actor->blocked}});
    shared.config.on_counters(counters);
}

// The calling thread: heartbeats, cancellation, counters, and the hard exit if native code will not return.
void supervise(Shared& shared) {
    std::optional<Clock::time_point> cancelling;
    auto publish_at = Clock::now() + 1s;
    while (!shared.all_done()) {
        if (Clock::now() >= publish_at) { publish(shared); publish_at = Clock::now() + 1s; }
        if (!shared.stopped) {
            const long long now = now_ms(), limit = shared.config.timeout.count();
            for (int i = 0; i < shared.config.actors; ++i) {
                const auto& actor = *shared.actors[i];
                if (!actor.done && now - actor.heartbeat >= limit)
                    shared.fail(std::make_exception_ptr(std::runtime_error{"actor timeout: actor " + std::to_string(i) +
                        " slot " + std::to_string(actor.current_slot.load())}));
            }
            for (int i = 0; i < shared.servers; ++i) {
                const auto lane = shared.inference.status(i);
                if (!lane.done && now - lane.heartbeat_ms >= limit)
                    shared.fail(std::make_exception_ptr(std::runtime_error{"inference server timeout"}));
            }
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
    run_actors(config, std::move(next), std::vector<Backend*>{&backend});
}

void run_actors(const ActorConfig& config, NextEpisode next, const std::vector<Backend*>& backends) {
    if (config.actors < 1 || config.slots < 1 || config.batch < 1 ||
        config.batch_wait.count() < 0 || config.timeout.count() <= 0 || config.shutdown_timeout.count() <= 0 || !next)
        throw std::invalid_argument{"invalid actor configuration"};
    const auto started = Clock::now();
    Shared shared{config, std::move(next), backends};
    std::vector<std::unique_ptr<Actor>> actors;
    for (int i = 0; i < config.actors; ++i) actors.push_back(std::make_unique<Actor>(shared, i));
    std::vector<std::jthread> server_threads, actor_threads;
    try {
        for (int i = 0; i < shared.servers; ++i) server_threads.emplace_back([&, i] {
            try { shared.inference.run(i); } catch (...) { shared.fail(std::current_exception()); }
            ++shared.servers_done;
        });
        for (int i = 0; i < config.actors; ++i) actor_threads.emplace_back([&, i] {
            const double cpu = thread_cpu_seconds();
            try { actors[i]->run(); } catch (...) { shared.fail(std::current_exception()); }
            shared.actors[i]->cpu_seconds = thread_cpu_seconds() - cpu;
            shared.actors[i]->done = true;
            if (++shared.actors_done == config.actors) shared.inference.close();
        });
    } catch (...) {  // thread creation failed: count the never-started threads as done
        shared.fail(std::current_exception());
        for (std::size_t i = actor_threads.size(); i < actors.size(); ++i) { shared.actors[i]->done = true; ++shared.actors_done; }
        shared.servers_done += shared.servers - int(server_threads.size());
    }
    supervise(shared);
    server_threads.clear();
    actor_threads.clear();
    publish(shared);  // final totals

    const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
    double actor_cpu = 0;
    for (const auto& actor : shared.actors) actor_cpu += actor->cpu_seconds;
    const auto inference = shared.inference.totals();
    std::fprintf(stderr, "actors_stats: {\"seconds\":%.3f,\"actors\":%d,\"slots\":%d,\"servers\":%d,\"batches\":%ld,\"requests\":%ld,"
                 "\"mean_batch\":%.1f,\"actor_cpu_pct\":%.1f,\"server_cpu_pct\":%.1f}\n",
                 seconds, config.actors, config.slots, shared.servers, inference.batches, inference.requests,
                 inference.batches ? double(inference.requests) / inference.batches : 0.,
                 100 * actor_cpu / (config.actors * seconds), 100 * inference.cpu_seconds / (shared.servers * seconds));
    if (shared.error) std::rethrow_exception(shared.error);
}
}  // namespace stsrl::infra
