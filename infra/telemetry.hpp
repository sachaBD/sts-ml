#pragma once
// Cumulative, monotonic counters. Readers compute windows by subtracting snapshots.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace stsrl::infra {
using TelemetryClock = std::chrono::steady_clock;

template<std::size_t N> class TimeIntegral {
    TelemetryClock::time_point since_;
    std::array<double, N> values_{}, totals_{};
public:
    explicit TimeIntegral(std::array<double, N> values = {}, TelemetryClock::time_point now = TelemetryClock::now())
        : since_{now}, values_{values} {}
    std::array<double, N> snapshot(TelemetryClock::time_point now = TelemetryClock::now()) const {
        auto result = totals_;
        const double seconds = std::max(0., std::chrono::duration<double>(now - since_).count());
        for (std::size_t i = 0; i < N; ++i) result[i] += seconds * values_[i];
        return result;
    }
    void set(std::array<double, N> values, TelemetryClock::time_point now = TelemetryClock::now()) {
        totals_ = snapshot(now); since_ = now; values_ = values;
    }
};

// Non-cumulative bins; the final bin has an unbounded upper edge. Counts are cumulative over the run.
inline constexpr std::array<double, 16> queue_wait_bounds_seconds{
    .0001, .00025, .0005, .001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 10, 60};
using QueueWaitCounts = std::array<std::uint64_t, queue_wait_bounds_seconds.size() + 1>;
inline void record_queue_wait(QueueWaitCounts& counts, double seconds) {
    ++counts[std::lower_bound(queue_wait_bounds_seconds.begin(), queue_wait_bounds_seconds.end(), seconds)
             - queue_wait_bounds_seconds.begin()];
}

struct ActorSnapshot {
    int id = 0, slots = 0;
    std::array<double, 3> slot_seconds{}; // ready (includes advancing), awaiting inference, empty
    double working_seconds = 0;
    std::uint64_t requests = 0, completed = 0;
};
struct QueueSnapshot {
    int id = 0, depth = 0, peak_depth = 0;
    double depth_seconds = 0, oldest_seconds = 0;
    QueueWaitCounts wait_counts{};
};
struct DispatcherSnapshot {
    int id = 0, backend_id = 0;
    std::uint64_t requests = 0, batches = 0, capacity_rows = 0;
};
struct LaneSnapshot {
    int id = 0;
    std::array<double, 6> time_seconds{}; // idle, packing, H2D, inference API, D2H/unpacking, CPU retry
    std::uint64_t rows = 0, batches = 0, capacity_rows = 0;
};
struct BackendSnapshot {
    int id = 0, max_batch = 0, tokens = 0, attended = 0;
    std::string kind = "unknown";
    std::vector<LaneSnapshot> lanes;
    std::uint64_t oversized_rows = 0, precision_retries = 0;
};
struct RuntimeSnapshot {
    std::uint64_t sequence = 0;
    double elapsed_seconds = 0, captured_unix_seconds = 0;
    std::vector<ActorSnapshot> actors;
    std::vector<QueueSnapshot> queues;
    std::vector<DispatcherSnapshot> dispatchers;
    std::vector<BackendSnapshot> backends;
};
} // namespace stsrl::infra
