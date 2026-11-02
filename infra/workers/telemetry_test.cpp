#include "infra/telemetry.hpp"
#include <gtest/gtest.h>
#include <numeric>

using namespace stsrl::infra;
using namespace std::chrono_literals;

TEST(Telemetry, SlotIntegralsConserveCapacityAndReadsDoNotAdvanceCounters) {
    const auto begin = TelemetryClock::time_point{};
    TimeIntegral<3> slots{{0, 0, 224}, begin};
    slots.set({100, 124, 0}, begin + 1s);
    const auto first = slots.snapshot(begin + 3s);
    EXPECT_EQ(first, (std::array<double, 3>{200, 248, 224}));
    EXPECT_DOUBLE_EQ(std::accumulate(first.begin(), first.end(), 0.), 224 * 3);
    EXPECT_EQ(slots.snapshot(begin + 3s), first);
    slots.set({50, 100, 74}, begin + 3s);
    const auto next = slots.snapshot(begin + 4s);
    EXPECT_DOUBLE_EQ(std::accumulate(next.begin(), next.end(), 0.), 224 * 4);
}

TEST(Telemetry, QueueHistogramCountsEachObservationOnceIncludingOverflow) {
    QueueWaitCounts counts{};
    for (double seconds : {0., .0001, .000101, .001, 3., 1000.}) record_queue_wait(counts, seconds);
    EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), std::uint64_t{0}), 6);
    EXPECT_EQ(counts.front(), 2);
    EXPECT_EQ(counts.back(), 1);
}
