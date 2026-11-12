#include <gtest/gtest.h>
#include <numeric>
#include <vector>
#include "ppl/latency_recorder.hpp"

using ppl::compute_stats;
using ppl::FrameTiming;
using ppl::LatencyRecorder;
using ppl::percentile_sorted;

TEST(Percentile, OneToHundred) {
    std::vector<std::int64_t> v(100);
    std::iota(v.begin(), v.end(), 1);  // 1, 2, ..., 100
    EXPECT_EQ(percentile_sorted(v, 50.0), 50);
    EXPECT_EQ(percentile_sorted(v, 99.0), 99);
    EXPECT_EQ(percentile_sorted(v, 100.0), 100);
}

TEST(Percentile, OneToThousandP999) {
    std::vector<std::int64_t> v(1000);
    std::iota(v.begin(), v.end(), 1);
    EXPECT_EQ(percentile_sorted(v, 99.9), 999);
}

TEST(Percentile, SingleValue) {
    std::vector<std::int64_t> v{42};
    EXPECT_EQ(percentile_sorted(v, 99.0), 42);
}

TEST(Percentile, EmptyReturnsZero) {
    std::vector<std::int64_t> v;
    EXPECT_EQ(percentile_sorted(v, 50.0), 0);
}

TEST(Stats, HandlesUnsortedInput) {
    const auto s = compute_stats({5, 1, 4, 2, 3});
    EXPECT_EQ(s.p50, 3);
    EXPECT_EQ(s.max, 5);
    EXPECT_DOUBLE_EQ(s.mean, 3.0);
}

TEST(Recorder, NoReallocationWithinCapacity) {
    LatencyRecorder r(1000);
    const FrameTiming* before = r.samples().data();
    for (int i = 0; i < 1000; ++i) r.record(FrameTiming{});
    EXPECT_EQ(before, r.samples().data());  // same buffer: no allocation happened
}