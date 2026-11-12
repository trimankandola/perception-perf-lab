#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ppl {

struct FrameTiming {
    std::uint64_t frame_id = 0;
    std::int64_t total_ns = 0;     // capture -> tracker done (end-to-end latency)
    std::int64_t preprocess_ns = 0;
    std::int64_t infer_ns = 0;
    std::int64_t track_ns = 0;
};

struct Stats {
    std::int64_t p50 = 0;
    std::int64_t p99 = 0;
    std::int64_t p999 = 0;
    std::int64_t max = 0;
    double mean = 0.0;
};

// Nearest-rank percentile on an already sorted vector. p is in (0, 100].
std::int64_t percentile_sorted(const std::vector<std::int64_t>& sorted, double p);

// Takes a copy on purpose: sorting must not reorder the caller's data.
Stats compute_stats(std::vector<std::int64_t> values);

class LatencyRecorder {
public:
    explicit LatencyRecorder(std::size_t expected_frames);
    void record(const FrameTiming& t);  // never allocates if expected_frames was right
    const std::vector<FrameTiming>& samples() const { return samples_; }
    std::vector<std::int64_t> column(std::int64_t FrameTiming::*field) const;
    bool write_csv(const std::string& path) const;
private:
    std::vector<FrameTiming> samples_;
};

}  // namespace ppl