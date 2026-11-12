#include "ppl/latency_recorder.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace ppl {

std::int64_t percentile_sorted(const std::vector<std::int64_t>& sorted, double p) {
    if (sorted.empty()) return 0;
    // 99.9 has no exact binary representation, so 99.9 / 100 * 1000 comes out as
    // 999.0000000000001 and ceil() would round it up to 1000. Subtracting a tiny
    // epsilon keeps exact ranks exact.
    const double rank = std::ceil(p / 100.0 * static_cast<double>(sorted.size()) - 1e-9);
    // Nearest-rank: the smallest value with at least p% of samples at or below it.
    // const double rank = std::ceil(p / 100.0 * static_cast<double>(sorted.size()));
    std::size_t idx = rank < 1.0 ? 0 : static_cast<std::size_t>(rank) - 1;
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return sorted[idx];
}

Stats compute_stats(std::vector<std::int64_t> values) {
    Stats s;
    if (values.empty()) return s;
    std::sort(values.begin(), values.end());
    s.p50 = percentile_sorted(values, 50.0);
    s.p99 = percentile_sorted(values, 99.0);
    s.p999 = percentile_sorted(values, 99.9);
    s.max = values.back();
    const long double sum = std::accumulate(values.begin(), values.end(), 0.0L);
    s.mean = static_cast<double>(sum / values.size());
    return s;
}

LatencyRecorder::LatencyRecorder(std::size_t expected_frames) {
    samples_.reserve(expected_frames);
}

void LatencyRecorder::record(const FrameTiming& t) { samples_.push_back(t); }

std::vector<std::int64_t> LatencyRecorder::column(std::int64_t FrameTiming::*field) const {
    std::vector<std::int64_t> out;
    out.reserve(samples_.size());
    for (const auto& s : samples_) out.push_back(s.*field);
    return out;
}

bool LatencyRecorder::write_csv(const std::string& path) const {
    std::ofstream out(path);
    if (!out) return false;
    out << "frame_id,total_ns,preprocess_ns,infer_ns,track_ns\n";
    for (const auto& s : samples_) {
        out << s.frame_id << ',' << s.total_ns << ',' << s.preprocess_ns << ','
            << s.infer_ns << ',' << s.track_ns << '\n';
    }
    return static_cast<bool>(out);
}

}  // namespace ppl