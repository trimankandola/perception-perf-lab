#pragma once
#include <chrono>
#include <cstdint>
#include <vector>

namespace ppl {

// steady_clock never jumps backwards (unlike system_clock, which NTP can adjust),
// so it is the only correct clock for measuring durations.
using Clock = std::chrono::steady_clock;

struct Frame {
    std::uint64_t id = 0;
    Clock::time_point capture_time{};
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;  // RGB bytes, width * height * 3
    std::vector<float> tensor;         // preprocess output, same element count
    std::vector<float> scores;         // "inference" output
    int tracked_objects = 0;           // tracker output
};

// Allocates every buffer once, up front. Nothing in the per-frame path allocates.
Frame make_frame(int width, int height, int num_scores);

}  // namespace ppl