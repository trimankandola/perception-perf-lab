#include "ppl/stages.hpp"
#include <cstddef>

namespace ppl {

Frame make_frame(int width, int height, int num_scores) {
    Frame f;
    f.width = width;
    f.height = height;
    const std::size_t n = static_cast<std::size_t>(width) * height * 3;
    f.pixels.resize(n);
    f.tensor.resize(n);
    f.scores.resize(static_cast<std::size_t>(num_scores));
    return f;
}

void capture(Frame& f, std::uint64_t id) {
    f.id = id;
    f.capture_time = Clock::now();
    const std::size_t n = f.pixels.size();
    for (std::size_t i = 0; i < n; ++i) {
        f.pixels[i] = static_cast<std::uint8_t>((i + id) & 0xFF);
    }
}

void preprocess(Frame& f) {
    constexpr float kScale = 1.0f / 255.0f;
    const std::size_t n = f.pixels.size();
    for (std::size_t i = 0; i < n; ++i) {
        f.tensor[i] = (static_cast<float>(f.pixels[i]) * kScale - 0.5f) * 2.0f;  // [-1, 1]
    }
}

FakeModel::FakeModel(int inputs, int outputs)
    : inputs_(inputs), outputs_(outputs),
      weights_(static_cast<std::size_t>(inputs) * outputs) {
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] = static_cast<float>((i * 7919) % 1000) / 1000.0f - 0.5f;
    }
}

void FakeModel::infer(Frame& f) const {
    // Reads the first inputs_ values of the tensor as the model input.
    for (int o = 0; o < outputs_; ++o) {
        const float* w = &weights_[static_cast<std::size_t>(o) * inputs_];
        float acc = 0.0f;
        for (int i = 0; i < inputs_; ++i) acc += w[i] * f.tensor[static_cast<std::size_t>(i)];
        f.scores[static_cast<std::size_t>(o)] = acc;
    }
}

void track(Frame& f, float threshold) {
    int count = 0;
    for (float s : f.scores) count += (s > threshold) ? 1 : 0;
    f.tracked_objects = count;
}

}  // namespace ppl