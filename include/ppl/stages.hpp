#pragma once
#include <cstdint>
#include <vector>
#include "ppl/frame.hpp"

namespace ppl {

// Stand-in for a camera: writes a pattern that changes every frame, so the
// compiler cannot skip the work and the caches see new data each time.
void capture(Frame& f, std::uint64_t id);

// Converts bytes to normalized floats (what real preprocessing does before inference).
void preprocess(Frame& f);

// Stand-in for a neural network: a fixed matrix-vector multiply.
class FakeModel {
public:
    FakeModel(int inputs, int outputs);
    void infer(Frame& f) const;
private:
    int inputs_;
    int outputs_;
    std::vector<float> weights_;  // outputs_ x inputs_, row-major
};

// Stand-in for a tracker: counts "objects" whose score passes a threshold.
void track(Frame& f, float threshold);

}  // namespace ppl