#include <benchmark/benchmark.h>
#include "ppl/stages.hpp"

namespace {
constexpr int kW = 1280, kH = 720, kOut = 256, kIn = 4096;
}

static void BM_Capture(benchmark::State& state) {
    auto f = ppl::make_frame(kW, kH, kOut);
    std::uint64_t id = 0;
    for (auto _ : state) {
        ppl::capture(f, id++);
        benchmark::DoNotOptimize(f.pixels.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Capture)->Unit(benchmark::kMillisecond);

static void BM_Preprocess(benchmark::State& state) {
    auto f = ppl::make_frame(kW, kH, kOut);
    ppl::capture(f, 0);
    for (auto _ : state) {
        ppl::preprocess(f);
        benchmark::DoNotOptimize(f.tensor.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Preprocess)->Unit(benchmark::kMillisecond);

static void BM_Infer(benchmark::State& state) {
    auto f = ppl::make_frame(kW, kH, kOut);
    ppl::capture(f, 0);
    ppl::preprocess(f);
    const ppl::FakeModel model(kIn, kOut);
    for (auto _ : state) {
        model.infer(f);
        benchmark::DoNotOptimize(f.scores.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Infer)->Unit(benchmark::kMillisecond);

static void BM_Track(benchmark::State& state) {
    auto f = ppl::make_frame(kW, kH, kOut);
    for (auto _ : state) {
        ppl::track(f, 0.0f);
        benchmark::DoNotOptimize(f.tracked_objects);
    }
}
BENCHMARK(BM_Track)->Unit(benchmark::kMicrosecond);