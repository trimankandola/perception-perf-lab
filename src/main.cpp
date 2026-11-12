#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "ppl/frame.hpp"
#include "ppl/latency_recorder.hpp"
#include "ppl/stages.hpp"

namespace {

struct Options {
    int frames = 3000;
    int warmup = 100;
    int fps = 30;
    int width = 1280;
    int height = 720;
    std::string csv = "results/baseline.csv";
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        const char* val = argv[i + 1];
        if (key == "--frames") o.frames = std::atoi(val);
        else if (key == "--warmup") o.warmup = std::atoi(val);
        else if (key == "--fps") o.fps = std::atoi(val);
        else if (key == "--width") o.width = std::atoi(val);
        else if (key == "--height") o.height = std::atoi(val);
        else if (key == "--csv") o.csv = val;
        else { std::fprintf(stderr, "unknown option %s\n", key.c_str()); std::exit(2); }
    }
    return o;
}

std::int64_t ns_between(ppl::Clock::time_point a, ppl::Clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

double ms(std::int64_t ns) { return static_cast<double>(ns) / 1e6; }

void print_stats(const char* name, const ppl::Stats& s) {
    std::printf("%-10s p50=%7.3f ms  p99=%7.3f ms  p99.9=%7.3f ms  max=%7.3f ms  mean=%7.3f ms\n",
                name, ms(s.p50), ms(s.p99), ms(s.p999), ms(s.max), s.mean / 1e6);
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    constexpr int kModelInputs = 4096;
    constexpr int kModelOutputs = 256;

    // Everything is allocated here, before the timed loop starts.
    ppl::Frame frame = ppl::make_frame(opt.width, opt.height, kModelOutputs);
    const ppl::FakeModel model(kModelInputs, kModelOutputs);
    ppl::LatencyRecorder recorder(static_cast<std::size_t>(opt.frames));

    const auto period = std::chrono::nanoseconds(1'000'000'000LL / opt.fps);
    auto next_capture = ppl::Clock::now();
    long long checksum = 0;  // consumed at the end so the compiler cannot delete the work
    int deadline_misses = 0;

    for (int i = 0; i < opt.warmup + opt.frames; ++i) {
        std::this_thread::sleep_until(next_capture);  // fixed-rate "camera"
        next_capture += period;

        ppl::capture(frame, static_cast<std::uint64_t>(i));
        const auto t0 = ppl::Clock::now();
        ppl::preprocess(frame);
        const auto t1 = ppl::Clock::now();
        model.infer(frame);
        const auto t2 = ppl::Clock::now();
        ppl::track(frame, 0.0f);
        const auto t3 = ppl::Clock::now();
        checksum += frame.tracked_objects;

        if (i < opt.warmup) continue;  // skip cold-cache, page-faulting first frames
        ppl::FrameTiming t;
        t.frame_id = frame.id;
        t.total_ns = ns_between(frame.capture_time, t3);
        t.preprocess_ns = ns_between(t0, t1);
        t.infer_ns = ns_between(t1, t2);
        t.track_ns = ns_between(t2, t3);
        if (t.total_ns > period.count()) ++deadline_misses;
        recorder.record(t);
    }

    std::printf("frames=%d fps=%d resolution=%dx%d deadline_misses=%d checksum=%lld\n",
                opt.frames, opt.fps, opt.width, opt.height, deadline_misses, checksum);
    print_stats("total", ppl::compute_stats(recorder.column(&ppl::FrameTiming::total_ns)));
    print_stats("preprocess", ppl::compute_stats(recorder.column(&ppl::FrameTiming::preprocess_ns)));
    print_stats("infer", ppl::compute_stats(recorder.column(&ppl::FrameTiming::infer_ns)));
    print_stats("track", ppl::compute_stats(recorder.column(&ppl::FrameTiming::track_ns)));

    if (!recorder.write_csv(opt.csv)) {
        std::fprintf(stderr, "could not write %s (does the folder exist?)\n", opt.csv.c_str());
        return 1;
    }
    std::printf("wrote %s\n", opt.csv.c_str());
    return 0;
}