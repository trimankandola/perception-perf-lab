# Perception Pipeline Performance Lab

A C++20 camera-perception pipeline built to study latency, tail latency and resource contention: the problems that decide whether an autonomy stack fits its compute budget.

**Status:** Step 1 of 10 complete: a single-threaded baseline with a zero-allocation latency harness, unit tests, microbenchmarks and sanitizer CI. Next: a real YOLO model through ONNX Runtime, then multithreaded, lock-free and time-triggered designs, profiling, and a contention lab.

## Environment

- CPU: 8 vCPUs at 3264 MHz; per core 48 KiB L1d, 2 MiB L2; 12 MiB shared L3 (as reported by Google Benchmark)
- CPU model: <!-- paste the "Model name" line from: lscpu | grep "Model name" -->
- OS: Ubuntu 24.04 on WSL2 (Windows), 8 vCPUs
- Compiler: GCC 13, C++20, CMake `RelWithDebInfo` (`-O2 -g -fno-omit-frame-pointer`)
- Note: Ubuntu's Google Benchmark package reports "Library was built as DEBUG". That refers to the benchmark library itself, not the code being measured, which is built with `-O2`.

## Build and run

```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/ppl_tests
./build/ppl_bench
./build/ppl_run --frames 3000 --csv results/baseline_run1.csv
python3 scripts/plot_latency.py results/baseline_run1.csv
```

## Pipeline (Step 1)

A simulated camera produces 1280×720 RGB frames at a fixed 30 FPS (33.3 ms budget per frame). Each frame passes through three stages on one thread:

1. **Preprocess:** converts 2.76 M bytes to normalized floats (memory-bandwidth-bound)
2. **Inference:** a fixed 4096 → 256 matrix-vector multiply standing in for a neural network (compute-bound)
3. **Tracking:** counts scores above a threshold

All buffers are allocated once at startup, so the timed loop never allocates. Latency is measured end to end, from capture to tracker output, with `std::chrono::steady_clock`. The first 100 frames are discarded as warm-up.

## Results

### Step 1: single-threaded baseline (tag `v0.1-baseline`)

3,000 frames at 30 FPS, 1 run so far.

| Stage | p50 (ms) | p99 (ms) | p99.9 (ms) | max (ms) | mean (ms) |
|---|---|---|---|---|---|
| **Total (end to end)** | **12.637** | **21.142** | **25.610** | **25.928** | **12.847** |
| Preprocess | 5.953 | 11.525 | 15.800 | 17.708 | 6.179 |
| Inference | 2.610 | 5.488 | 8.444 | 12.688 | 2.696 |
| Tracking | 0.001 | 0.002 | 0.057 | 0.165 | 0.001 |

Deadline misses: **0 of 3,000** (worst frame 25.9 ms against a 33.3 ms budget).

![Step 1 baseline latency](results/baseline_run1.png)

Microbenchmarks (Google Benchmark, each stage in isolation, hot loop):

| Benchmark | Time |
|---|---|
| Capture | 1.34 ms |
| Preprocess | 2.00 ms |
| Inference | 0.786 ms |
| Tracking | 0.204 µs |

### Observations

- **The tail is long relative to the median.** End-to-end p99 is 1.7× the p50, and the slowest frame is 2.1× the p50, on a single-threaded loop doing identical work every frame. That spread comes from the environment, not the code: scheduling, cache state and the hypervisor. It's the baseline every later design has to beat.
- **Every stage runs about 3× slower in the pipeline than in its microbenchmark.** Preprocess has a median of 5.95 ms in the pipeline against 2.00 ms isolated; inference 2.61 ms against 0.79 ms; capture about 4.0 ms (by mean subtraction) against 1.34 ms. Because the ratio is the same across a memory-bound and a compute-bound stage, it points to something systematic rather than to either stage's code. The leading hypothesis: the pipeline sleeps about 20 ms between frames, so each frame starts with a lowered CPU clock and caches that other work has partly evicted, while the microbenchmark runs back-to-back with a hot cache at full clock. **Planned test:** rerun with `--fps 1000` (no sleeping). If the per-stage times converge on the microbenchmarks, the cause is idle-state recovery, which matters for any real-time loop that sleeps between frames.
- **The 30 FPS budget has headroom:** the worst frame left 7.4 ms of slack.

## Roadmap

| Step | Topic | Status |
|---|---|---|
| 1 | Baseline pipeline, latency harness, tests, benchmarks, CI | Done |
| 2 | Real workload: YOLO11n through ONNX Runtime, letterbox preprocessing, NMS, IoU tracker | Next |
| 3 | Thread per stage with bounded mutex queues and a frame pool | |
| 4 | Lock-free SPSC ring buffer (acquire/release ordering, false-sharing study) | |
| 5 | Time-triggered single-threaded executor, CPU pinning, SCHED_FIFO, mlock | |
| 6 | Profiling: perf, flame graphs, Cachegrind, Tracy, heaptrack, data-driven optimizations | |
| 7 | Contention lab: host, CPU, memory-bandwidth and cache interference, with mitigations | |
| 8 | ONNX Runtime threading and INT8 quantization | |
| 9 | Performance-regression gate in CI | |
| 10 | Write-up | |