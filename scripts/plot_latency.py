#!/usr/bin/env python3
"""Plot end-to-end latency from a ppl_run CSV: over time and as a histogram."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files, no window needed
import matplotlib.pyplot as plt


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: plot_latency.py results/<file>.csv")
        sys.exit(2)
    path = Path(sys.argv[1])
    frames, total_ms = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            frames.append(int(row["frame_id"]))
            total_ms.append(int(row["total_ns"]) / 1e6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(frames, total_ms, linewidth=0.6)
    ax1.set_xlabel("frame")
    ax1.set_ylabel("end-to-end latency (ms)")
    ax1.set_title("Latency over time")
    ax2.hist(total_ms, bins=100)
    ax2.set_yscale("log")  # log scale makes the rare slow frames visible
    ax2.set_xlabel("end-to-end latency (ms)")
    ax2.set_ylabel("frames (log scale)")
    ax2.set_title("Latency distribution")
    fig.tight_layout()
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()