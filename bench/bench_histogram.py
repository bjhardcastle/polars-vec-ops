#!/usr/bin/env python3
"""
Benchmark harness for polars-vec-ops histogram function.

THIS FILE IS FROZEN — do not modify during the autoresearch loop.

Generates a fixed test dataset, runs the histogram function, measures three metrics,
and prints them in a machine-parseable format for the autoresearch loop to consume.

Metrics printed (stdout, after the '---' marker):
    output_elements:  total scalar elements across all histogram rows (n_rows * n_bins)
    peak_memory_mb:   delta RSS memory during the benchmark window (MB)
    wall_time_ms:     wall-clock time to run histogram across the full DataFrame (ms)
    output_size_mb:   Polars estimated_size of the result column (MB), for reference

Data generation uses PyArrow for memory efficiency (~800 MB RSS vs ~7 GB with Python lists).
Peak RSS is measured via /proc/self/status before/after the benchmark window.
"""

import sys
import time
import numpy as np
import pyarrow as pa
import polars as pl
import polars_vec_ops  # noqa: F401 — registers .vec namespace

# ---------------------------------------------------------------------------
# Configuration (frozen — do not change)
# ---------------------------------------------------------------------------
N_ROWS = 100_000
SPIKES_PER_ROW = 1_000
SPIKE_MIN = 0.0
SPIKE_MAX = 10.0
N_BINS = 50
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_rss_mb() -> float:
    """Read current RSS from /proc/self/status (Linux)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def generate_spike_times_df() -> pl.DataFrame:
    """
    Build a DataFrame with a single List[Float64] column 'spike_times'.

    Uses PyArrow to construct from a flat numpy array + offsets, avoiding
    the ~7 GB memory footprint of building 100M Python float objects.

    Data: N_ROWS rows, each a list of SPIKES_PER_ROW floats uniform in
    [SPIKE_MIN, SPIKE_MAX), seeded with RANDOM_SEED for reproducibility.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    flat = rng.uniform(SPIKE_MIN, SPIKE_MAX, size=N_ROWS * SPIKES_PER_ROW).astype(np.float64)
    offsets = np.arange(0, N_ROWS * SPIKES_PER_ROW + 1, SPIKES_PER_ROW, dtype=np.int32)
    arrow_list = pa.ListArray.from_arrays(offsets, flat)
    return pl.from_arrow(pa.table({"spike_times": arrow_list}))


def spot_check_correctness(df: pl.DataFrame, result: pl.DataFrame) -> None:
    """
    Validate histogram counts for 5 sampled rows against numpy.histogram.
    Raises AssertionError on mismatch so the loop can log a crash.
    """
    counts_col = result["spike_times"]
    check_rows = [0, 1_000, 10_000, 50_000, 99_999]

    for row_idx in check_rows:
        spike_vals = df["spike_times"][row_idx].to_numpy()
        plugin_counts = counts_col[row_idx].to_list()

        np_counts, _ = np.histogram(spike_vals, bins=N_BINS)
        assert list(np_counts) == plugin_counts, (
            f"Row {row_idx}: expected {np_counts[:5].tolist()}... "
            f"got {plugin_counts[:5]}..."
        )

    print("correctness_check: PASS", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    # 1. Generate test data (outside the measurement window)
    print("Generating test data...", file=sys.stderr, flush=True)
    df = generate_spike_times_df()
    print(
        f"  {N_ROWS} rows × {SPIKES_PER_ROW} spikes/row, "
        f"df RSS ≈ {_read_rss_mb():.0f} MB",
        file=sys.stderr,
        flush=True,
    )

    # 2. Warm-up pass (not counted)
    _ = df.head(10).with_columns(pl.col("spike_times").vec.histogram(bins=N_BINS))

    # 3. Benchmark window: RSS before → run → RSS after
    rss_before = _read_rss_mb()
    t0 = time.perf_counter()

    result = df.with_columns(pl.col("spike_times").vec.histogram(bins=N_BINS))

    wall_time_ms = (time.perf_counter() - t0) * 1000.0
    rss_after = _read_rss_mb()

    # 4. Compute metrics
    peak_memory_mb = max(0.0, rss_after - rss_before)
    output_elements = int(result["spike_times"].list.len().sum())
    output_size_mb = result["spike_times"].estimated_size("mb")

    # 5. Correctness spot-check
    spot_check_correctness(df, result)

    # 6. Print metrics (machine-parseable)
    print("---")
    print(f"output_elements: {output_elements}")
    print(f"peak_memory_mb: {peak_memory_mb:.1f}")
    print(f"wall_time_ms: {wall_time_ms:.1f}")
    print(f"output_size_mb: {output_size_mb:.1f}")


if __name__ == "__main__":
    run_benchmark()
