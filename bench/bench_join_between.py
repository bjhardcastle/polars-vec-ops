#!/usr/bin/env python3
"""
Benchmark: spike-clipping — NumPy searchsorted vs polars-vec-ops join_between.

Compares speed and memory for clipping sorted spike-time lists to trial
windows.  I/O is excluded from timing.

Results are validated: polars spike counts per window must match numpy exactly.

Outputs a human-readable table to stdout, followed by a machine-parseable
summary block after a '---' marker.
"""

import argparse
import gc
import json
import logging
import os
import pathlib
import sys
import time

import psutil
import numpy as np
import polars as pl
import polars_vec_ops  # noqa: F401 — registers .vec namespace

# ── Setup logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("BENCH_DEBUG") else logging.WARNING,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Locate data files ────────────────────────────────────────────────────
_DATA_DIRS = ["/root/capsule/scratch", "/root/capsule/code", pathlib.Path(__file__).parent, pathlib.Path(__file__).parent.parent]
DATA_DIR = None
for _d in _DATA_DIRS:
    if os.path.isfile(os.path.join(_d, "units.parquet")):
        DATA_DIR = _d
        break
if DATA_DIR is None:
    sys.exit("ERROR: Cannot find units.parquet / trials.parquet in any known location")

logger.info(f"Data directory: {DATA_DIR}")


# ── Helpers ──────────────────────────────────────────────────────────────

def _rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def measure(fn, *args, n_warmup, n_repeats):
    """Return (median_seconds, peak_memory_bytes, result)."""
    for _ in range(n_warmup):
        _ = fn(*args)
        gc.collect()

    timings = []
    peak_delta = 0
    result = None

    for _ in range(n_repeats):
        gc.collect()
        gc.disable()
        rss_before = _rss_bytes()
        t0 = time.perf_counter()
        result = fn(*args)
        elapsed = time.perf_counter() - t0
        rss_after = _rss_bytes()
        gc.enable()
        delta = max(0, rss_after - rss_before)
        timings.append(elapsed)
        peak_delta = max(peak_delta, delta)
        gc.collect()

    median_time = float(np.median(timings))
    return median_time, peak_delta, result


# ── Benchmark functions ──────────────────────────────────────────────────

def clip_numpy(
    spike_arrays: list,
    trial_starts: np.ndarray,
    trial_stops: np.ndarray,
) -> list:
    """Clip spike times to [start, stop) for every (unit, trial) pair via searchsorted."""
    results = []
    for spk in spike_arrays:
        los = np.searchsorted(spk, trial_starts)
        his = np.searchsorted(spk, trial_stops)
        for lo, hi in zip(los, his):
            results.append(spk[lo:hi])
    return results


def clip_polars(units_df: pl.DataFrame, trials_df: pl.DataFrame) -> pl.DataFrame:
    """Clip spike times to [start_time, stop_time) via join_between."""
    return units_df.vec.join_between(
        other=trials_df,
        values="spike_times",
        bounds=("start_time", "stop_time"),
        check_sortedness=False,
    )


def _counts_from_numpy(clipped: list) -> np.ndarray:
    return np.array([len(a) for a in clipped], dtype=np.uint32)


def _counts_from_polars(result_df: pl.DataFrame) -> np.ndarray:
    return np.array(result_df["spike_times"].list.len().to_list(), dtype=np.uint32)


# ── Main benchmark ───────────────────────────────────────────────────────

def run_benchmarks(n_units: int, n_trials: int, n_warmup: int, n_repeats: int) -> None:
    # ── Load data ────────────────────────────────────────────────────
    all_units = pl.read_parquet(
        os.path.join(DATA_DIR, "units.parquet"), columns=["spike_times"]
    )
    all_trials = pl.read_parquet(
        os.path.join(DATA_DIR, "trials.parquet"), columns=["start_time", "stop_time"]
    )

    if n_units > len(all_units):
        reps = -(-n_units // len(all_units))
        all_units = pl.concat([all_units] * reps)
    units_df = all_units.head(n_units)

    if n_trials > len(all_trials):
        reps = -(-n_trials // len(all_trials))
        all_trials = pl.concat([all_trials] * reps)
    trials_df = all_trials.head(n_trials)

    # Pre-extract numpy arrays (exclude I/O from timing)
    spike_arrays: list = [
        np.asarray(row, dtype=np.float64) for row in units_df["spike_times"].to_list()
    ]
    trial_starts: np.ndarray = trials_df["start_time"].to_numpy()
    trial_stops: np.ndarray = trials_df["stop_time"].to_numpy()

    logger.info(f"n_units={n_units}, n_trials={n_trials}")

    # ── Header ───────────────────────────────────────────────────────
    header = (
        f"{'n_units':>7} {'n_trials':>8} │ "
        f"{'pl_time':>9} {'pl_mem':>10} │ "
        f"{'np_time':>9} {'np_mem':>10} │ "
        f"{'speedup':>8} {'match':>6}"
    )
    sep = "─" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    # ── Polars ───────────────────────────────────────────────────────
    pl_time = pl_mem = None
    pl_result = None
    pl_success = False
    try:
        pl_time, pl_mem, pl_result = measure(
            clip_polars, units_df, trials_df,
            n_warmup=n_warmup, n_repeats=n_repeats,
        )
        pl_success = True
    except Exception as exc:
        logger.error(f"Polars FAILED: {type(exc).__name__}: {exc}")
        print(
            f"{n_units:>7} {n_trials:>8} │ "
            f"{'ERROR':>9} {'N/A':>10} │ "
            f"{'SKIP':>9} {'N/A':>10} │ "
            f"{'N/A':>8} {'✗':>6}"
        )

    # ── NumPy ────────────────────────────────────────────────────────
    np_time, np_mem, np_result = measure(
        clip_numpy, spike_arrays, trial_starts, trial_stops,
        n_warmup=n_warmup, n_repeats=n_repeats,
    )

    # ── Validate ─────────────────────────────────────────────────────
    match = False
    if pl_success and pl_result is not None:
        try:
            np_counts = _counts_from_numpy(np_result)
            pl_counts = _counts_from_polars(pl_result)
            match = bool(np.array_equal(np_counts, pl_counts))
        except Exception as exc:
            logger.error(f"Validation failed: {type(exc).__name__}: {exc}")

    speedup = (np_time / pl_time) if (pl_success and pl_time and pl_time > 0) else 0.0

    if pl_success:
        print(
            f"{n_units:>7} {n_trials:>8} │ "
            f"{pl_time:>8.3f}s {pl_mem / 1e6:>8.1f}MB │ "
            f"{np_time:>8.3f}s {np_mem / 1e6:>8.1f}MB │ "
            f"{speedup:>7.2f}x {'✓' if match else '✗':>6}"
        )

    print(sep)

    # ── Machine-parseable summary ─────────────────────────────────────
    print("---")
    print(f"n_units: {n_units}")
    print(f"n_trials: {n_trials}")
    print(f"pl_time: {pl_time:.4f}" if pl_time is not None else "pl_time: N/A")
    print(f"np_time: {np_time:.4f}")
    print(f"speedup: {speedup:.4f}")
    print(f"match: {str(match).lower()}")
    print(f"pl_mem_mb: {pl_mem / 1e6:.1f}" if pl_mem is not None else "pl_mem_mb: N/A")
    print(f"np_mem_mb: {np_mem / 1e6:.1f}")

    summary = {
        "n_units": n_units,
        "n_trials": n_trials,
        "pl_time": round(pl_time, 4) if pl_time is not None else None,
        "np_time": round(np_time, 4),
        "speedup": round(speedup, 4),
        "match": match,
        "pl_mem_mb": round(pl_mem / 1e6, 1) if pl_mem is not None else None,
        "np_mem_mb": round(np_mem / 1e6, 1),
    }
    json_path = os.path.join(os.path.dirname(__file__), "join_between_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"results_json: {json_path}", file=sys.stderr)


# ── Entry point ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark join_between vs NumPy searchsorted for spike clipping."
    )
    parser.add_argument("--n-units", type=int, default=10, metavar="N")
    parser.add_argument("--n-trials", type=int, default=500, metavar="N")
    parser.add_argument("--n-warmup", type=int, default=1, metavar="N")
    parser.add_argument("--n-repeats", type=int, default=3, metavar="N")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmarks(
        n_units=args.n_units,
        n_trials=args.n_trials,
        n_warmup=args.n_warmup,
        n_repeats=args.n_repeats,
    )
