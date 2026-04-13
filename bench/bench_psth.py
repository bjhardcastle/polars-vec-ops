#!/usr/bin/env python3
"""
Benchmark: PSTH computation — NumPy searchsorted vs polars-vec-ops hist.

THIS FILE IS FROZEN — do not modify during the autoresearch loop.

Compares speed and memory at multiple scales (num trials, bin sizes).
I/O is excluded from timing. Conversion costs (polars→numpy) are tracked
separately so neither method gets an unfair advantage.

Results are validated: polars-vec-ops counts must match numpy counts exactly.

Outputs a human-readable table to stdout, followed by a machine-parseable
summary block after a '---' marker.
"""
import pathlib

import gc
import os
import sys
import time
import json
import logging

import psutil
from itertools import product
from math import exp, log

import altair as alt
import numpy as np
import polars as pl
import polars_vec_ops as pvo

# ── Setup logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("BENCH_DEBUG") else logging.WARNING,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Benchmark parameters ────────────────────────────────────────────────
TRIAL_COUNTS = [500, 1000, 2000][:]
BIN_SPACINGS = [0.1, 0.01, 0.001, 0.0001][:]
PRE_TIME = -0.5
POST_TIME = 5.0
N_WARMUP = 1
N_REPEATS = 3
SAVE_CHARTS = os.environ.get("BENCH_CHARTS")

# ── Locate data files ──────────────────────────────────────────────────
# Code Ocean mounts code at /code; local dev may have them elsewhere.
_DATA_DIRS = ["/code", "/root/capsule/code", pathlib.Path(__file__).parent, pathlib.Path(__file__).parent.parent]
DATA_DIR = None
for _d in _DATA_DIRS:
    if os.path.isfile(os.path.join(_d, "units.parquet")):
        DATA_DIR = _d
        break
if DATA_DIR is None:
    sys.exit("ERROR: Cannot find units.parquet / trials.parquet in any known location")

logger.info(f"Data directory: {DATA_DIR}")

# ── Load data once (outside benchmarks) ─────────────────────────────────
UNITS = pl.read_parquet(os.path.join(DATA_DIR, "units.parquet"), columns=["spike_times", "unit_id"]).head(10)
TRIALS = pl.read_parquet(os.path.join(DATA_DIR, "trials.parquet"), columns=["start_time", "stop_time", "trial_index"])

# Tile trials to cover the largest requested count
_max_trials = max(TRIAL_COUNTS)
_n_reps = -(-_max_trials // len(TRIALS))
if _n_reps > 1:
    TRIALS = pl.concat([TRIALS] * _n_reps).head(_max_trials)

# Pre-extract numpy arrays for the numpy path (no I/O penalty)
SPIKE_TIMES_NP: list[np.ndarray] = [
    np.asarray(row, dtype=np.float64) for row in UNITS["spike_times"].to_list()
]
TRIAL_STARTS_NP: np.ndarray = TRIALS["start_time"].to_numpy()
TRIAL_STOPS_NP: np.ndarray = TRIALS["stop_time"].to_numpy()

N_UNITS = len(SPIKE_TIMES_NP)
N_TRIALS_FULL = len(TRIAL_STARTS_NP)


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_edges(spacing: float) -> np.ndarray:
    return np.arange(PRE_TIME, POST_TIME + spacing * 0.5, spacing)


def psth_numpy(
    spike_arrays: list[np.ndarray],
    trial_starts: np.ndarray,
    trial_stops: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Return (n_units, n_trials, n_bins) uint32 array via searchsorted."""
    n_units = len(spike_arrays)
    n_trials = len(trial_starts)
    n_bins = len(edges) - 1
    out = np.empty((n_units, n_trials, n_bins), dtype=np.uint32)

    for u, spk in enumerate(spike_arrays):
        sorted_spk = spk
        if not np.all(np.diff(sorted_spk) >= 0):
            sorted_spk = np.sort(spk)
        for t in range(n_trials):
            rel = sorted_spk - trial_starts[t]
            lo = np.searchsorted(rel, edges[0])
            hi = np.searchsorted(rel, edges[-1], side="right")
            windowed = rel[lo:hi]
            out[u, t, :] = np.diff(np.searchsorted(windowed, edges))

    return out


def psth_polars(
    units_df: pl.DataFrame,
    trials_df: pl.DataFrame,
    spacing: float,
) -> pl.DataFrame:
    """Return DataFrame with one row per (unit, trial) and a list column of counts."""
    result = (
        units_df
        .vec.join_between(
            other=trials_df,
            values="spike_times",
            bounds=("start_time", "stop_time"),
            check_sortedness=False,
            relative=True,
        )
        .with_columns(
            pvo.hist(
                "spike_times",
                start=PRE_TIME,
                stop=POST_TIME,
                spacing=spacing,
            ).alias("counts"),
        )
        .drop("spike_times")
    )
    # gc.collect()
    return result


def polars_counts_to_numpy(result_df: pl.DataFrame, units_df: pl.DataFrame, n_units: int, n_trials: int) -> np.ndarray:
    """Convert polars hist output to (n_units, n_trials, n_bins) numpy array."""
    unit_id_to_idx = {uid: idx for idx, uid in enumerate(units_df["unit_id"].to_list())}
    first_row = result_df["counts"][0]
    n_bins = len(first_row)
    out = np.zeros((n_units, n_trials, n_bins), dtype=np.uint32)
    counts_series = result_df["counts"]

    for i in range(len(result_df)):
        u = unit_id_to_idx[result_df["unit_id"][i]]
        t = int(result_df["trial_index"][i])
        out[u, t, :] = np.array(counts_series[i], dtype=np.uint32)

    return out


def _rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def measure(fn, *args, n_warmup=N_WARMUP, n_repeats=N_REPEATS):
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


# ── Main benchmark loop ─────────────────────────────────────────────────

def run_benchmarks():
    header = (
        f"{'trials':>7} {'spacing':>8} {'n_bins':>6} │ "
        f"{'pl_time':>9} {'pl_mem':>10} │ "
        f"{'np_time':>9} {'np_mem':>10} │ "
        f"{'speedup':>8} {'match':>6}"
    )
    sep = "─" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    rows = []  # collect per-cell results for summary

    for n_trials, spacing in product(TRIAL_COUNTS, BIN_SPACINGS):
        edges = _make_edges(spacing)
        n_bins = len(edges) - 1

        trial_starts = TRIAL_STARTS_NP[:n_trials]
        trial_stops = TRIAL_STOPS_NP[:n_trials]
        trials_sub = TRIALS.head(n_trials)
        units_sub = UNITS

        # ── Polars ──────────────────────────────────────────────────
        try:
            pl_time, pl_mem, pl_result = measure(
                psth_polars, units_sub, trials_sub, spacing
            )
        except Exception as e:
            logger.error(f"Polars FAILED: {type(e).__name__}: {e}")
            pl_time = pl_mem = None
            pl_result = None
            print(
                f"{n_trials:>7} {spacing:>8.4f} {n_bins:>6} │ "
                f"{'ERROR':>9} {'N/A':>10} │ "
                f"{'SKIP':>9} {'N/A':>10} │ "
                f"{'N/A':>8} {'✗':>6}"
            )
            rows.append({
                "n_trials": n_trials, "spacing": spacing, "n_bins": n_bins,
                "pl_time": None, "pl_mem": None, "np_time": None, "np_mem": None,
                "speedup": 0.0, "match": False,
            })
            gc.collect()
            continue

        del pl_result
        gc.collect()

        # ── NumPy ───────────────────────────────────────────────────
        np_time, np_mem, np_result = measure(
            psth_numpy, SPIKE_TIMES_NP, trial_starts, trial_stops, edges
        )

        # ── Validate ────────────────────────────────────────────────
        try:
            pl_result_validate = psth_polars(units_sub, trials_sub, spacing)
            pl_array = polars_counts_to_numpy(pl_result_validate, units_sub, N_UNITS, n_trials)
            match = bool(np.array_equal(np_result, pl_array))

            # ── Average PSTH per unit ────────────────────────────────
            if SAVE_CHARTS:
                mean_psth = (
                    pl_result_validate
                    .group_by("unit_id")
                    .agg(pvo.mean("counts"))
                )
                bin_centers = ((_make_edges(spacing)[:-1] + _make_edges(spacing)[1:]) / 2).tolist()
                rows_long = [
                    {"unit_id": str(row["unit_id"]), "time": t, "count": c}
                    for row in mean_psth.iter_rows(named=True)
                    for t, c in zip(bin_centers, row["counts"])
                ]
                chart = (
                    alt.Chart(pl.DataFrame(rows_long))
                    .mark_line(opacity=0.8)
                    .encode(
                        x=alt.X("time:Q", title="Time from trial start (s)"),
                        y=alt.Y("count:Q", title="Mean spike count"),
                        color=alt.Color("unit_id:N", title="Unit"),
                    )
                    .properties(title=f"Average PSTH — {n_trials} trials, spacing={spacing}")
                )
                chart_path = os.path.join(
                    os.path.dirname(__file__),
                    f"psth_chart_{n_trials}_{spacing}.html",
                )
                chart.save(chart_path)
                logger.info(f"Chart saved → {chart_path}")
                print(f"  [chart → {chart_path}]")

            del pl_result_validate, pl_array
        except Exception as e:
            logger.error(f"Validation failed: {type(e).__name__}: {e}")
            match = False

        speedup = np_time / pl_time if pl_time and pl_time > 0 else 0.0

        print(
            f"{n_trials:>7} {spacing:>8.4f} {n_bins:>6} │ "
            f"{pl_time:>8.3f}s {pl_mem / 1e6:>8.1f}MB │ "
            f"{np_time:>8.3f}s {np_mem / 1e6:>8.1f}MB │ "
            f"{speedup:>7.2f}x {'✓' if match else '✗':>6}"
        )

        rows.append({
            "n_trials": n_trials, "spacing": spacing, "n_bins": n_bins,
            "pl_time": pl_time, "pl_mem": pl_mem, "np_time": np_time, "np_mem": np_mem,
            "speedup": speedup, "match": match,
        })

        del np_result
        gc.collect()

    print(sep)

    # ── Machine-parseable summary ───────────────────────────────────
    valid_speedups = [r["speedup"] for r in rows if r["speedup"] and r["speedup"] > 0]
    all_match = all(r["match"] for r in rows)

    if valid_speedups:
        worst_speedup = min(valid_speedups)
        geomean_speedup = exp(sum(log(s) for s in valid_speedups) / len(valid_speedups))
        worst_row = min((r for r in rows if r["speedup"] and r["speedup"] > 0),
                        key=lambda r: r["speedup"])
        worst_case = f"{worst_row['n_trials']}_{worst_row['spacing']}"
    else:
        worst_speedup = 0.0
        geomean_speedup = 0.0
        worst_case = "N/A"

    pl_total_mem = sum(r["pl_mem"] for r in rows if r["pl_mem"] is not None) / 1e6
    np_total_mem = sum(r["np_mem"] for r in rows if r["np_mem"] is not None) / 1e6

    print("---")
    print(f"worst_speedup: {worst_speedup:.4f}")
    print(f"worst_case: {worst_case}")
    print(f"geomean_speedup: {geomean_speedup:.4f}")
    print(f"all_match: {str(all_match).lower()}")
    print(f"pl_total_mem_mb: {pl_total_mem:.1f}")
    print(f"np_total_mem_mb: {np_total_mem:.1f}")

    # Also write JSON for easy parsing
    summary = {
        "worst_speedup": round(worst_speedup, 4),
        "worst_case": worst_case,
        "geomean_speedup": round(geomean_speedup, 4),
        "all_match": all_match,
        "pl_total_mem_mb": round(pl_total_mem, 1),
        "np_total_mem_mb": round(np_total_mem, 1),
        "cells": rows,
    }
    json_path = os.path.join(os.path.dirname(__file__), "psth_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"results_json: {json_path}", file=sys.stderr)


if __name__ == "__main__":
    run_benchmarks()
