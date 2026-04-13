# Autoresearch Plan: `join_between` Speed & Memory Optimization

## Overview

Iteratively improve the speed and memory usage of `join_between` — a Python-level cross-join +
interval-clipping operation on sorted list columns in `polars-vec-ops`. The current implementation
uses `map_elements` with Python's `bisect` module, which creates massive per-row overhead.
The optimization target is the benchmark `bench/bench_join_between.py`.

---

## Goal

Maximize `join_between` throughput and minimize memory, measured by two metrics
(both evaluated mechanically on every iteration):

| # | Metric | Unit | Direction | Description |
|---|--------|------|-----------|-------------|
| 1 | **pl_time** | seconds | lower is better | Median Polars `join_between` execution time (excludes I/O) |
| 2 | **pl_mem_mb** | megabytes | lower is better | Peak RSS delta during Polars execution |

A composite **fitness score** combines these (see Evaluation section).

Correctness guard (independent of fitness): run `bench/bench_join_between.py` after every
kept experiment; if `match` is `false`, discard unconditionally regardless of speed or memory gain.

---

## Autoresearch Adaptation

### Three-file architecture

| Original autoresearch | This project | Role |
|-----------------------|--------------|------|
| `prepare.py` (read-only) | `bench/bench_join_between.py` (read-only) | Fixed benchmark harness — times Polars vs NumPy, validates correctness |
| `train.py` (agent edits) | `polars_vec_ops/join_between.py` | The implementation being optimized — agent modifies freely |
| `program.md` (human edits) | **This file** — `AUTORESEARCH.md` | Instructions for the autonomous loop |
| `results.tsv` (untracked) | `join_between_results.tsv` (untracked) | Experiment log |

### What CAN be modified
- `src/expressions/list_clip.rs` (new): the Rust `list_clip` expression plugin — primary optimization target
- `src/expressions/mod.rs`: to register the new module
- `polars_vec_ops/join_between.py`: Python-side orchestration (cross join, column setup, calling the Rust plugin)
- `polars_vec_ops/__init__.py`: to register the Rust plugin function via `register_plugin_function`
- `Cargo.toml`: if new dependencies are needed for the Rust plugin
- New helper modules under `polars_vec_ops/` if needed

### What CANNOT be modified
- The benchmark harness `bench/bench_join_between.py` (frozen)
- The test suite `tests/test_join_between.py` (frozen)
- The public Python API: the `VecOpsNamespace.join_between()` method signature and return type
  in `polars_vec_ops/join_between.py` must remain compatible (same parameters, same output shape)
- The benchmark data files (`units.parquet`, `trials.parquet`)
- Other existing Rust expressions or Python API (histogram, list_sum, etc.)

---

## Setup Phase

Before the loop begins:

1. **Create experiment branch**: `git checkout -b autoresearch/join_between` from the current commit
   (or continue on it if it already exists).
2. **Read all in-scope files**: Understand `polars_vec_ops/join_between.py`, `bench/bench_join_between.py`,
   `tests/test_join_between.py`.
3. **Initialize `join_between_results.tsv`**:
   ```
   commit	pl_time	pl_mem_mb	np_time	speedup	match	fitness	status	description
   ```

### Benchmark invocation

```bash
# Run benchmark (default: 10 units, 500 trials, 1 warmup, 3 repeats)
python bench/bench_join_between.py --n-units 10 --n-trials 500 --n-warmup 1 --n-repeats 3
```

Machine-parseable output appears after the `---` marker:
```
---
pl_time: 1.2345
np_time: 0.0678
speedup: 0.0549
match: true
pl_mem_mb: 45.2
np_mem_mb: 12.1
```

### Test invocation

```bash
pytest tests/test_join_between.py -v
```

---

## The Experiment Loop

**LOOP FOREVER** (until manually interrupted):

### Iteration 0 (first iteration only): Establish baseline

The very first iteration makes NO code changes. It compiles the current state, runs the
benchmark, and records the baseline that all future experiments are compared against.

1. **Compile**: `maturin develop --release` (ensure the Rust plugin is built).
2. **Test**: `pytest tests/test_join_between.py -v` (confirm tests pass on current code).
3. **Benchmark**: `python bench/bench_join_between.py` — extract metrics from `---` block.
4. **Log**: Write the first row to `join_between_results.tsv` with `status=keep` and
   `description=baseline`. Set `fitness=1.000` (by definition, baseline is 1.0).
5. **Plot**: Generate `bench/join_between_progress.png` (see Plotting section).
6. **Commit & push**: Commit the results TSV and plot, then push.

After iteration 0, proceed to the normal loop below.

### 1. Review state
- Check git log, current branch, last few entries in `join_between_results.tsv`.
- Identify what has been tried, what worked, what failed.

### 2. Plan next experiment
- Select ONE focused, atomic change from the Ideas Queue (below).
- Prefer ideas that target the worst-performing metric.

### 3. Modify
- Edit source files (Rust and/or Python) as needed.
- Keep changes small and focused — one idea per experiment.

### 4. Commit
- `git add polars_vec_ops/ src/ Cargo.toml && git commit -m "experiment: <short description>"`
- Commit BEFORE verification to enable clean rollback.

### 5. Compile, test & verify
- Compile: `maturin develop --release`
  - If compilation fails: attempt a fix (up to 2 tries) and amend the commit.
  - If still broken, log as `crash` and `git reset --hard HEAD~1`.
- Run tests: `pytest tests/test_join_between.py -v`
  - If tests fail: attempt a fix (up to 2 tries). If still broken, log as `crash` and revert.
- Run the benchmark: `python bench/bench_join_between.py`
- Extract metrics from the `---` block.
- If the benchmark itself crashes:
  - Read the error output.
  - Attempt a fix (up to 2 tries) and amend the commit.
  - If still broken, log as `crash` and `git revert HEAD --no-edit`.

### 6. Evaluate
Compute composite fitness (lower is better):

```
fitness = 0.7 × (pl_time / baseline_pl_time)
        + 0.3 × (pl_mem_mb / baseline_pl_mem_mb)
```

Speed is weighted 70% because it is the primary optimization target; memory is 30%.

**Correctness guard** (independent of fitness):
- If `match` is `false` → discard unconditionally.
- If any test in `test_join_between.py` fails → discard unconditionally.

**Keep** if: `fitness < previous_best_fitness` AND correctness passes.
**Also keep** if: either metric improved >5% with no metric regressing >2% (and correctness passes).
**Discard** if: fitness is equal or worse, or correctness fails.

### 7. Log & plot
Append to `join_between_results.tsv`:
```
<commit>	<pl_time>	<pl_mem_mb>	<np_time>	<speedup>	<match>	<fitness>	<keep|discard|crash>	<description>
```

Update the progress plot `bench/join_between_progress.png` (see Plotting section below).

### 8. Advance or revert, commit plot, and always push
- **Keep**: branch advances, new baseline for comparison.
- **Discard**: `git revert HEAD --no-edit` (preserves history of what was tried).
- **Crash**: `git reset --hard HEAD~1` (remove broken code entirely).

**Always** commit the updated results TSV and plot:
```bash
git add bench/join_between_progress.png join_between_results.tsv
git commit -m "results: <short description of experiment outcome>"
```

If github credentials permit, push to origin.

### 9. Repeat
Go to step 1. **NEVER STOP.**

---

## Plotting

After every iteration (including the baseline), generate or update
`bench/join_between_progress.png` using matplotlib. The plot should contain:

**Layout**: two subplots stacked vertically, sharing the x-axis (experiment index).

**Top subplot — Speed**:
- Y-axis: `pl_time` (seconds), log scale if range > 10×.
- Horizontal dashed line: `np_time` baseline (the NumPy reference to beat).
- Points colored by status: green = `keep`, red = `discard`, orange = `crash`.
- X-axis labels: short description from the TSV, rotated 45°.

**Bottom subplot — Memory**:
- Y-axis: `pl_mem_mb` (MB).
- Horizontal dashed line: `np_mem_mb` baseline.
- Same color coding as above.

**Title**: `"join_between optimization progress"` with the current best fitness in the subtitle.

Use a simple script or inline code — the plot does not need to be fancy, just informative.
Save to `bench/join_between_progress.png` (overwritten each iteration).

---

## Ideas Queue (Ordered by Expected Impact)

### Current bottleneck analysis

The implementation in `polars_vec_ops/join_between.py` is entirely Python. Three compounding
bottlenecks:

1. **`map_elements` with Python callback** (lines 151–179): The `_clip` function is called once
   per (unit, interval) pair via `map_elements`. For 10 units × 500 trials = 5,000 Python function
   calls, each involving Series→list conversion and Python-level `bisect`.

2. **Full cross-join materialization** (line 127): `df.join(other, how="cross")` creates N×M rows,
   duplicating every list column M times. For large lists this dominates memory.

3. **Per-row list materialization** (line 159): `vals.to_list()` converts each Polars Series to a
   Python list for `bisect.bisect_left`, adding per-row allocation and copy overhead.

### Tier 1: Rust `list_clip` expression plugin (foundation — do this FIRST)

All subsequent tiers build on a working Rust implementation. The Python `map_elements` + `bisect`
approach is the root cause of slowness and must be replaced with native Rust before any other
optimization is meaningful.

1. **Implement `list_clip` as a `#[polars_expr]`**: Write `src/expressions/list_clip.rs`.
   The function takes three input Series (values list, start scalar, stop scalar) plus kwargs
   (`relative: bool`, `as_counts: bool`). For each row: cast the list to f64 offsets, binary
   search for `lo = partition_point(|x| x < start)` and `hi = partition_point(|x| x < stop)`,
   then either emit the slice `values[lo..hi]` (optionally shifted by `-start`) or the count
   `(hi - lo) as u32`.
   Follow the existing patterns in `histogram.rs` and `list_sum.rs`:
   - Use `ensure_list_type` for Array→List conversion.
   - Define `list_clip_output_type` for return dtype.
   - Register in `src/expressions/mod.rs`.
   - Call from Python via `register_plugin_function` in `__init__.py`, invoked from `join_between.py`
     after the cross join (replacing the `map_elements` block).
   Expected: **10–50× speed improvement** over Python map_elements (binary search on Arrow
   buffers, no Python→Rust→Python per-row overhead).
   Memory: unchanged (still uses Polars cross join).

2. **Handle all input dtypes generically**: The initial implementation may hard-code f64.
   Generalize to all numeric types (i32, i64, u32, u64, f32, f64) using Polars' `downcast_iter`
   or by casting to f64 at the Rust boundary. Must pass all tests including the integer-dtype
   and mixed-dtype test cases in `test_join_between.py`.

3. **Handle `relative=True` in Rust**: After slicing `values[lo..hi]`, subtract `start` from each
   element. This avoids a second pass in Python. Implement in the same `list_clip` function
   controlled by the `relative` kwarg.

### Tier 2: Reduce cross-join memory overhead

The Polars cross join in `join_between.py` still duplicates the list column M times. These ideas
reduce peak memory while keeping the Rust `list_clip` expression as the core.

5. **Interval-loop with concat**: Process one interval (or small batch) at a time in Python.
   For each interval, add start/stop as literal columns to all units, run `list_clip`, collect.
   `pl.concat(results)` at the end. Avoids materializing the full N×M DataFrame.
   Expected: **50–90% memory reduction** (peak = O(n_units) vs O(n_units × n_intervals)).
   Speed: may regress slightly from Python loop overhead — benchmark to verify.

6. **Chunked interval processing**: Process intervals in configurable batches (e.g., 50 at a time)
   to balance memory savings against per-chunk overhead. Tune chunk size empirically.
   Expected: tunable memory/speed tradeoff.

7. **Move cross join into Rust**: Instead of Polars cross join + per-row `list_clip`, implement
   the full cross-join-and-clip as a single Rust function that takes the values column and
   start/stop arrays. Produces the output list column directly without materializing the N×M
   intermediate DataFrame. Requires a multi-input `#[polars_expr]` or a custom Python→Rust bridge.
   Expected: **large memory reduction + speed improvement** (no intermediate DataFrame at all).
   Effort: higher (must reconstruct cross-join output columns in Rust or Python).

### Tier 3: Rust-level performance tuning

Once the Rust foundation is working, optimize the Rust code itself.

8. **Zero-copy list slicing via Arrow offsets**: Arrow list arrays store data as a flat values
   buffer + offsets array. Instead of copying `values[lo..hi]` into a new allocation, construct
   the output list by adjusting offsets to point into the same backing buffer. This is O(1) per
   row instead of O(slice_len).
   Expected: significant speedup for large lists, major memory reduction.

9. **Avoid per-row `get_as_series`**: Instead of `list_chunked.get_as_series(i)` (which allocates),
   work directly with the underlying `LargeListArray`'s offsets and values buffers via
   `downcast_iter()`. This is the standard high-perf pattern in Polars plugins.
   Expected: moderate speedup from reduced allocation overhead.

10. **`as_counts` pure-arithmetic path**: When `as_counts=True`, the output is a flat `UInt32`
    column (not a list). Skip all list construction machinery — just binary search and subtract.
    Expected: large speedup for count-only queries.

11. **Batch binary search with SIMD-friendly layout**: For each row, the two binary searches
    (start, stop) are independent. Structure the inner loop so the compiler can auto-vectorize
    the comparison operations. (No explicit SIMD intrinsics — rely on `rustc` auto-vectorization
    which is portable across architectures.)
    Expected: modest speedup (depends on list sizes and compiler heuristics).

### Tier 4: Python-side structural optimizations

12. **Vectorized sortedness check**: Replace `map_elements(_is_sorted, ...)` with a Polars
    native expression or a small Rust plugin. Eliminates one `map_elements` call during validation.
    Expected: minor speed improvement (only affects `check_sortedness=True` path).

13. **Lazy evaluation**: Keep inputs as LazyFrames throughout the computation, letting Polars
    optimize the full query plan. Collect only at the very end.
    Expected: depends on Polars optimizer.

14. **Reduce intermediate DataFrame copies**: Chain `.with_columns()` calls, avoid re-materializing
    columns that don't change.
    Expected: minor speed and memory improvement.

---

## Guard Rails

- **Correctness**: `match` must be `true` in benchmark output. All tests in
  `tests/test_join_between.py` must pass. If either fails, the experiment is discarded.
- **API stability**: The `VecOpsNamespace.join_between()` method signature must remain unchanged.
  All existing parameters must keep their documented behavior.
- **No platform-specific optimizations**: Do not use platform-specific flags, SIMD intrinsics tied
  to a specific architecture, or OS-specific APIs. NumPy's `searchsorted` is acceptable (portable C).
  Polars native operations are acceptable (portable Rust/Arrow).
- **No hardware-specific tuning**: Do not hard-code batch sizes or parallelism levels based on
  specific CPU core counts, cache sizes, or memory amounts.

---

## Results Tracking

`join_between_results.tsv` (tab-separated, untracked by git):

```
commit	pl_time	pl_mem_mb	np_time	speedup	match	fitness	status	description
```

Example:
```
commit	pl_time	pl_mem_mb	np_time	speedup	match	fitness	status	description
a1b2c3d	1.234	45.2	0.068	0.055	true	2.000	keep	baseline (map_elements + bisect)
b2c3d4e	0.089	45.0	0.068	0.764	true	0.349	keep	numpy searchsorted + list.slice
c3d4e5f	0.085	12.1	0.068	0.800	true	0.329	keep	interval-loop avoids cross join memory
d4e5f6g	0.042	12.0	0.068	1.619	true	0.204	keep	as_counts fast path (index arithmetic)
e5f6g7h	0.035	11.8	0.068	1.943	false	—	discard	off-by-one in searchsorted (match failed)
```

---

## Timing Budget per Iteration

| Phase | Time budget |
|-------|-------------|
| Review + plan | 1 min |
| Code modification | 2 min |
| Commit | 30 s |
| Test + benchmark | ~60–120 s (not counted against budget) |
| Log | 30 s |
| **Total active time** | **~4 min** |

---

## Summary

This plan targets the **algorithmic and structural bottlenecks** in `join_between`:

- **Same core loop**: modify → commit → test → benchmark → keep/discard → repeat
- **Same git-as-memory**: every experiment is a commit; failures revert cleanly
- **Same TSV logging**: mechanical tracking of all experiments
- **Adapted metrics**: `pl_time` + `pl_mem_mb` (instead of compile times)
- **Adapted scope**: `polars_vec_ops/join_between.py`, `src/expressions/list_clip.rs`, and supporting files
- **Hard constraint**: `match == true` + all tests pass — correctness is never traded for speed

The Ideas Queue provides 14 experiments across 4 tiers. **Tier 1 (Rust `list_clip` plugin) is the
foundation** — it must be implemented first, replacing the Python `map_elements` + `bisect` approach
with native binary search on Arrow buffers. This alone should yield 10–50× speed improvement.
All subsequent tiers optimize on top of the Rust base: Tier 2 reduces cross-join memory overhead,
Tier 3 tunes the Rust implementation itself, and Tier 4 cleans up Python-side inefficiencies.
Combined, these should bring `join_between` well past parity with the NumPy baseline.
