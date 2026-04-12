#!/usr/bin/env python3
"""
Compile-time benchmark harness for polars-vec-ops.
THIS FILE IS FROZEN — do not modify during the autoresearch loop.

Measures:
  clean_compile_s       — time from `cargo clean` to importable plugin
  incremental_compile_s — time after `touch src/expressions.rs` to importable plugin
  wall_time_ms          — histogram benchmark wall time (runtime regression guard)

Prints metrics after the '---' marker (machine-parseable).
"""

import subprocess
import time
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent  # polars-vec-ops/

# Ensure Rust toolchain is on PATH
import os
os.environ["PATH"] = "/root/.cargo/bin:" + os.environ.get("PATH", "")

def run(cmd: str, cwd=ROOT) -> float:
    """Run shell command, return elapsed seconds. Raises on failure."""
    t0 = time.perf_counter()
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        print(result.stdout[-1000:], file=sys.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return elapsed

# Use venv Python/maturin directly (no bash source needed)
MATURIN = str(ROOT / ".venv" / "bin" / "maturin")
MATURIN_CMD = f"PATH=/root/.cargo/bin:$PATH {MATURIN} develop --release --skip-install 2>&1"
BENCH_PYTHON = str(ROOT / ".venv" / "bin" / "python")

print("Starting clean build...", file=sys.stderr, flush=True)
# 1. Clean build
run("cargo clean", cwd=ROOT)
clean_s = run(MATURIN_CMD, cwd=ROOT)
print(f"  clean build: {clean_s:.1f}s", file=sys.stderr, flush=True)

# 2. Incremental build (single-file touch)
# Touch expressions mod.rs as a representative hot-path file
(ROOT / "src" / "expressions" / "histogram.rs").touch()
print("Starting incremental build...", file=sys.stderr, flush=True)
incremental_s = run(MATURIN_CMD, cwd=ROOT)
print(f"  incremental build: {incremental_s:.1f}s", file=sys.stderr, flush=True)

# 3. Runtime performance guard — re-run histogram bench
print("Running histogram benchmark...", file=sys.stderr, flush=True)
out = subprocess.check_output(
    f"{BENCH_PYTHON} bench/bench_histogram.py 2>/dev/null",
    shell=True, cwd=ROOT, text=True
)
wall_ms_line = [l for l in out.splitlines() if l.startswith("wall_time_ms")]
if not wall_ms_line:
    raise RuntimeError(f"Could not find wall_time_ms in bench output:\n{out}")
wall_ms = float(wall_ms_line[0].split(":")[1].strip())

print("---")
print(f"clean_compile_s: {clean_s:.1f}")
print(f"incremental_compile_s: {incremental_s:.1f}")
print(f"wall_time_ms: {wall_ms:.1f}")
