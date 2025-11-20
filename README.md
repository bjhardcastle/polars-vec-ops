# Polars Vec Ops

A Polars plugin for vertical operations on list or array columns - aggregate across rows instead of within lists.

## Installation

```bash
uv add polars-vec-ops
```

## Quick Start

```python
import polars as pl
import polars_vec_ops # registers the `vec` namespace on columns/expressions

# Sum across rows at each position
df = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6]]})
df.select(pl.col("a").vec.sum())
# shape: (1, 1)
# ┌───────────┐
# │ a         │
# ╞═══════════╡
# │ [5, 7, 9] │
# └───────────┘
```

## Operations

All operations work vertically (across rows) on List or Array columns:

- **`sum()`** - Sum elements at each position
- **`mean()` / `avg()`** - Calculate mean at each position
- **`min()` / `max()`** - Find min/max at each position
- **`diff()`** - Calculate row-to-row differences

## Features

- Works with both List and Array dtypes
- Handles null rows and null elements
- Type preservation where possible (Int64, Float64, etc.)
- Fast Rust implementation via PyO3

## Development

```bash
# Install dev dependencies
uv sync

# Rebuild after modifying Rust code
maturin develop --release

# Run tests
pytest tests/
```

## License

MIT
