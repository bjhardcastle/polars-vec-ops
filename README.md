# Polars Vec Ops

A Polars plugin for vertical operations on columns 1D arrays or lists of equal length - aggregate across rows instead of within lists.

**🚧 Under Development**

**⚠️ Disclaimer:** The initial Rust extensions are entirely AI-generated, as are the majority of tests and documentation. Use at your own risk!

## Acknowledgments

Initialized from
[`https://github.com/MarcoGorelli/cookiecutter-polars-plugins`](https://github.com/MarcoGorelli/cookiecutter-polars-plugins):
thanks to Marco Gorelli for writing the excellent [Polars Plugins Tutorial](https://marcogorelli.github.io/polars-plugins-tutorial/).

## Installation

```bash
uv add polars-vec-ops
```

## Quick Start

```python
>>> import polars as pl
>>> import polars_vec_ops # registers the `vec` namespace on columns/expressions
>>> df = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6]]})
>>> df.select(pl.col("a").vec.sum())
shape: (1, 1)
┌───────────┐
│ a         │
│ ---       │
│ list[i64] │
╞═══════════╡
│ [5, 7, 9] │
└───────────┘

# alternatively, use functions on column names (with IDE hints and proper type checking):
>>> import polars_vec_ops as vec
>>> df.select(vec.sum("a"))
shape: (1, 1)
┌───────────┐
│ a         │
│ ---       │
│ list[i64] │
╞═══════════╡
│ [5, 7, 9] │
└───────────┘

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
pytest
```

## License

MIT
