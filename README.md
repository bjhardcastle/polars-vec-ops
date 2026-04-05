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

### Aggregation
- **`sum()`** - Sum elements at each position
- **`mean()` / `avg()`** - Calculate mean at each position
- **`min()` / `max()`** - Find min/max at each position

### Row-wise
- **`diff()`** - Calculate row-to-row differences

### Per-element
- **`convolve(kernel, fill_value, mode)`** - 1D convolution with a kernel
- **`histogram(bins, *, start, stop, spacing)`** - Compute per-row histograms

### Histogram

Computes a histogram for each row's list, returning a struct with `breakpoints` (bin edges)
and `counts`. Bins can be specified as:

```python
# fixed number of bins (auto-ranged from data)
pl.col("a").vec.histogram(bins=10)

# explicit bin edges
pl.col("a").vec.histogram(bins=[0, 10, 20, 30])

# evenly spaced range
pl.col("a").vec.histogram(start=0.0, stop=100.0, spacing=10.0)

# any scalar parameter can be an expression for per-row values
pl.col("a").vec.histogram(bins=pl.col("n_bins"))
pl.col("a").vec.histogram(start=pl.col("lo"), stop=pl.col("hi"), spacing=pl.col("step"))
```

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
