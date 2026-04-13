from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars._utils.parse import parse_into_expression, parse_into_list_of_expressions
from polars._utils.wrap import wrap_expr
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExprColumn

_LIB = Path(__file__).parent


@pl.api.register_expr_namespace("vec")
class VecOpsNamespace:
    """Custom namespace for vertical list operations."""

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def sum(self) -> pl.Expr:
        """
        Sum across rows for list columns (vertical aggregation).

        Returns a single row with a list where each element is the sum
        of elements at that position across all input lists.

        All lists must have the same length.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[0, 1, 2], [1, 2, 3]]})
        >>> df.select(pl.col("a").vec.sum())
        shape: (1, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 3, 5] │
        └───────────┘
        """
        return register_plugin_function(
            args=[self._expr],
            plugin_path=_LIB,
            function_name="list_sum",
            is_elementwise=False,
            returns_scalar=True,
        )

    def mean(self) -> pl.Expr:
        """
        Calculate mean across rows for list columns (vertical aggregation).

        Returns a single row with a list where each element is the mean
        of elements at that position across all input lists.

        All lists must have the same length.

        Returns
        -------
        pl.Expr
            Expression returning a list of Float64 values.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[1, 2, 3], [3, 4, 5]]})
        >>> df.select(pl.col("a").vec.mean())
        shape: (1, 1)
        ┌─────────────────┐
        │ a               │
        │ ---             │
        │ list[f64]       │
        ╞═════════════════╡
        │ [2.0, 3.0, 4.0] │
        └─────────────────┘
        """
        return register_plugin_function(
            args=[self._expr],
            plugin_path=_LIB,
            function_name="list_mean",
            is_elementwise=False,
            returns_scalar=True,
        )

    # Alias for mean
    def avg(self) -> pl.Expr:
        """
        Alias for mean(). Calculate average across rows for list columns.

        See mean() for full documentation.
        """
        return self.mean()

    def min(self) -> pl.Expr:
        """
        Find minimum element at each position across rows (vertical aggregation).

        Returns a single row with a list where each element is the minimum
        of elements at that position across all input lists.

        All lists must have the same length.

        Returns
        -------
        pl.Expr
            Expression returning a list with the same type as input.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[3, 5, 2], [1, 7, 4]]})
        >>> df.select(pl.col("a").vec.min())
        shape: (1, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 5, 2] │
        └───────────┘
        """
        return register_plugin_function(
            args=[self._expr],
            plugin_path=_LIB,
            function_name="list_min",
            is_elementwise=False,
            returns_scalar=True,
        )

    def max(self) -> pl.Expr:
        """
        Find maximum element at each position across rows (vertical aggregation).

        Returns a single row with a list where each element is the maximum
        of elements at that position across all input lists.

        All lists must have the same length.

        Returns
        -------
        pl.Expr
            Expression returning a list with the same type as input.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[3, 5, 2], [1, 7, 4]]})
        >>> df.select(pl.col("a").vec.max())
        shape: (1, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [3, 7, 4] │
        └───────────┘
        """
        return register_plugin_function(
            args=[self._expr],
            plugin_path=_LIB,
            function_name="list_max",
            is_elementwise=False,
            returns_scalar=True,
        )

    def diff(self) -> pl.Expr:
        """
        Calculate differences between consecutive rows at each position.

        Returns the same number of rows as input. The first row contains
        a list of nulls (no previous row to compare). Each subsequent row
        contains the element-wise difference from the previous row: row[i] - row[i-1].

        If either the current or previous row is null, the result is a list of nulls.

        All lists must have the same length.

        Returns
        -------
        pl.Expr
            Expression returning lists with differences, preserving input type.
            First row is always a list of nulls.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[5, 10, 15], [2, 15, 5], [0, 0, 0]]})
        >>> df.select(pl.col("a").vec.diff())
        shape: (3, 1)
        ┌────────────────────┐
        │ a                  │
        │ ---                │
        │ list[i64]          │
        ╞════════════════════╡
        │ [null, null, null] │
        │ [-3, 5, -10]       │
        │ [-2, -15, -5]      │
        └────────────────────┘
        """
        return register_plugin_function(
            args=[self._expr],
            plugin_path=_LIB,
            function_name="list_diff",
            is_elementwise=False,
            returns_scalar=False,  # Returns same number of rows
        )

    def convolve(
        self,
        kernel: list[float] | pl.Series | pl.Expr,
        fill_value: float = 0.0,
        mode: str = "same",
    ) -> pl.Expr:
        """
        Convolve each list in the column with the given kernel.

        Performs 1D convolution on each row's list independently.
        The convolution operation applies the kernel as a sliding window
        over each signal, computing weighted sums.

        Parameters
        ----------
        kernel
            The convolution kernel (filter). Can be a list of floats,
            a Polars Series, or an expression. Non-finite values are filtered out.
        fill_value
            Value to use for null elements in the signal. Default is 0.0.
        mode
            Convolution mode, one of:
            - "full": Full convolution, output length = signal_len + kernel_len - 1
            - "same": Same length as signal, centered (default)
            - "valid": Only where signal and kernel fully overlap,
                      output length = signal_len - kernel_len + 1
            - "left": Same length as signal, left-aligned
            - "right": Same length as signal, right-aligned

        Returns
        -------
        pl.Expr
            Expression returning lists of Float64 values.

        Examples
        --------
        >>> df = pl.DataFrame({"signal": [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]})
        >>> kernel = [0.25, 0.5, 0.25]  # Smoothing kernel
        >>> df.select(pl.col("signal").vec.convolve(kernel, mode="same"))
        shape: (2, 1)
        ┌───────────────────┐
        │ signal            │
        │ ---               │
        │ list[f64]         │
        ╞═══════════════════╡
        │ [1.0, 2.0, … 3.5] │
        │ [3.5, 4.0, … 1.0] │
        └───────────────────┘

        >>> # Full mode returns longer output
        >>> df.select(pl.col("signal").vec.convolve(kernel, mode="full"))
        shape: (2, 1)
        ┌─────────────────────┐
        │ signal              │
        │ ---                 │
        │ list[f64]           │
        ╞═════════════════════╡
        │ [0.25, 1.0, … 1.25] │
        │ [1.25, 3.5, … 0.25] │
        └─────────────────────┘

        >>> # Null rows return None (not empty list)
        >>> df_with_null = pl.DataFrame({"signal": [[1, 2, 3], None]})
        >>> df_with_null.select(pl.col("signal").vec.convolve(kernel))
        shape: (2, 1)
        ┌─────────────────┐
        │ signal          │
        │ ---             │
        │ list[f64]       │
        ╞═════════════════╡
        │ [1.0, 2.0, 2.0] │
        │ null            │
        └─────────────────┘
        """
        # Convert kernel to list if needed
        if isinstance(kernel, pl.Expr):
            # For expression kernels, we can't easily pass them
            # For now, require kernel to be materialized
            raise TypeError(
                "kernel as Expr not yet supported. "
                "Please pass a list or Series instead."
            )
        elif isinstance(kernel, pl.Series):
            kernel_list = kernel.to_list()
        else:
            kernel_list = list(kernel)

        return register_plugin_function(
            args=[self._expr],
            plugin_path=_LIB,
            function_name="list_convolve",
            is_elementwise=True,  # Operates on each row independently
            returns_scalar=False,
            kwargs={
                "kernel": kernel_list,
                "fill_value": fill_value,
                "mode": mode,
            },
        )

    def histogram(
        self,
        bins: int | list[float] | pl.Series | pl.Expr | str | None = None,
        *,
        start: float | pl.Expr | str | pl.Series | None = None,
        stop: float | pl.Expr | str | pl.Series | None = None,
        spacing: float | pl.Expr | str | pl.Series | None = None,
        count_dtype: pl.DataType | type[bool] | type[int] | None = None,
        include_breakpoints: bool = False,
    ) -> pl.Expr:
        """
        Compute a histogram for each list in the column.

        Bin specification supports two mutually exclusive modes:

        1. ``bins``: an integer number of bins (auto-ranged from data) or
           explicit bin edges as a list.
        2. ``start`` / ``stop`` / ``spacing``: evenly spaced bins.

        Any scalar parameter can be a ``pl.Expr`` to derive values per-row
        from other columns.

        Parameters
        ----------
        bins
            Number of bins (int) or explicit bin edges (list/Series).
            Can be a ``pl.Expr`` resolving to an integer column.
        start
            Left edge of the first bin. Required with ``stop`` and ``spacing``.
        stop
            Right edge of the last bin.
        spacing
            Width of each bin.
        count_dtype
            Data type for the counts field. Smaller types save memory.
            Accepts ``pl.Boolean`` / ``bool`` (1 bit: any count > 0),
            ``pl.UInt8`` (max 255), ``pl.UInt16`` (max 65535), or
            ``pl.UInt32`` / ``int`` (default, max ~4 billion).
        include_breakpoints
            If ``False``, omit the ``breakpoints`` field from the output
            struct, returning only ``counts``. Saves memory when bin edges
            are already known (e.g. from ``start``/``stop``/``spacing``
            or explicit ``bins`` list). Default ``True``.

        Returns
        -------
        pl.Expr
            If ``include_breakpoints=True`` (default when set), returns a
            Struct with fields:
            - ``breakpoints``: ``List[Float64]`` — n+1 bin edges
            - ``counts``: ``List[count_dtype]`` — n bin counts

            If ``include_breakpoints=False``, returns ``List[count_dtype]``
            directly (just the counts, no struct wrapper).

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]})
        >>> df.select(pl.col("a").vec.histogram(bins=3))
        shape: (2, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[u32] │
        ╞═══════════╡
        │ [2, 1, 2] │
        │ [2, 1, 2] │
        └───────────┘
        """
        has_bins = bins is not None
        has_range = any(x is not None for x in (start, stop, spacing))

        if has_bins and has_range:
            raise ValueError(
                "Cannot specify both 'bins' and 'start'/'stop'/'spacing'."
            )
        if not has_bins and not has_range:
            raise ValueError(
                "Must specify either 'bins' or all of 'start', 'stop', 'spacing'."
            )
        if has_range and any(x is None for x in (start, stop, spacing)):
            # Allow Expr for any of them, but all three must be provided
            non_none = [
                name
                for name, val in [("start", start), ("stop", stop), ("spacing", spacing)]
                if val is not None
            ]
            missing = [
                name
                for name in ("start", "stop", "spacing")
                if name not in non_none
            ]
            raise ValueError(
                f"All of 'start', 'stop', 'spacing' are required. Missing: {missing}"
            )

        # Resolve count_dtype to string for Rust
        _dtype_map: dict[object, str] = {
            bool: "bool",
            int: "u32",
            pl.Boolean: "bool",
            pl.UInt8: "u8",
            pl.UInt16: "u16",
            pl.UInt32: "u32",
        }
        dtype_str: str | None = None
        if count_dtype is not None:
            dtype_str = _dtype_map.get(count_dtype)
            if dtype_str is None:
                raise TypeError(
                    f"count_dtype must be pl.Boolean, pl.UInt8, pl.UInt16, pl.UInt32, bool, or int, got {count_dtype!r}"
                )

        # Fast path: bins=int, no breakpoints, no dtype cast → direct List(UInt32) output.
        # list_histogram_bins_int_fast returns List(UInt32) via zero-copy static buffers,
        # eliminating the Struct wrapper and the map_batches 20MB copy allocation.
        if isinstance(bins, int) and not include_breakpoints and dtype_str is None:
            return register_plugin_function(
                args=[self._expr],
                plugin_path=_LIB,
                function_name="list_histogram_bins_int_fast",
                is_elementwise=True,
                returns_scalar=False,
                kwargs={"bins_int": bins},
            )

        args: list[pl.Expr] = [self._expr]
        kwargs: dict = {"arg_positions": {}, "count_dtype": dtype_str, "include_breakpoints": include_breakpoints}

        if has_bins:
            if isinstance(bins, str):
                bins = wrap_expr(parse_into_expression(bins))
            if isinstance(bins, pl.Expr):
                kwargs["mode"] = "bins_int"
                kwargs["bins_int"] = None
                kwargs["arg_positions"]["bins_int"] = len(args)
                args.append(bins)
            elif isinstance(bins, (list, pl.Series)):
                bins_list = bins.to_list() if isinstance(bins, pl.Series) else list(bins)
                kwargs["mode"] = "edges"
                kwargs["bins_edges"] = [float(x) for x in bins_list]
            elif isinstance(bins, int):
                kwargs["mode"] = "bins_int"
                kwargs["bins_int"] = bins
            else:
                raise TypeError(
                    f"bins must be int, list, pl.Series, str, or pl.Expr, got {type(bins)}"
                )
        else:
            kwargs["mode"] = "range"
            for name, value in [("start", start), ("stop", stop), ("spacing", spacing)]:
                if isinstance(value, (str, pl.Series)):
                    value = wrap_expr(parse_into_expression(value))
                if isinstance(value, pl.Expr):
                    kwargs["arg_positions"][name] = len(args)
                    args.append(value)
                    kwargs[name] = None
                else:
                    kwargs[name] = float(value)  # type: ignore[arg-type]

        result = register_plugin_function(
            args=args,
            plugin_path=_LIB,
            function_name="list_histogram",
            is_elementwise=True,
            returns_scalar=False,
            kwargs=kwargs,
        )

        # Post-process: cast counts dtype and/or unwrap struct
        needs_cast = dtype_str is not None and dtype_str != "u32"
        cast_map = {"bool": pl.Boolean, "u8": pl.UInt8, "u16": pl.UInt16}
        target = cast_map.get(dtype_str or "") if needs_cast else None

        if not include_breakpoints and target is not None:
            # Unwrap to just counts list + cast dtype
            _target_type = target
            def _unwrap_and_cast(s: pl.Series, _target: type[pl.DataType] = _target_type) -> pl.Series:
                return (
                    s.struct.field("counts")
                    .list.eval(pl.element().cast(_target))
                    .rename(s.name)
                )
            result = result.map_batches(
                _unwrap_and_cast, return_dtype=pl.List(target),
            )
        elif not include_breakpoints:
            # Unwrap to just counts list (default u32)
            def _unwrap(s: pl.Series) -> pl.Series:
                return s.struct.field("counts").rename(s.name)
            result = result.map_batches(
                _unwrap, return_dtype=pl.List(pl.UInt32),
            )
        elif target is not None:
            # Keep struct, cast counts dtype
            _target_type = target
            def _cast_counts(s: pl.Series, _target: type[pl.DataType] = _target_type) -> pl.Series:
                fields = s.struct.unnest()
                fields = fields.with_columns(
                    pl.col("counts").list.eval(
                        pl.element().cast(_target)
                    )
                )
                return fields.to_struct(s.name)
            result = result.map_batches(_cast_counts, return_dtype=pl.Struct({
                "breakpoints": pl.List(pl.Float64),
                "counts": pl.List(target),
            }))

        return result

    hist = histogram


def sum(*exprs: IntoExprColumn) -> pl.Expr | list[pl.Expr]:
    """
    Sum across rows for list columns (vertical aggregation).

    Returns a single row with a list where each element is the sum
    of elements at that position across all input lists.

    All lists must have the same length.

    Examples
    --------
    >>> import polars_vec_ops as vec
    >>> df = pl.DataFrame({"a": [[0, 1, 2], [1, 2, 3]]})
    >>> df.select(vec.sum("a"))
    shape: (1, 1)
    ┌───────────┐
    │ a         │
    │ ---       │
    │ list[i64] │
    ╞═══════════╡
    │ [1, 3, 5] │
    └───────────┘

    Can be called with multiple columns:
    >>> df = pl.DataFrame({"a": [[0, 1], [1, 2]], "b": [[10, 20], [30, 40]]})
    >>> df.select(vec.sum("a", "b"))
    shape: (1, 2)
    ┌───────────┬───────────┐
    │ a         ┆ b         │
    │ ---       ┆ ---       │
    │ list[i64] ┆ list[i64] │
    ╞═══════════╪═══════════╡
    │ [1, 3]    ┆ [40, 60]  │
    └───────────┴───────────┘
    """
    results = [VecOpsNamespace(wrap_expr(e)).sum() for e in parse_into_list_of_expressions(*exprs)]
    return results[0] if len(results) == 1 else results


def mean(*exprs: IntoExprColumn) -> pl.Expr | list[pl.Expr]:
    """
    Calculate mean across rows for list columns (vertical aggregation).

    Returns a single row with a list where each element is the mean
    of elements at that position across all input lists.

    All lists must have the same length.

    Returns
    -------
    pl.Expr
        Expression returning a list of Float64 values.

    Examples
    --------
    >>> import polars_vec_ops as vec
    >>> df = pl.DataFrame({"a": [[1, 2, 3], [3, 4, 5]]})
    >>> df.select(vec.mean("a"))
    shape: (1, 1)
    ┌─────────────────┐
    │ a               │
    │ ---             │
    │ list[f64]       │
    ╞═════════════════╡
    │ [2.0, 3.0, 4.0] │
    └─────────────────┘

    Can be called with multiple columns:
    >>> df = pl.DataFrame({"a": [[1, 2], [3, 4]], "b": [[10, 20], [30, 40]]})
    >>> df.select(vec.mean("a", "b"))
    shape: (1, 2)
    ┌────────────┬──────────────┐
    │ a          ┆ b            │
    │ ---        ┆ ---          │
    │ list[f64]  ┆ list[f64]    │
    ╞════════════╪══════════════╡
    │ [2.0, 3.0] ┆ [20.0, 30.0] │
    └────────────┴──────────────┘
    """
    results = [VecOpsNamespace(wrap_expr(e)).mean() for e in parse_into_list_of_expressions(*exprs)]
    return results[0] if len(results) == 1 else results


def avg(*exprs: IntoExprColumn) -> pl.Expr | list[pl.Expr]:
    """
    Alias for mean(). Calculate average across rows for list columns.

    See mean() for full documentation.
    """
    return mean(*exprs)


def min(*exprs: IntoExprColumn) -> pl.Expr | list[pl.Expr]:
    """
    Find minimum element at each position across rows (vertical aggregation).

    Returns a single row with a list where each element is the minimum
    of elements at that position across all input lists.

    All lists must have the same length.

    Returns
    -------
    pl.Expr
        Expression returning a list with the same type as input.

    Examples
    --------
    >>> import polars_vec_ops as vec
    >>> df = pl.DataFrame({"a": [[3, 5, 2], [1, 7, 4]]})
    >>> df.select(vec.min("a"))
    shape: (1, 1)
    ┌───────────┐
    │ a         │
    │ ---       │
    │ list[i64] │
    ╞═══════════╡
    │ [1, 5, 2] │
    └───────────┘

    Can be called with multiple columns:
    >>> df = pl.DataFrame({"a": [[3, 5], [1, 7]], "b": [[10, 20], [5, 15]]})
    >>> df.select(vec.min("a", "b"))
    shape: (1, 2)
    ┌───────────┬───────────┐
    │ a         ┆ b         │
    │ ---       ┆ ---       │
    │ list[i64] ┆ list[i64] │
    ╞═══════════╪═══════════╡
    │ [1, 5]    ┆ [5, 15]   │
    └───────────┴───────────┘
    """
    results = [VecOpsNamespace(wrap_expr(e)).min() for e in parse_into_list_of_expressions(*exprs)]
    return results[0] if len(results) == 1 else results


def max(*exprs: IntoExprColumn) -> pl.Expr | list[pl.Expr]:
    """
    Find maximum element at each position across rows (vertical aggregation).

    Returns a single row with a list where each element is the maximum
    of elements at that position across all input lists.

    All lists must have the same length.

    Returns
    -------
    pl.Expr
        Expression returning a list with the same type as input.

    Examples
    --------
    >>> import polars_vec_ops as vec
    >>> df = pl.DataFrame({"a": [[3, 5, 2], [1, 7, 4]]})
    >>> df.select(vec.max("a"))
    shape: (1, 1)
    ┌───────────┐
    │ a         │
    │ ---       │
    │ list[i64] │
    ╞═══════════╡
    │ [3, 7, 4] │
    └───────────┘

    Can be called with multiple columns:
    >>> df = pl.DataFrame({"a": [[3, 5], [1, 7]], "b": [[10, 20], [30, 15]]})
    >>> df.select(vec.max("a", "b"))
    shape: (1, 2)
    ┌───────────┬───────────┐
    │ a         ┆ b         │
    │ ---       ┆ ---       │
    │ list[i64] ┆ list[i64] │
    ╞═══════════╪═══════════╡
    │ [3, 7]    ┆ [30, 20]  │
    └───────────┴───────────┘
    """
    results = [VecOpsNamespace(wrap_expr(e)).max() for e in parse_into_list_of_expressions(*exprs)]
    return results[0] if len(results) == 1 else results


def diff(*exprs: IntoExprColumn) -> pl.Expr | list[pl.Expr]:
    """
    Calculate differences between consecutive rows at each position.

    Returns the same number of rows as input. The first row contains
    a list of nulls (no previous row to compare). Each subsequent row
    contains the element-wise difference from the previous row: row[i] - row[i-1].

    If either the current or previous row is null, the result is a list of nulls.

    All lists must have the same length.

    Returns
    -------
    pl.Expr
        Expression returning lists with differences, preserving input type.
        First row is always a list of nulls.

    Examples
    --------
    >>> import polars_vec_ops as vec
    >>> df = pl.DataFrame({"a": [[5, 10, 15], [2, 15, 5], [0, 0, 0]]})
    >>> df.select(vec.diff("a"))
    shape: (3, 1)
    ┌────────────────────┐
    │ a                  │
    │ ---                │
    │ list[i64]          │
    ╞════════════════════╡
    │ [null, null, null] │
    │ [-3, 5, -10]       │
    │ [-2, -15, -5]      │
    └────────────────────┘

    Can be called with multiple columns:
    >>> df = pl.DataFrame({"a": [[5, 10], [2, 15], [0, 0]], "b": [[50, 100], [20, 150], [0, 0]]})
    >>> df.select(vec.diff("a", "b"))
    shape: (3, 2)
    ┌──────────────┬──────────────┐
    │ a            ┆ b            │
    │ ---          ┆ ---          │
    │ list[i64]    ┆ list[i64]    │
    ╞══════════════╪══════════════╡
    │ [null, null] ┆ [null, null] │
    │ [-3, 5]      ┆ [-30, 50]    │
    │ [-2, -15]    ┆ [-20, -150]  │
    └──────────────┴──────────────┘
    """
    results = [VecOpsNamespace(wrap_expr(e)).diff() for e in parse_into_list_of_expressions(*exprs)]
    return results[0] if len(results) == 1 else results


def convolve(
    expr: IntoExprColumn,
    kernel: list[float] | pl.Series,
    fill_value: float = 0.0,
    mode: str = "same",
) -> pl.Expr:
    """
    Convolve each list in the column with the given kernel.

    Performs 1D convolution on each row's list independently.
    The convolution operation applies the kernel as a sliding window
    over each signal, computing weighted sums.

    Parameters
    ----------
    expr
        Column name containing lists/arrays to convolve.
    kernel
        The convolution kernel (filter). Can be a list of floats or
        a Polars Series. Non-finite values are filtered out.
    fill_value
        Value to use for null elements in the signal. Default is 0.0.
    mode
        Convolution mode, one of:
        - "full": Full convolution, output length = signal_len + kernel_len - 1
        - "same": Same length as signal, centered (default)
        - "valid": Only where signal and kernel fully overlap,
                  output length = signal_len - kernel_len + 1
        - "left": Same length as signal, left-aligned
        - "right": Same length as signal, right-aligned

    Returns
    -------
    pl.Expr
        Expression returning lists of Float64 values.

    Examples
    --------
    >>> import polars_vec_ops as vec
    >>> df = pl.DataFrame({"signal": [[1, 2, 3, 4, 5]]})
    >>> kernel = [0.25, 0.5, 0.25]
    >>> df.select(vec.convolve("signal", kernel, mode="same"))
    shape: (1, 1)
    ┌───────────────────┐
    │ signal            │
    │ ---               │
    │ list[f64]         │
    ╞═══════════════════╡
    │ [1.0, 2.0, … 3.5] │
    └───────────────────┘
    """
    return VecOpsNamespace(wrap_expr(parse_into_expression(expr))).convolve(kernel, fill_value, mode)


def histogram(
    expr: IntoExprColumn,
    bins: int | list[float] | pl.Series | pl.Expr | str | None = None,
    *,
    start: float | pl.Expr | None = None,
    stop: float | pl.Expr | None = None,
    spacing: float | pl.Expr | None = None,
    count_dtype: pl.DataType | type[bool] | type[int] | None = None,
    include_breakpoints: bool = False,
) -> pl.Expr:
    """
    Compute a histogram for each list in the column.

    Bin specification supports two mutually exclusive modes:

    1. ``bins``: an integer number of bins (auto-ranged from data) or
       explicit bin edges as a list.
    2. ``start`` / ``stop`` / ``spacing``: evenly spaced bins.

    Any scalar parameter can be a ``pl.Expr`` to derive values per-row
    from other columns.

    Parameters
    ----------
    expr
        Column name containing lists/arrays.
    bins
        Number of bins (int) or explicit bin edges (list/Series).
        Can be a ``pl.Expr`` resolving to an integer column.
    start
        Left edge of the first bin.
    stop
        Right edge of the last bin.
    spacing
        Width of each bin.
    count_dtype
        Data type for the counts field. Smaller types save memory.
        Accepts ``pl.Boolean`` / ``bool``, ``pl.UInt8``, ``pl.UInt16``,
        or ``pl.UInt32`` / ``int`` (default).
    include_breakpoints
        If ``True``, return a Struct with ``breakpoints`` and ``counts``.
        If ``False`` (default), return ``List[count_dtype]`` directly.

    Returns
    -------
    pl.Expr
        If ``include_breakpoints=True``, returns a Struct with fields:
        - ``breakpoints``: ``List[Float64]`` — n+1 bin edges
        - ``counts``: ``List[count_dtype]`` — n bin counts

        If ``include_breakpoints=False`` (default), returns
        ``List[count_dtype]`` directly.

    Examples
    --------
    >>> import polars_vec_ops as vec
    >>> df = pl.DataFrame({"a": [[1, 2, 3, 4, 5]]})
    >>> df.select(vec.histogram("a", bins=3))
    shape: (1, 1)
    ┌───────────┐
    │ a         │
    │ ---       │
    │ list[u32] │
    ╞═══════════╡
    │ [2, 1, 2] │
    └───────────┘
    """
    return VecOpsNamespace(wrap_expr(parse_into_expression(expr))).histogram(
        bins, start=start, stop=stop, spacing=spacing,
        count_dtype=count_dtype, include_breakpoints=include_breakpoints,
    )


hist = histogram
