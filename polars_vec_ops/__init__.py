from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

from polars_vec_ops._internal import __version__ as __version__

LIB = Path(__file__).parent


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
            plugin_path=LIB,
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
            plugin_path=LIB,
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
            plugin_path=LIB,
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
            plugin_path=LIB,
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
            plugin_path=LIB,
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
            plugin_path=LIB,
            function_name="list_convolve",
            is_elementwise=True,  # Operates on each row independently
            returns_scalar=False,
            kwargs={
                "kernel": kernel_list,
                "fill_value": fill_value,
                "mode": mode,
            },
        )


def sum(*exprs: str) -> pl.Expr:
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
    return pl.col(exprs).vec.sum()  # type: ignore[attr-defined]


def mean(*exprs: str) -> pl.Expr:
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
    return pl.col(exprs).vec.mean()  # type: ignore[attr-defined]


def avg(*exprs: str) -> pl.Expr:
    """
    Alias for mean(). Calculate average across rows for list columns.

    See mean() for full documentation.
    """
    return mean(*exprs)


def min(*exprs: str) -> pl.Expr:
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
    return pl.col(exprs).vec.min()  # type: ignore[attr-defined]


def max(*exprs: str) -> pl.Expr:
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
    return pl.col(exprs).vec.max()  # type: ignore[attr-defined]


def diff(*exprs: str) -> pl.Expr:
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
    return pl.col(exprs).vec.diff()  # type: ignore[attr-defined]


def convolve(
    expr: str,
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
    return pl.col(expr).vec.convolve(kernel, fill_value, mode)  # type: ignore[attr-defined]
