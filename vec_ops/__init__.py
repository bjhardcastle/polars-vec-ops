from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from vec_ops._internal import __version__ as __version__

if TYPE_CHECKING:
    from vec_ops.typing import IntoExprColumn

LIB = Path(__file__).parent




@pl.api.register_expr_namespace("vec_ops")
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
        >>> df.select(pl.col("a").vec_ops.sum())
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
        >>> df.select(pl.col("a").vec_ops.mean())
        shape: (1, 1)
        ┌─────────────┐
        │ a           │
        │ ---         │
        │ list[f64]   │
        ╞═════════════╡
        │ [2.0, 3.0, 4.0] │
        └─────────────┘
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
        >>> df.select(pl.col("a").vec_ops.min())
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
        >>> df.select(pl.col("a").vec_ops.max())
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
        >>> df.select(pl.col("a").vec_ops.diff())
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