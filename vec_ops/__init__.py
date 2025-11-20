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