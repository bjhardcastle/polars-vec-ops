from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from polars._typing import IntoExpr, IntoExprColumn, FrameType
from polars.plugins import register_plugin_function

_LIB = Path(__file__).parent


@pl.api.register_dataframe_namespace("vec")
@pl.api.register_lazyframe_namespace("vec")
class VecOpsNamespace:
    """Custom namespace for vertical list operations."""

    def __init__(self, df: FrameType) -> None:
        self._df = df

    def join_between(
        self,
        other: FrameType,
        values: IntoExprColumn,
        bounds: tuple[IntoExpr, IntoExpr],
        *,
        relative: bool = False,
        as_counts: bool = False,
        allow_parallel: bool = True,
        force_parallel: bool = False,
        check_sortedness: bool = True,
    ) -> FrameType:
        """
        Cross-join with an intervals table, clipping a list or array column to each interval.

        Performs a cross join of ``self`` × ``other``, clips each row's ``values``
        list to the half-open window ``[start, stop)`` defined by ``bounds``.
        One row is returned for every (self-row, other-row) pair.

        Parameters
        ----------
        other
            DataFrame containing the interval boundaries.  One row per interval.
        values
            Column name or expression in ``self`` resolving to a sorted
            ``list[T]`` or ``array[T, N]`` where ``T`` is any numeric type.
        bounds
            Two-element tuple ``(start, stop)`` where each element is a column
            name, expression, or literal in ``other`` resolving to a scalar.
            Defines the half-open window ``[start, stop)`` applied to ``values``.
        relative
            If ``True``, shift retained values by ``-start``.  Default: ``False``.
        as_counts
            If ``True``, return the count of values in ``[start, stop)`` as
            ``UInt32`` instead of the clipped list.  Default: ``False``.
        allow_parallel
            Allow the cross-join to be parallelised.  Default: ``True``.
        force_parallel
            Force parallel execution.  Default: ``False``.
        check_sortedness
            Validate that every list in ``values`` is sorted ascending.
            Default: ``True``.
        """
        df = self._df
        is_lazy = isinstance(df, pl.LazyFrame)

        # ── Normalise values → (expr, output-column name) ──────────────────
        if isinstance(values, str):
            values_expr: pl.Expr = pl.col(values)
            val_col_name: str = values
        elif isinstance(values, pl.Series):
            values_expr = pl.lit(values)
            val_col_name = values.name
        else:
            values_expr = values
            val_col_name = values.meta.output_name()

        # ── Normalise bounds → (start_expr, stop_expr) ─────────────────────
        def _to_expr(raw: IntoExpr) -> pl.Expr:
            if isinstance(raw, str):
                return pl.col(raw)
            if isinstance(raw, pl.Expr):
                return raw
            return pl.lit(raw)

        start_expr = _to_expr(bounds[0])
        stop_expr  = _to_expr(bounds[1])

        # ── Collect lazy inputs ─────────────────────────────────────────────
        df_eager    = df.collect()    if is_lazy                    else df
        other_eager = other.collect() if isinstance(other, pl.LazyFrame) else other

        # ── Evaluate values expr; cast Array → List ─────────────────────────
        _TEMP_VAL   = "__vec_jb_values__"
        _TEMP_START = "__vec_jb_start__"
        _TEMP_STOP  = "__vec_jb_stop__"

        df_work = df_eager.with_columns(values_expr.alias(_TEMP_VAL))

        raw_dtype = df_work.schema[_TEMP_VAL]
        if isinstance(raw_dtype, pl.Array):
            inner_dtype = raw_dtype.inner
            df_work = df_work.with_columns(
                pl.col(_TEMP_VAL).cast(pl.List(inner_dtype))
            )
        else:
            inner_dtype = raw_dtype.inner

        # ── Sortedness check ────────────────────────────────────────────────
        if check_sortedness:
            def _is_sorted(val: Any) -> bool:
                if val is None:
                    return True
                s = val if isinstance(val, pl.Series) else pl.Series(val)
                return bool(s.is_sorted())

            all_sorted = df_work.select(
                pl.col(_TEMP_VAL)
                .map_elements(_is_sorted, return_dtype=pl.Boolean)
                .all()
            ).item()

            if not all_sorted:
                raise pl.exceptions.InvalidOperationError(
                    f"{val_col_name!r} contains lists that are not sorted in "
                    "ascending order; set check_sortedness=False to skip validation"
                )

        # ── Evaluate start/stop expressions in other_eager ─────────────────
        other_with_bounds = other_eager.with_columns(
            start_expr.alias(_TEMP_START),
            stop_expr.alias(_TEMP_STOP),
        )

        # ── Determine output dtype ──────────────────────────────────────────
        start_dtype = other_with_bounds.schema[_TEMP_START]

        if relative:
            out_inner = (
                pl.Series([0], dtype=inner_dtype)
                - pl.Series([0], dtype=start_dtype)
            ).dtype
        else:
            out_inner = inner_dtype

        return_dtype: pl.PolarsDataType = pl.UInt32 if as_counts else pl.List(out_inner)

        # ── Unit-loop: process one unit at a time, all intervals per unit ──
        # This preserves cross-join ordering: unit0×all_intervals, unit1×all_intervals, ...
        # while avoiding materializing the N×M cross-join DataFrame.
        #
        # For each unit row: repeat the single unit row M times (one per interval),
        # add start/stop columns from other_with_bounds, apply list_clip.
        # This avoids the huge cross-join memory cost while keeping correct ordering.
        #
        # Columns from other_eager (data cols, excluding temp bounds)
        other_data_cols = [c for c in other_with_bounds.columns if c not in (_TEMP_START, _TEMP_STOP)]
        n_intervals = len(other_with_bounds)
        n_units = len(df_work)

        # Pre-extract start/stop arrays for efficiency
        starts = other_with_bounds[_TEMP_START]
        stops = other_with_bounds[_TEMP_STOP]

        # Pre-extract other data columns as a DataFrame (already ordered by interval)
        other_data = other_with_bounds.select(other_data_cols) if other_data_cols else None

        # Build the Rust plugin expression (applied after column setup)
        def _make_clipped_expr() -> pl.Expr:
            clipped_expr = register_plugin_function(
                args=[pl.col(_TEMP_VAL), pl.col(_TEMP_START), pl.col(_TEMP_STOP)],
                plugin_path=_LIB,
                function_name="list_clip",
                is_elementwise=True,
                returns_scalar=False,
                kwargs={"relative": relative, "as_counts": False},
            )
            if as_counts:
                clipped_expr = clipped_expr.list.len().cast(pl.UInt32)
            elif return_dtype != pl.List(pl.Float64):
                clipped_expr = clipped_expr.cast(return_dtype)
            return clipped_expr

        clipped_expr = _make_clipped_expr()

        chunks = []
        for i in range(n_units):
            # Get this unit's row repeated N_intervals times
            unit_row = df_work[i]  # 1-row DataFrame
            # Tile the unit row n_intervals times using concat
            unit_tiled = pl.concat([unit_row] * n_intervals)

            # Add start/stop columns from intervals
            unit_tiled = unit_tiled.with_columns([
                starts.alias(_TEMP_START),
                stops.alias(_TEMP_STOP),
            ])

            # Apply Rust list_clip
            unit_result = unit_tiled.with_columns(clipped_expr.alias(val_col_name))
            unit_result = unit_result.drop([_TEMP_VAL, _TEMP_START, _TEMP_STOP])

            # Add other data columns from intervals
            if other_data is not None:
                unit_result = pl.concat([unit_result, other_data], how="horizontal")

            chunks.append(unit_result)

        if not chunks:
            # Return empty DataFrame with correct schema
            empty = df_work.head(0).drop([_TEMP_VAL])
            result = empty
        else:
            result = pl.concat(chunks)

        if is_lazy:
            return result.lazy()
        return result
