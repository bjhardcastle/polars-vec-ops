from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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

        # Extract starts/stops Series for Rust plugin
        starts_series = other_with_bounds[_TEMP_START].cast(pl.Float64)
        stops_series = other_with_bounds[_TEMP_STOP].cast(pl.Float64)
        n_intervals = len(starts_series)

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

        # ── Call cross_clip_series Rust plugin (no cross-join!) ───────────────
        # Passes starts/stops as Series inputs (avoids kwargs serialization overhead).
        # Takes values column (n_units rows) + starts + stops Series.
        # Produces n_units × n_intervals rows.
        cross_clip_expr = register_plugin_function(
            args=[pl.col(_TEMP_VAL), pl.lit(starts_series), pl.lit(stops_series)],
            plugin_path=_LIB,
            function_name="cross_clip_series",
            is_elementwise=False,
            returns_scalar=False,
            kwargs={"relative": relative},
        )

        # Post-process: convert to counts or cast to correct output dtype
        if as_counts:
            cross_clip_expr = cross_clip_expr.list.len().cast(pl.UInt32)
        elif return_dtype != pl.List(pl.Float64):
            cross_clip_expr = cross_clip_expr.cast(return_dtype)

        # Apply cross_clip to extract only _TEMP_VAL column (avoid gathering large lists)
        clipped_series = df_work.select(_TEMP_VAL).select(
            cross_clip_expr.alias(val_col_name)
        )[val_col_name]

        # ── Build the output DataFrame efficiently ────────────────────────
        # We need n_units × n_intervals rows in order:
        #   unit0×int0, unit0×int1, ..., unit0×intN, unit1×int0, ...
        n_units = len(df_work)
        other_data_cols = [c for c in other_with_bounds.columns if c not in (_TEMP_START, _TEMP_STOP)]

        # Tile df_eager rows (excluding val column): unit0 × n_intervals times, ...
        # Drop the values column from df_eager before gather to avoid copying large lists
        df_no_val = df_eager.drop(val_col_name) if val_col_name in df_eager.columns else df_eager

        # Build unit indices: [0,0,...,0, 1,1,...,1, ..., n_units-1, ...]
        # Use numpy for fast vectorized index construction
        unit_indices = pl.Series(
            np.repeat(np.arange(n_units, dtype=np.uint32), n_intervals)
        )
        # Build interval indices: [0,1,...,n-1, 0,1,...,n-1, ...]
        interval_indices = pl.Series(
            np.tile(np.arange(n_intervals, dtype=np.uint32), n_units)
        )

        # Gather df rows (now without large list column)
        df_tiled = df_no_val[unit_indices]

        # Gather other rows
        other_data = other_with_bounds.select(other_data_cols)
        other_tiled = other_data[interval_indices]

        # Build clipped column as a single-column DataFrame
        clipped_df = pl.DataFrame({val_col_name: clipped_series})

        # Stack horizontally: df columns + clipped values + other columns
        result = pl.concat([df_tiled, clipped_df, other_tiled], how="horizontal")

        if is_lazy:
            return result.lazy()
        return result
