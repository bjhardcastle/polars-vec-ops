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

        # Extract starts/stops as Python lists for passing to Rust kwargs
        starts_list: list[float] = other_with_bounds[_TEMP_START].cast(pl.Float64).to_list()
        stops_list: list[float] = other_with_bounds[_TEMP_STOP].cast(pl.Float64).to_list()
        n_intervals = len(starts_list)

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

        # ── Call cross_clip Rust plugin (no cross-join!) ─────────────────────
        # cross_clip takes the values column and produces n_units × n_intervals rows.
        # The output is in the same order as a cross-join: unit0×all_intervals, unit1×all_intervals, ...
        cross_clip_expr = register_plugin_function(
            args=[pl.col(_TEMP_VAL)],
            plugin_path=_LIB,
            function_name="cross_clip",
            is_elementwise=False,
            returns_scalar=False,
            kwargs={
                "starts": starts_list,
                "stops": stops_list,
                "relative": relative,
                "as_counts": False,
                "n_other_cols": 0,
            },
        )

        # Post-process: convert to counts or cast to correct output dtype
        if as_counts:
            cross_clip_expr = cross_clip_expr.list.len().cast(pl.UInt32)
        elif return_dtype != pl.List(pl.Float64):
            cross_clip_expr = cross_clip_expr.cast(return_dtype)

        # Apply cross_clip to the df_work (just the values column)
        clipped_series = df_work.select(
            cross_clip_expr.alias(val_col_name)
        )[val_col_name]

        # ── Build the output DataFrame ─────────────────────────────────────
        # The cross-join output has n_units × n_intervals rows.
        # Row i*n_intervals+j corresponds to unit i and interval j.
        # We need to tile df_eager rows (n_intervals times each) and
        # repeat other_eager (n_units times, interleaved).
        #
        # Build unit columns: repeat each unit row n_intervals times
        n_units = len(df_work)
        other_data_cols = [c for c in other_with_bounds.columns if c not in (_TEMP_START, _TEMP_STOP)]

        # Tile df_work rows: unit0×n_intervals, unit1×n_intervals, ...
        # Use row index to create the tiling
        unit_indices = pl.Series("__unit_idx__",
            [i for i in range(n_units) for _ in range(n_intervals)],
            dtype=pl.UInt32
        )
        interval_indices = pl.Series("__int_idx__",
            [j for _ in range(n_units) for j in range(n_intervals)],
            dtype=pl.UInt32
        )

        # Build output from df columns (tiled)
        df_tiled = df_eager.select([
            c for c in df_eager.columns if c != val_col_name or val_col_name not in df_eager.columns
        ]).take(unit_indices)

        # Build other columns (repeated per unit)
        other_data = other_with_bounds.select(other_data_cols)
        other_tiled = other_data.take(interval_indices)

        # Combine all columns
        result_frames = []
        # Add df_eager columns (excluding _TEMP_VAL if it was added)
        df_cols = [c for c in df_eager.columns]
        df_tiled2 = df_eager.take(unit_indices).select(df_cols)

        # Build clipped series as a DataFrame column
        clipped_df = pl.DataFrame({val_col_name: clipped_series})

        # Stack: df cols + clipped + other cols
        # Remove val_col_name from df_tiled2 if present (we use clipped instead)
        if val_col_name in df_tiled2.columns:
            df_base = df_tiled2.drop(val_col_name)
        else:
            df_base = df_tiled2

        result = pl.concat([df_base, clipped_df, other_tiled], how="horizontal")

        if is_lazy:
            return result.lazy()
        return result
