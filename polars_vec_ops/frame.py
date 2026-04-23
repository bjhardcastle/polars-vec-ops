from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from polars._typing import IntoExpr, IntoExprColumn, FrameType, PolarsDataType
from polars.plugins import register_plugin_function

_LIB = Path(__file__).parent


@pl.api.register_dataframe_namespace("vec")
@pl.api.register_lazyframe_namespace("vec")
class VecOpsNamespace:
    """Custom namespace for vertical list operations."""

    def __init__(self, df: FrameType) -> None:
        self._df: Any = df

    def join_between(
        self,
        other: FrameType,
        values: IntoExprColumn,
        bounds: tuple[IntoExpr, IntoExpr],
        *,
        on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        left_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        right_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        relative: bool = False,
        as_counts: bool = False,
        allow_parallel: bool = True,
        force_parallel: bool = False,
        check_sortedness: bool = True,
    ) -> FrameType:
        """
        Join with an intervals table, clipping a list or array column to each interval.

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
            Values outside ``[start, stop)`` are dropped.  Array columns are
            treated as variable-length lists; the output column is always
            ``list``.
        bounds
            Two-element tuple ``(start, stop)`` where each element is a column
            name or expression in ``other`` resolving to a scalar, or a literal value.
            Defines the half-open window ``[start, stop)`` applied to ``values``.
        on
            Join key or keys present in both frames. When provided, only
            matching rows are paired, following regular inner-join semantics.
        left_on, right_on
            Join key or keys to match between ``self`` and ``other`` when the
            column names differ. Mutually exclusive with ``on``.
        relative
            If ``True``, shift retained values by ``-start`` so they are
            expressed relative to interval onset.  Default: ``False``
            (values remain in their original coordinate space).
        as_counts
            If ``True``, return the number of ``values`` values that fall within
            ``[start, stop)`` as a ``UInt32`` scalar instead of materialising
            the clipped list.  Faster and more memory-efficient when the actual
            values are not needed — the count is derived from two binary
            searches on the sorted list with no allocation.
            Default: ``False``.
        allow_parallel
            Currently unused; reserved for future parallel-execution support.
            Default: ``True``.
        force_parallel
            Currently unused; reserved for future parallel-execution support.
            Default: ``False``.
        check_sortedness
            If ``True``, validate that every list in ``values`` is sorted in
            ascending order before joining.  Raises ``InvalidOperationError`` if
            not.  Disable only when sortedness is guaranteed upstream.
            Default: ``True``.

        Returns
        -------
        pl.DataFrame | pl.LazyFrame
            Without join keys, shape ``(n_self_rows * n_other_rows, ...)``.
            With join keys, one row per matched pair. Columns from ``self``
            appear first (with ``values`` replaced), followed by columns from
            ``other`` using Polars join-column semantics. The column resolved by
            ``values`` is replaced with either:

            - ``list[T]`` — clipped values (absolute coordinates by default,
              or shifted by ``-start`` when ``relative=True``), when
              ``as_counts=False`` (default).  The inner type ``T`` preserves
              the input type, or follows Polars type-promotion rules when
              ``relative=True`` (the subtraction ``value_element - start`` may
              widen the type).  Array input is always returned as ``list``.
            - ``UInt32`` — count of values within ``[start, stop)``,
              when ``as_counts=True``.

        Examples
        --------
        >>> import polars as pl
        >>> import polars_vec_ops  # noqa: F401 — registers .vec namespace
        >>> events = pl.DataFrame({
        ...     "event_id": [0, 1],
        ...     "event_times": [[0.1, 0.5, 1.2, 1.8, 2.3], [0.2, 0.9, 1.5, 2.1]],
        ... })
        >>> intervals = pl.DataFrame({
        ...     "interval_id": [0, 1, 2],
        ...     "start_time": [0.0, 1.0, 2.0],
        ...     "stop_time":  [1.0, 2.0, 3.0],
        ... })
        >>> events.vec.join_between(
        ...     other=intervals,
        ...     values="event_times",
        ...     bounds=("start_time", "stop_time"),
        ... )
        shape: (6, 5)
        ┌──────────┬─────────────┬─────────────┬────────────┬───────────┐
        │ event_id ┆ event_times ┆ interval_id ┆ start_time ┆ stop_time │
        │ ---      ┆ ---         ┆ ---         ┆ ---        ┆ ---       │
        │ i64      ┆ list[f64]   ┆ i64         ┆ f64        ┆ f64       │
        ╞══════════╪═════════════╪═════════════╪════════════╪═══════════╡
        │ 0        ┆ [0.1, 0.5]  ┆ 0           ┆ 0.0        ┆ 1.0       │
        │ 0        ┆ [1.2, 1.8]  ┆ 1           ┆ 1.0        ┆ 2.0       │
        │ 0        ┆ [2.3]       ┆ 2           ┆ 2.0        ┆ 3.0       │
        │ 1        ┆ [0.2, 0.9]  ┆ 0           ┆ 0.0        ┆ 1.0       │
        │ 1        ┆ [1.5]       ┆ 1           ┆ 1.0        ┆ 2.0       │
        │ 1        ┆ [2.1]       ┆ 2           ┆ 2.0        ┆ 3.0       │
        └──────────┴─────────────┴─────────────┴────────────┴───────────┘

        Pass ``relative=True`` to shift values to interval onset:

        >>> events.vec.join_between(
        ...     other=intervals,
        ...     values="event_times",
        ...     bounds=("start_time", "stop_time"),
        ...     relative=True,
        ... )
        shape: (6, 5)
        ┌──────────┬─────────────┬─────────────┬────────────┬───────────┐
        │ event_id ┆ event_times ┆ interval_id ┆ start_time ┆ stop_time │
        │ ---      ┆ ---         ┆ ---         ┆ ---        ┆ ---       │
        │ i64      ┆ list[f64]   ┆ i64         ┆ f64        ┆ f64       │
        ╞══════════╪═════════════╪═════════════╪════════════╪═══════════╡
        │ 0        ┆ [0.1, 0.5]  ┆ 0           ┆ 0.0        ┆ 1.0       │
        │ 0        ┆ [0.2, 0.8]  ┆ 1           ┆ 1.0        ┆ 2.0       │
        │ 0        ┆ [0.3]       ┆ 2           ┆ 2.0        ┆ 3.0       │
        │ 1        ┆ [0.2, 0.9]  ┆ 0           ┆ 0.0        ┆ 1.0       │
        │ 1        ┆ [0.5]       ┆ 1           ┆ 1.0        ┆ 2.0       │
        │ 1        ┆ [0.1]       ┆ 2           ┆ 2.0        ┆ 3.0       │
        └──────────┴─────────────┴─────────────┴────────────┴───────────┘

        Use ``as_counts=True`` for a fast scalar spike count per trial — no list
        allocation, just two binary searches on the sorted spike times:

        >>> events.vec.join_between(
        ...     other=intervals,
        ...     values="event_times",
        ...     bounds=("start_time", "stop_time"),
        ...     as_counts=True,
        ... )
        shape: (6, 5)
        ┌──────────┬─────────────┬─────────────┬────────────┬───────────┐
        │ event_id ┆ event_times ┆ interval_id ┆ start_time ┆ stop_time │
        │ ---      ┆ ---         ┆ ---         ┆ ---        ┆ ---       │
        │ i64      ┆ u32         ┆ i64         ┆ f64        ┆ f64       │
        ╞══════════╪═════════════╪═════════════╪════════════╪═══════════╡
        │ 0        ┆ 2           ┆ 0           ┆ 0.0        ┆ 1.0       │
        │ 0        ┆ 2           ┆ 1           ┆ 1.0        ┆ 2.0       │
        │ 0        ┆ 1           ┆ 2           ┆ 2.0        ┆ 3.0       │
        │ 1        ┆ 2           ┆ 0           ┆ 0.0        ┆ 1.0       │
        │ 1        ┆ 1           ┆ 1           ┆ 1.0        ┆ 2.0       │
        │ 1        ┆ 1           ┆ 2           ┆ 2.0        ┆ 3.0       │
        └──────────┴─────────────┴─────────────┴────────────┴───────────┘
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

        def _parse_join_keys(
            raw: str | pl.Expr | Sequence[str | pl.Expr],
        ) -> list[str | pl.Expr]:
            if isinstance(raw, (str, pl.Expr)):
                raw_items = [raw]
            elif isinstance(raw, Sequence):
                raw_items = list(raw)
            else:
                raise TypeError(
                    "join keys must be a column name, expression, or sequence of those"
                )

            join_key_items: list[str | pl.Expr] = []
            for item in raw_items:
                if not isinstance(item, (str, pl.Expr)):
                    raise TypeError(
                        "join keys must be a column name, expression, or sequence of those"
                    )
                join_key_items.append(item)

            return join_key_items

        def _normalise_join_keys() -> tuple[list[str | pl.Expr], list[str | pl.Expr]] | None:
            if on is not None:
                if left_on is not None or right_on is not None:
                    raise ValueError(
                        "cannot specify both `on` and `left_on`/`right_on`"
                    )
                parsed = _parse_join_keys(on)
                return parsed, parsed

            if left_on is None and right_on is None:
                return None
            if left_on is None or right_on is None:
                raise ValueError("`left_on` and `right_on` must both be provided")

            return _parse_join_keys(left_on), _parse_join_keys(right_on)

        join_keys = _normalise_join_keys()
        if join_keys is not None and len(join_keys[0]) != len(join_keys[1]):
            raise ValueError("`left_on` and `right_on` must have the same number of keys")

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
            out_inner: PolarsDataType = (
                pl.Series([0], dtype=inner_dtype)
                - pl.Series([0], dtype=start_dtype)
            ).dtype
        else:
            out_inner = inner_dtype

        return_dtype: PolarsDataType = pl.UInt32 if as_counts else pl.List(out_inner)

        if join_keys is not None:
            _TEMP_LEFT_IDX = "__vec_jb_left_idx__"
            _TEMP_RIGHT_IDX = "__vec_jb_right_idx__"
            _TEMP_LEFT_GROUP = "__vec_jb_left_group__"
            _TEMP_RIGHT_GROUP = "__vec_jb_right_group__"

            def _join_expr(raw: str | pl.Expr, alias: str) -> pl.Expr:
                return _to_expr(raw).alias(alias)

            def _counts_from_clipped(clipped: pl.Series) -> pl.Series:
                return (
                    clipped
                    .to_frame()
                    .select(
                        pl.when(pl.col(val_col_name).is_null())
                        .then(pl.lit(None).cast(pl.UInt32))
                        .otherwise(pl.col(val_col_name).list.len().cast(pl.UInt32))
                        .alias(val_col_name)
                    )
                    [val_col_name]
                )

            left_key_names = [
                f"__vec_jb_left_key_{idx}__"
                for idx in range(len(join_keys[0]))
            ]
            right_key_names = [
                f"__vec_jb_right_key_{idx}__"
                for idx in range(len(join_keys[1]))
            ]
            left_key_exprs = [
                _join_expr(raw, alias)
                for raw, alias in zip(join_keys[0], left_key_names)
            ]
            right_key_exprs = [
                _join_expr(raw, alias)
                for raw, alias in zip(join_keys[1], right_key_names)
            ]

            df_no_val = df_eager.drop(val_col_name) if val_col_name in df_eager.columns else df_eager
            other_data_cols = [c for c in other_with_bounds.columns if c not in (_TEMP_START, _TEMP_STOP)]
            other_data = other_with_bounds.select(other_data_cols)

            left_meta = df_no_val.with_row_index(_TEMP_LEFT_IDX)
            right_meta = other_data.with_row_index(_TEMP_RIGHT_IDX)

            if on is not None:
                joined_meta = left_meta.join(
                    right_meta,
                    on=on,
                    how="inner",
                )
            else:
                joined_meta = left_meta.join(
                    right_meta,
                    left_on=left_on,
                    right_on=right_on,
                    how="inner",
                )

            joined_meta = joined_meta.sort([_TEMP_LEFT_IDX, _TEMP_RIGHT_IDX])

            values_only = df_work.select(_TEMP_VAL)

            left_group_source = df_work.with_row_index(_TEMP_LEFT_IDX).with_columns(left_key_exprs)
            right_group_source = other_with_bounds.with_row_index(_TEMP_RIGHT_IDX).with_columns(right_key_exprs)

            left_groups = (
                left_group_source
                .select(_TEMP_LEFT_IDX, *left_key_names)
                .group_by(left_key_names, maintain_order=True)
                .agg(pl.col(_TEMP_LEFT_IDX).alias(_TEMP_LEFT_GROUP))
            )
            right_groups = (
                right_group_source
                .select(_TEMP_RIGHT_IDX, *right_key_names)
                .group_by(right_key_names, maintain_order=True)
                .agg(pl.col(_TEMP_RIGHT_IDX).alias(_TEMP_RIGHT_GROUP))
            )

            matched_groups = left_groups.join(
                right_groups,
                left_on=left_key_names,
                right_on=right_key_names,
                how="inner",
            )

            clipped_chunks: list[pl.Series] = []
            left_index_chunks: list[np.ndarray] = []
            right_index_chunks: list[np.ndarray] = []

            for left_idx_group, right_idx_group in matched_groups.select(
                _TEMP_LEFT_GROUP,
                _TEMP_RIGHT_GROUP,
            ).iter_rows():
                left_idx_array = np.asarray(left_idx_group, dtype=np.uint32)
                right_idx_array = np.asarray(right_idx_group, dtype=np.uint32)

                right_idx_series = pl.Series(right_idx_array)
                bucket_clip_expr = register_plugin_function(
                    args=[
                        pl.col(_TEMP_VAL),
                        pl.lit(starts_series[right_idx_series]),
                        pl.lit(stops_series[right_idx_series]),
                    ],
                    plugin_path=_LIB,
                    function_name="cross_clip_series",
                    is_elementwise=False,
                    returns_scalar=False,
                    kwargs={"relative": relative},
                )

                if not as_counts and return_dtype != pl.List(pl.Float64):
                    bucket_clip_expr = bucket_clip_expr.cast(return_dtype)

                left_idx_series = pl.Series(left_idx_array)
                clipped_chunk = values_only[left_idx_series].select(
                    bucket_clip_expr.alias(val_col_name)
                )[val_col_name]

                if as_counts:
                    clipped_chunk = _counts_from_clipped(clipped_chunk)

                clipped_chunks.append(clipped_chunk)
                left_index_chunks.append(
                    np.repeat(left_idx_array, len(right_idx_array))
                )
                right_index_chunks.append(
                    np.tile(right_idx_array, len(left_idx_array))
                )

            if clipped_chunks:
                clipped_series = pl.concat(
                    [chunk.to_frame() for chunk in clipped_chunks],
                    how="vertical",
                )[val_col_name]

                left_indices = np.concatenate(left_index_chunks)
                right_indices = np.concatenate(right_index_chunks)
                order = pl.Series(
                    np.lexsort((right_indices, left_indices)).astype(np.uint32)
                )
                clipped_series = clipped_series[order]
            else:
                clipped_series = pl.Series(val_col_name, [], dtype=return_dtype)

            left_cols = list(df_no_val.columns)
            right_cols = [
                c
                for c in joined_meta.columns
                if c not in left_cols and c not in (_TEMP_LEFT_IDX, _TEMP_RIGHT_IDX)
            ]

            result_parts: list[pl.DataFrame] = []
            if left_cols:
                result_parts.append(joined_meta.select(left_cols))
            result_parts.append(pl.DataFrame({val_col_name: clipped_series}))
            if right_cols:
                result_parts.append(joined_meta.select(right_cols))

            result = pl.concat(result_parts, how="horizontal")

            if is_lazy:
                return result.lazy()  # type: ignore[return-value]
            return result  # type: ignore[return-value]

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

        # Post-process: cast to correct output dtype
        if not as_counts and return_dtype != pl.List(pl.Float64):
            cross_clip_expr = cross_clip_expr.cast(return_dtype)

        # Apply cross_clip to extract only _TEMP_VAL column (avoid gathering large lists)
        clipped_series = df_work.select(_TEMP_VAL).select(
            cross_clip_expr.alias(val_col_name)
        )[val_col_name]

        # Convert list results to counts after clipping, preserving nulls.
        # This is done after the clip rather than inside the expression because
        # list.len() returns 0 for null lists in older polars versions.
        if as_counts:
            clipped_series = (
                clipped_series
                .to_frame()
                .select(
                    pl.when(pl.col(val_col_name).is_null())
                    .then(pl.lit(None).cast(pl.UInt32))
                    .otherwise(pl.col(val_col_name).list.len().cast(pl.UInt32))
                    .alias(val_col_name)
                )
                [val_col_name]
            )

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

        # Gather other rows
        other_data = other_with_bounds.select(other_data_cols)
        other_tiled = other_data[interval_indices]

        # Build clipped column as a single-column DataFrame
        clipped_df = pl.DataFrame({val_col_name: clipped_series})

        # Stack horizontally: df columns (if any) + clipped values + other columns
        if df_no_val.width > 0:
            df_tiled = df_no_val[unit_indices]
            result = pl.concat([df_tiled, clipped_df, other_tiled], how="horizontal")
        else:
            result = pl.concat([clipped_df, other_tiled], how="horizontal")

        if is_lazy:
            return result.lazy()  # type: ignore[return-value]
        return result  # type: ignore[return-value]
