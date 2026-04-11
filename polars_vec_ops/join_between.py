from __future__ import annotations

import polars as pl
from polars._typing import IntoExpr, IntoExprColumn, FrameType


@pl.api.register_dataframe_namespace("vec")
@pl.api.register_lazyframe_namespace("vec")
class VecOpsNamespace:
    """Custom namespace for vertical list operations."""

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
        list to the half-open window ``[start, stop)`` defined by ``bounds``,
        and realigns the retained values to the lower bound (``value - start``).
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
            name or expression in ``other`` resolving to a scalar Float64.
            Defines the half-open window ``[start, stop)`` applied to ``values``.
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
            Allow the cross-join to be parallelised across partitions.
            Default: ``True``.
        force_parallel
            Force parallel execution even if the cost model would choose serial.
            Default: ``False``.
        check_sortedness
            If ``True``, validate that every list in ``values`` is sorted in
            ascending order before joining.  Raises ``InvalidOperationError`` if
            not.  Disable only when sortedness is guaranteed upstream.
            Default: ``True``.

        Returns
        -------
        pl.DataFrame
            Shape ``(n_self_rows * n_other_rows, ...)``.  All columns from both
            DataFrames are retained.  The column resolved by ``values`` is replaced
            with either:

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
        shape: (6, 4)
        ┌─────────┬──────────┬────────────┬─────────────┐
        │ event_id ┆ interval_id ┆ start_time ┆ event_times │
        │ ---     ┆ ---      ┆ ---        ┆ ---         │
        │ i64     ┆ i64      ┆ f64        ┆ list[f64]   │
        ╞═════════╪══════════╪════════════╪═════════════╡
        │ 0       ┆ 0        ┆ 0.0        ┆ [0.1, 0.5]  │
        │ 0       ┆ 1        ┆ 1.0        ┆ [1.2, 1.8]  │
        │ 0       ┆ 2        ┆ 2.0        ┆ [2.3]       │
        │ 1       ┆ 0        ┆ 0.0        ┆ [0.2, 0.9]  │
        │ 1       ┆ 1        ┆ 1.0        ┆ [1.5]       │
        │ 1       ┆ 2        ┆ 2.0        ┆ [2.1]       │
        └─────────┴──────────┴────────────┴─────────────┘

        Pass ``relative=True`` to shift values to interval onset:

        >>> events.vec.join_between(
        ...     other=intervals,
        ...     values="event_times",
        ...     bounds=("start_time", "stop_time"),
        ...     relative=True,
        ... )
        shape: (6, 4)
        ┌─────────┬──────────┬────────────┬─────────────┐
        │ event_id ┆ interval_id ┆ start_time ┆ event_times │
        │ ---     ┆ ---      ┆ ---        ┆ ---         │
        │ i64     ┆ i64      ┆ f64        ┆ list[f64]   │
        ╞═════════╪══════════╪════════════╪═════════════╡
        │ 0       ┆ 0        ┆ 0.0        ┆ [0.1, 0.5]  │
        │ 0       ┆ 1        ┆ 1.0        ┆ [0.2, 0.8]  │
        │ 0       ┆ 2        ┆ 2.0        ┆ [0.3]       │
        │ 1       ┆ 0        ┆ 0.0        ┆ [0.2, 0.9]  │
        │ 1       ┆ 1        ┆ 1.0        ┆ [0.5]       │
        │ 1       ┆ 2        ┆ 2.0        ┆ [0.1]       │
        └─────────┴──────────┴────────────┴─────────────┘

        Use ``as_counts=True`` for a fast scalar spike count per trial — no list
        allocation, just two binary searches on the sorted spike times:

        >>> events.vec.join_between(
        ...     other=intervals,
        ...     values="event_times",
        ...     bounds=("start_time", "stop_time"),
        ...     as_counts=True,
        ... )
        shape: (6, 4)
        ┌─────────┬──────────┬────────────┬─────────────┐
        │ event_id ┆ interval_id ┆ start_time ┆ event_times │
        │ ---     ┆ ---      ┆ ---        ┆ ---         │
        │ i64     ┆ i64      ┆ f64        ┆ u32         │
        ╞═════════╪══════════╪════════════╪═════════════╡
        │ 0       ┆ 0        ┆ 0.0        ┆ 2           │
        │ 0       ┆ 1        ┆ 1.0        ┆ 2           │
        │ 0       ┆ 2        ┆ 2.0        ┆ 1           │
        │ 1       ┆ 0        ┆ 0.0        ┆ 2           │
        │ 1       ┆ 1        ┆ 1.0        ┆ 1           │
        │ 1       ┆ 2        ┆ 2.0        ┆ 1           │
        └─────────┴──────────┴────────────┴─────────────┘
        
        """
        ...