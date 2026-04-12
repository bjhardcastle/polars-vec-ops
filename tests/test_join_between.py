import polars as pl
import pytest

import polars_vec_ops  # noqa: F401 — registers .vec namespace

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def units():
    return pl.DataFrame({
        "event_id": [0, 1],
        "event_times": [[0.1, 0.5, 1.2, 1.8, 2.3], [0.2, 0.9, 1.5, 2.1]],
    })


@pytest.fixture
def intervals():
    return pl.DataFrame({
        "interval_id": [0, 1, 2],
        "start_time": [0.0, 1.0, 2.0],
        "stop_time":  [1.0, 2.0, 3.0],
    })


# ── Shape & columns ───────────────────────────────────────────────────────────

def test_join_between_row_count(units, intervals):
    result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
    )
    assert len(result) == len(units) * len(intervals)


def test_join_between_retains_all_columns(units, intervals):
    result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
    )
    for col in units.columns + intervals.columns:
        assert col in result.columns


def test_join_between_vec_column_dtype_float(units, intervals):
    """Float64 input preserves Float64 inner type."""
    result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
    )
    assert result.schema["event_times"] == pl.List(pl.Float64)


def test_join_between_vec_column_dtype_int():
    """Integer inner type is preserved (output follows type-promotion, not forced to Float64)."""
    units_int = pl.DataFrame({
        "event_id": [0],
        "values": [[1, 3, 7, 12, 20]],
    })
    intervals_int = pl.DataFrame({
        "interval_id": [0, 1],
        "start_time": [0, 10],
        "stop_time":  [5, 15],
    })
    result = units_int.vec.join_between(
        other=intervals_int,
        values="values",
        bounds=("start_time", "stop_time"),
    )
    # Without relative=True, inner type is unchanged and values are absolute
    assert result.schema["values"] == pl.List(pl.Int64)
    assert result.filter(pl.col("interval_id") == 0)["values"][0].to_list() == [1, 3]
    assert result.filter(pl.col("interval_id") == 1)["values"][0].to_list() == [12]

    result_rel = units_int.vec.join_between(
        other=intervals_int,
        values="values",
        bounds=("start_time", "stop_time"),
        relative=True,
    )
    # Int64 - Int64 → Int64; values are shifted
    assert result_rel.schema["values"] == pl.List(pl.Int64)
    assert result_rel.filter(pl.col("interval_id") == 0)["values"][0].to_list() == [1, 3]
    assert result_rel.filter(pl.col("interval_id") == 1)["values"][0].to_list() == [2]


def test_join_between_relative_int_vec_float_start():
    """Int vec with float start/stop promotes to float when relative=True."""
    units_int = pl.DataFrame({
        "event_id": [0],
        "values": [[1, 3, 7, 12, 20]],
    })
    intervals_float_start = pl.DataFrame({
        "interval_id": [0, 1],
        "start_time": [0.0, 10.0],   # Float64
        "stop_time":  [5.0, 15.0],
    })
    result = units_int.vec.join_between(
        other=intervals_float_start,
        values="values",
        bounds=("start_time", "stop_time"),
        relative=True,
    )
    # Int64 - Float64 → Float64 via Polars type promotion
    assert result.schema["values"] == pl.List(pl.Float64)


# ── Clipping & realignment ────────────────────────────────────────────────────

def test_join_between_clipped_values_absolute(units, intervals):
    """By default, retained values are in their original coordinate space."""
    result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
    )
    # unit 0, trial 0: window [0.0, 1.0) — start is 0 so absolute == relative
    assert result.filter(
        (pl.col("event_id") == 0) & (pl.col("interval_id") == 0)
    )["event_times"][0].to_list() == pytest.approx([0.1, 0.5])

    # unit 0, trial 1: window [1.0, 2.0) — values stay absolute
    assert result.filter(
        (pl.col("event_id") == 0) & (pl.col("interval_id") == 1)
    )["event_times"][0].to_list() == pytest.approx([1.2, 1.8])


def test_join_between_relative_shifts_to_onset(units, intervals):
    """relative=True shifts retained values by -start."""
    result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
        relative=True,
    )
    # unit 0, trial 1: absolute spikes [1.2, 1.8], start=1.0 → [0.2, 0.8]
    assert result.filter(
        (pl.col("event_id") == 0) & (pl.col("interval_id") == 1)
    )["event_times"][0].to_list() == pytest.approx([0.2, 0.8])

    # unit 1, trial 2: spike at 2.1, start=2.0 → 0.1
    assert result.filter(
        (pl.col("event_id") == 1) & (pl.col("interval_id") == 2)
    )["event_times"][0].to_list() == pytest.approx([0.1])


def test_join_between_relative_false_no_shift(units, intervals):
    """relative=False (default) does not subtract start."""
    result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
        relative=False,
    )
    assert result.filter(
        (pl.col("event_id") == 1) & (pl.col("interval_id") == 2)
    )["event_times"][0].to_list() == pytest.approx([2.1])


def test_join_between_no_spikes_in_window(units):
    """Window containing no spikes produces an empty list."""
    intervals_empty = pl.DataFrame({
        "interval_id": [0],
        "start_time": [10.0],
        "stop_time":  [11.0],
    })
    result = units.vec.join_between(
        other=intervals_empty,
        values="event_times",
        bounds=("start_time", "stop_time"),
    )
    for row in result["event_times"]:
        assert row.to_list() == []


def test_join_between_boundary_stop_excluded(units):
    """A spike exactly at stop is excluded (half-open interval)."""
    intervals_boundary = pl.DataFrame({
        "interval_id": [0],
        "start_time": [0.0],
        "stop_time":  [0.5],  # spike at 0.5 should be excluded
    })
    result = units.vec.join_between(
        other=intervals_boundary,
        values="event_times",
        bounds=("start_time", "stop_time"),
    )
    # unit 0: only 0.1 is in [0.0, 0.5); 0.5 is excluded
    row = result.filter(pl.col("event_id") == 0)["event_times"][0].to_list()
    assert row == pytest.approx([0.1])


def test_join_between_boundary_start_included(units):
    """A spike exactly at start is included."""
    intervals_boundary = pl.DataFrame({
        "interval_id": [0],
        "start_time": [0.1],
        "stop_time":  [1.0],
    })
    result = units.vec.join_between(
        other=intervals_boundary,
        values="event_times",
        bounds=("start_time", "stop_time"),
    )
    # unit 0: spike at 0.1 is exactly at start → included, kept as absolute 0.1
    row = result.filter(pl.col("event_id") == 0)["event_times"][0].to_list()
    assert row[0] == pytest.approx(0.1)


def test_join_between_boundary_start_included_relative(units):
    """With relative=True, a spike exactly at start maps to 0.0."""
    intervals_boundary = pl.DataFrame({
        "interval_id": [0],
        "start_time": [0.1],
        "stop_time":  [1.0],
    })
    result = units.vec.join_between(
        other=intervals_boundary,
        values="event_times",
        bounds=("start_time", "stop_time"),
        relative=True,
    )
    row = result.filter(pl.col("event_id") == 0)["event_times"][0].to_list()
    assert row[0] == pytest.approx(0.0)


# ── as_counts fastpath ─────────────────────────────────────────────────────────

def test_join_between_as_counts_dtype(units, intervals):
    result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
        as_counts=True,
    )
    assert result.schema["event_times"] == pl.UInt32


def test_join_between_as_counts_values(units, intervals):
    result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
        as_counts=True,
    )
    counts = (
        result
        .sort("event_id", "interval_id")
        ["event_times"]
        .to_list()
    )
    assert counts == [2, 2, 1, 2, 1, 1]


def test_join_between_as_counts_matches_list_len(units, intervals):
    """as_counts result equals len of list from the default mode."""
    list_result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
    ).sort("event_id", "interval_id")

    count_result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
        as_counts=True,
    ).sort("event_id", "interval_id")

    for list_val, count_val in zip(
        list_result["event_times"], count_result["event_times"]
    ):
        assert len(list_val) == count_val


def test_join_between_as_counts_empty_window(units):
    """as_counts returns 0 when no spikes fall in the window."""
    intervals_empty = pl.DataFrame({
        "interval_id": [0],
        "start_time": [10.0],
        "stop_time":  [11.0],
    })
    result = units.vec.join_between(
        other=intervals_empty,
        values="event_times",
        bounds=("start_time", "stop_time"),
        as_counts=True,
    )
    assert result["event_times"].to_list() == [0, 0]


def test_join_between_as_counts_retains_columns(units, intervals):
    result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
        as_counts=True,
    )
    for col in units.columns + intervals.columns:
        assert col in result.columns


# ── IntoExpr support ──────────────────────────────────────────────────────────

def test_join_between_vec_as_list_expr(units, intervals):
    """vec accepts arbitrary list expressions; clipping uses the transformed list.

    Adding 1 to the list shifts all values up by 1, producing verifiably
    different clipping results than the original column.
    """
    result = units.vec.join_between(
        other=intervals,
        values=pl.col("event_times") + 1,
        bounds=("start_time", "stop_time"),
    ).sort("event_id", "interval_id")

    # unit 0: +1 → [1.1, 1.5, 2.2, 2.8, 3.3]
    # trial 0 [0.0,1.0): nothing in window (all values >= 1.1)
    # trial 1 [1.0,2.0): [1.1, 1.5]  (differs from unshifted [1.2, 1.8])
    # trial 2 [2.0,3.0): [2.2, 2.8]
    assert result.filter((pl.col("event_id") == 0) & (pl.col("interval_id") == 0))["event_times"][0].to_list() == []
    assert result.filter((pl.col("event_id") == 0) & (pl.col("interval_id") == 1))["event_times"][0].to_list() == pytest.approx([1.1, 1.5])
    assert result.filter((pl.col("event_id") == 0) & (pl.col("interval_id") == 2))["event_times"][0].to_list() == pytest.approx([2.2, 2.8])


def test_join_between_interval_start_expr(units, intervals):
    """interval start accepts an expression; here start_time + 0.5 narrows the window."""
    result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=(pl.col("start_time") + 0.5, pl.col("stop_time")),
    ).sort("event_id", "interval_id")

    # trial 0 window becomes [0.5, 1.0) instead of [0.0, 1.0)
    assert result.filter((pl.col("event_id") == 0) & (pl.col("interval_id") == 0))["event_times"][0].to_list() == pytest.approx([0.5])
    # trial 1 window becomes [1.5, 2.0): only 1.8 for unit 0
    assert result.filter((pl.col("event_id") == 0) & (pl.col("interval_id") == 1))["event_times"][0].to_list() == pytest.approx([1.8])


def test_join_between_interval_literals(units):
    """interval tuple accepts plain Python scalars as literal bounds."""
    other = pl.DataFrame({"interval_id": [0]})  # single-row — literal bounds apply uniformly
    result = units.vec.join_between(
        other=other,
        values="event_times",
        bounds=(1.0, 2.0),
    )
    # window [1.0, 2.0) for all units
    assert result.filter(pl.col("event_id") == 0)["event_times"][0].to_list() == pytest.approx([1.2, 1.8])
    assert result.filter(pl.col("event_id") == 1)["event_times"][0].to_list() == pytest.approx([1.5])


# ── check_sortedness ──────────────────────────────────────────────────────────

def test_join_between_check_sortedness_raises_on_unsorted(intervals):
    unsorted_units = pl.DataFrame({
        "event_id": [0],
        "event_times": [[0.9, 0.1, 0.5]],  # unsorted
    })
    with pytest.raises(Exception):
        unsorted_units.vec.join_between(
            other=intervals,
            values="event_times",
            bounds=("start_time", "stop_time"),
            check_sortedness=True,
        )


def test_join_between_check_sortedness_false_skips_check(intervals):
    """check_sortedness=False does not raise even for unsorted input."""
    unsorted_units = pl.DataFrame({
        "event_id": [0],
        "event_times": [[0.9, 0.1, 0.5]],
    })
    # Should not raise
    unsorted_units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
        check_sortedness=False,
    )


# ── Null handling ─────────────────────────────────────────────────────────────

def test_join_between_null_vec_row_propagates(intervals):
    """A null list in the vec column produces a null output row."""
    units_with_null = pl.DataFrame({
        "event_id": [0, 1],
        "event_times": [[0.1, 0.5], None],
    })
    result = units_with_null.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
    )
    null_rows = result.filter(pl.col("event_id") == 1)
    assert null_rows["event_times"].is_null().all()


def test_join_between_null_vec_row_as_counts(intervals):
    """Null list in vec column produces null count when as_counts=True."""
    units_with_null = pl.DataFrame({
        "event_id": [0, 1],
        "event_times": [[0.1, 0.5], None],
    })
    result = units_with_null.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
        as_counts=True,
    )
    null_rows = result.filter(pl.col("event_id") == 1)
    assert null_rows["event_times"].is_null().all()


# ── Array column support ──────────────────────────────────────────────────────

def test_join_between_array_col_accepted(intervals):
    """vec accepts array[f64, N] columns; output is list (variable-length after clipping)."""
    units_arr = pl.DataFrame({
        "event_id": [0, 1],
        "event_times": [[0.1, 0.5, 1.2, 1.8, 2.3], [0.2, 0.9, 1.5, 2.1, 3.0]],
    }).with_columns(pl.col("event_times").cast(pl.Array(pl.Float64, 5)))

    result = units_arr.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
    )
    assert result.schema["event_times"].base_type() == pl.List
    assert len(result) == len(intervals) * 2


def test_join_between_array_col_values_match_list(intervals):
    """Array and List inputs produce identical clipped output."""
    spikes = [[0.1, 0.5, 1.2, 1.8, 2.3], [0.2, 0.9, 1.5, 2.1, 3.0]]

    units_list = pl.DataFrame({"event_id": [0, 1], "event_times": spikes})
    units_arr = units_list.with_columns(
        pl.col("event_times").cast(pl.Array(pl.Float64, 5))
    )

    result_list = units_list.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
    ).sort("event_id", "interval_id")

    result_arr = units_arr.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
    ).sort("event_id", "interval_id")

    assert result_list["event_times"].to_list() == result_arr["event_times"].to_list()


def test_join_between_array_col_as_counts(intervals):
    """as_counts=True works with array[f64, N] input."""
    units_arr = pl.DataFrame({
        "event_id": [0, 1],
        "event_times": [[0.1, 0.5, 1.2, 1.8, 2.3], [0.2, 0.9, 1.5, 2.1, 3.0]],
    }).with_columns(pl.col("event_times").cast(pl.Array(pl.Float64, 5)))

    result = units_arr.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
        as_counts=True,
    )
    assert result.schema["event_times"] == pl.UInt32


# ── Single-row edge cases ─────────────────────────────────────────────────────

def test_join_between_single_unit_single_interval():
    units = pl.DataFrame({"event_times": [[0.2, 0.4, 0.8]]})
    intervals = pl.DataFrame({"start_time": [0.0], "stop_time": [1.0]})
    result = units.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
    )
    assert len(result) == 1
    assert result["event_times"][0].to_list() == pytest.approx([0.2, 0.4, 0.8])


def test_join_between_empty_vec_list(intervals):
    """Unit with no spikes at all produces empty lists for every trial."""
    units_empty = pl.DataFrame({
        "event_id": [0],
        "event_times": [[]],
    })
    result = units_empty.vec.join_between(
        other=intervals,
        values="event_times",
        bounds=("start_time", "stop_time"),
    )
    assert len(result) == len(intervals)
    for row in result["event_times"]:
        assert row.to_list() == []
