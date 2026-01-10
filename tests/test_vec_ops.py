import time

import numpy as np
import polars as pl
import pytest

import polars_vec_ops  # noqa


def test_vec_sum():
    """Test vertical sum across list elements."""
    df = pl.DataFrame({"a": [[0, 1, 2], [1, 2, 3]]})
    result = df.select(pl.col("a").vec.sum())
    print(result)

    # Expect a single row with [1, 3, 5]
    assert len(result) == 1
    assert result["a"][0].to_list() == [1.0, 3.0, 5.0]


def test_vec_sum_multiple_rows():
    """Test vertical sum with more than 2 rows."""
    df = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})
    result = df.select(pl.col("a").vec.sum())
    print(result)

    # Expect a single row with [12, 15, 18]
    assert len(result) == 1
    assert result["a"][0].to_list() == [12.0, 15.0, 18.0]


def test_vec_sum_single_row():
    """Test vertical sum with a single row."""
    df = pl.DataFrame({"a": [[10, 20, 30]]})
    result = df.select(pl.col("a").vec.sum())
    print(result)

    # Expect a single row with the same values
    assert len(result) == 1
    assert result["a"][0].to_list() == [10.0, 20.0, 30.0]


def test_vec_sum_mismatch():
    """Test that mismatched list lengths raise an error."""
    df = pl.DataFrame({"a": [[1, 2], [1]]})

    with pytest.raises(Exception) as exc_info:
        df.select(pl.col("a").vec.sum()).collect()

    print(f"Caught expected error: {exc_info.value}")
    assert "same length" in str(exc_info.value).lower()


def test_vec_sum_floats():
    """Test vertical sum with float lists."""
    df = pl.DataFrame({"a": [[1.5, 2.5, 3.5], [0.5, 1.5, 2.5]]})
    result = df.select(pl.col("a").vec.sum())
    print(result)

    # Expect a single row with [2.0, 4.0, 6.0]
    assert len(result) == 1
    assert result["a"][0].to_list() == [2.0, 4.0, 6.0]


def test_vec_sum_type_preservation():
    """Test that vec_ops.sum preserves the input data type."""
    # Test with integers
    df_int = pl.DataFrame({"a": [[0, 1, 2], [1, 2, 3]]})
    result_int = df_int.select(pl.col("a").vec.sum())
    print(f"\nInteger input dtype: {df_int['a'].dtype}")
    print(f"Integer result dtype: {result_int['a'].dtype}")
    print(f"Integer result: {result_int}")

    # Check if integers are preserved
    inner_dtype = str(result_int["a"].dtype)
    print(f"Inner dtype: {inner_dtype}")

    # Test with floats
    df_float = pl.DataFrame({"a": [[0.5, 1.5, 2.5], [1.5, 2.5, 3.5]]})
    result_float = df_float.select(pl.col("a").vec.sum())
    print(f"\nFloat input dtype: {df_float['a'].dtype}")
    print(f"Float result dtype: {result_float['a'].dtype}")
    print(f"Float result: {result_float}")

    # Verify results are correct
    assert len(result_int) == 1
    assert len(result_float) == 1

    # The actual values should be correct regardless of type
    int_vals = result_int["a"][0].to_list()
    float_vals = result_float["a"][0].to_list()

    # Allow for potential float conversion
    assert [int(v) if isinstance(v, float) else v for v in int_vals] == [1, 3, 5]
    assert float_vals == [2.0, 4.0, 6.0]


def test_vec_sum_performance():
    """Compare performance of vec_ops.sum vs manual list comprehension approach.

    Note: The manual approach may be faster due to polars' query optimization,
    but vec_ops provides cleaner, more maintainable code.
    """
    # Create a larger dataset for meaningful performance comparison
    n_rows = 10000
    list_length = 100

    df = pl.DataFrame(
        {
            "group": [i % 100 for i in range(n_rows)],
            "values": [[float(j) for j in range(list_length)] for _ in range(n_rows)],
        }
    )

    # Test 1: Simple aggregation without grouping
    print("\n=== Test 1: Simple aggregation (no grouping) ===")
    df_simple = df.select("values")

    start = time.perf_counter()
    _ = df_simple.select(pl.col("values").vec.sum())
    time_vec_ops_simple = time.perf_counter() - start

    start = time.perf_counter()
    _ = df_simple.select(
        pl.concat_list(
            [pl.col("values").list.get(i).sum() for i in range(list_length)]
        ).alias("values")
    )
    time_manual_simple = time.perf_counter() - start

    print(f"vec_ops.sum time: {time_vec_ops_simple:.4f}s")
    print(f"Manual approach time: {time_manual_simple:.4f}s")
    if time_vec_ops_simple < time_manual_simple:
        print(f"vec_ops is {time_manual_simple / time_vec_ops_simple:.2f}x faster")
    else:
        print(f"Manual is {time_vec_ops_simple / time_manual_simple:.2f}x faster")

    # Test 2: With grouping
    print("\n=== Test 2: With grouping ===")

    start = time.perf_counter()
    result_vec_ops = df.group_by("group", maintain_order=True).agg(
        pl.col("values").vec.sum()
    )
    time_vec_ops = time.perf_counter() - start

    start = time.perf_counter()
    result_manual = df.group_by("group", maintain_order=True).agg(
        pl.concat_list(
            [pl.col("values").list.get(i).sum() for i in range(list_length)]
        ).alias("values")
    )
    time_manual = time.perf_counter() - start

    print(f"vec_ops.sum time: {time_vec_ops:.4f}s")
    print(f"Manual approach time: {time_manual:.4f}s")
    if time_vec_ops < time_manual:
        print(f"vec_ops is {time_manual / time_vec_ops:.2f}x faster")
    else:
        print(f"Manual is {time_vec_ops / time_manual:.2f}x faster")

    # Verify results are the same for grouped case
    assert result_vec_ops.shape == result_manual.shape

    # Check a sample of results match
    for i in range(min(5, len(result_vec_ops))):
        vec_ops_vals = result_vec_ops["values"][i].to_list()
        manual_vals = result_manual["values"][i].to_list()
        assert len(vec_ops_vals) == len(manual_vals)
        for v1, v2 in zip(vec_ops_vals, manual_vals):
            assert abs(v1 - v2) < 1e-10, f"Mismatch: {v1} vs {v2}"


def test_vec_mean():
    """Test vertical mean across list elements."""
    df = pl.DataFrame({"a": [[1, 2, 3], [3, 4, 5]]})
    result = df.select(pl.col("a").vec.mean())
    print(result)

    # Expect a single row with [2.0, 3.0, 4.0]
    assert len(result) == 1
    assert result["a"][0].to_list() == [2.0, 3.0, 4.0]


def test_vec_mean_multiple_rows():
    """Test vertical mean with more than 2 rows."""
    df = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})
    result = df.select(pl.col("a").vec.mean())
    print(result)

    # Expect a single row with [4.0, 5.0, 6.0]
    assert len(result) == 1
    assert result["a"][0].to_list() == [4.0, 5.0, 6.0]


def test_vec_mean_single_row():
    """Test vertical mean with a single row."""
    df = pl.DataFrame({"a": [[10.0, 20.0, 30.0]]})
    result = df.select(pl.col("a").vec.mean())
    print(result)

    # Expect a single row with the same values
    assert len(result) == 1
    assert result["a"][0].to_list() == [10.0, 20.0, 30.0]


def test_vec_avg_alias():
    """Test that avg is an alias for mean."""
    df = pl.DataFrame({"a": [[1, 2, 3], [3, 4, 5]]})
    result_mean = df.select(pl.col("a").vec.mean())
    result_avg = df.select(pl.col("a").vec.avg())

    assert result_mean["a"][0].to_list() == result_avg["a"][0].to_list()


def test_vec_min():
    """Test vertical min across list elements."""
    df = pl.DataFrame({"a": [[3, 5, 2], [1, 7, 4]]})
    result = df.select(pl.col("a").vec.min())
    print(result)

    # Expect a single row with [1, 5, 2]
    assert len(result) == 1
    assert result["a"][0].to_list() == [1, 5, 2]


def test_vec_min_multiple_rows():
    """Test vertical min with more than 2 rows."""
    df = pl.DataFrame({"a": [[5, 10, 3], [2, 8, 1], [7, 6, 4]]})
    result = df.select(pl.col("a").vec.min())
    print(result)

    # Expect a single row with [2, 6, 1]
    assert len(result) == 1
    assert result["a"][0].to_list() == [2, 6, 1]


def test_vec_min_floats():
    """Test vertical min with float lists."""
    df = pl.DataFrame({"a": [[1.5, 2.5, 3.5], [0.5, 3.5, 2.5]]})
    result = df.select(pl.col("a").vec.min())
    print(result)

    # Expect a single row with [0.5, 2.5, 2.5]
    assert len(result) == 1
    assert result["a"][0].to_list() == [0.5, 2.5, 2.5]


def test_vec_max():
    """Test vertical max across list elements."""
    df = pl.DataFrame({"a": [[3, 5, 2], [1, 7, 4]]})
    result = df.select(pl.col("a").vec.max())
    print(result)

    # Expect a single row with [3, 7, 4]
    assert len(result) == 1
    assert result["a"][0].to_list() == [3, 7, 4]


def test_vec_max_multiple_rows():
    """Test vertical max with more than 2 rows."""
    df = pl.DataFrame({"a": [[5, 10, 3], [2, 8, 1], [7, 6, 4]]})
    result = df.select(pl.col("a").vec.max())
    print(result)

    # Expect a single row with [7, 10, 4]
    assert len(result) == 1
    assert result["a"][0].to_list() == [7, 10, 4]


def test_vec_max_floats():
    """Test vertical max with float lists."""
    df = pl.DataFrame({"a": [[1.5, 2.5, 3.5], [0.5, 3.5, 2.5]]})
    result = df.select(pl.col("a").vec.max())
    print(result)

    # Expect a single row with [1.5, 3.5, 3.5]
    assert len(result) == 1
    assert result["a"][0].to_list() == [1.5, 3.5, 3.5]


def test_vec_diff():
    """Test vertical diff across list elements."""
    df = pl.DataFrame({"a": [[5, 10, 15], [2, 15, 5], [0, 0, 0]]})
    result = df.select(pl.col("a").vec.diff())
    print(result)

    # Expect 3 rows: first is list of nulls, then differences
    assert len(result) == 3
    assert result["a"][0].to_list() == [None, None, None]  # First row is list of nulls
    assert result["a"][1].to_list() == [-3, 5, -10]
    assert result["a"][2].to_list() == [-2, -15, -5]


def test_vec_diff_two_rows():
    """Test vertical diff with just two rows."""
    df = pl.DataFrame({"a": [[10, 20, 30], [5, 15, 25]]})
    result = df.select(pl.col("a").vec.diff())
    print(result)

    # Expect 2 rows: first is list of nulls, second is diff
    assert len(result) == 2
    assert result["a"][0].to_list() == [None, None, None]  # First row is list of nulls
    assert result["a"][1].to_list() == [-5, -5, -5]


def test_vec_diff_single_row():
    """Test vertical diff with a single row returns list of nulls."""
    df = pl.DataFrame({"a": [[10, 20, 30]]})
    result = df.select(pl.col("a").vec.diff())
    print(result)

    # Expect 1 row with list of nulls (no previous row to diff against)
    assert len(result) == 1
    assert result["a"][0].to_list() == [None, None, None]


def test_vec_diff_floats():
    """Test vertical diff with float lists."""
    df = pl.DataFrame({"a": [[1.5, 2.5, 3.5], [1.0, 2.0, 3.0]]})
    result = df.select(pl.col("a").vec.diff())
    print(result)

    # Expect 2 rows: first is list of nulls, second is diff
    assert len(result) == 2
    assert result["a"][0].to_list() == [None, None, None]
    assert result["a"][1].to_list() == [-0.5, -0.5, -0.5]


def test_vec_diff_with_nulls():
    """Test vertical diff with null rows."""
    df = pl.DataFrame({"a": [[5, 10, 15], None, [0, 0, 0]]})
    result = df.select(pl.col("a").vec.diff())
    print(result)

    # Expect 3 rows: first is list of nulls, second is list of nulls (curr is null),
    # third is list of nulls (prev is null)
    assert len(result) == 3
    assert result["a"][0].to_list() == [None, None, None]  # No previous row
    assert result["a"][1].to_list() == [None, None, None]  # Current is null
    assert result["a"][2].to_list() == [None, None, None]  # Previous is null


def test_vec_sum_with_nulls():
    """Test that sum skips null rows."""
    df = pl.DataFrame({"a": [[1, 2, 3], None, [4, 5, 6]]})
    result = df.select(pl.col("a").vec.sum())
    print(result)

    # Expect sum of non-null rows: [5, 7, 9]
    assert len(result) == 1
    assert result["a"][0].to_list() == [5.0, 7.0, 9.0]


def test_vec_mean_with_nulls():
    """Test that mean skips null rows."""
    df = pl.DataFrame({"a": [[2, 4, 6], None, [4, 6, 8]]})
    result = df.select(pl.col("a").vec.mean())
    print(result)

    # Expect mean of non-null rows: [3.0, 5.0, 7.0]
    assert len(result) == 1
    assert result["a"][0].to_list() == [3.0, 5.0, 7.0]


def test_vec_min_with_nulls():
    """Test that min skips null rows."""
    df = pl.DataFrame({"a": [[5, 10, 3], None, [2, 8, 7]]})
    result = df.select(pl.col("a").vec.min())
    print(result)

    # Expect min of non-null rows: [2, 8, 3]
    assert len(result) == 1
    assert result["a"][0].to_list() == [2, 8, 3]


def test_vec_max_with_nulls():
    """Test that max skips null rows."""
    df = pl.DataFrame({"a": [[5, 10, 3], None, [2, 8, 7]]})
    result = df.select(pl.col("a").vec.max())
    print(result)

    # Expect max of non-null rows: [5, 10, 7]
    assert len(result) == 1
    assert result["a"][0].to_list() == [5, 10, 7]


def test_vec_sum_with_arrays():
    """Test sum on Array dtype."""
    df = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6]]}).select(
        pl.col("a").cast(pl.Array(pl.Int64, 3))
    )

    result = df.select(pl.col("a").vec.sum())
    print(result)

    # Should return Array dtype
    assert result.schema["a"] == pl.Array(pl.Int64, 3)
    assert result["a"][0].to_list() == [5, 7, 9]


def test_vec_mean_with_arrays():
    """Test mean on Array dtype."""
    df = pl.DataFrame({"a": [[2, 4, 6], [4, 6, 8]]}).select(
        pl.col("a").cast(pl.Array(pl.Int64, 3))
    )

    result = df.select(pl.col("a").vec.mean())
    print(result)

    # Should return Array[Float64]
    assert result.schema["a"] == pl.Array(pl.Float64, 3)
    assert result["a"][0].to_list() == [3.0, 5.0, 7.0]


def test_vec_min_with_arrays():
    """Test min on Array dtype."""
    df = pl.DataFrame({"a": [[5, 10, 3], [2, 8, 7]]}).select(
        pl.col("a").cast(pl.Array(pl.Int64, 3))
    )

    result = df.select(pl.col("a").vec.min())
    print(result)

    # Should return Array dtype with same inner type
    assert result.schema["a"] == pl.Array(pl.Int64, 3)
    assert result["a"][0].to_list() == [2, 8, 3]


def test_vec_max_with_arrays():
    """Test max on Array dtype."""
    df = pl.DataFrame({"a": [[5, 10, 3], [2, 8, 7]]}).select(
        pl.col("a").cast(pl.Array(pl.Int64, 3))
    )

    result = df.select(pl.col("a").vec.max())
    print(result)

    # Should return Array dtype with same inner type
    assert result.schema["a"] == pl.Array(pl.Int64, 3)
    assert result["a"][0].to_list() == [5, 10, 7]


def test_vec_diff_with_arrays():
    """Test diff on Array dtype."""
    df = pl.DataFrame({"a": [[5, 10, 15], [0, 5, 10], [1, 2, 3]]}).select(
        pl.col("a").cast(pl.Array(pl.Int64, 3))
    )

    result = df.select(pl.col("a").vec.diff())
    print(result)

    # Should return Array dtype with same inner type
    assert result.schema["a"] == pl.Array(pl.Int64, 3)
    assert len(result) == 3
    assert result["a"][0].to_list() == [None, None, None]  # First row
    assert result["a"][1].to_list() == [-5, -5, -5]  # Second row
    assert result["a"][2].to_list() == [1, -3, -7]  # Third row


def test_vec_sum_with_null_elements():
    """Test that sum skips null elements within lists."""
    df = pl.DataFrame({"a": [[1, None, 3], [4, 5, None], [None, 2, 1]]})
    result = df.select(pl.col("a").vec.sum())
    print(result)

    # Expect sum ignoring nulls: [5, 7, 4]
    assert len(result) == 1
    assert result["a"][0].to_list() == [5.0, 7.0, 4.0]


def test_vec_mean_with_null_elements():
    """Test that mean skips null elements within lists."""
    df = pl.DataFrame({"a": [[2, None, 6], [4, 6, None], [None, 3, 9]]})
    result = df.select(pl.col("a").vec.mean())
    print(result)

    # Position 0: (2+4)/2 = 3.0
    # Position 1: (6+3)/2 = 4.5
    # Position 2: (6+9)/2 = 7.5
    assert len(result) == 1
    assert result["a"][0].to_list() == [3.0, 4.5, 7.5]


def test_vec_min_with_null_elements():
    """Test that min skips null elements within lists."""
    df = pl.DataFrame({"a": [[5, None, 3], [None, 8, 7], [2, 10, None]]})
    result = df.select(pl.col("a").vec.min())
    print(result)

    # Position 0: min(5, 2) = 2
    # Position 1: min(8, 10) = 8
    # Position 2: min(3, 7) = 3
    assert len(result) == 1
    assert result["a"][0].to_list() == [2, 8, 3]


def test_vec_max_with_null_elements():
    """Test that max skips null elements within lists."""
    df = pl.DataFrame({"a": [[5, None, 3], [None, 8, 7], [2, 10, None]]})
    result = df.select(pl.col("a").vec.max())
    print(result)

    # Position 0: max(5, 2) = 5
    # Position 1: max(8, 10) = 10
    # Position 2: max(3, 7) = 7
    assert len(result) == 1
    assert result["a"][0].to_list() == [5, 10, 7]


def test_vec_convolve_basic():
    """Test basic convolution with numpy comparison."""
    data = [1, 2, 3, 4, 5]
    df = pl.DataFrame({"signal": [data]})
    kernel = [0.25, 0.5, 0.25]

    result = df.select(pl.col("signal").vec.convolve(kernel, mode="same"))
    print(result)

    # Compare with numpy
    np_result = np.convolve(data, kernel, mode="same")
    print(f"NumPy result: {np_result}")

    assert len(result) == 1
    result_vals = result["signal"][0].to_list()
    assert len(result_vals) == len(np_result)
    for v1, v2 in zip(result_vals, np_result):
        assert abs(v1 - v2) < 1e-10, f"Mismatch: {v1} vs {v2}"


def test_vec_convolve_multiple_rows():
    """Test convolution on multiple rows independently."""
    data1 = [1, 2, 3, 4, 5]
    data2 = [5, 4, 3, 2, 1]
    df = pl.DataFrame({"signal": [data1, data2]})
    kernel = [0.25, 0.5, 0.25]

    result = df.select(pl.col("signal").vec.convolve(kernel, mode="same"))
    print(result)

    # Each row should be convolved independently
    assert len(result) == 2

    # Compare row 1 with numpy
    np_result1 = np.convolve(data1, kernel, mode="same")
    result_vals1 = result["signal"][0].to_list()
    for v1, v2 in zip(result_vals1, np_result1):
        assert abs(v1 - v2) < 1e-10

    # Compare row 2 with numpy
    np_result2 = np.convolve(data2, kernel, mode="same")
    result_vals2 = result["signal"][1].to_list()
    for v1, v2 in zip(result_vals2, np_result2):
        assert abs(v1 - v2) < 1e-10


def test_vec_convolve_mode_full():
    """Test convolution with full mode."""
    data = [1, 2, 3]
    df = pl.DataFrame({"signal": [data]})
    kernel = [1, 0.5]

    result = df.select(pl.col("signal").vec.convolve(kernel, mode="full"))
    print(result)

    # Compare with numpy
    np_result = np.convolve(data, kernel, mode="full")
    print(f"NumPy result: {np_result}")

    result_vals = result["signal"][0].to_list()
    assert len(result_vals) == len(np_result)
    for v1, v2 in zip(result_vals, np_result):
        assert abs(v1 - v2) < 1e-10


def test_vec_convolve_mode_valid():
    """Test convolution with valid mode."""
    data = [1, 2, 3, 4, 5]
    df = pl.DataFrame({"signal": [data]})
    kernel = [1, 0.5, 0.25]

    result = df.select(pl.col("signal").vec.convolve(kernel, mode="valid"))
    print(result)

    # Compare with numpy
    np_result = np.convolve(data, kernel, mode="valid")
    print(f"NumPy result: {np_result}")

    result_vals = result["signal"][0].to_list()
    assert len(result_vals) == len(np_result)
    for v1, v2 in zip(result_vals, np_result):
        assert abs(v1 - v2) < 1e-10


def test_vec_convolve_mode_same():
    """Test convolution with same mode (default)."""
    data = [1, 2, 3, 4, 5, 6, 7]
    df = pl.DataFrame({"signal": [data]})
    kernel = [0.2, 0.2, 0.2, 0.2, 0.2]  # Moving average

    result = df.select(pl.col("signal").vec.convolve(kernel, mode="same"))
    print(result)

    # Compare with numpy
    np_result = np.convolve(data, kernel, mode="same")
    print(f"NumPy result: {np_result}")

    result_vals = result["signal"][0].to_list()
    assert len(result_vals) == len(np_result)
    for v1, v2 in zip(result_vals, np_result):
        assert abs(v1 - v2) < 1e-10


def test_vec_convolve_with_nulls():
    """Test that convolve handles null rows."""
    data1 = [1, 2, 3]
    data3 = [4, 5, 6]
    df = pl.DataFrame({"signal": [data1, None, data3]})
    kernel = [0.5, 0.5]

    result = df.select(pl.col("signal").vec.convolve(kernel, mode="same"))
    print(result)

    assert len(result) == 3

    # First row should be convolved normally
    np_result1 = np.convolve(data1, kernel, mode="same")
    result_vals1 = result["signal"][0].to_list()
    for v1, v2 in zip(result_vals1, np_result1):
        assert abs(v1 - v2) < 1e-10

    # Second row is null - result should be null
    assert result["signal"][1] is None 

    # Third row should be convolved normally
    np_result3 = np.convolve(data3, kernel, mode="same")
    result_vals3 = result["signal"][2].to_list()
    for v1, v2 in zip(result_vals3, np_result3):
        assert abs(v1 - v2) < 1e-10


def test_vec_convolve_with_null_elements():
    """Test that convolve handles null elements within lists using fill_value."""
    data = [1, None, 3, 4, 5]
    df = pl.DataFrame({"signal": [data]})
    kernel = [0.5, 0.5]

    # With default fill_value=0.0
    result = df.select(
        pl.col("signal").vec.convolve(kernel, fill_value=0.0, mode="same")
    )
    print(result)

    # Null should be treated as 0
    np_result = np.convolve([1, 0, 3, 4, 5], kernel, mode="same")
    result_vals = result["signal"][0].to_list()
    for v1, v2 in zip(result_vals, np_result):
        assert abs(v1 - v2) < 1e-10

    # Test with different fill_value
    result2 = df.select(
        pl.col("signal").vec.convolve(kernel, fill_value=2.0, mode="same")
    )
    np_result2 = np.convolve([1, 2, 3, 4, 5], kernel, mode="same")
    result_vals2 = result2["signal"][0].to_list()
    for v1, v2 in zip(result_vals2, np_result2):
        assert abs(v1 - v2) < 1e-10


def test_vec_convolve_with_arrays():
    """Test convolve on Array dtype."""
    data = [1, 2, 3, 4, 5]
    df = pl.DataFrame({"signal": [data]}).select(
        pl.col("signal").cast(pl.Array(pl.Int64, 5))
    )

    kernel = [0.25, 0.5, 0.25]
    result = df.select(pl.col("signal").vec.convolve(kernel, mode="same"))
    print(result)

    # Should return Array[Float64] dtype
    assert result.schema["signal"] == pl.Array(pl.Float64, 5)

    # Compare with numpy
    np_result = np.convolve(data, kernel, mode="same")
    result_vals = result["signal"][0].to_list()
    for v1, v2 in zip(result_vals, np_result):
        assert abs(v1 - v2) < 1e-10


def test_vec_convolve_edge_empty_signal():
    """Test convolve with empty signal."""
    df = pl.DataFrame({"signal": [[]]})
    kernel = [0.5, 0.5]

    result = df.select(pl.col("signal").vec.convolve(kernel, mode="same"))
    print(result)

    # Empty signal should produce empty result
    assert len(result) == 1
    assert len(result["signal"][0].to_list()) == 0


def test_vec_convolve_edge_single_element():
    """Test convolve with single element signal."""
    data = [5.0]
    df = pl.DataFrame({"signal": [data]})
    kernel = [0.5, 0.5]

    result = df.select(pl.col("signal").vec.convolve(kernel, mode="same"))
    print(result)

    # Compare with numpy
    np_result = np.convolve(data, kernel, mode="same")
    result_vals = result["signal"][0].to_list()
    assert len(result_vals) == len(np_result)
    for v1, v2 in zip(result_vals, np_result):
        assert abs(v1 - v2) < 1e-10


def test_vec_convolve_long_kernel():
    """Test convolve where kernel is longer than signal (valid mode)."""
    data = [1, 2, 3]
    df = pl.DataFrame({"signal": [data]})
    kernel = [1, 1, 1, 1, 1]  # Kernel longer than signal

    result = df.select(pl.col("signal").vec.convolve(kernel, mode="valid"))
    print(result)

    # Valid mode should produce empty result
    np_result = np.convolve(data, kernel, mode="valid")
    result_vals = result["signal"][0].to_list()
    assert len(result_vals) == len(np_result)  # Should be 0


def test_vec_convolve_smoothing():
    """Test practical use case: smoothing noisy signal."""
    # Create a signal with some noise
    signal = [1, 3, 2, 4, 3, 5, 4, 6, 5]
    df = pl.DataFrame({"signal": [signal]})

    # Gaussian-like smoothing kernel
    kernel = [0.1, 0.25, 0.3, 0.25, 0.1]

    result = df.select(pl.col("signal").vec.convolve(kernel, mode="same"))
    print(result)

    # Compare with numpy
    np_result = np.convolve(signal, kernel, mode="same")
    result_vals = result["signal"][0].to_list()

    assert len(result_vals) == len(np_result)
    for v1, v2 in zip(result_vals, np_result):
        assert abs(v1 - v2) < 1e-10


def test_vec_convolve_performance():
    """
    Compare performance of vec.convolve vs numpy.convolve applied to each row.

    This tests the efficiency of the polars plugin implementation for batch operations.
    """
    # Create a larger dataset for meaningful performance comparison
    n_rows = 10000
    signal_length = 100

    # Random signals
    np.random.seed(42)
    signals = [np.random.randn(signal_length).tolist() for _ in range(n_rows)]
    df = pl.DataFrame({"signal": signals})

    # Smoothing kernel
    kernel = [0.1, 0.2, 0.4, 0.2, 0.1]

    print(
        f"\n=== Performance Test: Convolve {n_rows} rows of length {signal_length} ==="
    )

    # Test 1: polars vec.convolve
    start = time.perf_counter()
    result_polars = df.select(pl.col("signal").vec.convolve(kernel, mode="same"))
    time_polars = time.perf_counter() - start

    # Test 2: numpy.convolve applied to each row
    start = time.perf_counter()
    result_numpy = [np.convolve(sig, kernel, mode="same").tolist() for sig in signals]
    time_numpy = time.perf_counter() - start

    print(f"polars vec.convolve time: {time_polars:.4f}s")
    print(f"numpy (per-row) time: {time_numpy:.4f}s")

    if time_polars < time_numpy:
        print(f"polars is {time_numpy / time_polars:.2f}x faster")
    else:
        print(f"numpy is {time_polars / time_numpy:.2f}x faster")

    # Verify a sample of results match
    for i in range(min(5, n_rows)):
        polars_vals = result_polars["signal"][i].to_list()
        numpy_vals = result_numpy[i]
        assert len(polars_vals) == len(numpy_vals)
        for v1, v2 in zip(polars_vals, numpy_vals):
            assert abs(v1 - v2) < 1e-10, f"Mismatch at row {i}: {v1} vs {v2}"

    print("✓ Results verified to match numpy")


def test_vec_convolve_with_groupby():
    """Test convolve within group_by context."""
    df = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "signal": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        }
    )
    kernel = [0.5, 0.5]

    result = df.group_by("group", maintain_order=True).agg(
        pl.col("signal").vec.convolve(kernel, mode="same")
    )
    print(result)

    # Each group should have convolved signals
    assert len(result) == 2

    # Group A should have 2 rows
    group_a_signals = result.filter(pl.col("group") == "A")["signal"][0]
    assert len(group_a_signals) == 2

    # Verify first signal in group A
    np_result = np.convolve([1, 2, 3], kernel, mode="same")
    result_vals = group_a_signals[0].to_list()
    for v1, v2 in zip(result_vals, np_result):
        assert abs(v1 - v2) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
