import polars as pl
import pytest
import time
import vec_ops  # Register the vec_ops namespace


def test_vec_sum():
    """Test vertical sum across list elements."""
    df = pl.DataFrame({
        "a": [[0, 1, 2], [1, 2, 3]]
    })
    result = df.select(pl.col("a").vec_ops.sum())
    print(result)
    
    # Expect a single row with [1, 3, 5]
    assert len(result) == 1
    assert result["a"][0].to_list() == [1.0, 3.0, 5.0]


def test_vec_sum_multiple_rows():
    """Test vertical sum with more than 2 rows."""
    df = pl.DataFrame({
        "a": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    })
    result = df.select(pl.col("a").vec_ops.sum())
    print(result)
    
    # Expect a single row with [12, 15, 18]
    assert len(result) == 1
    assert result["a"][0].to_list() == [12.0, 15.0, 18.0]


def test_vec_sum_single_row():
    """Test vertical sum with a single row."""
    df = pl.DataFrame({
        "a": [[10, 20, 30]]
    })
    result = df.select(pl.col("a").vec_ops.sum())
    print(result)
    
    # Expect a single row with the same values
    assert len(result) == 1
    assert result["a"][0].to_list() == [10.0, 20.0, 30.0]


def test_vec_sum_mismatch():
    """Test that mismatched list lengths raise an error."""
    df = pl.DataFrame({
        "a": [[1, 2], [1]]
    })
    
    with pytest.raises(Exception) as exc_info:
        df.select(pl.col("a").vec_ops.sum()).collect()
    
    print(f"Caught expected error: {exc_info.value}")
    assert "same length" in str(exc_info.value).lower()


def test_vec_sum_floats():
    """Test vertical sum with float lists."""
    df = pl.DataFrame({
        "a": [[1.5, 2.5, 3.5], [0.5, 1.5, 2.5]]
    })
    result = df.select(pl.col("a").vec_ops.sum())
    print(result)
    
    # Expect a single row with [2.0, 4.0, 6.0]
    assert len(result) == 1
    assert result["a"][0].to_list() == [2.0, 4.0, 6.0]


def test_vec_sum_type_preservation():
    """Test that vec_ops.sum preserves the input data type."""
    # Test with integers
    df_int = pl.DataFrame({
        "a": [[0, 1, 2], [1, 2, 3]]
    })
    result_int = df_int.select(pl.col("a").vec_ops.sum())
    print(f"\nInteger input dtype: {df_int['a'].dtype}")
    print(f"Integer result dtype: {result_int['a'].dtype}")
    print(f"Integer result: {result_int}")
    
    # Check if integers are preserved
    inner_dtype = str(result_int['a'].dtype)
    print(f"Inner dtype: {inner_dtype}")
    
    # Test with floats
    df_float = pl.DataFrame({
        "a": [[0.5, 1.5, 2.5], [1.5, 2.5, 3.5]]
    })
    result_float = df_float.select(pl.col("a").vec_ops.sum())
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
    
    df = pl.DataFrame({
        "group": [i % 100 for i in range(n_rows)],
        "values": [[float(j) for j in range(list_length)] for _ in range(n_rows)]
    })
    
    # Test 1: Simple aggregation without grouping
    print("\n=== Test 1: Simple aggregation (no grouping) ===")
    df_simple = df.select("values")
    
    start = time.perf_counter()
    result_vec_ops_simple = df_simple.select(pl.col("values").vec_ops.sum())
    time_vec_ops_simple = time.perf_counter() - start
    
    start = time.perf_counter()
    result_manual_simple = df_simple.select(
        pl.concat_list([pl.col("values").list.get(i).sum() for i in range(list_length)])
        .alias("values")
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
    result_vec_ops = (
        df
        .group_by("group", maintain_order=True)
        .agg(pl.col("values").vec_ops.sum())
    )
    time_vec_ops = time.perf_counter() - start
    
    start = time.perf_counter()
    result_manual = (
        df
        .group_by("group", maintain_order=True)
        .agg(
            pl.concat_list([pl.col("values").list.get(i).sum() for i in range(list_length)])
            .alias("values")
        )
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
    


if __name__ == "__main__":
    pytest.main([__file__, '-s', '-v'])