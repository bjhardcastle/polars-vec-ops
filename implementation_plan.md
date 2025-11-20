# Polars Vec Ops - Comprehensive Implementation Plan

**Project**: Vertical List Operations for Polars
**Date**: November 19, 2025
**Status**: In Progress
**Last Updated**: November 20, 2025

## Overview

This document outlines the complete implementation plan for vertical list operations in `polars-vec-ops`. These operations work across rows of list columns, performing element-wise operations at each position.

**Current Status**: ✅ High priority operations implemented (sum, mean, min, max, diff) - all tests passing

---

## 1. Arithmetic Operations

Operations that perform element-wise arithmetic across lists.

### 1.1 Basic Arithmetic
- [x] **sum** - Sum elements at each position
  - Status: ✅ Implemented
  - Function: `list_sum`
  - Test: ✅ Passing

## 2. Aggregation Operations

Statistical aggregations across lists.

### 2.1 Central Tendency
- [x] **mean** / **avg** - Mean of elements at each position
  - Status: ✅ Implemented
  - Function: `list_mean`
  - Signature: `vec_ops.mean()` or `vec_ops.avg()`
  - Example: `[[1, 2, 3], [3, 4, 5]]` → `[2.0, 3.0, 4.0]`
  - Test: ✅ Passing

### 2.1b Differences
- [x] **diff** - Difference between elements at each position in consecutive rows
  - Status: ✅ Implemented
  - Function: `list_diff`
  - Signature: `vec_ops.diff()`
  - Example: `[[5, 10, 15], [2, 15, 5], [0, 0, 0]]` → `[[null, null, null], [-3, 5, -10], [-2, -15, -5]]`
  - Note: Returns same number of rows; first row is list of nulls (consistent with scalar diff behavior)
  - Test: ✅ Passing 
  
- [ ] **median** - Median of elements at each position
  - Function: `list_median`
  - Signature: `vec_ops.median()`
  - Example: `[[1, 2, 3], [3, 4, 5], [2, 3, 4]]` → `[2.0, 3.0, 4.0]`
  
- [ ] **mode** - Most frequent value at each position
  - Function: `list_mode`
  - Signature: `vec_ops.mode()`
  - Note: Handle ties appropriately

### 2.2 Dispersion
- [ ] **std** / **std_dev** - Standard deviation at each position
  - Function: `list_std`
  - Signature: `vec_ops.std(ddof=1)` or `vec_ops.std_dev(ddof=1)`
  - Parameter: `ddof` (delta degrees of freedom, default=1 for sample std)
  - Example: `[[1, 2, 3], [3, 4, 5]]` → `[1.414, 1.414, 1.414]`
  
- [ ] **var** / **variance** - Variance at each position
  - Function: `list_var`
  - Signature: `vec_ops.var(ddof=1)` or `vec_ops.variance(ddof=1)`
  - Parameter: `ddof` (delta degrees of freedom, default=1)
  - Example: `[[1, 2, 3], [3, 4, 5]]` → `[2.0, 2.0, 2.0]`
  
- [ ] **range** - Range (max - min) at each position
  - Function: `list_range`
  - Signature: `vec_ops.range()`
  - Example: `[[1, 2, 3], [5, 4, 1]]` → `[4, 2, 2]`

### 2.3 Extrema
- [x] **min** - Minimum element at each position
  - Status: ✅ Implemented
  - Function: `list_min`
  - Signature: `vec_ops.min()`
  - Example: `[[3, 5, 2], [1, 7, 4]]` → `[1, 5, 2]`
  - Test: ✅ Passing
  
- [x] **max** - Maximum element at each position
  - Status: ✅ Implemented
  - Function: `list_max`
  - Signature: `vec_ops.max()`
  - Example: `[[3, 5, 2], [1, 7, 4]]` → `[3, 7, 4]`
  - Test: ✅ Passing
  
- [ ] **argmin** - Index of minimum value at each position
  - Function: `list_argmin`
  - Signature: `vec_ops.argmin()`
  - Returns: List of indices (0-based)
  
- [ ] **argmax** - Index of maximum value at each position
  - Function: `list_argmax`
  - Signature: `vec_ops.argmax()`
  - Returns: List of indices (0-based)

### 2.4 Quantiles
- [ ] **quantile** - Quantile at each position
  - Function: `list_quantile`
  - Signature: `vec_ops.quantile(q)` where `q` in [0, 1]
  - Example: `vec_ops.quantile(0.5)` for median
  
- [ ] **percentile** - Percentile at each position
  - Function: `list_percentile`
  - Signature: `vec_ops.percentile(p)` where `p` in [0, 100]
  - Example: `vec_ops.percentile(95)` for 95th percentile

### 2.5 Other Aggregations
- [ ] **prod** / **product** - Product of elements at each position
  - Function: `list_prod`
  - Signature: `vec_ops.prod()` or `vec_ops.product()`
  - Example: `[[2, 3, 4], [5, 6, 7]]` → `[10, 18, 28]`
  
- [ ] **count** - Count of non-null elements at each position
  - Function: `list_count`
  - Signature: `vec_ops.count()`
  - Useful when lists contain nulls
  
- [ ] **any** - Logical OR at each position
  - Function: `list_any`
  - Signature: `vec_ops.any()`
  - Works with boolean lists
  
- [ ] **all** - Logical AND at each position
  - Function: `list_all`
  - Signature: `vec_ops.all()`
  - Works with boolean lists

---

## 4. Mathematical Functions

Element-wise mathematical transformations applied vertically.


### 4.4 Rounding & Sign
- [ ] **abs** - Absolute value at each position
  - Function: `list_abs`
  - Signature: `vec_ops.abs()`
  
- [ ] **round** - Round to n decimal places at each position
  - Function: `list_round`
  - Signature: `vec_ops.round(decimals=0)`
  
- [ ] **floor** - Floor at each position
  - Function: `list_floor`
  - Signature: `vec_ops.floor()`
  
- [ ] **ceil** - Ceiling at each position
  - Function: `list_ceil`
  - Signature: `vec_ops.ceil()`
  
- [ ] **sign** - Sign (-1, 0, 1) at each position
  - Function: `list_sign`
  - Signature: `vec_ops.sign()`

---

## 7. Utility Operations

Helper operations for working with vertical lists.
  
- [ ] **clip** - Clip values at each position to range [min, max]
  - Function: `list_clip`
  - Signature: `vec_ops.clip(min_val, max_val)`

---

## Implementation Priority

### Phase 1: High Priority
✅ sum 
✅ mean
✅ min
✅ max
✅ diff - **Priority**: Essential for time-series analysis and change detection



---

## Technical Implementation Notes

### Rust Function Template

Each operation should follow this pattern:

```rust
fn list_<operation>_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::List(inner) => {
            // For operations returning same shape:
            Ok(Field::new(field.name().clone(), DataType::List(inner.clone())))
            
            // For scalar reductions (like norm, dot):
            // Ok(Field::new(field.name().clone(), inner.as_ref().clone()))
        }
        _ => polars_bail!(InvalidOperation: "Expected List type, got {:?}", field.dtype()),
    }
}

#[polars_expr(output_type_func=list_<operation>_output_type)]
fn list_<operation>(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let list_chunked = series.list()?;
    
    // Validate all lists have same length
    // Extract all series
    // Perform operation
    // Return result wrapped in appropriate type
}
```

### Python Function Template

Each operation should be added to `VecOpsNamespace`:

```python
def <operation>(self, **kwargs) -> pl.Expr:
    """
    <Operation description>
    
    Parameters
    ----------
    <parameters if any>
    
    Returns
    -------
    pl.Expr
        <return description>
    
    Examples
    --------
    >>> df = pl.DataFrame({"a": [[...], [...]]})
    >>> df.select(pl.col("a").vec_ops.<operation>())
    """
    return register_plugin_function(
        args=[self._expr],
        plugin_path=LIB,
        function_name="list_<operation>",
        is_elementwise=False,
        returns_scalar=True,  # or False depending on operation
        kwargs=kwargs,  # if operation has parameters
    )
```

### Test Template

Each operation should have comprehensive tests:

```python
def test_vec_<operation>():
    """Test basic functionality."""
    df = pl.DataFrame({"a": [[...], [...]]})
    result = df.select(pl.col("a").vec_ops.<operation>())
    assert result["a"][0].to_list() == [expected]

def test_vec_<operation>_edge_cases():
    """Test edge cases: single row, empty, nulls, etc."""
    pass

def test_vec_<operation>_type_preservation():
    """Test that appropriate types are preserved/converted."""
    pass

def test_vec_<operation>_error_handling():
    """Test error conditions: length mismatch, invalid inputs, etc."""
    pass
```

---

## Documentation Requirements

For each implemented operation:

1. **Docstring** in Python with clear examples
2. **README.md** update with operation description
3. **Type hints** in `_internal.pyi`
4. **Performance notes** where relevant
5. **Edge case documentation** (nulls, empty lists, single value, etc.)

---

## Testing Strategy

### Unit Tests
- Basic functionality with 2-3 lists
- Single row edge case
- Empty lists
- Type preservation (int vs float)
- Error handling (length mismatch, nulls)

### Integration Tests
- Combined with group_by operations
- Large datasets (performance testing)
- Mixed with other polars operations
- Chaining multiple vec_ops operations

### Property-Based Tests (Optional)
- Use hypothesis for random test generation
- Verify mathematical properties (associativity, commutativity where applicable)

---

## Performance Considerations

1. **Type Casting**: Minimize unnecessary type conversions
2. **Memory Allocation**: Pre-allocate result vectors when possible
3. **Parallelization**: Consider using Polars' parallel capabilities for independent operations
4. **Chunked Processing**: Handle large ChunkedArrays efficiently
5. **Null Handling**: Optimize null checking and propagation

---

## Related Polars Functions

Users might be interested in these native Polars functions:

- `list.eval()` - Apply expressions to list elements (horizontal)
- `list.get()` - Get element at index
- `list.sum()` - Sum within each list (horizontal, not vertical)
- `list.mean()` - Mean within each list (horizontal)
- `arr` namespace - Array operations (fixed-size lists)

The key difference: `vec_ops` works **across rows** (vertically), while native list operations work **within each list** (horizontally).

---

## Future Enhancements

1. **Weighted Operations**: Add weights parameter to mean, std, etc.
2. **Custom Kernels**: Allow users to define custom reduction functions
3. **Multi-column Support**: Operations across multiple list columns
4. **Sparse Support**: Optimize for sparse list data
5. **GPU Acceleration**: Leverage GPU for large-scale operations
6. **Parallel Group Operations**: Optimize group_by + vec_ops patterns

---

## Questions & Decisions

- [ ] Should operations handle nulls by propagating them or skipping them?
- [ ] Should we provide both camel_case and snake_case aliases (e.g., `stdDev` vs `std_dev`)?
- [ ] Should scalar reductions (norm, dot) return single values or 1-element lists?
- [ ] How to handle division by zero - return NaN, null, or error?
- [ ] Should we support broadcasting of different-length lists?

---

## Progress Tracking

| Category | Total | Implemented | Percentage |
|----------|-------|-------------|------------|
| Arithmetic | 7 | 1 | 14% |
| Aggregations | 20 | 5 | 25% |
| Linear Algebra | 11 | 0 | 0% |
| Mathematical | 23 | 0 | 0% |
| Comparison | 6 | 0 | 0% |
| Windowing | 5 | 0 | 0% |
| Utility | 5 | 0 | 0% |
| **TOTAL** | **77** | **5** | **6%** |

---

## Contact & Maintenance

- **Repository**: (add GitHub link)
- **Maintainer**: Ben Hardcastle
- **Last Updated**: November 19, 2025

---

## Appendix: Example Use Cases

### Use Case 1: Time Series Aggregation
```python
# Aggregate multiple time series vertically
df = pl.DataFrame({
    "sensor_id": [1, 1, 2, 2],
    "readings": [
        [10.1, 10.2, 10.3],
        [10.0, 10.1, 10.2],
        [20.1, 20.2, 20.3],
        [20.0, 20.1, 20.2],
    ]
})

# Get mean reading at each time point per sensor
df.group_by("sensor_id").agg(
    pl.col("readings").vec_ops.mean().alias("avg_readings")
)
```

### Use Case 2: Machine Learning Feature Aggregation
```python
# Aggregate feature vectors across samples
df = pl.DataFrame({
    "batch": [1, 1, 1],
    "features": [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
    ]
})

# Normalize features
df.select(
    pl.col("features")
      .vec_ops.mean().alias("mean_features"),
    pl.col("features")
      .vec_ops.std().alias("std_features")
)
```

### Use Case 3: Vector Operations
```python
# Calculate similarity between document vectors
df = pl.DataFrame({
    "doc_embeddings": [
        [0.5, 0.3, 0.2],
        [0.4, 0.4, 0.2],
    ]
})

df.select(
    pl.col("doc_embeddings").vec_ops.dot().alias("similarity"),
    pl.col("doc_embeddings").vec_ops.norm().alias("magnitude")
)
```

---

**END OF IMPLEMENTATION PLAN**
