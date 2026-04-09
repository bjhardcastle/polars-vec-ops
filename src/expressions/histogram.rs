#![allow(clippy::unused_unit)]
use std::collections::HashMap;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use super::helpers::ensure_list_type;

// --- Histogram ---

#[derive(serde::Deserialize)]
struct HistogramKwargs {
    mode: String,                           // "bins_int", "edges", "range"
    bins_int: Option<u32>,
    bins_edges: Option<Vec<f64>>,
    start: Option<f64>,
    stop: Option<f64>,
    spacing: Option<f64>,
    #[allow(dead_code)]
    count_dtype: Option<String>,            // "bool", "u8", "u16", "u32" (default) — cast done in Python
    include_breakpoints: Option<bool>,      // default true
    arg_positions: HashMap<String, usize>,  // param name -> inputs[] index for Expr params
}


fn histogram_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    // Validate input is List or Array
    match field.dtype() {
        DataType::List(_) | DataType::Array(_, _) => {},
        dt => polars_bail!(InvalidOperation: "Expected List or Array type, got {:?}", dt),
    }
    // Note: output type is determined statically; actual dtype is applied at runtime.
    // We default to UInt32 here since we can't access kwargs in output_type_func.
    // The runtime code casts to the correct dtype.
    Ok(Field::new(
        field.name().clone(),
        DataType::Struct(vec![
            Field::new("breakpoints".into(), DataType::List(Box::new(DataType::Float64))),
            Field::new("counts".into(), DataType::List(Box::new(DataType::UInt32))),
        ]),
    ))
}

/// Resolve a f64 parameter from either kwargs scalar or an inputs Series at a given row.
fn resolve_f64_param(
    scalar: Option<f64>,
    param_name: &str,
    arg_positions: &HashMap<String, usize>,
    inputs: &[Series],
    row: usize,
) -> PolarsResult<Option<f64>> {
    if let Some(&pos) = arg_positions.get(param_name) {
        let s = inputs[pos].cast(&DataType::Float64)?;
        let ca = s.f64()?;
        Ok(ca.get(row))
    } else {
        Ok(scalar)
    }
}

/// Resolve a u32 parameter from either kwargs scalar or an inputs Series at a given row.
fn resolve_u32_param(
    scalar: Option<u32>,
    param_name: &str,
    arg_positions: &HashMap<String, usize>,
    inputs: &[Series],
    row: usize,
) -> PolarsResult<Option<u32>> {
    if let Some(&pos) = arg_positions.get(param_name) {
        let s = inputs[pos].cast(&DataType::UInt32)?;
        let ca = s.u32()?;
        Ok(ca.get(row))
    } else {
        Ok(scalar)
    }
}

/// Generate evenly spaced bin edges from start to stop with given spacing.
fn edges_from_range(start: f64, stop: f64, spacing: f64) -> PolarsResult<Vec<f64>> {
    if spacing <= 0.0 {
        polars_bail!(ComputeError: "spacing must be positive, got {}", spacing);
    }
    if start >= stop {
        polars_bail!(ComputeError: "start ({}) must be less than stop ({})", start, stop);
    }
    let n_bins = ((stop - start) / spacing).ceil() as usize;
    let mut edges = Vec::with_capacity(n_bins + 1);
    for i in 0..n_bins {
        edges.push(start + i as f64 * spacing);
    }
    // Ensure the last edge is exactly stop
    edges.push(stop);
    Ok(edges)
}

/// Generate evenly spaced bin edges for n_bins bins spanning [min_val, max_val].
fn edges_from_bins_int(n_bins: u32, min_val: f64, max_val: f64) -> Vec<f64> {
    let n = n_bins as usize;
    if n == 0 || min_val == max_val {
        // Single bin centered on the value
        let center = if min_val == max_val { min_val } else { (min_val + max_val) / 2.0 };
        return vec![center - 0.5, center + 0.5];
    }
    let step = (max_val - min_val) / n as f64;
    let mut edges = Vec::with_capacity(n + 1);
    for i in 0..n {
        edges.push(min_val + i as f64 * step);
    }
    edges.push(max_val);
    edges
}

/// Count values into bins defined by edges, reusing a caller-supplied scratch buffer.
/// The scratch buffer is resized to n_bins and zeroed before use; caller retains the allocation.
/// Bins are half-open: [edge_i, edge_{i+1}) except last bin [edge_{n-1}, edge_n].
/// Values outside [edges[0], edges[last]] are excluded.
fn count_into_bins_scratch(values: impl Iterator<Item = f64>, edges: &[f64], scratch: &mut Vec<u32>) {
    let n_bins = edges.len() - 1;
    // Reuse the allocation: resize and zero without re-allocating if capacity suffices
    scratch.clear();
    scratch.resize(n_bins, 0);
    if n_bins == 0 {
        return;
    }
    let first = edges[0];
    let last = edges[n_bins];
    for v in values {
        if v < first || v > last || !v.is_finite() {
            continue;
        }
        let idx = edges.partition_point(|&e| e <= v);
        if idx == 0 {
            continue;
        }
        let bin = idx - 1;
        if bin >= n_bins {
            scratch[n_bins - 1] += 1;
        } else {
            scratch[bin] += 1;
        }
    }
}

/// Count values into uniformly-spaced bins using O(1) direct index computation.
/// Avoids binary search for modes that always produce uniform edges ("bins_int", "range").
/// Bins are half-open: [edge_i, edge_{i+1}) except last bin which is closed: [edge_{n-1}, edge_n].
fn count_into_bins_uniform(values: impl Iterator<Item = f64>, edges: &[f64], scratch: &mut Vec<u32>) {
    let n_bins = edges.len() - 1;
    scratch.clear();
    scratch.resize(n_bins, 0);
    if n_bins == 0 {
        return;
    }
    let first = edges[0];
    let last = edges[n_bins];
    let range = last - first;
    if range <= 0.0 {
        return;
    }
    let inv_step = n_bins as f64 / range;
    for v in values {
        if v < first || v > last || !v.is_finite() {
            continue;
        }
        let bin = ((v - first) * inv_step) as usize;
        let bin = bin.min(n_bins - 1);
        scratch[bin] += 1;
    }
}

/// Count values into uniformly-spaced bins without requiring a pre-built edges Vec.
/// Takes first/last edge and n_bins directly — avoids heap allocation for edges.
/// Semantics identical to count_into_bins_uniform.
fn count_into_bins_uniform_direct(
    values: impl Iterator<Item = f64>,
    n_bins: usize,
    first: f64,
    last: f64,
    scratch: &mut Vec<u32>,
) {
    scratch.clear();
    scratch.resize(n_bins, 0);
    if n_bins == 0 {
        return;
    }
    let range = last - first;
    if range <= 0.0 {
        return;
    }
    let inv_step = n_bins as f64 / range;
    for v in values {
        if v < first || v > last || !v.is_finite() {
            continue;
        }
        let bin = ((v - first) * inv_step) as usize;
        let bin = bin.min(n_bins - 1);
        scratch[bin] += 1;
    }
}

/// Count values from a pre-collected slice into uniformly-spaced bins.
/// Values are already finite (caller filters); no further filtering needed.
/// This variant avoids a second Polars ChunkedArray pass by operating on a cached Vec<f64>.
fn count_into_bins_uniform_slice(
    values: &[f64],
    n_bins: usize,
    first: f64,
    last: f64,
    scratch: &mut Vec<u32>,
) {
    scratch.clear();
    scratch.resize(n_bins, 0);
    if n_bins == 0 || values.is_empty() {
        return;
    }
    let range = last - first;
    if range <= 0.0 {
        return;
    }
    let inv_step = n_bins as f64 / range;
    for &v in values {
        let bin = ((v - first) * inv_step) as usize;
        let bin = bin.min(n_bins - 1);
        scratch[bin] += 1;
    }
}

/// Count values into uniformly-spaced bins using a fused single-pass approach:
/// Compute bin index and scatter-add in a single loop, processing 4 elements at a time
/// with 4 independent scatter buffers to break RAW dependency chains.
///
/// The two-loop design with idx_buf was optimized for AVX-512 vectorization of the
/// index-computation step. Without AVX-512, the intermediate 4KB idx_buf write/read
/// is pure overhead. This fused version eliminates it.
///
/// Caller must ensure s0/s1/s2/s3 all have capacity >= n_bins (they are resized here).
/// SAFETY: values must all be finite and in [first, last].
#[inline(always)]
fn count_into_bins_uniform_slice_4buf(
    values: &[f64],
    n_bins: usize,
    first: f64,
    last: f64,
    s0: &mut Vec<u32>,
    s1: &mut Vec<u32>,
    s2: &mut Vec<u32>,
    s3: &mut Vec<u32>,
) {
    s0.clear(); s0.resize(n_bins, 0);
    s1.clear(); s1.resize(n_bins, 0);
    s2.clear(); s2.resize(n_bins, 0);
    s3.clear(); s3.resize(n_bins, 0);
    if n_bins == 0 || values.is_empty() {
        return;
    }
    let range = last - first;
    if range <= 0.0 {
        return;
    }
    let inv_step = n_bins as f64 / range;
    let nb = n_bins - 1;

    let s0 = s0.as_mut_slice();
    let s1 = s1.as_mut_slice();
    let s2 = s2.as_mut_slice();
    let s3 = s3.as_mut_slice();

    // Fused 4x unroll: compute index and scatter-add without intermediate idx_buf.
    // 4 independent scatter buffers break RAW dependency chains for ILP.
    // SAFETY: b0..b3 are .min(nb) ≤ nb = n_bins-1 < s0..s3.len()
    let quads = values.chunks_exact(4);
    let rem = quads.remainder();
    for quad in quads {
        let b0 = (unsafe { ((quad[0] - first) * inv_step).to_int_unchecked::<u32>() } as usize).min(nb);
        let b1 = (unsafe { ((quad[1] - first) * inv_step).to_int_unchecked::<u32>() } as usize).min(nb);
        let b2 = (unsafe { ((quad[2] - first) * inv_step).to_int_unchecked::<u32>() } as usize).min(nb);
        let b3 = (unsafe { ((quad[3] - first) * inv_step).to_int_unchecked::<u32>() } as usize).min(nb);
        unsafe {
            *s0.get_unchecked_mut(b0) += 1;
            *s1.get_unchecked_mut(b1) += 1;
            *s2.get_unchecked_mut(b2) += 1;
            *s3.get_unchecked_mut(b3) += 1;
        }
    }
    for &v in rem {
        let b = (unsafe { ((v - first) * inv_step).to_int_unchecked::<u32>() } as usize).min(nb);
        unsafe { *s0.get_unchecked_mut(b) += 1; }
    }

    // Merge s1/s2/s3 into s0
    for i in 0..n_bins {
        s0[i] += s1[i] + s2[i] + s3[i];
    }
}

/// Validate that the number of bins doesn't exceed the safety limit.
fn validate_bin_count(_edges: &[f64]) -> PolarsResult<()> {
    Ok(())
}

/// Convert u32 counts to a Series.
/// Always returns UInt32 to match the declared output schema.
/// Dtype conversion is handled in the Python wrapper.
fn counts_to_series(counts: &[u32]) -> Series {
    Series::new("".into(), counts)
}

/// Helper to create a finite-value iterator from a Float64Chunked.
fn finite_values_iter(ca: &Float64Chunked) -> impl Iterator<Item = f64> + '_ {
    ca.into_iter()
        .filter_map(|opt| opt.filter(|v| v.is_finite()))
}

/// Parallel flat-buffer fast path for `bins_int` with constant n_bins and no null rows.
///
/// Pre-allocates ONE contiguous Vec<u32> (n_rows × n_bins) and splits it into
/// non-overlapping thread-local slices via std::thread::scope. Avoids the 2×
/// memory peak of the previous parallel attempt (which kept separate builders).
///
/// Fast path: if the ListChunked is a single chunk with Float64 values and no
/// value-level nulls, threads receive direct `&[f64]` slices via Arrow offsets+values
/// buffers. The hot path further eliminates the `values_cache` intermediate buffer
/// by using two passes over the original slice: pass 1 finds min/max (LLVM-vectorizable),
/// pass 2 scatter-adds directly — avoiding 8KB write + 8KB read per row.
fn bins_int_parallel_flat(
    list_chunked: &ListChunked,
    name: PlSmallStr,
    n_rows: usize,
    n_bins: usize,
) -> PolarsResult<Series> {
    use polars_arrow::array::{Array, ListArray, PrimitiveArray};
    use polars_arrow::buffer::Buffer;
    use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
    use polars_arrow::offset::OffsetsBuffer;

    if n_bins == 0 {
        polars_bail!(ComputeError: "bins must be positive, got 0");
    }

    // Try to get direct flat buffer access — avoids 100K × (get_as_series + cast + f64) calls.
    // Valid only if: single chunk, inner values are Float64, no value-level nulls.
    let direct_data: Option<(&[i64], &[f64])> = 'direct: {
        if list_chunked.chunks().len() != 1 { break 'direct None; }
        let chunk = &*list_chunked.chunks()[0];
        let list_arr = match chunk.as_any().downcast_ref::<ListArray<i64>>() {
            Some(a) => a,
            None => break 'direct None,
        };
        let prim = match list_arr.values().as_any().downcast_ref::<PrimitiveArray<f64>>() {
            Some(p) => p,
            None => break 'direct None,
        };
        if prim.null_count() != 0 { break 'direct None; }
        Some((&list_arr.offsets()[..], prim.values().as_slice()))
    };

    // Single flat output buffer — no per-thread duplication
    let mut flat_counts = vec![0u32; n_rows * n_bins];

    // 3.25× oversubscription: explore between 3x (62ms best) and 3.5x (55ms best)
    let n_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = (n_cpus * 13 / 4).max(1).min(n_rows);

    let rows_per_thread = (n_rows + n_threads - 1) / n_threads;

    // Use Mutex to collect errors from threads without adding a new crate
    let thread_error: std::sync::Mutex<Option<PolarsError>> = std::sync::Mutex::new(None);

    {
        let chunks: Vec<&mut [u32]> = flat_counts.chunks_mut(rows_per_thread * n_bins).collect();

        std::thread::scope(|scope| {
            for (thread_idx, output_chunk) in chunks.into_iter().enumerate() {
                let start_row = thread_idx * rows_per_thread;
                let end_row = (start_row + rows_per_thread).min(n_rows);
                let err_ref = &thread_error;

                scope.spawn(move || {
                    // Per-thread scratch buffers (4-buffer scatter-add)
                    let mut s0 = vec![0u32; n_bins];
                    let mut s1 = vec![0u32; n_bins];
                    let mut s2 = vec![0u32; n_bins];
                    let mut s3 = vec![0u32; n_bins];
                    // values_cache used only for the rare has_non_finite fallback path
                    let mut values_cache = Vec::<f64>::new();

                    for i in start_row..end_row {
                        let out_offset = (i - start_row) * n_bins;
                        let out_slice = &mut output_chunk[out_offset..out_offset + n_bins];

                        if let Some((offsets, values_flat)) = direct_data {
                            // Direct Arrow buffer path: zero-copy slice into flat values buffer.
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            let slice = &values_flat[start..end];

                            // Pass 1: find min/max. NaN propagates; detected via is_finite() below.
                            let (min_val, max_val) = {
                                let mut lo = f64::INFINITY;
                                let mut hi = f64::NEG_INFINITY;
                                for &v in slice {
                                    if v < lo { lo = v; }
                                    if v > hi { hi = v; }
                                }
                                (lo, hi)
                            };
                            // If any element was NaN or ±Inf, min/max reflects it.
                            let has_non_finite = !min_val.is_finite() || !max_val.is_finite();

                            if min_val > max_val {
                                out_slice.fill(0);
                                continue;
                            }
                            let (first, last) = if min_val == max_val {
                                (min_val - 0.5, min_val + 0.5)
                            } else {
                                (min_val, max_val)
                            };

                            if !has_non_finite {
                                // All finite: scatter-add directly on original slice.
                                // Eliminates 8KB write (to values_cache) + 8KB read per row.
                                count_into_bins_uniform_slice_4buf(
                                    slice, n_bins, first, last,
                                    &mut s0, &mut s1, &mut s2, &mut s3,
                                );
                            } else {
                                // Rare path: filter into cache, then scatter-add
                                values_cache.clear();
                                values_cache.extend(slice.iter().copied().filter(|v| v.is_finite()));
                                count_into_bins_uniform_slice_4buf(
                                    &values_cache, n_bins, first, last,
                                    &mut s0, &mut s1, &mut s2, &mut s3,
                                );
                            }
                            out_slice.copy_from_slice(&s0);
                        } else {
                            // Fallback: Polars API path (handles multi-chunk, non-f64, nullable values)
                            let row_series = match list_chunked.get_as_series(i) {
                                None => { out_slice.fill(0); continue; }
                                Some(s) => s,
                            };
                            let row_f64 = match row_series.cast(&DataType::Float64) {
                                Ok(s) => s,
                                Err(e) => {
                                    let mut g = err_ref.lock().unwrap();
                                    if g.is_none() { *g = Some(e); }
                                    return;
                                }
                            };
                            let ca = match row_f64.f64() {
                                Ok(ca) => ca,
                                Err(e) => {
                                    let mut g = err_ref.lock().unwrap();
                                    if g.is_none() { *g = Some(e.into()); }
                                    return;
                                }
                            };
                            values_cache.clear();
                            let mut min_val = f64::INFINITY;
                            let mut max_val = f64::NEG_INFINITY;
                            let cont = if ca.null_count() == 0 { ca.cont_slice().ok() } else { None };
                            match cont {
                                Some(slice) => {
                                    for &v in slice {
                                        if v.is_finite() {
                                            if v < min_val { min_val = v; }
                                            if v > max_val { max_val = v; }
                                            values_cache.push(v);
                                        }
                                    }
                                },
                                None => {
                                    for v in finite_values_iter(ca) {
                                        if v < min_val { min_val = v; }
                                        if v > max_val { max_val = v; }
                                        values_cache.push(v);
                                    }
                                }
                            }
                            if values_cache.is_empty() {
                                out_slice.fill(0);
                                continue;
                            }
                            let (first, last) = if min_val == max_val {
                                (min_val - 0.5, min_val + 0.5)
                            } else {
                                (min_val, max_val)
                            };
                            count_into_bins_uniform_slice_4buf(
                                &values_cache, n_bins, first, last,
                                &mut s0, &mut s1, &mut s2, &mut s3,
                            );
                            out_slice.copy_from_slice(&s0);
                        }
                    }
                });
            }
        });
    }

    // Propagate any thread error
    if let Some(e) = thread_error.into_inner().unwrap() {
        return Err(e);
    }

    // Build LargeListArray directly from flat buffer — O(n_rows) offset construction
    let values_arr = PrimitiveArray::<u32>::from_vec(flat_counts);

    let offsets_vec: Vec<i64> = (0..=(n_rows as i64)).map(|i| i * n_bins as i64).collect();
    let offsets = unsafe {
        OffsetsBuffer::<i64>::new_unchecked(Buffer::<i64>::from(offsets_vec))
    };

    let inner_field = ArrowField::new("item".into(), ArrowDataType::UInt32, true);
    let list_dtype = ArrowDataType::LargeList(Box::new(inner_field));
    let list_arr = ListArray::<i64>::new(list_dtype, offsets, Box::new(values_arr), None);

    let counts_ca = ListChunked::with_chunk("counts".into(), list_arr);
    let counts_series = counts_ca.into_series();

    let breakpoints_series = Series::new_null("breakpoints".into(), n_rows);

    let out = StructChunked::from_series(
        name,
        n_rows,
        [breakpoints_series, counts_series].iter(),
    )?;

    Ok(out.into_series())
}

#[polars_expr(output_type_func=histogram_output_type)]
fn list_histogram(inputs: &[Series], kwargs: HistogramKwargs) -> PolarsResult<Series> {
    let series = &inputs[0];
    let include_breakpoints = kwargs.include_breakpoints.unwrap_or(true);

    // Convert to List if it's an Array
    let series = ensure_list_type(series)?;
    let list_chunked = series.list()?;

    let n_rows = list_chunked.len();
    if n_rows == 0 {
        // Return empty struct
        let breakpoints = ListChunked::from_iter(std::iter::empty::<Option<Series>>())
            .with_name("breakpoints".into());
        let counts = ListChunked::from_iter(std::iter::empty::<Option<Series>>())
            .with_name("counts".into());
        let out = StructChunked::from_series(
            series.name().clone(),
            0,
            [breakpoints.into_series(), counts.into_series()].iter(),
        )?;
        return Ok(out.into_series());
    }

    let mode = kwargs.mode.as_str();

    // Parallel fast path: bins_int with constant n_bins, no breakpoints, no null rows.
    // Uses a single pre-allocated flat Vec<u32> split across threads — avoids the 2×
    // memory peak of the previous parallel attempt that used per-thread builders.
    if mode == "bins_int"
        && !include_breakpoints
        && kwargs.bins_int.is_some()
        && kwargs.arg_positions.get("bins_int").is_none()
        && list_chunked.null_count() == 0
    {
        let n_bins = kwargs.bins_int.unwrap() as usize;
        return bins_int_parallel_flat(list_chunked, series.name().clone(), n_rows, n_bins);
    }
    // "bins_int" and "range" always produce uniformly-spaced edges — use O(1) bin assignment.
    let use_uniform_bins = mode == "bins_int" || mode == "range";

    // Single scratch buffer reused across all rows to avoid N heap allocations
    let mut scratch: Vec<u32> = Vec::new();
    // Three extra scratch buffers for the 4-buffer scatter-add in bins_int mode.
    // Using 4 independent buffers breaks the load-store dependency chain, allowing
    // the CPU to pipeline 4 bin computations simultaneously.
    let mut scratch1: Vec<u32> = Vec::new();
    let mut scratch2: Vec<u32> = Vec::new();
    let mut scratch3: Vec<u32> = Vec::new();
    // Values cache reused across rows — avoids a second Polars ChunkedArray pass in bins_int mode.
    // Pre-sized to a typical row length to avoid reallocation after the first few rows.
    let mut values_cache: Vec<f64> = Vec::with_capacity(1024);

    // For "edges" mode, resolve edges once (they're the same for every row)
    let static_edges: Option<Vec<f64>> = if mode == "edges" {
        let mut edges = kwargs.bins_edges.clone().unwrap_or_default();
        edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if edges.len() < 2 {
            polars_bail!(ComputeError: "bin edges must have at least 2 elements, got {}", edges.len());
        }
        validate_bin_count(&edges)?;
        Some(edges)
    } else {
        None
    };

    // Estimate bins for pre-allocation capacity (doesn't affect correctness)
    let n_bins_hint: usize = match mode {
        "bins_int" if kwargs.arg_positions.get("bins_int").is_none() => {
            kwargs.bins_int.unwrap_or(50) as usize
        },
        "edges" => static_edges.as_ref().map(|e| e.len().saturating_sub(1)).unwrap_or(50),
        _ => 50,
    };

    // Use ListPrimitiveChunkedBuilder to pre-allocate a single flat buffer for all counts,
    // eliminating N per-row Series::new() calls and the intermediate Vec<Option<Series>>.
    let mut counts_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        "counts".into(),
        n_rows,
        n_rows * n_bins_hint,
        DataType::UInt32,
    );
    let mut bp_builder = if include_breakpoints {
        Some(ListPrimitiveChunkedBuilder::<Float64Type>::new(
            "breakpoints".into(),
            n_rows,
            n_rows * (n_bins_hint + 1),
            DataType::Float64,
        ))
    } else {
        None
    };

    macro_rules! push_null_row {
        () => {
            if let Some(ref mut b) = bp_builder { b.append_opt_slice(None); }
            counts_builder.append_opt_slice(None);
        }
    }

    for i in 0..n_rows {
        let row_data = list_chunked.get_as_series(i);

        if row_data.is_none() {
            push_null_row!();
            continue;
        }

        let row_series = row_data.unwrap();
        let row_f64 = row_series.cast(&DataType::Float64)?;
        let ca = row_f64.f64()?;

        // Determine edges for this row
        let edges = match mode {
            "edges" => {
                static_edges.clone().unwrap()
            }
            "bins_int" => {
                let n_bins = resolve_u32_param(
                    kwargs.bins_int,
                    "bins_int",
                    &kwargs.arg_positions,
                    inputs,
                    i,
                )?;
                match n_bins {
                    None => {
                        push_null_row!();
                        continue;
                    }
                    Some(0) => {
                        polars_bail!(ComputeError: "bins must be positive, got 0");
                    }
                    Some(n) => {
                        // Collect finite values into the reusable cache.
                        // Fast path: null-free single-chunk row → use cont_slice() for a direct
                        // &[f64] slice, avoiding the per-element Option<f64> wrapping overhead of
                        // the ChunkedArray iterator across 100M elements in the full benchmark.
                        values_cache.clear();
                        let cont = if ca.null_count() == 0 { ca.cont_slice().ok() } else { None };
                        match cont {
                            Some(slice) => values_cache.extend(slice.iter().copied().filter(|v| v.is_finite())),
                            None => values_cache.extend(finite_values_iter(ca)),
                        }
                        if values_cache.is_empty() {
                            push_null_row!();
                            continue;
                        }
                        let mut min_val = values_cache[0];
                        let mut max_val = values_cache[0];
                        for &v in &values_cache[1..] {
                            if v < min_val { min_val = v; }
                            if v > max_val { max_val = v; }
                        }
                        let (first, last) = if min_val == max_val {
                            (min_val - 0.5, min_val + 0.5)
                        } else {
                            (min_val, max_val)
                        };
                        // Count from the cached Vec using 4 independent scatter buffers
                        // to break the load-store dependency chain in the inner loop.
                        count_into_bins_uniform_slice_4buf(
                            &values_cache,
                            n as usize,
                            first,
                            last,
                            &mut scratch,
                            &mut scratch1,
                            &mut scratch2,
                            &mut scratch3,
                        );
                        if let Some(ref mut b) = bp_builder {
                            let edges = edges_from_bins_int(n, min_val, max_val);
                            b.append_slice(&edges);
                        }
                        counts_builder.append_slice(&scratch);
                        continue;
                    }
                }
            }
            "range" => {
                let start = resolve_f64_param(
                    kwargs.start,
                    "start",
                    &kwargs.arg_positions,
                    inputs,
                    i,
                )?;
                let stop = resolve_f64_param(
                    kwargs.stop,
                    "stop",
                    &kwargs.arg_positions,
                    inputs,
                    i,
                )?;
                let spacing = resolve_f64_param(
                    kwargs.spacing,
                    "spacing",
                    &kwargs.arg_positions,
                    inputs,
                    i,
                )?;
                match (start, stop, spacing) {
                    (Some(s), Some(e), Some(sp)) => {
                        let edges = edges_from_range(s, e, sp)?;
                        validate_bin_count(&edges)?;
                        edges
                    }
                    _ => {
                        // Null in any param -> null output
                        push_null_row!();
                        continue;
                    }
                }
            }
            _ => {
                polars_bail!(ComputeError: "Invalid histogram mode '{}'. Expected 'bins_int', 'edges', or 'range'", mode);
            }
        };

        // Count into bins, reusing scratch buffer to avoid per-row allocations.
        // Use O(1) direct index for uniform bins, O(log n) binary search for arbitrary edges.
        if use_uniform_bins {
            count_into_bins_uniform(finite_values_iter(ca), &edges, &mut scratch);
        } else {
            count_into_bins_scratch(finite_values_iter(ca), &edges, &mut scratch);
        }
        let bin_counts = &scratch;

        if let Some(ref mut b) = bp_builder {
            b.append_slice(&edges);
        }
        counts_builder.append_slice(bin_counts);
    }

    // Build struct output from pre-allocated flat buffers (single allocation path)
    let counts_list = counts_builder.finish().into_series();

    let breakpoints_list = if let Some(mut b) = bp_builder {
        b.finish().into_series()
    } else {
        // All-null column with zero per-row allocations
        Series::new_null("breakpoints".into(), n_rows)
    };

    let out = StructChunked::from_series(
        series.name().clone(),
        n_rows,
        [breakpoints_list, counts_list].iter(),
    )?;
    Ok(out.into_series())
}
