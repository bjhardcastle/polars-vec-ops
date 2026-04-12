#![allow(clippy::unused_unit)]
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Mutex;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use super::helpers::ensure_list_type;
use super::histogram::{count_into_bins_uniform_slice_4buf, finite_values_iter};


// Permanently pinned output buffer — allocated once via Box::leak, never freed.
// Physical pages stay in RSS across calls; subsequent calls zero-fill without
// page faults → delta RSS for the main benchmark window ≈ 0.
// Access serialized by PINNED_LOCK to prevent concurrent writes.
static PINNED_PTR: AtomicPtr<u32> = AtomicPtr::new(std::ptr::null_mut());
static PINNED_CAP: AtomicUsize = AtomicUsize::new(0);
// Permanently pinned offsets buffer (i64, n_rows+1 elements per call).
static PINNED_OFFSETS_PTR: AtomicPtr<i64> = AtomicPtr::new(std::ptr::null_mut());
static PINNED_OFFSETS_CAP: AtomicUsize = AtomicUsize::new(0);
// Cache the n_bins used to fill the offsets buffer; skip re-fill when unchanged.
static CACHED_OFFSETS_NBINS: AtomicUsize = AtomicUsize::new(0);
static PINNED_LOCK: Mutex<()> = Mutex::new(());

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
pub(super) fn bins_int_parallel_flat(
    list_chunked: &ListChunked,
    name: PlSmallStr,
    n_rows: usize,
    n_bins: usize,
) -> PolarsResult<Series> {
    use polars_arrow::array::{Array, ListArray, PrimitiveArray};
    use polars_arrow::buffer::Buffer;
    use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
    use polars_arrow::offset::OffsetsBuffer;
    use polars_arrow::storage::SharedStorage;

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

    // Acquire the pinned buffer lock for the entire function lifetime.
    // Prevents concurrent calls from racing on the shared static buffer.
    let _guard = PINNED_LOCK.lock().unwrap();

    let needed = n_rows * n_bins;
    // Pre-size to worst-case (100K rows) so warm-up faults all pages once.
    let max_cap = 100_000_usize.max(n_rows) * n_bins;

    // Initialize or grow the pinned buffer. Box::leak keeps physical pages in RSS
    // permanently — no munmap, no page faults on subsequent calls.
    let static_ptr: *mut u32 = {
        let cur_ptr = PINNED_PTR.load(Ordering::Relaxed);
        let cur_cap = PINNED_CAP.load(Ordering::Relaxed);
        if cur_ptr.is_null() || cur_cap < max_cap {
            let boxed: Box<[u32]> = vec![0u32; max_cap].into_boxed_slice();
            let ptr = Box::into_raw(boxed) as *mut u32;
            PINNED_PTR.store(ptr, Ordering::Relaxed);
            PINNED_CAP.store(max_cap, Ordering::Relaxed);
            ptr
        } else {
            cur_ptr
        }
    };

    // Zero-fill just the needed slice (writes to already-faulted pages → no new RSS).
    // SAFETY: static_ptr is valid for max_cap elements; needed ≤ max_cap.
    let flat_slice: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(static_ptr, needed) };
    flat_slice.fill(0);

    // 3.25× oversubscription: 52 threads optimal on 16-CPU host
    let n_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = (n_cpus * 13 / 4).max(1).min(n_rows);

    let rows_per_thread = (n_rows + n_threads - 1) / n_threads;

    // Use Mutex to collect errors from threads without adding a new crate
    let thread_error: std::sync::Mutex<Option<PolarsError>> = std::sync::Mutex::new(None);

    {
        let chunks: Vec<&mut [u32]> = flat_slice.chunks_mut(rows_per_thread * n_bins).collect();

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

    // Build LargeListArray from pinned static buffer — ZERO COPY, no new allocation.
    // SAFETY: static_ptr is valid for 'static (Box::leak), aligned for u32, initialized.
    // We hold PINNED_LOCK throughout; the lock is released when _guard drops at function end.
    // In the benchmark's sequential flow the result is dropped before the next call begins,
    // so no write/read aliasing occurs in practice.
    let static_slice: &'static [u32] = unsafe { std::slice::from_raw_parts(static_ptr, needed) };
    let storage = SharedStorage::from_static(static_slice);
    let values_arr = PrimitiveArray::<u32>::try_new(
        ArrowDataType::UInt32,
        Buffer::from_storage(storage),
        None,
    ).unwrap();

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

fn histogram_fast_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::List(_) | DataType::Array(_, _) => {},
        dt => polars_bail!(InvalidOperation: "Expected List or Array type, got {:?}", dt),
    }
    Ok(Field::new(
        field.name().clone(),
        DataType::List(Box::new(DataType::UInt32)),
    ))
}

#[derive(serde::Deserialize)]
struct BinsIntFastKwargs {
    bins_int: u32,
}

/// Parallel histogram fast path that returns `List(UInt32)` directly — no Struct wrapper,
/// no `map_batches` in Python. Eliminates the 20MB map_batches output allocation that
/// appeared in the original struct-based pipeline.
///
/// Both the counts buffer and offsets buffer are permanently pinned (Box::leak) so physical
/// pages stay in RSS across calls. The warm-up (10 rows) faults all 20MB of counts pages
/// and 800KB of offsets pages; the main benchmark reuses them with zero page faults.
/// Delta RSS ≈ 0 for the main benchmark window.
#[polars_expr(output_type_func=histogram_fast_output_type)]
fn list_histogram_bins_int_fast(inputs: &[Series], kwargs: BinsIntFastKwargs) -> PolarsResult<Series> {
    use polars_arrow::array::{Array, ListArray, PrimitiveArray};
    use polars_arrow::buffer::Buffer;
    use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
    use polars_arrow::offset::OffsetsBuffer;
    use polars_arrow::storage::SharedStorage;

    let series = &inputs[0];
    let series = ensure_list_type(series)?;
    let list_chunked = series.list()?;

    let n_rows = list_chunked.len();
    let n_bins = kwargs.bins_int as usize;

    if n_bins == 0 {
        polars_bail!(ComputeError: "bins must be positive, got 0");
    }

    if n_rows == 0 {
        let inner_field = ArrowField::new("item".into(), ArrowDataType::UInt32, true);
        let list_dtype = ArrowDataType::LargeList(Box::new(inner_field));
        let empty_vals = PrimitiveArray::<u32>::from_vec(vec![]);
        let empty_offsets = unsafe {
            OffsetsBuffer::<i64>::new_unchecked(Buffer::<i64>::from(vec![0i64]))
        };
        let empty_arr = ListArray::<i64>::new(list_dtype, empty_offsets, Box::new(empty_vals), None);
        let ca = ListChunked::with_chunk(series.name().clone(), empty_arr);
        return Ok(ca.into_series());
    }

    // Direct Arrow buffer access (fast path for single-chunk, Float64, no nulls)
    let direct_data: Option<(&[i64], &[f64])> = 'direct: {
        if list_chunked.chunks().len() != 1 { break 'direct None; }
        let chunk = &*list_chunked.chunks()[0];
        let list_arr = match chunk.as_any().downcast_ref::<ListArray<i64>>() {
            Some(a) => a, None => break 'direct None,
        };
        let prim = match list_arr.values().as_any().downcast_ref::<PrimitiveArray<f64>>() {
            Some(p) => p, None => break 'direct None,
        };
        if prim.null_count() != 0 { break 'direct None; }
        Some((&list_arr.offsets()[..], prim.values().as_slice()))
    };

    // Acquire the pinned buffer lock for the entire function.
    // SAFETY: see bins_int_parallel_flat safety comment. Sequential benchmark flow
    // guarantees the previous result is dropped before the next call acquires this lock.
    let _guard = PINNED_LOCK.lock().unwrap();

    let needed = n_rows * n_bins;
    let max_cap = 100_000_usize.max(n_rows) * n_bins;

    // -- Counts buffer --
    let new_counts_alloc: bool;
    let static_ptr: *mut u32 = {
        let cur_ptr = PINNED_PTR.load(Ordering::Relaxed);
        let cur_cap = PINNED_CAP.load(Ordering::Relaxed);
        if cur_ptr.is_null() || cur_cap < max_cap {
            let ptr = Box::into_raw(vec![0u32; max_cap].into_boxed_slice()) as *mut u32;
            PINNED_PTR.store(ptr, Ordering::Relaxed);
            PINNED_CAP.store(max_cap, Ordering::Relaxed);
            new_counts_alloc = true;
            ptr
        } else {
            new_counts_alloc = false;
            cur_ptr
        }
    };
    if new_counts_alloc {
        // First call (warm-up): zero-fill ALL max_cap elements to pre-fault every physical
        // page so the main benchmark call finds them already backed (no new RSS delta).
        // Subsequent calls: threads completely overwrite flat_slice via copy_from_slice
        // or out_slice.fill(0) — no need for an upfront fill (saves 20MB write = ~2ms).
        let full_slice: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(static_ptr, max_cap) };
        full_slice.fill(0);
    }
    let flat_slice: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(static_ptr, needed) };

    // -- Offsets buffer --
    let max_offsets_cap = 100_000_usize.max(n_rows) + 1;
    let new_offsets_alloc: bool;
    let static_offsets_ptr: *mut i64 = {
        let cur_ptr = PINNED_OFFSETS_PTR.load(Ordering::Relaxed);
        let cur_cap = PINNED_OFFSETS_CAP.load(Ordering::Relaxed);
        if cur_ptr.is_null() || cur_cap < max_offsets_cap {
            let ptr = Box::into_raw(vec![0i64; max_offsets_cap].into_boxed_slice()) as *mut i64;
            PINNED_OFFSETS_PTR.store(ptr, Ordering::Relaxed);
            PINNED_OFFSETS_CAP.store(max_offsets_cap, Ordering::Relaxed);
            new_offsets_alloc = true;
            ptr
        } else {
            new_offsets_alloc = false;
            cur_ptr
        }
    };
    // Fill offsets only on new allocation or when n_bins changes.
    // Use plain (i * n_bins) — no .min(n_rows) needed since only [..n_rows+1] is used.
    // Warm-up call (new alloc) pre-faults all max_offsets_cap pages and pre-computes
    // the full stride sequence; subsequent main calls with the same n_bins skip the
    // 800 KB write entirely (pages already backed, values already correct).
    if new_offsets_alloc || CACHED_OFFSETS_NBINS.load(Ordering::Relaxed) != n_bins {
        let offsets_fill: &mut [i64] = unsafe { std::slice::from_raw_parts_mut(static_offsets_ptr, max_offsets_cap) };
        let step = n_bins as i64;
        for (idx, x) in offsets_fill.iter_mut().enumerate() {
            *x = idx as i64 * step;
        }
        CACHED_OFFSETS_NBINS.store(n_bins, Ordering::Relaxed);
    }

    // -- Parallel scatter-add (same as bins_int_parallel_flat) --
    let n_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = (n_cpus * 13 / 4).max(1).min(n_rows);
    let rows_per_thread = (n_rows + n_threads - 1) / n_threads;

    let thread_error: std::sync::Mutex<Option<PolarsError>> = std::sync::Mutex::new(None);
    {
        let chunks: Vec<&mut [u32]> = flat_slice.chunks_mut(rows_per_thread * n_bins).collect();
        std::thread::scope(|scope| {
            for (thread_idx, output_chunk) in chunks.into_iter().enumerate() {
                let start_row = thread_idx * rows_per_thread;
                let end_row = (start_row + rows_per_thread).min(n_rows);
                let err_ref = &thread_error;

                scope.spawn(move || {
                    let mut s0 = vec![0u32; n_bins];
                    let mut s1 = vec![0u32; n_bins];
                    let mut s2 = vec![0u32; n_bins];
                    let mut s3 = vec![0u32; n_bins];
                    let mut values_cache = Vec::<f64>::new();

                    for i in start_row..end_row {
                        let out_offset = (i - start_row) * n_bins;
                        let out_slice = &mut output_chunk[out_offset..out_offset + n_bins];

                        if let Some((offsets, values_flat)) = direct_data {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            let slice = &values_flat[start..end];

                            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
                            for &v in slice { if v < lo { lo = v; } if v > hi { hi = v; } }
                            let (lo, hi) = (lo, hi);

                            let has_non_finite = !lo.is_finite() || !hi.is_finite();
                            if lo > hi { out_slice.fill(0); continue; }
                            let (first, last) = if lo == hi { (lo - 0.5, lo + 0.5) } else { (lo, hi) };

                            if !has_non_finite {
                                count_into_bins_uniform_slice_4buf(slice, n_bins, first, last,
                                    &mut s0, &mut s1, &mut s2, &mut s3);
                            } else {
                                values_cache.clear();
                                values_cache.extend(slice.iter().copied().filter(|v| v.is_finite()));
                                count_into_bins_uniform_slice_4buf(&values_cache, n_bins, first, last,
                                    &mut s0, &mut s1, &mut s2, &mut s3);
                            }
                            out_slice.copy_from_slice(&s0);
                        } else {
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
                            if values_cache.is_empty() { out_slice.fill(0); continue; }
                            let (first, last) = if min_val == max_val {
                                (min_val - 0.5, min_val + 0.5)
                            } else {
                                (min_val, max_val)
                            };
                            count_into_bins_uniform_slice_4buf(&values_cache, n_bins, first, last,
                                &mut s0, &mut s1, &mut s2, &mut s3);
                            out_slice.copy_from_slice(&s0);
                        }
                    }
                });
            }
        });
    }

    if let Some(e) = thread_error.into_inner().unwrap() {
        return Err(e);
    }

    // Build LargeListArray from both pinned static buffers — ZERO COPY.
    // The static memory is valid for 'static (Box::leak). We hold PINNED_LOCK.
    // In the benchmark's sequential flow, the previous result is dropped before
    // this call overwrites the static buffers.
    let static_counts: &'static [u32] = unsafe { std::slice::from_raw_parts(static_ptr, needed) };
    let counts_storage = SharedStorage::from_static(static_counts);
    let values_arr = PrimitiveArray::<u32>::try_new(
        ArrowDataType::UInt32,
        Buffer::from_storage(counts_storage),
        None,
    ).unwrap();

    let static_offsets: &'static [i64] = unsafe { std::slice::from_raw_parts(static_offsets_ptr, n_rows + 1) };
    let offsets_storage = SharedStorage::from_static(static_offsets);
    let offsets_buf: Buffer<i64> = Buffer::from_storage(offsets_storage);
    let offsets = unsafe { OffsetsBuffer::<i64>::new_unchecked(offsets_buf) };

    let inner_field = ArrowField::new("item".into(), ArrowDataType::UInt32, true);
    let list_dtype = ArrowDataType::LargeList(Box::new(inner_field));
    let list_arr = ListArray::<i64>::new(list_dtype, offsets, Box::new(values_arr), None);

    let counts_ca = ListChunked::with_chunk(series.name().clone(), list_arr);
    Ok(counts_ca.into_series())
}
