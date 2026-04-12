#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use super::helpers::ensure_list_type;

#[derive(serde::Deserialize)]
struct ListClipKwargs {
    relative: bool,
    as_counts: bool,
}

fn list_clip_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(Field::new(
        field.name().clone(),
        DataType::List(Box::new(DataType::Float64)),
    ))
}

#[polars_expr(output_type_func=list_clip_output_type)]
fn list_clip(inputs: &[Series], kwargs: ListClipKwargs) -> PolarsResult<Series> {
    let values_series = ensure_list_type(&inputs[0])?;
    let start_series = inputs[1].cast(&DataType::Float64)?;
    let stop_series = inputs[2].cast(&DataType::Float64)?;

    let list_ca = values_series.list()?;
    let start_ca = start_series.f64()?;
    let stop_ca = stop_series.f64()?;

    let n = list_ca.len();

    // Try direct Arrow buffer access for the fast path
    // Valid when: single chunk, inner values are Float64, no value-level nulls
    let direct_data: Option<(&[i64], &[f64])> = 'direct: {
        use polars_arrow::array::{Array, ListArray, PrimitiveArray};
        if list_ca.chunks().len() != 1 { break 'direct None; }
        let chunk = &*list_ca.chunks()[0];
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

    // Estimate capacity: average 10 values per row
    let cap_hint = n * 10;
    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        values_series.name().clone(),
        n,
        cap_hint,
        DataType::Float64,
    );

    if let Some((offsets, values_flat)) = direct_data {
        // Fast path: direct Arrow buffer access, no per-row allocation
        // Check outer nullity once upfront
        let outer_validity: Option<&polars_arrow::bitmap::Bitmap> = {
            use polars_arrow::array::Array;
            let chunk = &*list_ca.chunks()[0];
            chunk.validity()
        };

        for i in 0..n {
            let start_opt = start_ca.get(i);
            let stop_opt = stop_ca.get(i);

            // Check for null in this row (list array outer null)
            let is_null = outer_validity.map_or(false, |v| !v.get_bit(i));

            match (start_opt, stop_opt) {
                (Some(start), Some(stop)) if !is_null => {
                    let row_start = offsets[i] as usize;
                    let row_end = offsets[i + 1] as usize;
                    let slice = &values_flat[row_start..row_end];

                    let lo = slice.partition_point(|&x| x < start);
                    let hi = slice.partition_point(|&x| x < stop);
                    let clipped = &slice[lo..hi];

                    if kwargs.relative {
                        let shifted: Vec<f64> = clipped.iter().map(|&v| v - start).collect();
                        builder.append_slice(&shifted);
                    } else {
                        builder.append_slice(clipped);
                    }
                }
                _ => {
                    builder.append_null();
                }
            }
        }
    } else {
        // Fallback path: use get_as_series (handles non-f64, multi-chunk, nullable values)
        for i in 0..n {
            let row_opt = list_ca.get_as_series(i);
            let start_opt = start_ca.get(i);
            let stop_opt = stop_ca.get(i);

            match (row_opt, start_opt, stop_opt) {
                (Some(row_series), Some(start), Some(stop)) => {
                    let float_series = row_series.cast(&DataType::Float64)?;
                    let float_ca = float_series.f64()?;

                    // Use cont_slice for null-free rows (faster)
                    if float_ca.null_count() == 0 {
                        if let Ok(slice) = float_ca.cont_slice() {
                            let lo = slice.partition_point(|&x| x < start);
                            let hi = slice.partition_point(|&x| x < stop);
                            let clipped = &slice[lo..hi];
                            if kwargs.relative {
                                let shifted: Vec<f64> = clipped.iter().map(|&v| v - start).collect();
                                builder.append_slice(&shifted);
                            } else {
                                builder.append_slice(clipped);
                            }
                            continue;
                        }
                    }

                    let vals: Vec<f64> = float_ca.into_no_null_iter().collect();
                    let lo = vals.partition_point(|&x| x < start);
                    let hi = vals.partition_point(|&x| x < stop);
                    let clipped = &vals[lo..hi];

                    if kwargs.relative {
                        let shifted: Vec<f64> = clipped.iter().map(|&v| v - start).collect();
                        builder.append_slice(&shifted);
                    } else {
                        builder.append_slice(clipped);
                    }
                }
                _ => {
                    builder.append_null();
                }
            }
        }
    }

    Ok(builder.finish().into_series())
}

// --- Cross-clip: full cross-product clip without cross-join ---
// Two variants:
//   cross_clip: receives starts/stops as kwargs Vec<f64> (small intervals)
//   cross_clip_series: receives starts/stops as additional Series inputs (larger intervals)

#[derive(serde::Deserialize)]
struct CrossClipKwargs {
    starts: Vec<f64>,
    stops: Vec<f64>,
    relative: bool,
    as_counts: bool,
    n_other_cols: usize,  // number of other df columns to repeat (not used in Rust, just for reference)
}

fn cross_clip_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(Field::new(
        field.name().clone(),
        DataType::List(Box::new(DataType::Float64)),
    ))
}

// cross_clip_series: receives values + starts + stops as 3 Series inputs
// Avoids kwargs serialization for large starts/stops arrays
#[derive(serde::Deserialize)]
struct CrossClipSeriesKwargs {
    relative: bool,
}

fn cross_clip_series_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(Field::new(
        field.name().clone(),
        DataType::List(Box::new(DataType::Float64)),
    ))
}

/// Cross-clip via Series inputs for starts/stops (avoids kwargs serialization overhead).
/// inputs[0] = values (List[f64], n_units rows)
/// inputs[1] = starts (Float64, n_intervals rows)
/// inputs[2] = stops (Float64, n_intervals rows)
/// Output: List[Float64] with n_units × n_intervals rows
#[polars_expr(output_type_func=cross_clip_series_output_type)]
fn cross_clip_series(inputs: &[Series], kwargs: CrossClipSeriesKwargs) -> PolarsResult<Series> {
    use rayon::prelude::*;
    use polars_arrow::array::{Array, ListArray, PrimitiveArray};

    let values_series = ensure_list_type(&inputs[0])?;
    let starts_f64 = inputs[1].cast(&DataType::Float64)?;
    let stops_f64 = inputs[2].cast(&DataType::Float64)?;

    let list_ca = values_series.list()?;
    let starts_ca = starts_f64.f64()?;
    let stops_ca = stops_f64.f64()?;

    let n_units = list_ca.len();
    let n_intervals = starts_ca.len();
    let n_out = n_units * n_intervals;
    let relative = kwargs.relative;

    // Extract starts/stops as slices for fast access
    let starts: Vec<f64> = starts_ca.into_no_null_iter().collect();
    let stops: Vec<f64> = stops_ca.into_no_null_iter().collect();

    // Try direct Arrow buffer access
    let direct_data: Option<(&[i64], &[f64], Option<&polars_arrow::bitmap::Bitmap>)> = 'direct: {
        if list_ca.chunks().len() != 1 { break 'direct None; }
        let chunk = &*list_ca.chunks()[0];
        let list_arr = match chunk.as_any().downcast_ref::<ListArray<i64>>() {
            Some(a) => a,
            None => break 'direct None,
        };
        let prim = match list_arr.values().as_any().downcast_ref::<PrimitiveArray<f64>>() {
            Some(p) => p,
            None => break 'direct None,
        };
        if prim.null_count() != 0 { break 'direct None; }
        Some((&list_arr.offsets()[..], prim.values().as_slice(), chunk.validity()))
    };

    if let Some((offsets, values_flat, outer_validity)) = direct_data {
        // Parallel over units; each unit thread does binary search + flat data collection.
        // Returns (flat_data: Vec<f64>, row_lengths: Vec<u32>) per unit.
        // Row lengths = [len(j=0), len(j=1), ..., len(j=n_intervals-1)] for this unit.
        let has_nulls = outer_validity.is_some();

        let unit_outputs: Vec<(Vec<f64>, Vec<u32>, bool)> = (0..n_units)
            .into_par_iter()
            .map(|u| {
                let is_null = outer_validity.map_or(false, |v| !v.get_bit(u));
                if is_null {
                    return (vec![], vec![u32::MAX; n_intervals], true);
                }
                let row_start = offsets[u] as usize;
                let row_end = offsets[u + 1] as usize;
                let unit_slice = &values_flat[row_start..row_end];

                // Estimate output capacity: assume ~10% of spikes in average window
                let avg_clip = unit_slice.len() / 10;
                let mut flat: Vec<f64> = Vec::with_capacity(avg_clip * n_intervals);
                let mut lens: Vec<u32> = Vec::with_capacity(n_intervals);

                if !relative {
                    for j in 0..n_intervals {
                        let start = starts[j];
                        let stop = stops[j];
                        let lo = unit_slice.partition_point(|&x| x < start);
                        let hi = unit_slice.partition_point(|&x| x < stop);
                        flat.extend_from_slice(&unit_slice[lo..hi]);
                        lens.push((hi - lo) as u32);
                    }
                } else {
                    for j in 0..n_intervals {
                        let start = starts[j];
                        let stop = stops[j];
                        let lo = unit_slice.partition_point(|&x| x < start);
                        let hi = unit_slice.partition_point(|&x| x < stop);
                        for &v in &unit_slice[lo..hi] {
                            flat.push(v - start);
                        }
                        lens.push((hi - lo) as u32);
                    }
                }
                (flat, lens, false)
            })
            .collect();

        // Build Arrow output from unit_outputs
        use polars_arrow::array::{ListArray, PrimitiveArray};
        use polars_arrow::buffer::Buffer;
        use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
        use polars_arrow::offset::OffsetsBuffer;
        use polars_arrow::bitmap::MutableBitmap;

        // Compute total flat values size
        let total_vals: usize = unit_outputs.iter().map(|(f, _, _)| f.len()).sum();
        let mut flat_values: Vec<f64> = Vec::with_capacity(total_vals);
        let mut out_offsets: Vec<i64> = Vec::with_capacity(n_out + 1);
        let mut validity_buf: Vec<bool> = if has_nulls { Vec::with_capacity(n_out) } else { Vec::new() };
        out_offsets.push(0);
        let mut cur_offset: i64 = 0;

        for (unit_flat, unit_lens, is_null_unit) in &unit_outputs {
            if *is_null_unit {
                // All n_intervals rows for this unit are null
                for _ in 0..n_intervals {
                    if has_nulls { validity_buf.push(false); }
                    out_offsets.push(cur_offset);
                }
            } else {
                if has_nulls {
                    for _ in 0..n_intervals { validity_buf.push(true); }
                }
                flat_values.extend_from_slice(unit_flat);
                for &len in unit_lens {
                    cur_offset += len as i64;
                    out_offsets.push(cur_offset);
                }
            }
        }

        let values_arr = PrimitiveArray::<f64>::from_vec(flat_values);
        let offsets_buf = unsafe { OffsetsBuffer::<i64>::new_unchecked(Buffer::from(out_offsets)) };
        let inner_field = ArrowField::new("item".into(), ArrowDataType::Float64, true);
        let list_dtype = ArrowDataType::LargeList(Box::new(inner_field));

        let validity = if has_nulls {
            let mut bm = MutableBitmap::with_capacity(n_out);
            for &v in &validity_buf { bm.push(v); }
            Some(bm.freeze())
        } else {
            None
        };

        let list_arr = ListArray::<i64>::new(list_dtype, offsets_buf, Box::new(values_arr), validity);
        let ca = ListChunked::with_chunk(values_series.name().clone(), list_arr);
        Ok(ca.into_series())
    } else {
        // Fallback
        let cap_hint = n_out * 5;
        let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
            values_series.name().clone(), n_out, cap_hint, DataType::Float64,
        );
        for u in 0..n_units {
            let row_opt = list_ca.get_as_series(u);
            match row_opt {
                None => { for _ in 0..n_intervals { builder.append_null(); } }
                Some(row_series) => {
                    let float_series = row_series.cast(&DataType::Float64)?;
                    let float_ca = float_series.f64()?;
                    let vals: Vec<f64> = float_ca.into_no_null_iter().collect();
                    for j in 0..n_intervals {
                        let start = starts[j];
                        let stop = stops[j];
                        let lo = vals.partition_point(|&x| x < start);
                        let hi = vals.partition_point(|&x| x < stop);
                        let clipped = &vals[lo..hi];
                        if relative {
                            builder.append_slice(&clipped.iter().map(|&v| v - start).collect::<Vec<_>>());
                        } else {
                            builder.append_slice(clipped);
                        }
                    }
                }
            }
        }
        Ok(builder.finish().into_series())
    }
}

/// Perform cross-product clip entirely in Rust using parallel processing.
/// Input: values column (n_units rows)
/// kwargs.starts/stops: n_intervals values
/// Output: List[Float64] with n_units * n_intervals rows
///   Row order: unit0×int0, unit0×int1, ..., unit0×intN, unit1×int0, ...
#[polars_expr(output_type_func=cross_clip_output_type)]
fn cross_clip(inputs: &[Series], kwargs: CrossClipKwargs) -> PolarsResult<Series> {
    use rayon::prelude::*;
    use polars_arrow::array::{Array, ListArray, PrimitiveArray};

    let values_series = ensure_list_type(&inputs[0])?;
    let list_ca = values_series.list()?;
    let n_units = list_ca.len();
    let n_intervals = kwargs.starts.len();
    let n_out = n_units * n_intervals;

    let starts = &kwargs.starts;
    let stops = &kwargs.stops;
    let relative = kwargs.relative;

    // Try direct Arrow buffer access
    let direct_data: Option<(&[i64], &[f64], Option<&polars_arrow::bitmap::Bitmap>)> = 'direct: {
        if list_ca.chunks().len() != 1 { break 'direct None; }
        let chunk = &*list_ca.chunks()[0];
        let list_arr = match chunk.as_any().downcast_ref::<ListArray<i64>>() {
            Some(a) => a,
            None => break 'direct None,
        };
        let prim = match list_arr.values().as_any().downcast_ref::<PrimitiveArray<f64>>() {
            Some(p) => p,
            None => break 'direct None,
        };
        if prim.null_count() != 0 { break 'direct None; }
        Some((&list_arr.offsets()[..], prim.values().as_slice(), chunk.validity()))
    };

    if let Some((offsets, values_flat, outer_validity)) = direct_data {
        // Parallel fast path: compute all (unit, interval) pairs in parallel.
        // Output ordering: unit0×all_intervals, unit1×all_intervals, ...
        // Flatten to a single Vec<Option<Vec<f64>>> with n_units * n_intervals entries.
        let all_results: Vec<Option<Vec<f64>>> = (0..n_out)
            .into_par_iter()
            .map(|idx| {
                let u = idx / n_intervals;
                let j = idx % n_intervals;
                let is_null = outer_validity.map_or(false, |v| !v.get_bit(u));
                if is_null {
                    return None;
                }
                let row_start = offsets[u] as usize;
                let row_end = offsets[u + 1] as usize;
                let unit_slice = &values_flat[row_start..row_end];
                let start = starts[j];
                let stop = stops[j];
                let lo = unit_slice.partition_point(|&x| x < start);
                let hi = unit_slice.partition_point(|&x| x < stop);
                let clipped = &unit_slice[lo..hi];
                if relative {
                    Some(clipped.iter().map(|&v| v - start).collect::<Vec<f64>>())
                } else {
                    Some(clipped.to_vec())
                }
            })
            .collect();

        // Build the output Series from parallel results
        let cap_hint = n_out * 5;
        let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
            values_series.name().clone(),
            n_out,
            cap_hint,
            DataType::Float64,
        );
        for row in all_results {
            match row {
                Some(slice) => builder.append_slice(&slice),
                None => builder.append_null(),
            }
        }
        Ok(builder.finish().into_series())
    } else {
        // Fallback: sequential get_as_series per unit
        let cap_hint = n_out * 5;
        let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
            values_series.name().clone(),
            n_out,
            cap_hint,
            DataType::Float64,
        );
        for u in 0..n_units {
            let row_opt = list_ca.get_as_series(u);
            match row_opt {
                None => {
                    for _ in 0..n_intervals {
                        builder.append_null();
                    }
                }
                Some(row_series) => {
                    let float_series = row_series.cast(&DataType::Float64)?;
                    let float_ca = float_series.f64()?;
                    let vals: Vec<f64> = if float_ca.null_count() == 0 {
                        if let Ok(slice) = float_ca.cont_slice() {
                            slice.to_vec()
                        } else {
                            float_ca.into_no_null_iter().collect()
                        }
                    } else {
                        float_ca.into_no_null_iter().collect()
                    };

                    for j in 0..n_intervals {
                        let start = starts[j];
                        let stop = stops[j];
                        let lo = vals.partition_point(|&x| x < start);
                        let hi = vals.partition_point(|&x| x < stop);
                        let clipped = &vals[lo..hi];
                        if relative {
                            let shifted: Vec<f64> = clipped.iter().map(|&v| v - start).collect();
                            builder.append_slice(&shifted);
                        } else {
                            builder.append_slice(clipped);
                        }
                    }
                }
            }
        }
        Ok(builder.finish().into_series())
    }
}
