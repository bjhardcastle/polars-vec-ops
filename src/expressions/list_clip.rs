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
        for i in 0..n {
            let start_opt = start_ca.get(i);
            let stop_opt = stop_ca.get(i);

            // Check for null in this row (list array outer null)
            let is_null = list_ca.null_count() > 0 && {
                use polars_arrow::array::Array;
                let chunk = &*list_ca.chunks()[0];
                chunk.validity().map_or(false, |v| !v.get_bit(i))
            };

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
