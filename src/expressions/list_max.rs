#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use super::helpers::ensure_list_type;

fn list_max_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::List(inner) => Ok(Field::new(
            field.name().clone(),
            DataType::List(inner.clone()),
        )),
        DataType::Array(inner, width) => Ok(Field::new(
            field.name().clone(),
            DataType::Array(inner.clone(), *width),
        )),
        _ => polars_bail!(InvalidOperation: "Expected List or Array type, got {:?}", field.dtype()),
    }
}

#[polars_expr(output_type_func=list_max_output_type)]
fn list_max(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let input_dtype = series.dtype().clone();

    // Convert to List if it's an Array
    let series = ensure_list_type(series)?;
    let list_chunked = series.list()?;

    let n_lists = list_chunked.len();
    if n_lists == 0 {
        return Ok(series.slice(0, 0));
    }

    // Find first non-null list to determine length and type
    let mut expected_len = 0;
    let mut inner_dtype = DataType::Null;
    let mut found_valid = false;

    for i in 0..n_lists {
        if let Some(s) = list_chunked.get_as_series(i) {
            expected_len = s.len();
            inner_dtype = s.dtype().clone();
            found_valid = true;
            break;
        }
    }

    if !found_valid {
        // All rows are null
        return Ok(ListChunked::full_null(series.name().clone(), n_lists).into_series());
    }

    // Collect all non-null series references and validate
    let mut all_series = Vec::new();

    for i in 0..n_lists {
        if let Some(s) = list_chunked.get_as_series(i) {
            if s.len() != expected_len {
                polars_bail!(
                    ComputeError:
                    "All lists must have the same length for vertical max. Expected {}, got {}",
                    expected_len, s.len()
                );
            }
            all_series.push(s);
        }
        // Skip null rows
    }

    if all_series.is_empty() {
        return Ok(ListChunked::full_null(series.name().clone(), 1).into_series());
    }

    // Calculate element-wise maximum, ignoring nulls
    // For max with null handling: if result is null, take s; if s is null, keep result; otherwise take maximum
    let mut result = all_series[0].clone();
    for s in all_series.iter().skip(1) {
        let result_is_null = result.is_null();
        let both_not_null = result.is_not_null() & s.is_not_null();

        // Where both are not null, compare and take maximum
        let comparison_mask = result.lt(s)? & both_not_null;
        let take_s = &comparison_mask | &result_is_null;
        let take_s_not_s_null = take_s & s.is_not_null();

        result = s.zip_with(&take_s_not_s_null, &result)?;
    }

    // Cast back to original inner dtype to preserve type
    result = result.cast(&inner_dtype)?;

    // Wrap in a single-row list
    let result_list = ListChunked::full(series.name().clone(), &result, 1);

    // Cast back to Array if input was Array
    let result_series = result_list.into_series();
    match &input_dtype {
        DataType::Array(_, width) => {
            result_series.cast(&DataType::Array(Box::new(inner_dtype), *width))
        },
        _ => Ok(result_series),
    }
}

