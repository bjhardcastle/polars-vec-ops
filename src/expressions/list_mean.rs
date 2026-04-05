#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use super::helpers::ensure_list_type;

fn list_mean_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::List(_) => {
            // Mean always returns Float64
            let float_inner = Box::new(DataType::Float64);
            Ok(Field::new(
                field.name().clone(),
                DataType::List(float_inner),
            ))
        },
        DataType::Array(_, width) => {
            // Mean always returns Float64
            let float_inner = Box::new(DataType::Float64);
            Ok(Field::new(
                field.name().clone(),
                DataType::Array(float_inner, *width),
            ))
        },
        _ => polars_bail!(InvalidOperation: "Expected List or Array type, got {:?}", field.dtype()),
    }
}

#[polars_expr(output_type_func=list_mean_output_type)]
fn list_mean(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let input_dtype = series.dtype().clone();

    // Convert to List if it's an Array
    let series = ensure_list_type(series)?;
    let list_chunked = series.list()?;

    let n_lists = list_chunked.len();
    if n_lists == 0 {
        return Ok(series.slice(0, 0));
    }

    // Find first non-null list to determine length
    let mut expected_len = 0;
    let mut found_valid = false;

    for i in 0..n_lists {
        if let Some(s) = list_chunked.get_as_series(i) {
            expected_len = s.len();
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
                    "All lists must have the same length for vertical mean. Expected {}, got {}",
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

    // Sum all series (nulls treated as 0), then divide by count of non-nulls per position
    let mut sum_result = all_series[0]
        .cast(&DataType::Float64)?
        .fill_null(FillNullStrategy::Zero)?;
    let mut count_result = all_series[0].is_not_null().cast(&DataType::UInt32)?;

    for s in all_series.iter().skip(1) {
        let s_float = s
            .cast(&DataType::Float64)?
            .fill_null(FillNullStrategy::Zero)?;
        sum_result = (&sum_result + &s_float)?;

        let s_not_null = s.is_not_null().cast(&DataType::UInt32)?;
        count_result = (&count_result + &s_not_null)?;
    }

    // Divide sum by count to get mean (handle division by zero)
    let count_float = count_result.cast(&DataType::Float64)?;
    let result = sum_result.divide(&count_float)?;

    // Wrap in a single-row list
    let result_list = ListChunked::full(series.name().clone(), &result, 1);

    // Cast back to Array if input was Array
    let result_series = result_list.into_series();
    match &input_dtype {
        DataType::Array(_, width) => {
            result_series.cast(&DataType::Array(Box::new(DataType::Float64), *width))
        },
        _ => Ok(result_series),
    }
}

