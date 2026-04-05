#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use super::helpers::ensure_list_type;

fn list_diff_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
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

#[polars_expr(output_type_func=list_diff_output_type)]
fn list_diff(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let input_dtype = series.dtype().clone();

    // Convert to List if it's an Array
    let series = ensure_list_type(series)?;
    let list_chunked = series.list()?;

    let n_lists = list_chunked.len();
    if n_lists == 0 {
        return Ok(series.slice(0, 0));
    }

    // Determine expected length and dtype from first non-null list
    let mut expected_len = 0;
    let mut inner_dtype = DataType::Null;

    for i in 0..n_lists {
        if let Some(s) = list_chunked.get_as_series(i) {
            expected_len = s.len();
            inner_dtype = s.dtype().clone();
            break;
        }
    }

    if inner_dtype == DataType::Null {
        // All rows are null
        return Ok(series.clone());
    }

    // Build result: first row is null, then compute differences
    let mut diff_chunks = Vec::with_capacity(n_lists);

    // First row is always null (no previous row to compare)
    // Create a null Series with the correct type and length, then wrap in list
    let null_series = Series::full_null("".into(), expected_len, &inner_dtype);
    diff_chunks.push(ListChunked::full(series.name().clone(), &null_series, 1));

    // Calculate differences for remaining rows
    for i in 1..n_lists {
        let curr_opt = list_chunked.get_as_series(i);
        let prev_opt = list_chunked.get_as_series(i - 1);

        match (prev_opt, curr_opt) {
            (Some(prev), Some(curr)) => {
                // Both non-null: validate lengths and compute diff
                if prev.len() != expected_len || curr.len() != expected_len {
                    polars_bail!(
                        ComputeError:
                        "All lists must have the same length for vertical diff. Expected {}",
                        expected_len
                    );
                }
                let diff = (&curr - &prev)?;
                let diff_casted = diff.cast(&inner_dtype)?;
                let diff_list = ListChunked::full(series.name().clone(), &diff_casted, 1);
                diff_chunks.push(diff_list);
            },
            _ => {
                // Either current or previous is null: result is null list
                let null_series = Series::full_null("".into(), expected_len, &inner_dtype);
                diff_chunks.push(ListChunked::full(series.name().clone(), &null_series, 1));
            },
        }
    }

    // Concatenate all chunks vertically
    let result_list = unsafe {
        ListChunked::from_chunks(
            series.name().clone(),
            diff_chunks
                .iter()
                .flat_map(|c| c.chunks())
                .cloned()
                .collect(),
        )
    };

    // Cast back to Array if input was Array
    let result_series = result_list.into_series();
    match &input_dtype {
        DataType::Array(_, width) => {
            result_series.cast(&DataType::Array(Box::new(inner_dtype), *width))
        },
        _ => Ok(result_series),
    }
}

