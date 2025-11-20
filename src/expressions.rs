#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn list_sum_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::List(inner) => Ok(Field::new(
            field.name().clone(),
            DataType::List(inner.clone()),
        )),
        _ => polars_bail!(InvalidOperation: "Expected List type, got {:?}", field.dtype()),
    }
}

#[polars_expr(output_type_func=list_sum_output_type)]
fn list_sum(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let list_chunked = series.list()?;

    let n_lists = list_chunked.len();
    if n_lists == 0 {
        return Ok(series.slice(0, 0));
    }

    // Get first list to determine length and type
    let first_series = list_chunked
        .get_as_series(0)
        .ok_or_else(|| polars_err!(ComputeError: "No data in list column"))?;
    let expected_len = first_series.len();
    let original_dtype = first_series.dtype().clone();

    // Collect all series references and validate
    let mut all_series = Vec::with_capacity(n_lists);
    all_series.push(first_series);

    for i in 1..n_lists {
        let s = list_chunked
            .get_as_series(i)
            .ok_or_else(|| polars_err!(ComputeError: "Null value in list column at index {}", i))?;
        if s.len() != expected_len {
            polars_bail!(
                ComputeError:
                "All lists must have the same length for vertical sum. Expected {}, got {}",
                expected_len, s.len()
            );
        }
        all_series.push(s);
    }

    // Sum all series
    let mut result = all_series[0].clone();
    for s in all_series.iter().skip(1) {
        result = (&result + s)?;
    }

    // Cast back to original dtype to preserve integer types
    result = result.cast(&original_dtype)?;

    // Wrap in a single-row list
    let result_list = ListChunked::full(series.name().clone(), &result, 1);

    Ok(result_list.into_series())
}

fn list_mean_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::List(inner) => {
            // Mean always returns Float64
            let float_inner = Box::new(DataType::Float64);
            Ok(Field::new(
                field.name().clone(),
                DataType::List(float_inner),
            ))
        },
        _ => polars_bail!(InvalidOperation: "Expected List type, got {:?}", field.dtype()),
    }
}

#[polars_expr(output_type_func=list_mean_output_type)]
fn list_mean(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let list_chunked = series.list()?;

    let n_lists = list_chunked.len();
    if n_lists == 0 {
        return Ok(series.slice(0, 0));
    }

    // Get first list to determine length
    let first_series = list_chunked
        .get_as_series(0)
        .ok_or_else(|| polars_err!(ComputeError: "No data in list column"))?;
    let expected_len = first_series.len();

    // Collect all series references and validate
    let mut all_series = Vec::with_capacity(n_lists);
    all_series.push(first_series);

    for i in 1..n_lists {
        let s = list_chunked
            .get_as_series(i)
            .ok_or_else(|| polars_err!(ComputeError: "Null value in list column at index {}", i))?;
        if s.len() != expected_len {
            polars_bail!(
                ComputeError:
                "All lists must have the same length for vertical mean. Expected {}, got {}",
                expected_len, s.len()
            );
        }
        all_series.push(s);
    }

    // Sum all series, then divide by count
    let mut result = all_series[0].cast(&DataType::Float64)?;
    for s in all_series.iter().skip(1) {
        let s_float = s.cast(&DataType::Float64)?;
        result = (&result + &s_float)?;
    }

    // Divide by number of lists to get mean
    let n_lists_f64 = n_lists as f64;
    result = result.divide(&Series::new("".into(), &[n_lists_f64]))?;

    // Wrap in a single-row list
    let result_list = ListChunked::full(series.name().clone(), &result, 1);

    Ok(result_list.into_series())
}

fn list_min_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::List(inner) => Ok(Field::new(
            field.name().clone(),
            DataType::List(inner.clone()),
        )),
        _ => polars_bail!(InvalidOperation: "Expected List type, got {:?}", field.dtype()),
    }
}

#[polars_expr(output_type_func=list_min_output_type)]
fn list_min(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let list_chunked = series.list()?;

    let n_lists = list_chunked.len();
    if n_lists == 0 {
        return Ok(series.slice(0, 0));
    }

    // Get first list to determine length and type
    let first_series = list_chunked
        .get_as_series(0)
        .ok_or_else(|| polars_err!(ComputeError: "No data in list column"))?;
    let expected_len = first_series.len();
    let original_dtype = first_series.dtype().clone();

    // Collect all series references and validate
    let mut all_series = Vec::with_capacity(n_lists);
    all_series.push(first_series);

    for i in 1..n_lists {
        let s = list_chunked
            .get_as_series(i)
            .ok_or_else(|| polars_err!(ComputeError: "Null value in list column at index {}", i))?;
        if s.len() != expected_len {
            polars_bail!(
                ComputeError:
                "All lists must have the same length for vertical min. Expected {}, got {}",
                expected_len, s.len()
            );
        }
        all_series.push(s);
    }

    // Calculate element-wise minimum
    let mut result = all_series[0].clone();
    for s in all_series.iter().skip(1) {
        // Compare element by element: where result > s, take s (smaller value)
        let mask = result.gt(s)?;
        result = s.zip_with(&mask, &result)?;
    }

    // Cast back to original dtype to preserve type
    result = result.cast(&original_dtype)?;

    // Wrap in a single-row list
    let result_list = ListChunked::full(series.name().clone(), &result, 1);

    Ok(result_list.into_series())
}

fn list_max_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::List(inner) => Ok(Field::new(
            field.name().clone(),
            DataType::List(inner.clone()),
        )),
        _ => polars_bail!(InvalidOperation: "Expected List type, got {:?}", field.dtype()),
    }
}

#[polars_expr(output_type_func=list_max_output_type)]
fn list_max(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let list_chunked = series.list()?;

    let n_lists = list_chunked.len();
    if n_lists == 0 {
        return Ok(series.slice(0, 0));
    }

    // Get first list to determine length and type
    let first_series = list_chunked
        .get_as_series(0)
        .ok_or_else(|| polars_err!(ComputeError: "No data in list column"))?;
    let expected_len = first_series.len();
    let original_dtype = first_series.dtype().clone();

    // Collect all series references and validate
    let mut all_series = Vec::with_capacity(n_lists);
    all_series.push(first_series);

    for i in 1..n_lists {
        let s = list_chunked
            .get_as_series(i)
            .ok_or_else(|| polars_err!(ComputeError: "Null value in list column at index {}", i))?;
        if s.len() != expected_len {
            polars_bail!(
                ComputeError:
                "All lists must have the same length for vertical max. Expected {}, got {}",
                expected_len, s.len()
            );
        }
        all_series.push(s);
    }

    // Calculate element-wise maximum
    let mut result = all_series[0].clone();
    for s in all_series.iter().skip(1) {
        // Compare element by element: where result < s, take s (larger value)
        let mask = result.lt(s)?;
        result = s.zip_with(&mask, &result)?;
    }

    // Cast back to original dtype to preserve type
    result = result.cast(&original_dtype)?;

    // Wrap in a single-row list
    let result_list = ListChunked::full(series.name().clone(), &result, 1);

    Ok(result_list.into_series())
}

fn list_diff_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::List(inner) => Ok(Field::new(
            field.name().clone(),
            DataType::List(inner.clone()),
        )),
        _ => polars_bail!(InvalidOperation: "Expected List type, got {:?}", field.dtype()),
    }
}

#[polars_expr(output_type_func=list_diff_output_type)]
fn list_diff(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let list_chunked = series.list()?;

    let n_lists = list_chunked.len();
    if n_lists == 0 {
        return Ok(series.slice(0, 0));
    }

    // Determine expected length and dtype from first non-null list
    let mut expected_len = 0;
    let mut original_dtype = DataType::Null;
    
    for i in 0..n_lists {
        if let Some(s) = list_chunked.get_as_series(i) {
            expected_len = s.len();
            original_dtype = s.dtype().clone();
            break;
        }
    }
    
    if original_dtype == DataType::Null {
        // All rows are null
        return Ok(series.clone());
    }

    // Build result: first row is null, then compute differences
    let mut diff_chunks = Vec::with_capacity(n_lists);

    // First row is always null (no previous row to compare)
    // Create a null Series with the correct type and length, then wrap in list
    let null_series = Series::full_null("".into(), expected_len, &original_dtype);
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
                let diff_casted = diff.cast(&original_dtype)?;
                let diff_list = ListChunked::full(series.name().clone(), &diff_casted, 1);
                diff_chunks.push(diff_list);
            }
            _ => {
                // Either current or previous is null: result is null list
                let null_series = Series::full_null("".into(), expected_len, &original_dtype);
                diff_chunks.push(ListChunked::full(series.name().clone(), &null_series, 1));
            }
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

    Ok(result_list.into_series())
}
