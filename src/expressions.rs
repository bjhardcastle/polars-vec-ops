#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

// Helper function to convert Array to List if needed
fn ensure_list_type(series: &Series) -> PolarsResult<Series> {
    match series.dtype() {
        DataType::Array(inner, _width) => {
            // Convert Array to List
            let arr_chunked = series.array()?;
            let list_chunked = arr_chunked.cast(&DataType::List(inner.clone()))?;
            Ok(list_chunked)
        },
        DataType::List(_) => Ok(series.clone()),
        dt => polars_bail!(InvalidOperation: "Expected List or Array type, got {:?}", dt),
    }
}

fn list_sum_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
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

#[polars_expr(output_type_func=list_sum_output_type)]
fn list_sum(inputs: &[Series]) -> PolarsResult<Series> {
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
    
    for i in 0..n_lists {
        if let Some(s) = list_chunked.get_as_series(i) {
            expected_len = s.len();
            inner_dtype = s.dtype().clone();
            break;
        }
    }
    
    if expected_len == 0 {
        // All rows are null, return a null series
        return Ok(ListChunked::full_null(series.name().clone(), n_lists).into_series());
    }

    // Collect all non-null series references and validate
    let mut all_series = Vec::new();

    for i in 0..n_lists {
        if let Some(s) = list_chunked.get_as_series(i) {
            if s.len() != expected_len {
                polars_bail!(
                    ComputeError:
                    "All lists must have the same length for vertical sum. Expected {}, got {}",
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

    // Sum all series, treating nulls as 0 (ignoring them)
    let mut result = all_series[0].fill_null(FillNullStrategy::Zero)?;
    for s in all_series.iter().skip(1) {
        let s_filled = s.fill_null(FillNullStrategy::Zero)?;
        result = (&result + &s_filled)?;
    }

    // Cast back to original inner dtype to preserve integer types
    result = result.cast(&inner_dtype)?;

    // Wrap in a single-row list
    let result_list = ListChunked::full(series.name().clone(), &result, 1);

    // Cast back to Array if input was Array
    let result_series = result_list.into_series();
    match &input_dtype {
        DataType::Array(_, width) => result_series.cast(&DataType::Array(Box::new(inner_dtype), *width)),
        _ => Ok(result_series),
    }
}

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
    let mut sum_result = all_series[0].cast(&DataType::Float64)?.fill_null(FillNullStrategy::Zero)?;
    let mut count_result = all_series[0].is_not_null().cast(&DataType::UInt32)?;
    
    for s in all_series.iter().skip(1) {
        let s_float = s.cast(&DataType::Float64)?.fill_null(FillNullStrategy::Zero)?;
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
        DataType::Array(_, width) => result_series.cast(&DataType::Array(Box::new(DataType::Float64), *width)),
        _ => Ok(result_series),
    }
}

fn list_min_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
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

#[polars_expr(output_type_func=list_min_output_type)]
fn list_min(inputs: &[Series]) -> PolarsResult<Series> {
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
                    "All lists must have the same length for vertical min. Expected {}, got {}",
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

    // Calculate element-wise minimum, ignoring nulls
    // We use Series min_horizontal-like logic: for each position, take minimum of non-null values
    let mut result = all_series[0].clone();
    for s in all_series.iter().skip(1) {
        // For min with null handling: if result is null, take s; if s is null, keep result; otherwise take minimum
        let result_is_null = result.is_null();
        let both_not_null = result.is_not_null() & s.is_not_null();
        
        // Where both are not null, compare and take minimum
        let comparison_mask = result.gt(s)? & both_not_null;
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
        DataType::Array(_, width) => result_series.cast(&DataType::Array(Box::new(inner_dtype), *width)),
        _ => Ok(result_series),
    }
}

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
        DataType::Array(_, width) => result_series.cast(&DataType::Array(Box::new(inner_dtype), *width)),
        _ => Ok(result_series),
    }
}

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
            }
            _ => {
                // Either current or previous is null: result is null list
                let null_series = Series::full_null("".into(), expected_len, &inner_dtype);
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

    // Cast back to Array if input was Array
    let result_series = result_list.into_series();
    match &input_dtype {
        DataType::Array(_, width) => result_series.cast(&DataType::Array(Box::new(inner_dtype), *width)),
        _ => Ok(result_series),
    }
}
