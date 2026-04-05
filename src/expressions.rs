#![allow(clippy::unused_unit)]
use std::collections::HashMap;
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
        DataType::Array(_, width) => {
            result_series.cast(&DataType::Array(Box::new(inner_dtype), *width))
        },
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
        DataType::Array(_, width) => {
            result_series.cast(&DataType::Array(Box::new(inner_dtype), *width))
        },
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
        DataType::Array(_, width) => {
            result_series.cast(&DataType::Array(Box::new(inner_dtype), *width))
        },
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

fn list_convolve_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::List(_) => {
            // Convolution produces Float64 output
            Ok(Field::new(
                field.name().clone(),
                DataType::List(Box::new(DataType::Float64)),
            ))
        },
        DataType::Array(_, width) => {
            // Convolution produces Float64 output, preserve Array type
            Ok(Field::new(
                field.name().clone(),
                DataType::Array(Box::new(DataType::Float64), *width),
            ))
        },
        _ => polars_bail!(InvalidOperation: "Expected List or Array type, got {:?}", field.dtype()),
    }
}

#[derive(serde::Deserialize)]
struct ConvolveKwargs {
    kernel: Vec<f64>,
    fill_value: f64,
    mode: String,
}

#[polars_expr(output_type_func=list_convolve_output_type)]
fn list_convolve(inputs: &[Series], kwargs: ConvolveKwargs) -> PolarsResult<Series> {
    let series = &inputs[0];
    let input_dtype = series.dtype().clone();

    // Convert to List if it's an Array
    let series = ensure_list_type(series)?;
    let list_chunked = series.list()?;

    let n_lists = list_chunked.len();
    if n_lists == 0 {
        return Ok(series.slice(0, 0));
    }

    // Parse kernel from kwargs
    let kernel: Vec<f64> = kwargs
        .kernel
        .iter()
        .filter(|x| x.is_finite())
        .copied()
        .collect();

    if kernel.is_empty() {
        polars_bail!(ComputeError: "Kernel cannot be empty or contain only non-finite values");
    }

    let mode = kwargs.mode.as_str();

    // Build result: convolve each row's list with kernel
    let mut result_series_vec: Vec<Option<Series>> = Vec::with_capacity(n_lists);

    for i in 0..n_lists {
        if let Some(s) = list_chunked.get_as_series(i) {
            // Convert series to f64 and handle nulls
            let signal = s.cast(&DataType::Float64)?;
            let signal_f64 = signal.f64()?;

            // Extract signal values, filling nulls with fill_value
            let signal_vec: Vec<f64> = signal_f64
                .into_iter()
                .map(|opt| opt.unwrap_or(kwargs.fill_value))
                .collect();

            // Perform convolution
            let convolved = convolve_1d(&signal_vec, &kernel, mode)?;

            // Create series from result
            let result = Series::new("".into(), convolved);
            result_series_vec.push(Some(result));
        } else {
            // Null row: return None
            result_series_vec.push(None);
        }
    }

    // Create a ListChunked from the vector of optional series
    let result_list =
        ListChunked::from_iter(result_series_vec.into_iter()).with_name(series.name().clone());

    // Cast back to Array if input was Array
    let result_series = result_list.into_series();
    match &input_dtype {
        DataType::Array(_, width) => {
            result_series.cast(&DataType::Array(Box::new(DataType::Float64), *width))
        },
        _ => Ok(result_series),
    }
}

// Perform 1D convolution
fn convolve_1d(signal: &[f64], kernel: &[f64], mode: &str) -> PolarsResult<Vec<f64>> {
    let signal_len = signal.len();
    let kernel_len = kernel.len();

    if signal_len == 0 {
        return Ok(Vec::new());
    }

    if kernel_len == 0 {
        polars_bail!(ComputeError: "Kernel length cannot be 0");
    }

    // Determine output length and offset for mapping to full convolution indices
    let (output_len, offset_to_full) = match mode {
        "full" => (signal_len + kernel_len - 1, 0),
        "same" => {
            // NumPy's same mode: output has length max(signal_len, kernel_len)
            // The output is centered relative to the full convolution
            let out_len = signal_len.max(kernel_len);
            let offset = (kernel_len as isize - 1) / 2;
            (out_len, offset)
        },
        "valid" => {
            // Valid mode: where one array fully overlaps the other
            // NumPy treats inputs symmetrically: result length is max(M, N) - min(M, N) + 1
            let output_length = if signal_len >= kernel_len {
                signal_len - kernel_len + 1
            } else {
                kernel_len - signal_len + 1
            };
            let offset = kernel_len as isize - 1;
            (output_length, offset)
        },
        "left" => (signal_len, 0),
        "right" => (signal_len, kernel_len as isize - 1),
        _ => {
            polars_bail!(ComputeError: "Invalid mode '{}'. Must be one of: full, same, valid, left, right", mode)
        },
    };

    let mut result = vec![0.0; output_len];

    // Perform convolution
    // Convolution formula: out[n] = sum_k kernel_reversed[k] * signal[n - (kernel_len - 1) + k]
    // where kernel_reversed[k] = kernel[kernel_len - 1 - k]
    for (out_idx, result_val) in result.iter_mut().enumerate() {
        let mut sum = 0.0;

        // Map output index to full convolution coordinates
        let full_idx = out_idx as isize + offset_to_full;

        // Iterate through kernel positions
        for k_idx in 0..kernel_len {
            // Position in signal for this kernel element
            let sig_pos = full_idx - (kernel_len as isize - 1) + k_idx as isize;

            // Check if signal position is valid
            if sig_pos >= 0 && sig_pos < signal_len as isize {
                // Kernel is reversed in convolution
                let kernel_val = kernel[kernel_len - 1 - k_idx];
                sum += signal[sig_pos as usize] * kernel_val;
            }
        }

        *result_val = sum;
    }

    Ok(result)
}

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

/// Count values into uniformly-spaced bins using 4 independent scatter buffers.
/// Breaks the load-store dependency chain in the inner loop: the 4 bin index computations
/// are independent (CPU can pipeline them), and the 4 stores go to different arrays
/// (no RAW hazards between s0/s1/s2/s3 even when they hit the same bin index).
/// Result is accumulated into s0 at the end.  s1/s2/s3 are pre-allocated scratch reused
/// across rows to avoid per-row allocation overhead.
///
/// Caller must ensure s0/s1/s2/s3 all have capacity >= n_bins (they are resized here).
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

    let chunks = values.chunks_exact(4);
    let rem = chunks.remainder();
    for chunk in chunks {
        // 4 independent bin computations — CPU can pipeline all 4 simultaneously
        let b0 = (((chunk[0] - first) * inv_step) as usize).min(nb);
        let b1 = (((chunk[1] - first) * inv_step) as usize).min(nb);
        let b2 = (((chunk[2] - first) * inv_step) as usize).min(nb);
        let b3 = (((chunk[3] - first) * inv_step) as usize).min(nb);
        // Stores to 4 separate arrays — no RAW hazards, CPU can execute in parallel
        s0[b0] += 1;
        s1[b1] += 1;
        s2[b2] += 1;
        s3[b3] += 1;
    }
    // Handle remaining elements (< 4) in s0
    for &v in rem {
        let bin = (((v - first) * inv_step) as usize).min(nb);
        s0[bin] += 1;
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
                        // Collect all finite values into the reusable cache — single Polars pass.
                        // Then do 2 cheap Vec passes (min/max + count) instead of 2 Polars passes.
                        values_cache.clear();
                        values_cache.extend(finite_values_iter(ca));
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
