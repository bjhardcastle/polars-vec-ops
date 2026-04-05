#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use super::helpers::ensure_list_type;

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

