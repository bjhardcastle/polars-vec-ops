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
    let first_series = list_chunked.get_as_series(0)
        .ok_or_else(|| polars_err!(ComputeError: "No data in list column"))?;
    let expected_len = first_series.len();
    let original_dtype = first_series.dtype().clone();

    // Collect all series references and validate
    let mut all_series = Vec::with_capacity(n_lists);
    all_series.push(first_series);

    for i in 1..n_lists {
        let s = list_chunked.get_as_series(i)
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
