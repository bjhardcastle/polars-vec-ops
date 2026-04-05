#![allow(clippy::unused_unit)]
use polars::prelude::*;

// Helper function to convert Array to List if needed
pub(super) fn ensure_list_type(series: &Series) -> PolarsResult<Series> {
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
