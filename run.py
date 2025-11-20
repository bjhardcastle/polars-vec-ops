import polars as pl
from polars_vec_ops import *

df = pl.DataFrame({
    'a': [1, 1, None],
    'b': [4.1, 5.2, 6.3],
    'c': ['hello', 'everybody!', '!']
})
print(df.with_columns(noop(pl.all()).name.suffix('_noop')))
print(df.with_columns(abs_i64('a').name.suffix('_abs')))