from __future__ import annotations

import polars_vec_ops.expr  # noqa: F401 - registers .vec namespace
import polars_vec_ops.frame  # noqa: F401 - registers .vec namespace
from polars_vec_ops._internal import __version__ as __version__
from polars_vec_ops.expr import (
    avg,
    convolve,
    diff,
    hist,
    histogram,
    max,
    mean,
    min,
    sum,
)  # noqa: F401 - re-export for convenience

__all__ = [
    "__version__",
    "sum",
    "mean", 
    "avg", 
    "min", 
    "max", 
    "diff", 
    "convolve",
    "histogram", 
    "hist",
]