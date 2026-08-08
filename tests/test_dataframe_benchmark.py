import pandas as pd

from benchmarks.benchmark_dataframe_engines import (
    pandas_workload,
    polars_workload,
    synthetic_arrays,
)


def test_dataframe_engine_benchmark_has_numerical_parity() -> None:
    arrays = synthetic_arrays(10_000, seed=42)
    pandas_result = pandas_workload(arrays)
    polars_result = polars_workload(arrays)
    pd.testing.assert_frame_equal(
        pandas_result,
        polars_result,
        check_exact=False,
        check_dtype=False,
        rtol=1e-12,
        atol=1e-15,
    )
