from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Pandas and Polars on split assignment plus regime "
            "aggregation. This does not benchmark model fitting."
        )
    )
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def synthetic_arrays(
    n_rows: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    rng = np.random.default_rng(seed)
    return {
        "event_id": np.arange(n_rows, dtype=np.int64),
        "timestamp_ns": (
            pd.Timestamp("2024-03-01", tz="UTC").value
            + np.arange(n_rows, dtype=np.int64) * 1_000_000
        ),
        "relative_spread": rng.lognormal(-11.0, 0.3, n_rows),
        "realized_vol_20": rng.lognormal(-10.0, 0.6, n_rows),
        "trade_intensity_1s": rng.poisson(25, n_rows).astype(np.int32),
        "correct": rng.integers(0, 2, n_rows, dtype=np.int8),
    }


def pandas_workload(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame(arrays, copy=False)
    train_end = int(len(frame) * 0.60)
    validation_end = int(len(frame) * 0.80)
    frame["split"] = np.select(
        [
            frame["event_id"] < train_end,
            frame["event_id"] < validation_end,
        ],
        ["train", "validation"],
        default="test",
    )
    spread_cut = frame.loc[
        frame["split"] == "train",
        "relative_spread",
    ].median()
    frame["spread_regime"] = np.where(
        frame["relative_spread"] <= spread_cut,
        "tight",
        "wide",
    )
    return (
        frame.groupby(["split", "spread_regime"], observed=True)
        .agg(
            n_obs=("event_id", "size"),
            mean_correct=("correct", "mean"),
            mean_volatility=("realized_vol_20", "mean"),
            mean_intensity=("trade_intensity_1s", "mean"),
        )
        .reset_index()
        .sort_values(["split", "spread_regime"])
        .reset_index(drop=True)
    )


def polars_workload(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pl.DataFrame(arrays)
    train_end = int(frame.height * 0.60)
    validation_end = int(frame.height * 0.80)
    frame = frame.with_columns(
        pl.when(pl.col("event_id") < train_end)
        .then(pl.lit("train"))
        .when(pl.col("event_id") < validation_end)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )
    spread_cut = frame.filter(pl.col("split") == "train").select(
        pl.col("relative_spread").median()
    ).item()
    result = (
        frame.with_columns(
            pl.when(pl.col("relative_spread") <= spread_cut)
            .then(pl.lit("tight"))
            .otherwise(pl.lit("wide"))
            .alias("spread_regime")
        )
        .group_by(["split", "spread_regime"])
        .agg(
            pl.len().alias("n_obs"),
            pl.col("correct").mean().alias("mean_correct"),
            pl.col("realized_vol_20").mean().alias("mean_volatility"),
            pl.col("trade_intensity_1s").mean().alias("mean_intensity"),
        )
        .sort(["split", "spread_regime"])
    )
    return result.to_pandas()


def timed_run(function, arrays: dict[str, np.ndarray]) -> tuple[float, pd.DataFrame]:
    start = perf_counter()
    result = function(arrays)
    return perf_counter() - start, result


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive.")
    arrays = synthetic_arrays(args.rows, args.seed)
    rows = []
    reference: pd.DataFrame | None = None

    for repeat in range(1, args.repeats + 1):
        for engine, function in (
            ("pandas", pandas_workload),
            ("polars", polars_workload),
        ):
            elapsed, result = timed_run(function, arrays)
            if reference is None:
                reference = result
            else:
                pd.testing.assert_frame_equal(
                    result,
                    reference,
                    check_exact=False,
                    check_dtype=False,
                    rtol=1e-12,
                    atol=1e-15,
                )
            rows.append(
                {
                    "engine": engine,
                    "repeat": repeat,
                    "rows": args.rows,
                    "elapsed_seconds": elapsed,
                }
            )

    results = pd.DataFrame(rows)
    medians = (
        results.groupby("engine", as_index=False)["elapsed_seconds"]
        .median()
        .rename(columns={"elapsed_seconds": "median_seconds"})
    )
    pandas_seconds = float(
        medians.loc[medians["engine"] == "pandas", "median_seconds"].iloc[0]
    )
    polars_seconds = float(
        medians.loc[medians["engine"] == "polars", "median_seconds"].iloc[0]
    )
    medians["speedup_vs_pandas"] = pandas_seconds / medians["median_seconds"]

    print(results.to_string(index=False))
    print()
    print(medians.to_string(index=False))
    print(
        "\nDecision rule: retain the PyArrow/Pandas production path unless "
        "Polars shows a material, reproducible gain on the real bottleneck. "
        f"Observed Polars speedup here: {pandas_seconds / polars_seconds:.2f}x."
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.merge(medians, on="engine").to_csv(args.output, index=False)
        print(f"Saved benchmark: {args.output}")


if __name__ == "__main__":
    main()
