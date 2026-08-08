from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import parse_date  # noqa: E402
from src.artifact_naming import tagged_artifact_stem  # noqa: E402
from src.model_dataset_io import resolve_model_dataset  # noqa: E402
from src.protocol import (  # noqa: E402
    DEFAULT_HORIZONS,
    TERNARY_LABELS,
    validate_protocol_symbol,
)


CLASS_LABELS = TERNARY_LABELS
CLASS_NAMES = ("down", "unchanged", "up")


def parse_horizons(raw: str) -> tuple[int, ...]:
    horizons = tuple(
        int(x.strip())
        for x in raw.replace(" ", "").split(",")
        if x.strip()
    )

    if not horizons:
        raise ValueError("At least one horizon must be provided.")

    if any(h <= 0 for h in horizons):
        raise ValueError(f"All horizons must be positive integers. Got: {horizons}")

    return horizons


def label_columns(horizons: tuple[int, ...]) -> list[str]:
    return [f"y_{h}" for h in horizons]


def json_safe_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_experiment_id(
    symbol: str,
    start: str,
    end: str,
    experiment_tag: str | None,
    experiment_id: str | None,
) -> str:
    if experiment_id is not None:
        return experiment_id

    base = f"{symbol}_{start}_to_{end}"

    if experiment_tag:
        return f"{base}_{experiment_tag}"

    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run post-experiment research audit diagnostics. "
            "The script produces systematic CSV/JSON outputs for later notebook interpretation."
        )
    )

    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help=(
            "Optional filename tag inserted after the artifact kind, e.g. "
            "'v3_fixed_window_features'."
        ),
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated label horizons, e.g. '10,20,50'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=10)

    parser.add_argument(
        "--experiment-id",
        default=None,
        help=(
            "Optional explicit experiment id. If omitted, the script uses "
            "SYMBOL_START_to_END plus optional --experiment-tag."
        ),
    )
    parser.add_argument(
        "--experiment-tag",
        default=None,
        help=(
            "Optional suffix for the output folder, e.g. 'full_resolution_v1', "
            "'v3_fixed_window_features', or '14day_replication'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional explicit audit output directory. If omitted, outputs are saved under "
            "outputs/reports/research_audit/<experiment_id>/."
        ),
    )

    parser.add_argument(
        "--primary-model",
        default="full_logistic",
        help="Primary model name as stored in final test output files.",
    )
    parser.add_argument(
        "--secondary-model",
        default="full_logistic_thresholded",
        help="Secondary/comparison model name as stored in final test output files.",
    )
    parser.add_argument(
        "--final-prefix",
        default=None,
        help=(
            "Optional prefix for final test files. If omitted, uses the tagged "
            "final_test stem."
        ),
    )

    parser.add_argument(
        "--expected-boundary-drop",
        type=int,
        default=50,
        help=(
            "Expected number of boundary-dropped rows in train and validation. "
            "Default 50 matches the maximum horizon in the original protocol."
        ),
    )

    parser.add_argument(
        "--mid-return-warning-abs-tol",
        type=float,
        default=1e-6,
        help=(
            "Maximum absolute error below which mid_return_5 mismatch is treated "
            "as a warning rather than a critical failure."
        ),
    )
    parser.add_argument(
        "--realized-vol-warning-abs-tol",
        type=float,
        default=1e-5,
        help=(
            "Maximum absolute error below which realized_vol_20 mismatch is treated "
            "as a warning rather than a critical failure."
        ),
    )

    return parser.parse_args()


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def parquet_file_paths(path: Path) -> tuple[Path, ...]:
    """Return deterministic Parquet inputs without admitting sidecar files."""
    path = Path(path)
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"Expected a Parquet file, got: {path}")
        return (path,)
    if not path.is_dir():
        raise FileNotFoundError(f"Missing Parquet dataset: {path}")

    paths = tuple(sorted(path.glob("*.parquet")))
    if not paths:
        raise FileNotFoundError(
            f"No top-level Parquet files found in dataset directory: {path}"
        )
    return paths


def parquet_dataset(path: Path) -> ds.Dataset:
    return ds.dataset(
        [str(file_path) for file_path in parquet_file_paths(path)],
        format="parquet",
    )


def parquet_columns(path: Path) -> set[str]:
    return set(parquet_dataset(path).schema.names)


def parquet_num_rows(path: Path) -> int:
    return sum(
        int(pq.ParquetFile(file_path).metadata.num_rows)
        for file_path in parquet_file_paths(path)
    )


def iter_parquet_batches(
    path: Path,
    columns: list[str],
    *,
    batch_size: int,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return parquet_dataset(path).scanner(
        columns=columns,
        batch_size=batch_size,
        use_threads=False,
    ).to_batches()


def ensure_columns(path: Path, required_columns: list[str], context: str) -> None:
    available = parquet_columns(path)
    missing = sorted(set(required_columns) - available)

    if missing:
        raise ValueError(
            f"{context} is missing required columns in {path}: {missing}"
        )


def choose_event_id_sample(
    *,
    num_rows: int,
    rng: np.random.Generator,
    sample_size: int,
    min_event_id: int,
    max_future_horizon: int = 0,
    lookback: int = 0,
) -> np.ndarray:
    if num_rows <= 0:
        raise ValueError("num_rows must be positive.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if min_event_id < 0 or max_future_horizon < 0 or lookback < 0:
        raise ValueError(
            "min_event_id, max_future_horizon, and lookback must be non-negative."
        )

    lower = max(min_event_id, lookback)
    upper_exclusive = num_rows - max_future_horizon

    if upper_exclusive <= lower:
        raise ValueError(
            "Dataset is too small for requested audit sample. "
            f"num_rows={num_rows}, lower={lower}, upper_exclusive={upper_exclusive}"
        )

    population_size = upper_exclusive - lower
    size = min(sample_size, population_size)
    return (
        rng.choice(population_size, size=size, replace=False)
        .astype(np.int64, copy=False)
        + lower
    )


def read_parquet_by_event_ids(
    path: Path,
    event_ids: list[int] | np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    event_ids = sorted(set(int(x) for x in event_ids))

    dataset = parquet_dataset(path)
    table = dataset.to_table(
        columns=columns,
        filter=ds.field("event_id").isin(event_ids),
    )

    return (
        table.to_pandas()
        .sort_values("event_id")
        .reset_index(drop=True)
    )


def summarize_values(values: np.ndarray, prefix: str) -> dict:
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_q90": np.nan,
            f"{prefix}_q95": np.nan,
            f"{prefix}_q99": np.nan,
        }

    return {
        f"{prefix}_count": int(len(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
        f"{prefix}_q99": float(np.quantile(values, 0.99)),
    }


def audit_label_alignment(
    model_path: Path,
    events_path: Path,
    rng: np.random.Generator,
    output_dir: Path,
    sample_size: int,
    horizons: tuple[int, ...],
) -> dict:
    print("Running label alignment audit...")

    y_cols = label_columns(horizons)

    ensure_columns(
        model_path,
        ["event_id", "timestamp", "split", "model_eligible", *y_cols],
        "model dataset for label alignment audit",
    )
    ensure_columns(
        events_path,
        ["event_id", "timestamp", "midprice", "bid_price", "ask_price"],
        "quote events for label alignment audit",
    )

    num_rows = parquet_num_rows(model_path)

    sample_event_ids = choose_event_id_sample(
        num_rows=num_rows,
        rng=rng,
        sample_size=sample_size,
        min_event_id=100,
        max_future_horizon=max(horizons),
        lookback=0,
    )

    needed_model_ids = sample_event_ids.tolist()
    needed_event_ids = set(sample_event_ids.tolist())

    for h in horizons:
        needed_event_ids.update((sample_event_ids + h).tolist())

    model_sample = read_parquet_by_event_ids(
        model_path,
        needed_model_ids,
        columns=[
            "event_id",
            "timestamp",
            "split",
            "model_eligible",
            *y_cols,
        ],
    )

    event_sample = read_parquet_by_event_ids(
        events_path,
        list(needed_event_ids),
        columns=[
            "event_id",
            "timestamp",
            "midprice",
            "bid_price",
            "ask_price",
        ],
    )

    event_by_id = event_sample.set_index("event_id")

    rows = []

    for _, row in model_sample.iterrows():
        eid = int(row["event_id"])
        mid_t = float(event_by_id.loc[eid, "midprice"])

        for h in horizons:
            future_eid = eid + h
            mid_future = float(event_by_id.loc[future_eid, "midprice"])

            manual_label = int(np.sign(mid_future - mid_t))
            stored_label = int(row[f"y_{h}"])

            rows.append(
                {
                    "event_id": eid,
                    "split": row["split"],
                    "timestamp": row["timestamp"],
                    "horizon": h,
                    "midprice_t": mid_t,
                    "future_event_id": future_eid,
                    "midprice_t_plus_h": mid_future,
                    "manual_label": manual_label,
                    "stored_label": stored_label,
                    "matches": manual_label == stored_label,
                }
            )

    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "label_alignment_audit.csv", index=False)

    return {
        "all_manual_labels_match_stored_labels": bool(audit["matches"].all()),
        "label_alignment_sample_size": int(len(audit)),
    }


def audit_split_boundaries(
    model_path: Path,
    output_dir: Path,
    horizons: tuple[int, ...],
    expected_boundary_drop: int,
) -> dict:
    print("Running split-boundary audit...")

    y_cols = label_columns(horizons)

    cols = [
        "event_id",
        "timestamp",
        "split",
        "feature_complete",
        "is_boundary_drop",
        "model_eligible",
        *y_cols,
    ]

    ensure_columns(model_path, cols, "model dataset for split-boundary audit")

    summary_state = {
        split: {
            "rows_total": 0,
            "feature_complete_rows": 0,
            "boundary_drop_rows": 0,
            "model_eligible_rows": 0,
            "timestamp_start": None,
            "timestamp_end": None,
        }
        for split in ("train", "validation", "test")
    }
    boundary_frames = []
    no_eligible_boundary = True
    eligible_features_complete = True

    for batch in iter_parquet_batches(
        model_path,
        cols,
        batch_size=1_000_000,
    ):
        frame = batch.to_pandas()
        unexpected = sorted(
            set(frame["split"].dropna().unique()) - set(summary_state)
        )
        if unexpected:
            raise ValueError(f"Unexpected split values: {unexpected}")

        boundary_mask = frame["is_boundary_drop"].astype(bool)
        eligible_mask = frame["model_eligible"].astype(bool)
        no_eligible_boundary &= not bool(
            (boundary_mask & eligible_mask).any()
        )
        eligible_features_complete &= bool(
            frame.loc[eligible_mask, "feature_complete"].all()
        )
        if bool(boundary_mask.any()):
            boundary_frames.append(
                frame.loc[
                    boundary_mask,
                    [
                        "event_id",
                        "timestamp",
                        "split",
                        "is_boundary_drop",
                        "model_eligible",
                        *y_cols,
                    ],
                ].copy()
            )

        for split, state in summary_state.items():
            part = frame.loc[frame["split"] == split]
            if part.empty:
                continue
            state["rows_total"] += len(part)
            state["feature_complete_rows"] += int(
                part["feature_complete"].sum()
            )
            state["boundary_drop_rows"] += int(
                part["is_boundary_drop"].sum()
            )
            state["model_eligible_rows"] += int(
                part["model_eligible"].sum()
            )
            part_start = part["timestamp"].min()
            part_end = part["timestamp"].max()
            if state["timestamp_start"] is None:
                state["timestamp_start"] = part_start
            state["timestamp_end"] = part_end

    split_summary = pd.DataFrame(
        [
            {"split": split, **state}
            for split, state in summary_state.items()
            if state["rows_total"] > 0
        ]
    )

    split_summary.to_csv(output_dir / "split_boundary_summary.csv", index=False)

    boundary_rows = (
        pd.concat(boundary_frames, ignore_index=True)
        if boundary_frames
        else pd.DataFrame(
            columns=[
                "event_id",
                "timestamp",
                "split",
                "is_boundary_drop",
                "model_eligible",
                *y_cols,
            ]
        )
    )

    boundary_rows.to_csv(output_dir / "split_boundary_rows.csv", index=False)

    split_names = set(split_summary["split"])
    by_split = split_summary.set_index("split")

    checks = {
        "train_split_present": "train" in split_names,
        "validation_split_present": "validation" in split_names,
        "test_split_present": "test" in split_names,
        "train_boundary_drop_expected": (
            int(by_split.loc["train", "boundary_drop_rows"])
            == expected_boundary_drop
        ) if "train" in by_split.index else False,
        "validation_boundary_drop_expected": (
            int(by_split.loc["validation", "boundary_drop_rows"])
            == expected_boundary_drop
        ) if "validation" in by_split.index else False,
        "test_boundary_drop_0": (
            int(by_split.loc["test", "boundary_drop_rows"])
            == 0
        ) if "test" in by_split.index else False,
        "no_boundary_rows_are_model_eligible": no_eligible_boundary,
        "all_model_eligible_rows_are_feature_complete": (
            eligible_features_complete
        ),
    }

    return checks


def audit_feature_time(
    model_path: Path,
    events_path: Path,
    rng: np.random.Generator,
    output_dir: Path,
    sample_size: int,
) -> tuple[dict, list[int]]:
    print("Running feature-time audit...")

    audited_feature_cols = [
        "delta_bid_size",
        "delta_ask_size",
        "mid_return_5",
        "realized_vol_20",
        "relative_spread",
        "queue_imbalance",
    ]

    ensure_columns(
        model_path,
        ["event_id", "timestamp", *audited_feature_cols],
        "model dataset for feature-time audit",
    )
    ensure_columns(
        events_path,
        [
            "event_id",
            "timestamp",
            "bid_size",
            "ask_size",
            "midprice",
            "bid_price",
            "ask_price",
        ],
        "quote events for feature-time audit",
    )

    num_rows = parquet_num_rows(model_path)

    feature_audit_ids = choose_event_id_sample(
        num_rows=num_rows,
        rng=rng,
        sample_size=sample_size,
        min_event_id=1000,
        max_future_horizon=0,
        lookback=25,
    )

    needed_ids = set()

    for eid in feature_audit_ids:
        needed_ids.update(range(int(eid) - 25, int(eid) + 1))

    event_window = read_parquet_by_event_ids(
        events_path,
        list(needed_ids),
        columns=[
            "event_id",
            "timestamp",
            "bid_size",
            "ask_size",
            "midprice",
            "bid_price",
            "ask_price",
        ],
    )

    model_feature_sample = read_parquet_by_event_ids(
        model_path,
        feature_audit_ids,
        columns=[
            "event_id",
            "timestamp",
            *audited_feature_cols,
        ],
    )

    rows = []

    for _, row in model_feature_sample.iterrows():
        eid = int(row["event_id"])

        w = (
            event_window.loc[
                (event_window["event_id"] >= eid - 25)
                & (event_window["event_id"] <= eid)
            ]
            .sort_values("event_id")
            .reset_index(drop=True)
        )

        if len(w) < 26:
            raise ValueError(
                f"Not enough event history for event_id={eid}. "
                f"Expected 26 rows from eid-25 to eid, got {len(w)}."
            )

        if not np.all(np.diff(w["event_id"].to_numpy(dtype=np.int64)) == 1):
            raise ValueError(f"Event history window is not contiguous for event_id={eid}.")

        w_by_id = w.set_index("event_id")

        current = w_by_id.loc[eid]
        prev = w_by_id.loc[eid - 1]
        past_5 = w_by_id.loc[eid - 5]

        log_mid = np.log(w["midprice"].to_numpy(dtype=np.float64))
        one_event_returns = np.diff(log_mid)

        manual_rv20 = np.sqrt(np.sum(one_event_returns[-20:] ** 2))
        manual_delta_bid = float(current["bid_size"] - prev["bid_size"])
        manual_delta_ask = float(current["ask_size"] - prev["ask_size"])
        manual_mid_return_5 = float(np.log(current["midprice"]) - np.log(past_5["midprice"]))
        manual_relative_spread = float(
            (current["ask_price"] - current["bid_price"]) / current["midprice"]
        )
        manual_queue_imbalance = float(
            (current["bid_size"] - current["ask_size"])
            / (current["bid_size"] + current["ask_size"])
        )

        rows.append(
            {
                "event_id": eid,
                "timestamp": row["timestamp"],
                "manual_delta_bid_size": manual_delta_bid,
                "stored_delta_bid_size": row["delta_bid_size"],
                "delta_bid_size_match": np.isclose(
                    manual_delta_bid,
                    row["delta_bid_size"],
                    rtol=1e-5,
                    atol=1e-8,
                ),
                "manual_delta_ask_size": manual_delta_ask,
                "stored_delta_ask_size": row["delta_ask_size"],
                "delta_ask_size_match": np.isclose(
                    manual_delta_ask,
                    row["delta_ask_size"],
                    rtol=1e-5,
                    atol=1e-8,
                ),
                "manual_mid_return_5": manual_mid_return_5,
                "stored_mid_return_5": row["mid_return_5"],
                "mid_return_5_match_float64": np.isclose(
                    manual_mid_return_5,
                    row["mid_return_5"],
                    rtol=1e-5,
                    atol=1e-8,
                ),
                "manual_realized_vol_20": manual_rv20,
                "stored_realized_vol_20": row["realized_vol_20"],
                "realized_vol_20_match_float64": np.isclose(
                    manual_rv20,
                    row["realized_vol_20"],
                    rtol=1e-5,
                    atol=1e-8,
                ),
                "manual_relative_spread": manual_relative_spread,
                "stored_relative_spread": row["relative_spread"],
                "relative_spread_match": np.isclose(
                    manual_relative_spread,
                    row["relative_spread"],
                    rtol=1e-5,
                    atol=1e-8,
                ),
                "manual_queue_imbalance": manual_queue_imbalance,
                "stored_queue_imbalance": row["queue_imbalance"],
                "queue_imbalance_match": np.isclose(
                    manual_queue_imbalance,
                    row["queue_imbalance"],
                    rtol=1e-5,
                    atol=1e-8,
                ),
            }
        )

    audit = pd.DataFrame(rows)

    audit["mid_return_5_abs_error"] = (
        audit["manual_mid_return_5"] - audit["stored_mid_return_5"]
    ).abs()

    audit["realized_vol_20_abs_error"] = (
        audit["manual_realized_vol_20"] - audit["stored_realized_vol_20"]
    ).abs()

    audit.to_csv(output_dir / "feature_time_audit.csv", index=False)

    checks = {
        "delta_bid_size_all_match": bool(audit["delta_bid_size_match"].all()),
        "delta_ask_size_all_match": bool(audit["delta_ask_size_match"].all()),
        "mid_return_5_all_match_float64": bool(audit["mid_return_5_match_float64"].all()),
        "realized_vol_20_all_match_float64": bool(audit["realized_vol_20_match_float64"].all()),
        "relative_spread_all_match": bool(audit["relative_spread_match"].all()),
        "queue_imbalance_all_match": bool(audit["queue_imbalance_match"].all()),
        "max_mid_return_5_abs_error": float(audit["mid_return_5_abs_error"].max()),
        "max_realized_vol_20_abs_error": float(audit["realized_vol_20_abs_error"].max()),
    }

    problem_ids = audit.loc[
        (~audit["mid_return_5_match_float64"])
        | (~audit["realized_vol_20_match_float64"]),
        "event_id",
    ].astype(int).tolist()

    return checks, problem_ids


def audit_local_float32_log_recompute(
    model_path: Path,
    events_path: Path,
    problem_event_ids: list[int],
    output_dir: Path,
) -> dict:
    """
    Attempts a local float32-storage-style recomputation for rows where strict
    float64 recomputation did not exactly match stored log-return-derived features.

    This is intentionally not treated as a critical leakage check. It is a diagnostic
    to help interpret whether differences are small numerical/storage effects.
    """
    print("Running local float32 log-feature recomputation audit...")

    if not problem_event_ids:
        checks = {
            "mid_return_5_matches_local_float32_recompute": True,
            "realized_vol_20_matches_local_float32_recompute": True,
            "float32_precision_audit_n_problem_ids": 0,
            "max_mid_return_5_local_float32_abs_error": np.nan,
            "max_realized_vol_20_local_float32_abs_error": np.nan,
        }
        pd.DataFrame([checks]).to_csv(
            output_dir / "float32_log_feature_audit.csv",
            index=False,
        )
        return checks

    ensure_columns(
        model_path,
        ["event_id", "timestamp", "mid_return_5", "realized_vol_20"],
        "model dataset for local float32 log-feature recomputation audit",
    )
    ensure_columns(
        events_path,
        ["event_id", "timestamp", "bid_price", "ask_price", "midprice"],
        "quote events for local float32 log-feature recomputation audit",
    )

    rows = []

    needed_ids = set()
    for eid in problem_event_ids:
        needed_ids.update(range(int(eid) - 35, int(eid) + 1))

    event_window = read_parquet_by_event_ids(
        events_path,
        list(needed_ids),
        columns=[
            "event_id",
            "timestamp",
            "bid_price",
            "ask_price",
            "midprice",
        ],
    )

    stored = read_parquet_by_event_ids(
        model_path,
        problem_event_ids,
        columns=[
            "event_id",
            "timestamp",
            "mid_return_5",
            "realized_vol_20",
        ],
    )

    for _, row in stored.iterrows():
        eid = int(row["event_id"])

        w = (
            event_window[
                (event_window["event_id"] >= eid - 35)
                & (event_window["event_id"] <= eid)
            ]
            .sort_values("event_id")
            .reset_index(drop=True)
        )

        if len(w) < 36:
            raise ValueError(
                f"Not enough event history for event_id={eid}. "
                f"Expected 36 rows from eid-35 to eid, got {len(w)}."
            )

        if int(w["event_id"].iloc[-1]) != eid:
            raise ValueError(
                f"Event window for event_id={eid} does not end at the target event."
            )

        if not np.all(np.diff(w["event_id"].to_numpy(dtype=np.int64)) == 1):
            raise ValueError(
                f"Event window for event_id={eid} is not contiguous."
            )

        bid = w["bid_price"].to_numpy(dtype=np.float64)
        ask = w["ask_price"].to_numpy(dtype=np.float64)

        recomputed_mid64 = ((bid + ask) / 2.0).astype(np.float64)
        log_mid64 = np.log(recomputed_mid64)

        stored_mid_return_5 = float(row["mid_return_5"])
        stored_rv20 = float(row["realized_vol_20"])

        mid_return_5_float64 = log_mid64[-1] - log_mid64[-6]
        returns64 = np.diff(log_mid64)
        rv20_float64 = np.sqrt(np.sum(returns64[-20:] ** 2))

        mid_return_5_local_float32 = np.float32(mid_return_5_float64)
        rv20_local_float32 = np.float32(rv20_float64)

        rows.append(
            {
                "event_id": eid,
                "stored_mid_return_5": stored_mid_return_5,
                "mid_return_5_float64": mid_return_5_float64,
                "mid_return_5_local_float32": float(mid_return_5_local_float32),
                "mid_return_5_float64_abs_error": abs(
                    stored_mid_return_5 - mid_return_5_float64
                ),
                "mid_return_5_local_float32_abs_error": abs(
                    stored_mid_return_5 - float(mid_return_5_local_float32)
                ),
                "stored_realized_vol_20": stored_rv20,
                "rv20_float64": rv20_float64,
                "rv20_local_float32": float(rv20_local_float32),
                "rv20_float64_abs_error": abs(stored_rv20 - rv20_float64),
                "rv20_local_float32_abs_error": abs(
                    stored_rv20 - float(rv20_local_float32)
                ),
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "float32_log_feature_audit.csv", index=False)

    return {
        "mid_return_5_matches_local_float32_recompute": bool(
            np.isclose(
                audit["stored_mid_return_5"],
                audit["mid_return_5_local_float32"],
                rtol=1e-6,
                atol=1e-10,
            ).all()
        ),
        "realized_vol_20_matches_local_float32_recompute": bool(
            np.isclose(
                audit["stored_realized_vol_20"],
                audit["rv20_local_float32"],
                rtol=1e-6,
                atol=1e-10,
            ).all()
        ),
        "float32_precision_audit_n_problem_ids": int(len(audit)),
        "max_mid_return_5_local_float32_abs_error": float(
            audit["mid_return_5_local_float32_abs_error"].max()
        ),
        "max_realized_vol_20_local_float32_abs_error": float(
            audit["rv20_local_float32_abs_error"].max()
        ),
    }


def audit_trade_flow_timing(
    model_path: Path,
    trades_path: Path,
    rng: np.random.Generator,
    output_dir: Path,
    sample_size: int,
) -> dict:
    print("Running trade-flow timing audit...")

    audited_trade_cols = [
        "trade_intensity_1s",
        "signed_trade_count_imbalance_1s",
        "signed_trade_volume_imbalance_1s",
    ]

    ensure_columns(
        model_path,
        ["event_id", "timestamp", *audited_trade_cols],
        "model dataset for trade-flow timing audit",
    )
    ensure_columns(
        trades_path,
        ["timestamp", "quantity", "buyer_is_maker"],
        "raw trades for trade-flow timing audit",
    )

    num_rows = parquet_num_rows(model_path)

    trade_audit_ids = choose_event_id_sample(
        num_rows=num_rows,
        rng=rng,
        sample_size=sample_size,
        min_event_id=1000,
        max_future_horizon=0,
        lookback=0,
    )

    trade_feature_sample = read_parquet_by_event_ids(
        model_path,
        trade_audit_ids,
        columns=[
            "event_id",
            "timestamp",
            *audited_trade_cols,
        ],
    )

    min_ts = trade_feature_sample["timestamp"].min() - pd.Timedelta(seconds=2)
    max_ts = trade_feature_sample["timestamp"].max() + pd.Timedelta(milliseconds=1)

    trades_window = pd.read_parquet(
        trades_path,
        columns=["timestamp", "quantity", "buyer_is_maker"],
        filters=[
            ("timestamp", ">=", min_ts),
            ("timestamp", "<=", max_ts),
        ],
    )

    rows = []

    for _, row in trade_feature_sample.iterrows():
        eid = int(row["event_id"])
        ts = row["timestamp"]

        in_window = trades_window[
            (trades_window["timestamp"] > ts - pd.Timedelta(seconds=1))
            & (trades_window["timestamp"] < ts)
        ].copy()

        equal_timestamp_trades = trades_window[trades_window["timestamp"] == ts].copy()

        manual_count = len(in_window)

        if manual_count > 0:
            sign = np.where(in_window["buyer_is_maker"].to_numpy(), -1.0, 1.0)
            qty = in_window["quantity"].to_numpy()

            manual_signed_count_imbalance = sign.sum() / manual_count
            manual_signed_volume_imbalance = (sign * qty).sum() / qty.sum()
        else:
            manual_signed_count_imbalance = 0.0
            manual_signed_volume_imbalance = 0.0

        rows.append(
            {
                "event_id": eid,
                "timestamp": ts,
                "manual_trade_count_excluding_equal_ts": manual_count,
                "stored_trade_intensity_1s": row["trade_intensity_1s"],
                "trade_count_match": manual_count == int(row["trade_intensity_1s"]),
                "equal_timestamp_trade_count": len(equal_timestamp_trades),
                "manual_signed_trade_count_imbalance": manual_signed_count_imbalance,
                "stored_signed_trade_count_imbalance": row["signed_trade_count_imbalance_1s"],
                "signed_count_imbalance_match": np.isclose(
                    manual_signed_count_imbalance,
                    row["signed_trade_count_imbalance_1s"],
                    rtol=1e-5,
                    atol=1e-8,
                ),
                "manual_signed_trade_volume_imbalance": manual_signed_volume_imbalance,
                "stored_signed_trade_volume_imbalance": row["signed_trade_volume_imbalance_1s"],
                "signed_volume_imbalance_match": np.isclose(
                    manual_signed_volume_imbalance,
                    row["signed_trade_volume_imbalance_1s"],
                    rtol=1e-5,
                    atol=1e-8,
                ),
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "trade_flow_timing_audit.csv", index=False)

    return {
        "trade_intensity_all_match": bool(audit["trade_count_match"].all()),
        "signed_count_imbalance_all_match": bool(audit["signed_count_imbalance_match"].all()),
        "signed_volume_imbalance_all_match": bool(audit["signed_volume_imbalance_match"].all()),
        "at_least_some_equal_timestamp_cases_seen": bool(
            (audit["equal_timestamp_trade_count"] > 0).any()
        ),
    }


def build_target_drift_summary(
    target_diag_path: Path,
    output_dir: Path,
) -> dict:
    print("Building target drift summary...")

    target_diag = pd.read_csv(target_diag_path)

    required = {
        "split",
        "horizon",
        "nonzero_fraction",
        "majority_class_accuracy",
    }
    missing = sorted(required - set(target_diag.columns))
    if missing:
        raise ValueError(f"target diagnostics file missing required columns: {missing}")

    rows = []

    for h in sorted(target_diag["horizon"].unique()):
        h_df = target_diag[target_diag["horizon"] == h].set_index("split")

        if not {"train", "validation", "test"}.issubset(set(h_df.index)):
            raise ValueError(
                f"target diagnostics for horizon {h} must include train, validation, and test."
            )

        train = h_df.loc["train"]
        validation = h_df.loc["validation"]
        test = h_df.loc["test"]

        rows.append(
            {
                "horizon": h,
                "train_nonzero_fraction": train["nonzero_fraction"],
                "validation_nonzero_fraction": validation["nonzero_fraction"],
                "test_nonzero_fraction": test["nonzero_fraction"],
                "validation_minus_train_nonzero": (
                    validation["nonzero_fraction"] - train["nonzero_fraction"]
                ),
                "test_minus_train_nonzero": (
                    test["nonzero_fraction"] - train["nonzero_fraction"]
                ),
                "train_majority_accuracy": train["majority_class_accuracy"],
                "validation_majority_accuracy": validation["majority_class_accuracy"],
                "test_majority_accuracy": test["majority_class_accuracy"],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "target_drift_summary.csv", index=False)

    return {
        "max_test_minus_train_nonzero": float(summary["test_minus_train_nonzero"].max()),
        "max_validation_minus_train_nonzero": float(
            summary["validation_minus_train_nonzero"].max()
        ),
    }


def build_daily_stability_summary(
    final_daily_path: Path,
    output_dir: Path,
) -> dict:
    print("Building daily stability summary...")

    final_daily = pd.read_csv(final_daily_path)

    required = {
        "model",
        "horizon",
        "utc_day",
        "macro_f1",
        "balanced_accuracy",
    }
    missing = sorted(required - set(final_daily.columns))
    if missing:
        raise ValueError(f"final daily file missing required columns: {missing}")

    summary = (
        final_daily
        .groupby(["model", "horizon"])
        .agg(
            n_daily_blocks=("utc_day", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_min=("balanced_accuracy", "min"),
            balanced_accuracy_max=("balanced_accuracy", "max"),
        )
        .reset_index()
        .sort_values(["horizon", "model"])
    )

    summary.to_csv(output_dir / "daily_stability_summary.csv", index=False)

    return {
        "n_test_daily_blocks": int(final_daily["utc_day"].nunique()),
    }


def build_coefficient_interpretation(
    coefficient_path: Path,
    output_dir: Path,
    primary_model: str,
    horizons: tuple[int, ...],
) -> dict:
    print("Building coefficient interpretation...")

    coef = pd.read_csv(coefficient_path)

    required = {
        "model",
        "horizon",
        "class_label",
        "feature",
        "standardized_coefficient",
    }
    missing = sorted(required - set(coef.columns))
    if missing:
        raise ValueError(f"coefficient file missing required columns: {missing}")

    primary_coef = coef[coef["model"] == primary_model].copy()

    if primary_coef.empty:
        raise ValueError(
            f"No coefficient rows found for primary_model={primary_model!r}."
        )

    rows = []

    for h in horizons:
        h_coef = primary_coef[primary_coef["horizon"] == h]

        if h_coef.empty:
            continue

        available_classes = set(h_coef["class_label"].unique())
        if not set(CLASS_LABELS).issubset(available_classes):
            raise ValueError(
                f"Coefficient file for horizon {h} does not contain all classes "
                f"{CLASS_LABELS}. Found: {sorted(available_classes)}"
            )

        down = (
            h_coef[h_coef["class_label"] == -1]
            .set_index("feature")["standardized_coefficient"]
        )

        unchanged = (
            h_coef[h_coef["class_label"] == 0]
            .set_index("feature")["standardized_coefficient"]
        )

        up = (
            h_coef[h_coef["class_label"] == 1]
            .set_index("feature")["standardized_coefficient"]
        )

        common_features = sorted(set(down.index) & set(unchanged.index) & set(up.index))

        for feature in common_features:
            rows.append(
                {
                    "model": primary_model,
                    "horizon": h,
                    "feature": feature,
                    "coef_down": down.loc[feature],
                    "coef_unchanged": unchanged.loc[feature],
                    "coef_up": up.loc[feature],
                    "up_minus_down": up.loc[feature] - down.loc[feature],
                    "opposite_sign_down_up": (
                        np.sign(down.loc[feature]) != np.sign(up.loc[feature])
                    ),
                    "move_vs_unchanged_pattern": (
                        abs(down.loc[feature]) + abs(up.loc[feature])
                    ) / (abs(unchanged.loc[feature]) + 1e-12),
                }
            )

    interpretation = pd.DataFrame(rows)

    if interpretation.empty:
        raise ValueError("Coefficient interpretation produced no rows.")

    interpretation.to_csv(output_dir / "coefficient_interpretation.csv", index=False)

    sign_stability = (
        interpretation
        .assign(
            sign_down=lambda x: np.sign(x["coef_down"]),
            sign_unchanged=lambda x: np.sign(x["coef_unchanged"]),
            sign_up=lambda x: np.sign(x["coef_up"]),
        )
        .groupby("feature")
        .agg(
            down_signs=("sign_down", lambda s: tuple(s)),
            unchanged_signs=("sign_unchanged", lambda s: tuple(s)),
            up_signs=("sign_up", lambda s: tuple(s)),
            opposite_sign_down_up_all_horizons=("opposite_sign_down_up", "all"),
            mean_abs_up_minus_down=("up_minus_down", lambda s: s.abs().mean()),
        )
        .reset_index()
        .sort_values("mean_abs_up_minus_down", ascending=False)
    )

    sign_stability.to_csv(output_dir / "coefficient_sign_stability.csv", index=False)

    top_feature = sign_stability.iloc[0]["feature"]

    return {
        "top_directional_asymmetry_feature": str(top_feature),
    }


def build_confusion_recall_comparison(
    final_confusion_path: Path,
    output_dir: Path,
    model_names: tuple[str, ...],
    horizons: tuple[int, ...],
) -> dict:
    print("Building confusion recall comparison...")

    confusion = pd.read_csv(final_confusion_path)

    required = {
        "model",
        "horizon",
        "true_name",
        "pred_name",
        "value",
    }
    missing = sorted(required - set(confusion.columns))
    if missing:
        raise ValueError(f"confusion file missing required columns: {missing}")

    rows = []

    for model_name in model_names:
        for h in horizons:
            subset = confusion[
                (confusion["model"] == model_name)
                & (confusion["horizon"] == h)
            ]

            if subset.empty:
                continue

            mat = subset.pivot_table(
                index="true_name",
                columns="pred_name",
                values="value",
                fill_value=0,
            )

            mat = mat.reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0)
            row_totals = mat.sum(axis=1).replace(0, np.nan)
            row_norm = mat.div(row_totals, axis=0).fillna(0.0)

            rows.append(
                {
                    "model": model_name,
                    "horizon": h,
                    "down_recall": row_norm.loc["down", "down"],
                    "unchanged_recall": row_norm.loc["unchanged", "unchanged"],
                    "up_recall": row_norm.loc["up", "up"],
                    "balanced_accuracy_manual": (
                        row_norm.loc["down", "down"]
                        + row_norm.loc["unchanged", "unchanged"]
                        + row_norm.loc["up", "up"]
                    ) / 3,
                }
            )

    recall = pd.DataFrame(rows)

    if recall.empty:
        raise ValueError("Confusion recall comparison produced no rows.")

    recall.to_csv(output_dir / "confusion_recall_comparison.csv", index=False)

    summary = {
        "confusion_recall_models_compared": "|".join(sorted(recall["model"].unique())),
        "confusion_recall_rows": int(len(recall)),
    }

    for model_name in model_names:
        model_rows = recall[recall["model"] == model_name]

        if not model_rows.empty:
            safe_model_name = model_name.replace(" ", "_")
            summary[f"{safe_model_name}_min_balanced_accuracy_manual"] = float(
                model_rows["balanced_accuracy_manual"].min()
            )
            summary[f"{safe_model_name}_max_balanced_accuracy_manual"] = float(
                model_rows["balanced_accuracy_manual"].max()
            )

    return summary


def build_spread_relative_summary(
    model_path: Path,
    events_path: Path,
    output_dir: Path,
    horizons: tuple[int, ...],
) -> dict:
    print("Building spread-relative move-size summary...")

    y_cols = label_columns(horizons)

    ensure_columns(
        model_path,
        ["event_id", "timestamp", "split", "model_eligible", *y_cols],
        "model dataset for spread-relative summary",
    )
    ensure_columns(
        events_path,
        ["event_id", "timestamp", "bid_price", "ask_price", "midprice"],
        "quote events for spread-relative summary",
    )

    model_ds = parquet_dataset(model_path)

    test_meta_table = model_ds.to_table(
        columns=[
            "event_id",
            "timestamp",
            *y_cols,
        ],
        filter=(
            (ds.field("split") == "test")
            & ds.field("model_eligible")
        ),
    )

    test_meta = (
        test_meta_table
        .to_pandas()
        .sort_values("event_id")
        .reset_index(drop=True)
    )

    if test_meta.empty:
        raise ValueError("No model-eligible test rows found for spread-relative summary.")

    test_event_ids = test_meta["event_id"].to_numpy(dtype=np.int64)

    event_start = int(test_event_ids[0])
    event_end = int(test_event_ids[-1] + max(horizons))

    events_ds = ds.dataset(events_path, format="parquet")

    test_events_table = events_ds.to_table(
        columns=[
            "event_id",
            "timestamp",
            "bid_price",
            "ask_price",
            "midprice",
        ],
        filter=(
            (ds.field("event_id") >= event_start)
            & (ds.field("event_id") <= event_end)
        ),
    )

    test_events = (
        test_events_table
        .to_pandas()
        .sort_values("event_id")
        .reset_index(drop=True)
    )

    event_range_checks = {
        "test_event_ids_monotonic": bool(np.all(np.diff(test_event_ids) >= 0)),
        "test_event_ids_contiguous": bool(np.all(np.diff(test_event_ids) == 1)),
        "events_start_matches": int(test_events["event_id"].iloc[0]) == event_start,
        "events_end_covers_future_horizon": int(test_events["event_id"].iloc[-1]) >= event_end,
        "events_are_contiguous": bool(
            np.all(np.diff(test_events["event_id"].to_numpy(dtype=np.int64)) == 1)
        ),
    }

    if not all(event_range_checks.values()):
        raise ValueError(f"Event range checks failed: {event_range_checks}")

    mid = test_events["midprice"].to_numpy(dtype=np.float64)
    bid = test_events["bid_price"].to_numpy(dtype=np.float64)
    ask = test_events["ask_price"].to_numpy(dtype=np.float64)

    n_test = len(test_meta)

    mid_now = mid[:n_test]
    spread_now = ask[:n_test] - bid[:n_test]

    spread_checks = {
        "all_spreads_positive": bool(np.all(spread_now > 0)),
        "spread_min": float(np.min(spread_now)),
        "spread_median": float(np.median(spread_now)),
        "spread_q90": float(np.quantile(spread_now, 0.90)),
        "spread_q99": float(np.quantile(spread_now, 0.99)),
    }

    rows = []

    for h in horizons:
        y = test_meta[f"y_{h}"].to_numpy(dtype=np.int8)

        future_mid = mid[h : h + n_test]
        change = future_mid - mid_now
        abs_change = np.abs(change)

        spread_ratio = abs_change / spread_now
        manual_y = np.sign(change).astype(np.int8)

        groups = [
            ("all", np.ones(n_test, dtype=bool)),
            ("down", y == -1),
            ("unchanged", y == 0),
            ("up", y == 1),
            ("nonzero", y != 0),
        ]

        for group_name, mask in groups:
            group_abs_change = abs_change[mask]
            group_spread_ratio = spread_ratio[mask]
            group_change = change[mask]

            row = {
                "horizon": h,
                "group": group_name,
                "n_obs": int(mask.sum()),
                "label_manual_match_fraction": (
                    float(np.mean(manual_y[mask] == y[mask])) if mask.sum() else np.nan
                ),
                "mean_signed_mid_change": (
                    float(np.mean(group_change)) if mask.sum() else np.nan
                ),
                "median_signed_mid_change": (
                    float(np.median(group_change)) if mask.sum() else np.nan
                ),
                "fraction_abs_move_ge_1_spread": (
                    float(np.mean(group_spread_ratio >= 1.0)) if mask.sum() else np.nan
                ),
                "fraction_abs_move_ge_2_spreads": (
                    float(np.mean(group_spread_ratio >= 2.0)) if mask.sum() else np.nan
                ),
                "fraction_abs_move_ge_5_spreads": (
                    float(np.mean(group_spread_ratio >= 5.0)) if mask.sum() else np.nan
                ),
            }

            row.update(summarize_values(group_abs_change, "abs_mid_change"))
            row.update(summarize_values(group_spread_ratio, "abs_move_over_spread"))

            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "spread_relative_summary.csv", index=False)

    nonzero_summary = (
        summary[
            summary["group"].isin(["down", "up", "nonzero"])
        ]
        .sort_values(["horizon", "group"])
    )

    nonzero_summary.to_csv(output_dir / "spread_relative_nonzero_summary.csv", index=False)

    pd.DataFrame([spread_checks]).to_csv(output_dir / "spread_checks.csv", index=False)
    pd.DataFrame([event_range_checks]).to_csv(
        output_dir / "spread_event_range_checks.csv",
        index=False,
    )

    nonzero_only = nonzero_summary[nonzero_summary["group"] == "nonzero"]

    result = {
        **event_range_checks,
        **spread_checks,
    }

    for h in horizons:
        h_rows = nonzero_only[nonzero_only["horizon"] == h]
        if not h_rows.empty:
            result[f"h{h}_nonzero_abs_move_over_spread_median"] = float(
                h_rows["abs_move_over_spread_median"].iloc[0]
            )
            result[f"h{h}_nonzero_fraction_abs_move_ge_1_spread"] = float(
                h_rows["fraction_abs_move_ge_1_spread"].iloc[0]
            )
            result[f"h{h}_nonzero_fraction_abs_move_ge_2_spreads"] = float(
                h_rows["fraction_abs_move_ge_2_spreads"].iloc[0]
            )
            result[f"h{h}_nonzero_fraction_abs_move_ge_5_spreads"] = float(
                h_rows["fraction_abs_move_ge_5_spreads"].iloc[0]
            )

    return result


def classify_log_feature_precision(
    feature_time_checks: dict,
    float32_checks: dict,
    mid_return_warning_abs_tol: float,
    realized_vol_warning_abs_tol: float,
) -> dict:
    mid_float64_match = bool(feature_time_checks["mid_return_5_all_match_float64"])
    rv_float64_match = bool(feature_time_checks["realized_vol_20_all_match_float64"])

    max_mid_error = float(feature_time_checks["max_mid_return_5_abs_error"])
    max_rv_error = float(feature_time_checks["max_realized_vol_20_abs_error"])

    precision_issue_found = (not mid_float64_match) or (not rv_float64_match)

    mid_error_within_warning = max_mid_error <= mid_return_warning_abs_tol
    rv_error_within_warning = max_rv_error <= realized_vol_warning_abs_tol

    if not precision_issue_found:
        severity = "none"
        status = "pass"
        interpretation = (
            "log-return-derived features matched strict float64 manual recomputation "
            "on sampled rows."
        )
        critical_failure = False

    elif mid_error_within_warning and rv_error_within_warning:
        severity = "warning"
        status = "pass_with_precision_warning"
        interpretation = (
            "log-return-derived features did not exactly match strict float64 manual "
            "recomputation, but maximum absolute errors were within configured warning "
            "tolerances. This is treated as a numerical precision / reproducibility "
            "warning, not leakage evidence."
        )
        critical_failure = False

    else:
        severity = "critical"
        status = "fail_precision_check"
        interpretation = (
            "log-return-derived features did not match strict float64 manual "
            "recomputation and maximum absolute errors exceeded configured warning "
            "tolerances. This requires investigation before interpreting the result."
        )
        critical_failure = True

    return {
        "log_feature_precision_issue_found": precision_issue_found,
        "log_feature_precision_status": status,
        "log_feature_precision_severity": severity,
        "mid_return_5_float64_match": mid_float64_match,
        "realized_vol_20_float64_match": rv_float64_match,
        "max_mid_return_5_abs_error": max_mid_error,
        "max_realized_vol_20_abs_error": max_rv_error,
        "mid_return_warning_abs_tol": mid_return_warning_abs_tol,
        "realized_vol_warning_abs_tol": realized_vol_warning_abs_tol,
        "mid_return_error_within_warning_tol": mid_error_within_warning,
        "realized_vol_error_within_warning_tol": rv_error_within_warning,
        "mid_return_5_matches_local_float32_recompute": bool(
            float32_checks.get("mid_return_5_matches_local_float32_recompute", False)
        ),
        "realized_vol_20_matches_local_float32_recompute": bool(
            float32_checks.get("realized_vol_20_matches_local_float32_recompute", False)
        ),
        "max_mid_return_5_local_float32_abs_error": float(
            float32_checks.get("max_mid_return_5_local_float32_abs_error", np.nan)
        ),
        "max_realized_vol_20_local_float32_abs_error": float(
            float32_checks.get("max_realized_vol_20_local_float32_abs_error", np.nan)
        ),
        "precision_issue_interpretation": interpretation,
        "precision_issue_is_critical_failure": critical_failure,
    }


def build_critical_checks(
    label_checks: dict,
    boundary_checks: dict,
    feature_time_checks: dict,
    trade_flow_checks: dict,
    spread_info: dict,
) -> dict:
    return {
        "label_alignment_passed": label_checks["all_manual_labels_match_stored_labels"],
        "split_boundary_passed": (
            boundary_checks["train_split_present"]
            and boundary_checks["validation_split_present"]
            and boundary_checks["test_split_present"]
            and boundary_checks["train_boundary_drop_expected"]
            and boundary_checks["validation_boundary_drop_expected"]
            and boundary_checks["test_boundary_drop_0"]
            and boundary_checks["no_boundary_rows_are_model_eligible"]
            and boundary_checks["all_model_eligible_rows_are_feature_complete"]
        ),
        "quote_state_feature_timing_passed": (
            feature_time_checks["delta_bid_size_all_match"]
            and feature_time_checks["delta_ask_size_all_match"]
            and feature_time_checks["relative_spread_all_match"]
            and feature_time_checks["queue_imbalance_all_match"]
        ),
        "trade_flow_timing_passed": (
            trade_flow_checks["trade_intensity_all_match"]
            and trade_flow_checks["signed_count_imbalance_all_match"]
            and trade_flow_checks["signed_volume_imbalance_all_match"]
        ),
        "spread_event_range_passed": (
            spread_info["test_event_ids_monotonic"]
            and spread_info["test_event_ids_contiguous"]
            and spread_info["events_start_matches"]
            and spread_info["events_end_covers_future_horizon"]
            and spread_info["events_are_contiguous"]
            and spread_info["all_spreads_positive"]
        ),
    }


def build_spread_finding_text(
    spread_info: dict,
    horizons: tuple[int, ...],
) -> str:
    parts = []

    for h in horizons:
        key = f"h{h}_nonzero_abs_move_over_spread_median"
        if key in spread_info and pd.notna(spread_info[key]):
            parts.append(f"{spread_info[key]:.2f}x at h={h}")

    if parts:
        return (
            "nonzero labels often correspond to midprice moves larger than current spread; "
            "median nonzero abs(move)/spread is " + ", ".join(parts)
        )

    return (
        "spread-relative move-size summary was generated, but no nonzero median "
        "move/spread values were available."
    )


def write_experiment_record(
    output_dir: Path,
    experiment_id: str,
    symbol: str,
    start: str,
    end: str,
    horizons: tuple[int, ...],
    final_threshold_path: Path,
    primary_model: str,
    secondary_model: str,
) -> dict:
    record = {
        "experiment_id": experiment_id,
        "symbol": symbol,
        "start": start,
        "end": end,
        "horizons": ",".join(str(h) for h in horizons),
        "status": "final_test_evaluated_no_retuning",
        "important_note": (
            "The test set has been evaluated. No further threshold tuning, "
            "feature selection, model selection, or label changes should be made "
            "using this test result."
        ),
        "primary_model": primary_model,
        "secondary_model": secondary_model,
        "frozen_threshold_source": str(final_threshold_path.relative_to(PROJECT_ROOT)),
    }

    pd.DataFrame([record]).to_csv(output_dir / "experiment_record.csv", index=False)

    return record


def write_feature_precision_finding(
    output_dir: Path,
    precision_status: dict,
) -> dict:
    precision_issue_found = precision_status[
        "log_feature_precision_issue_found"
    ]
    feature_precision_finding = {
        "finding": (
            "log-return-derived features were checked against manual recomputation. "
            f"mid_return_5 strict float64 match: "
            f"{precision_status['mid_return_5_float64_match']}. "
            f"realized_vol_20 strict float64 match: "
            f"{precision_status['realized_vol_20_float64_match']}."
        ),
        "severity": precision_status["log_feature_precision_severity"],
        "status": precision_status["log_feature_precision_status"],
        "max_mid_return_5_abs_error": precision_status["max_mid_return_5_abs_error"],
        "max_realized_vol_20_abs_error": precision_status["max_realized_vol_20_abs_error"],
        "likely_cause": (
            "numerical precision, storage dtype, or implementation-window "
            "differences in log-return-derived features"
            if precision_issue_found
            else "not_applicable_no_precision_issue_detected"
        ),
        "impact_on_current_experiment": precision_status["precision_issue_interpretation"],
        "action_for_next_experiment": (
            "Standardize log-return-derived feature computation using explicit "
            "float64 arithmetic and rerun the audit before any new test "
            "evaluation."
            if precision_issue_found
            else "Retain explicit float64 arithmetic and the strict manual "
            "recomputation audit in future experiments."
        ),
    }

    pd.DataFrame([feature_precision_finding]).to_csv(
        output_dir / "feature_precision_finding.csv",
        index=False,
    )

    return feature_precision_finding


def write_final_conclusion(
    output_dir: Path,
    experiment_id: str,
    critical_audit_passed: bool,
    failed_critical_checks: list[str],
    precision_status: dict,
    spread_info: dict,
    horizons: tuple[int, ...],
) -> dict:
    if critical_audit_passed:
        if precision_status["log_feature_precision_issue_found"]:
            main_result_status = (
                "frozen_test_result_supported_by_audit_with_precision_caveat"
            )
        else:
            main_result_status = "frozen_test_result_supported_by_audit"

        leakage_evidence = "no_evidence_found"
        claim_allowed = (
            "can claim a statistical signal result for this frozen experiment, "
            "subject to the documented audit scope and limitations"
        )
        next_step = (
            "freeze the protocol and replicate on a longer untouched sample; "
            "do not retune on the already-evaluated test set"
        )

    else:
        main_result_status = "frozen_test_result_not_supported_by_audit"
        leakage_evidence = "failed_critical_audit_checks_require_investigation"
        claim_allowed = (
            "do not make a signal claim until failed critical audit checks are resolved"
        )
        next_step = (
            "inspect failed critical checks, fix the pipeline if needed, and rerun the audit"
        )

    if precision_status["log_feature_precision_issue_found"]:
        main_caveat = precision_status["precision_issue_interpretation"]
    else:
        main_caveat = (
            "no critical issue found within the sampled audit scope; this does "
            "not establish tradability or exclude every possible implementation error"
        )

    conclusion = {
        "experiment_id": experiment_id,
        "audit_status": "complete",
        "main_result_status": main_result_status,
        "critical_audit_passed": critical_audit_passed,
        "failed_critical_checks": "|".join(failed_critical_checks),
        "leakage_evidence": leakage_evidence,
        "main_caveat": main_caveat,
        "precision_issue_interpretation": precision_status["precision_issue_interpretation"],
        "spread_relative_finding": build_spread_finding_text(spread_info, horizons),
        "claim_allowed": claim_allowed,
        "claim_not_allowed": (
            "cannot claim tradability or PnL without execution-aware backtest, fees, "
            "latency, queue position, slippage, and adverse selection modeling"
        ),
        "next_step": next_step,
    }

    pd.DataFrame([conclusion]).to_csv(
        output_dir / "final_audit_conclusion.csv",
        index=False,
    )

    return conclusion


def main() -> None:
    args = parse_args()

    validate_protocol_symbol(args.symbol)
    symbol = args.symbol
    start = parse_date(args.start).strftime("%Y-%m-%d")
    end = parse_date(args.end).strftime("%Y-%m-%d")
    horizons = parse_horizons(args.horizons)

    experiment_id = make_experiment_id(
        symbol=symbol,
        start=start,
        end=end,
        experiment_tag=args.experiment_tag,
        experiment_id=args.experiment_id,
    )

    data_dir = PROJECT_ROOT / "data"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    results_dir = PROJECT_ROOT / "outputs" / "results"

    events_path = processed_dir / f"quote_events_{symbol}_{start}_to_{end}.parquet"
    trades_path = interim_dir / f"raw_trades_{symbol}_{start}_to_{end}.parquet"
    model_stem = tagged_artifact_stem(
        "model_dataset",
        symbol,
        start,
        end,
        args.artifact_tag,
    )
    model_path = resolve_model_dataset(processed_dir, model_stem).path

    target_diag_stem = tagged_artifact_stem(
        "target_diagnostics",
        symbol,
        start,
        end,
        args.artifact_tag,
    )
    target_diag_path = reports_dir / f"{target_diag_stem}.csv"

    final_prefix = args.final_prefix or tagged_artifact_stem(
        "final_test",
        symbol,
        start,
        end,
        args.artifact_tag,
    )

    final_daily_path = results_dir / f"{final_prefix}_daily_blocks.csv"
    final_confusion_path = results_dir / f"{final_prefix}_confusion_counts.csv"
    final_threshold_path = reports_dir / f"{final_prefix}_frozen_thresholds.csv"
    final_coefficients_path = results_dir / f"{final_prefix}_logistic_coefficients.csv"

    for path in [
        events_path,
        trades_path,
        model_path,
        target_diag_path,
        final_daily_path,
        final_confusion_path,
        final_threshold_path,
        final_coefficients_path,
    ]:
        ensure_exists(path)

    if args.output_dir is None:
        output_dir = reports_dir / "research_audit" / experiment_id
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print("=" * 80)
    print("RESEARCH AUDIT")
    print("=" * 80)
    print(f"Experiment: {experiment_id}")
    print(f"Symbol:     {symbol}")
    print(f"Start:      {start}")
    print(f"End:        {end}")
    print(f"Horizons:   {horizons}")
    print(f"Output dir: {output_dir}")

    experiment_record = write_experiment_record(
        output_dir=output_dir,
        experiment_id=experiment_id,
        symbol=symbol,
        start=start,
        end=end,
        horizons=horizons,
        final_threshold_path=final_threshold_path,
        primary_model=args.primary_model,
        secondary_model=args.secondary_model,
    )

    label_checks = audit_label_alignment(
        model_path=model_path,
        events_path=events_path,
        rng=rng,
        output_dir=output_dir,
        sample_size=args.sample_size,
        horizons=horizons,
    )

    boundary_checks = audit_split_boundaries(
        model_path=model_path,
        output_dir=output_dir,
        horizons=horizons,
        expected_boundary_drop=args.expected_boundary_drop,
    )

    feature_time_checks, problem_event_ids = audit_feature_time(
        model_path=model_path,
        events_path=events_path,
        rng=rng,
        output_dir=output_dir,
        sample_size=args.sample_size,
    )

    float32_checks = audit_local_float32_log_recompute(
        model_path=model_path,
        events_path=events_path,
        problem_event_ids=problem_event_ids,
        output_dir=output_dir,
    )

    trade_flow_checks = audit_trade_flow_timing(
        model_path=model_path,
        trades_path=trades_path,
        rng=rng,
        output_dir=output_dir,
        sample_size=args.sample_size,
    )

    target_drift_info = build_target_drift_summary(
        target_diag_path=target_diag_path,
        output_dir=output_dir,
    )

    daily_info = build_daily_stability_summary(
        final_daily_path=final_daily_path,
        output_dir=output_dir,
    )

    coefficient_info = build_coefficient_interpretation(
        coefficient_path=final_coefficients_path,
        output_dir=output_dir,
        primary_model=args.primary_model,
        horizons=horizons,
    )

    confusion_info = build_confusion_recall_comparison(
        final_confusion_path=final_confusion_path,
        output_dir=output_dir,
        model_names=(args.primary_model, args.secondary_model),
        horizons=horizons,
    )

    spread_info = build_spread_relative_summary(
        model_path=model_path,
        events_path=events_path,
        output_dir=output_dir,
        horizons=horizons,
    )

    critical_checks = build_critical_checks(
        label_checks=label_checks,
        boundary_checks=boundary_checks,
        feature_time_checks=feature_time_checks,
        trade_flow_checks=trade_flow_checks,
        spread_info=spread_info,
    )

    precision_status = classify_log_feature_precision(
        feature_time_checks=feature_time_checks,
        float32_checks=float32_checks,
        mid_return_warning_abs_tol=args.mid_return_warning_abs_tol,
        realized_vol_warning_abs_tol=args.realized_vol_warning_abs_tol,
    )

    failed_critical_checks = [
        name for name, passed in critical_checks.items()
        if not bool(passed)
    ]

    if precision_status["precision_issue_is_critical_failure"]:
        failed_critical_checks.append("log_feature_precision_check")

    critical_audit_passed = len(failed_critical_checks) == 0

    feature_precision_finding = write_feature_precision_finding(
        output_dir=output_dir,
        precision_status=precision_status,
    )

    final_conclusion = write_final_conclusion(
        output_dir=output_dir,
        experiment_id=experiment_id,
        critical_audit_passed=critical_audit_passed,
        failed_critical_checks=failed_critical_checks,
        precision_status=precision_status,
        spread_info=spread_info,
        horizons=horizons,
    )

    audit_summary = {
        **experiment_record,
        **label_checks,
        **boundary_checks,
        **feature_time_checks,
        **float32_checks,
        **trade_flow_checks,
        **target_drift_info,
        **daily_info,
        **coefficient_info,
        **confusion_info,
        **spread_info,
        **critical_checks,
        **precision_status,
        "critical_audit_passed": critical_audit_passed,
        "failed_critical_checks": (
            "|".join(failed_critical_checks) if failed_critical_checks else ""
        ),
        "feature_precision_finding_saved": True,
        "feature_precision_finding_status": feature_precision_finding["status"],
        **final_conclusion,
    }

    pd.DataFrame([audit_summary]).to_csv(
        output_dir / "audit_summary.csv",
        index=False,
    )

    json_summary = {
        key: json_safe_scalar(value)
        for key, value in audit_summary.items()
    }
    with open(output_dir / "audit_summary.json", "w") as f:
        json.dump(json_summary, f, indent=2, allow_nan=False)

    print()
    print("Audit summary:")
    for key, value in audit_summary.items():
        print(f"{key}: {value}")

    print()
    print("Done. Research audit completed.")


if __name__ == "__main__":
    main()
