from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_HORIZONS = (10, 20, 50)

DEFAULT_FEATURE_COLUMNS = [
    "relative_spread",
    "queue_imbalance",
    "log_bid_size",
    "log_ask_size",
    "delta_bid_size",
    "delta_ask_size",
    "mid_return_5",
    "realized_vol_20",
    "trade_intensity_1s",
    "signed_trade_count_imbalance_1s",
    "signed_trade_volume_imbalance_1s",
]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Computes SHA256 hash of a file.

    This helps freeze the dataset version before modeling.
    """
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def validate_model_dataset(
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
) -> None:
    """
    Checks that the model dataset has the expected columns and flags.
    """
    required = [
        "event_id",
        "timestamp",
        "split",
        "feature_complete",
        "is_boundary_drop",
        "model_eligible",
    ]

    required += [f"y_{h}" for h in horizons]
    required += list(feature_columns)

    missing = sorted(set(required) - set(data.columns))

    if missing:
        raise ValueError(f"model dataset missing required columns: {missing}")

    if data.empty:
        raise ValueError("model dataset is empty.")

    expected_splits = {"train", "validation", "test"}
    observed_splits = set(data["split"].dropna().unique())

    if observed_splits != expected_splits:
        raise ValueError(
            f"Expected splits {expected_splits}, observed {observed_splits}"
        )

    expected_event_ids = pd.Series(
        range(len(data)),
        index=data.index,
        name="event_id",
    )

    if not data["event_id"].equals(expected_event_ids):
        raise ValueError("event_id must be exactly 0, 1, 2, ..., len(data)-1.")

    if data["timestamp"].isna().any():
        raise ValueError("timestamp contains missing values.")

    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamp must be monotonic increasing.")

    if data["model_eligible"].isna().any():
        raise ValueError("model_eligible contains missing values.")

    if data["feature_complete"].isna().any():
        raise ValueError("feature_complete contains missing values.")

    label_cols = [f"y_{h}" for h in horizons]
    if data[label_cols].isna().any().any():
        raise ValueError("label columns contain missing values.")


def build_dataset_manifest(
        data: pd.DataFrame,
        dataset_path: Path,
        start: str,
        end: str,
        symbol: str = "BTCUSDT",
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        feature_columns: list[str] = DEFAULT_FEATURE_COLUMNS,
) -> dict:
    """
    Creates a dataset manifest dictionary.

    This is not analysis. It is a record of exactly what dataset version
    was used before modeling started.
    """
    validate_model_dataset(data, horizons=horizons, feature_columns=feature_columns)

    split_counts = (
        data.groupby("split")
        .agg(
            rows_total=("event_id", "count"),
            feature_complete_rows=("feature_complete", "sum"),
            boundary_drop_rows=("is_boundary_drop", "sum"),
            model_eligible_rows=("model_eligible", "sum"),
            timestamp_start=("timestamp", "min"),
            timestamp_end=("timestamp", "max"),
        )
        .reset_index()
    )

    split_counts_records = split_counts.copy()

    for col in ["timestamp_start", "timestamp_end"]:
        split_counts_records[col] = split_counts_records[col].astype(str)

    manifest = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "n_rows_total": int(len(data)),
        "horizons": list(horizons),
        "feature_columns": feature_columns,
        "label_columns": [f"y_{h}" for h in horizons],
        "split_column": "split",
        "eligibility_column": "model_eligible",
        "feature_complete_column": "feature_complete",
        "boundary_drop_column": "is_boundary_drop",
        "modeling_rule": "Use rows where model_eligible == True.",
        "normalization_rule": "Fit scalers on train only; apply to validation and test.",
        "class_weight_rule": "No class weights in primary models.",
        "split_counts": split_counts_records.to_dict(orient="records"),
    }

    return manifest


def save_dataset_manifest(manifest: dict, output_path: Path) -> None:
    """
    Saves dataset manifest as JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved dataset manifest: {output_path}")


def target_diagnostics_by_split(
        data: pd.DataFrame,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        eligible_only: bool = True
    ) -> pd.DataFrame:
    """
    Computes target diagnostics for each horizon and split.

    For each split and horizon, reports:
    - n observations
    - class counts
    - class proportions
    - non-zero fraction
    - majority class
    - majority-class accuracy
    """
    validate_model_dataset(data, horizons=horizons)

    if eligible_only:
        d = data.loc[data["model_eligible"]].copy()
    else:
        d = data.copy()

    rows = []

    for split_name in ["train", "validation", "test"]:
        split_df = d.loc[d["split"] == split_name].copy()

        for h in horizons:
            label_col = f"y_{h}"

            y = split_df[label_col]

            n = int(len(y))

            count_down = int((y == -1).sum())
            count_unchanged = int((y == 0).sum())
            count_up = int((y == 1).sum())

            if n == 0:
                prop_down = float("nan")
                prop_unchanged = float("nan")
                prop_up = float("nan")
                nonzero_fraction = float("nan")
                majority_class = None
                majority_class_accuracy = float("nan")
            else:
                prop_down = count_down / n
                prop_unchanged = count_unchanged / n
                prop_up = count_up / n
                nonzero_fraction = (count_down + count_up) / n

                counts = {
                    -1: count_down,
                    0: count_unchanged,
                    1: count_up,
                }

                majority_class = max(counts, key=counts.get)
                majority_class_accuracy = counts[majority_class] / n
            
            rows.append(
                {
                    "split": split_name,
                    "horizon": h,
                    "eligible_only": eligible_only,
                    "n_obs": n,
                    "count_down": count_down,
                    "count_unchanged": count_unchanged,
                    "count_up": count_up,
                    "prop_down": prop_down,
                    "prop_unchanged": prop_unchanged,
                    "prop_up": prop_up,
                    "nonzero_fraction": nonzero_fraction,
                    "majority_class": majority_class,
                    "majority_class_accuracy": majority_class_accuracy,
                }
            )

    return pd.DataFrame(rows)

def save_target_diagnostics(diagnostics: pd.DataFrame, output_path: Path) -> None:
    """
    Saves target diagnostics as CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_path, index=False)
    print(f"Saved target diagnostics: {output_path}")      