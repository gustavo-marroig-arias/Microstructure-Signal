from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.model_dataset_io import (
    ModelDatasetLocation,
    iter_model_split_batches,
)
from src.protocol import DEFAULT_HORIZONS, FEATURE_COLUMNS


DEFAULT_FEATURE_COLUMNS = FEATURE_COLUMNS
SPLIT_ORDER = ("train", "validation", "test")


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
        feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
        dataset_sha256_override: str | None = None,
) -> dict:
    """
    Creates a dataset manifest dictionary.

    This is not analysis. It is a record of exactly what dataset version
    was used before modeling started.
    """
    validate_model_dataset(data, horizons=horizons, feature_columns=())

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
        "dataset_sha256": (
            dataset_sha256_override
            if dataset_sha256_override is not None
            else file_sha256(dataset_path)
        ),
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


def streaming_target_diagnostics(
    location: ModelDatasetLocation,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    batch_size: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Aggregate manifest and eligible-label statistics without full loading."""
    columns = [
        "event_id",
        "timestamp",
        "split",
        "feature_complete",
        "is_boundary_drop",
        "model_eligible",
        *(f"y_{horizon}" for horizon in horizons),
    ]
    split_states = {
        split: {
            "rows_total": 0,
            "feature_complete_rows": 0,
            "boundary_drop_rows": 0,
            "model_eligible_rows": 0,
            "timestamp_start": None,
            "timestamp_end": None,
            "label_counts": {
                horizon: {-1: 0, 0: 0, 1: 0}
                for horizon in horizons
            },
        }
        for split in SPLIT_ORDER
    }
    previous_event_id: int | None = None
    previous_timestamp: pd.Timestamp | None = None

    for split in SPLIT_ORDER:
        state = split_states[split]
        for frame in iter_model_split_batches(
            location,
            split,
            columns,
            eligible_only=False,
            batch_size=batch_size,
        ):
            event_ids = frame["event_id"].to_numpy(dtype=np.int64, copy=False)
            expected_first = (
                0 if previous_event_id is None else previous_event_id + 1
            )
            if event_ids[0] != expected_first or (
                len(event_ids) > 1
                and not bool(np.all(np.diff(event_ids) == 1))
            ):
                raise ValueError(
                    "event_id is not globally contiguous while streaming "
                    f"target diagnostics at split {split!r}."
                )
            timestamps = pd.to_datetime(frame["timestamp"], utc=True)
            if not timestamps.is_monotonic_increasing or (
                previous_timestamp is not None
                and timestamps.iloc[0] < previous_timestamp
            ):
                raise ValueError(
                    "timestamp is not globally monotone while streaming "
                    "target diagnostics."
                )
            if frame["feature_complete"].isna().any():
                raise ValueError("feature_complete contains missing values.")
            if frame["is_boundary_drop"].isna().any():
                raise ValueError("is_boundary_drop contains missing values.")
            if frame["model_eligible"].isna().any():
                raise ValueError("model_eligible contains missing values.")
            for flag in (
                "feature_complete",
                "is_boundary_drop",
                "model_eligible",
            ):
                if not pd.api.types.is_bool_dtype(frame[flag]):
                    raise TypeError(f"{flag} must have boolean dtype.")
            expected_eligibility = (
                frame["feature_complete"] & ~frame["is_boundary_drop"]
            )
            if not frame["model_eligible"].equals(expected_eligibility):
                raise ValueError(
                    "model_eligible must equal feature_complete AND NOT "
                    "is_boundary_drop."
                )

            state["rows_total"] += len(frame)
            state["feature_complete_rows"] += int(
                frame["feature_complete"].sum()
            )
            state["boundary_drop_rows"] += int(
                frame["is_boundary_drop"].sum()
            )
            state["model_eligible_rows"] += int(
                frame["model_eligible"].sum()
            )
            if state["timestamp_start"] is None:
                state["timestamp_start"] = timestamps.iloc[0]
            state["timestamp_end"] = timestamps.iloc[-1]

            for horizon in horizons:
                all_labels = frame[f"y_{horizon}"]
                if all_labels.isna().any():
                    raise ValueError(f"y_{horizon} contains missing values.")
                observed = set(all_labels.unique())
                if not observed.issubset({-1, 0, 1}):
                    raise ValueError(
                        f"y_{horizon} contains invalid labels: {observed}"
                    )
                labels = all_labels.loc[frame["model_eligible"]]
                if labels.isna().any():
                    raise ValueError(f"y_{horizon} contains missing values.")
                counts = labels.value_counts()
                for label in (-1, 0, 1):
                    state["label_counts"][horizon][label] += int(
                        counts.get(label, 0)
                    )

            previous_event_id = int(event_ids[-1])
            previous_timestamp = timestamps.iloc[-1]

    split_summary = pd.DataFrame(
        [
            {
                "split": split,
                **{
                    key: value
                    for key, value in state.items()
                    if key != "label_counts"
                },
            }
            for split, state in split_states.items()
        ]
    )
    diagnostic_rows = []
    for split, state in split_states.items():
        for horizon in horizons:
            counts = state["label_counts"][horizon]
            n = sum(counts.values())
            majority_class = max(counts, key=counts.get) if n else None
            diagnostic_rows.append(
                {
                    "split": split,
                    "horizon": horizon,
                    "eligible_only": True,
                    "n_obs": n,
                    "count_down": counts[-1],
                    "count_unchanged": counts[0],
                    "count_up": counts[1],
                    "prop_down": counts[-1] / n if n else np.nan,
                    "prop_unchanged": counts[0] / n if n else np.nan,
                    "prop_up": counts[1] / n if n else np.nan,
                    "nonzero_fraction": (
                        (counts[-1] + counts[1]) / n if n else np.nan
                    ),
                    "majority_class": majority_class,
                    "majority_class_accuracy": (
                        counts[majority_class] / n
                        if majority_class is not None
                        else np.nan
                    ),
                }
            )

    total_rows = sum(state["rows_total"] for state in split_states.values())
    return pd.DataFrame(diagnostic_rows), split_summary, total_rows


def build_streaming_dataset_manifest(
    *,
    split_summary: pd.DataFrame,
    total_rows: int,
    dataset_path: Path,
    dataset_sha256: str,
    start: str,
    end: str,
    symbol: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
) -> dict:
    records = split_summary.copy()
    for column in ("timestamp_start", "timestamp_end"):
        records[column] = records[column].astype(str)
    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "n_rows_total": total_rows,
        "horizons": list(horizons),
        "feature_columns": feature_columns,
        "label_columns": [f"y_{horizon}" for horizon in horizons],
        "split_column": "split",
        "eligibility_column": "model_eligible",
        "feature_complete_column": "feature_complete",
        "boundary_drop_column": "is_boundary_drop",
        "modeling_rule": "Use rows where model_eligible == True.",
        "normalization_rule": (
            "Fit scalers on train only; apply to validation and test."
        ),
        "class_weight_rule": "No class weights in primary models.",
        "split_counts": records.to_dict(orient="records"),
    }


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
    validate_model_dataset(data, horizons=horizons, feature_columns=())

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
