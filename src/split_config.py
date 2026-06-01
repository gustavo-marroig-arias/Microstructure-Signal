from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplitConfig:
    """
    Chronological split configuration.

    Protocol:
    - train: first 60%
    - validation: next 20%
    - test: final 20%
    - split by event timestamp, not shuffled row order
    - drop last 50 labeled observations of train
    - drop last 50 labeled observations of validation
    """
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    boundary_drop_events: int = 50


def validate_feature_table(feature_table: pd.DataFrame) -> None:
    """
    Checks that feature table has the columns needed for splitting.
    """
    required = [
        "event_id",
        "timestamp",
        "feature_complete",
        "y_10",
        "y_20",
        "y_50",
    ]

    missing = sorted(set(required) - set(feature_table.columns))

    if missing:
        raise ValueError(f"feature_table missing required columns: {missing}")

    if feature_table.empty:
        raise ValueError("feature_table is empty.")

    expected_event_ids = pd.Series(
        range(len(feature_table)),
        index=feature_table.index,
        name="event_id",
    )

    if not feature_table["event_id"].equals(expected_event_ids):
        raise ValueError("event_id must be exactly 0, 1, 2, ..., len(feature_table)-1.")

    if feature_table["timestamp"].isna().any():
        raise ValueError("timestamp contains missing values.")

    if not feature_table["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamp must be monotonic increasing.")

    if feature_table["feature_complete"].isna().any():
        raise ValueError("feature_complete contains missing values.")

    label_cols = ["y_10", "y_20", "y_50"]
    if feature_table[label_cols].isna().any().any():
        raise ValueError("label columns contain missing values.")


def sample_window_from_dates(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Converts inclusive YYYY-MM-DD start/end dates into a UTC sample window.

    Example:
        start = 2024-03-01
        end   = 2024-03-03

    Returns:
        sample_start = 2024-03-01 00:00:00 UTC
        sample_end_exclusive = 2024-03-04 00:00:00 UTC
    """
    sample_start = pd.Timestamp(start, tz="UTC")
    sample_end_exclusive = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)

    if sample_end_exclusive <= sample_start:
        raise ValueError("end date must be >= start date.")

    return sample_start, sample_end_exclusive


def compute_time_boundaries(
    start: str,
    end: str,
    config: ChronologicalSplitConfig = ChronologicalSplitConfig(),
) -> dict:
    """
    Computes timestamp split boundaries from the full sample time window.

    This avoids splitting by model-filtered row count.
    """
    if config.train_fraction + config.validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1.0")
    
    sample_start, sample_end_exclusive = sample_window_from_dates(start, end)

    total_duration = sample_end_exclusive - sample_start

    train_end = sample_start + config.train_fraction * total_duration

    validation_end = sample_start + (
        config.train_fraction + config.validation_fraction
    ) * total_duration

    return {
        "sample_start": sample_start,
        "train_end": train_end,
        "validation_end": validation_end,
        "sample_end_exclusive": sample_end_exclusive,
    }


def assign_chronological_split(
    feature_table: pd.DataFrame,
    start: str,
    end: str,
    config: ChronologicalSplitConfig = ChronologicalSplitConfig(),
) -> pd.DataFrame:
    """
    Assigns train / validation / test split based on event timestamp.

    Rules:
    - timestamp < train_end -> train
    - train_end <= timestamp < validation_end -> validation
    - validation_end <= timestamp < sample_end_exclusive -> test
    """
    validate_feature_table(feature_table)

    boundaries = compute_time_boundaries(start, end, config=config)

    out = feature_table.copy()

    out["split"] = pd.NA

    train_mask = (
        (out["timestamp"] >= boundaries["sample_start"])
        & (out["timestamp"] < boundaries["train_end"])
    )

    validation_mask = (
        (out["timestamp"] >= boundaries["train_end"])
        & (out["timestamp"] < boundaries["validation_end"])
    )

    test_mask = (
        (out["timestamp"] >= boundaries["validation_end"])
        & (out["timestamp"] < boundaries["sample_end_exclusive"])
    )

    out.loc[train_mask, "split"] = "train"
    out.loc[validation_mask, "split"] = "validation"
    out.loc[test_mask, "split"] = "test"

    missing_split = out["split"].isna().sum()

    if missing_split > 0:
        raise ValueError(
            f"{missing_split} rows were not assigned to a split. "
            "Check sample start/end dates and timestamps."
        )

    return out


def mark_boundary_drops(
    split_table: pd.DataFrame,
    config: ChronologicalSplitConfig = ChronologicalSplitConfig(),
) -> pd.DataFrame:
    """
    Marks the last N labeled observations of train and validation as boundary drops.

    This prevents target spillover across split boundaries.

    Protocol:
    - drop last 50 labeled observations of training
    - drop last 50 labeled observations of validation
    - do not drop the first 50 observations after each boundary
    """
    out = split_table.copy()

    out["is_boundary_drop"] = False

    for split_name in ["train", "validation"]:
        split_idx = out.index[out["split"] == split_name]

        if len(split_idx) == 0:
            raise ValueError(f"No rows found for split: {split_name}")

        n_drop = min(config.boundary_drop_events, len(split_idx))

        if n_drop < config.boundary_drop_events:
            print(
                f"Warning: split {split_name} has only {len(split_idx)} rows; "
                f"marking {n_drop} rows as boundary drops."
            )

        drop_idx = split_idx[-n_drop:]

        out.loc[drop_idx, "is_boundary_drop"] = True

    return out


def add_model_eligibility(split_table: pd.DataFrame) -> pd.DataFrame:
    """
    Adds model_eligible flag.

    A row is model-eligible if:
    - all feature lookbacks are complete
    - it is not a split-boundary drop row
    """
    out = split_table.copy()

    out["model_eligible"] = (
        out["feature_complete"].astype(bool)
        & ~out["is_boundary_drop"].astype(bool)
    )

    return out


def build_model_dataset(
    feature_table: pd.DataFrame,
    start: str,
    end: str,
    config: ChronologicalSplitConfig = ChronologicalSplitConfig(),
) -> tuple[pd.DataFrame, dict]:
    """
    Full split-building pipeline.

    Returns:
        model_dataset
        split boundary dictionary
    """
    with_splits = assign_chronological_split(
        feature_table,
        start=start,
        end=end,
        config=config,
    )

    with_boundary_drops = mark_boundary_drops(with_splits, config=config)

    model_dataset = add_model_eligibility(with_boundary_drops)

    boundaries = compute_time_boundaries(start, end, config=config)

    return model_dataset, boundaries


def split_summary(model_dataset: pd.DataFrame, boundaries: dict) -> pd.DataFrame:
    """
    Summarizes rows by split.
    """
    rows = []

    for split_name in ["train", "validation", "test"]:
        g = model_dataset[model_dataset["split"] == split_name]

        rows.append(
            {
                "split": split_name,
                "rows_total": int(len(g)),
                "feature_complete_rows": int(g["feature_complete"].sum()),
                "boundary_drop_rows": int(g["is_boundary_drop"].sum()),
                "model_eligible_rows": int(g["model_eligible"].sum()),
                "timestamp_start": g["timestamp"].min(),
                "timestamp_end": g["timestamp"].max(),
                "y_10_down": int((g["y_10"] == -1).sum()),
                "y_10_unchanged": int((g["y_10"] == 0).sum()),
                "y_10_up": int((g["y_10"] == 1).sum()),
                "y_20_down": int((g["y_20"] == -1).sum()),
                "y_20_unchanged": int((g["y_20"] == 0).sum()),
                "y_20_up": int((g["y_20"] == 1).sum()),
                "y_50_down": int((g["y_50"] == -1).sum()),
                "y_50_unchanged": int((g["y_50"] == 0).sum()),
                "y_50_up": int((g["y_50"] == 1).sum()),
            }
        )

    boundary_rows = [
        {
            "split": "BOUNDARIES",
            "rows_total": pd.NA,
            "feature_complete_rows": pd.NA,
            "boundary_drop_rows": pd.NA,
            "model_eligible_rows": pd.NA,
            "timestamp_start": boundaries["sample_start"],
            "timestamp_end": boundaries["sample_end_exclusive"],
            "y_10_down": pd.NA,
            "y_10_unchanged": pd.NA,
            "y_10_up": pd.NA,
            "y_20_down": pd.NA,
            "y_20_unchanged": pd.NA,
            "y_20_up": pd.NA,
            "y_50_down": pd.NA,
            "y_50_unchanged": pd.NA,
            "y_50_up": pd.NA,
        },
        {
            "split": "TRAIN_END",
            "rows_total": pd.NA,
            "feature_complete_rows": pd.NA,
            "boundary_drop_rows": pd.NA,
            "model_eligible_rows": pd.NA,
            "timestamp_start": boundaries["train_end"],
            "timestamp_end": boundaries["train_end"],
            "y_10_down": pd.NA,
            "y_10_unchanged": pd.NA,
            "y_10_up": pd.NA,
            "y_20_down": pd.NA,
            "y_20_unchanged": pd.NA,
            "y_20_up": pd.NA,
            "y_50_down": pd.NA,
            "y_50_unchanged": pd.NA,
            "y_50_up": pd.NA,
        },
        {
            "split": "VALIDATION_END",
            "rows_total": pd.NA,
            "feature_complete_rows": pd.NA,
            "boundary_drop_rows": pd.NA,
            "model_eligible_rows": pd.NA,
            "timestamp_start": boundaries["validation_end"],
            "timestamp_end": boundaries["validation_end"],
            "y_10_down": pd.NA,
            "y_10_unchanged": pd.NA,
            "y_10_up": pd.NA,
            "y_20_down": pd.NA,
            "y_20_unchanged": pd.NA,
            "y_20_up": pd.NA,
            "y_50_down": pd.NA,
            "y_50_unchanged": pd.NA,
            "y_50_up": pd.NA,
        },
    ]

    return pd.DataFrame(rows + boundary_rows)


def save_model_dataset(model_dataset: pd.DataFrame, output_path: Path) -> None:
    """
    Saves model dataset as parquet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dataset.to_parquet(output_path, index=False)
    print(f"Saved model dataset: {output_path}")


def save_split_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """
    Saves split summary as CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved split summary: {output_path}")