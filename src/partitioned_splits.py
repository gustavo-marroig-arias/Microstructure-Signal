from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.model_dataset_io import VALID_SPLITS
from src.protocol import DEFAULT_HORIZONS, ExperimentSpec, FEATURE_COLUMNS
from src.split_config import (
    DEFAULT_SPLIT_CONFIG,
    ChronologicalSplitConfig,
    compute_time_boundaries,
)


REQUIRED_SPLIT_COLUMNS = (
    "event_id",
    "timestamp",
    "feature_complete",
    "has_full_trade_lookback_1s",
    "y_10",
    "y_20",
    "y_50",
    *FEATURE_COLUMNS,
)


def build_partitioned_model_dataset(
    feature_path: Path,
    output_dir: Path,
    *,
    start: str,
    end: str,
    config: ChronologicalSplitConfig = DEFAULT_SPLIT_CONFIG,
    overwrite: bool = False,
    compression: str = "zstd",
) -> tuple[pd.DataFrame, dict[str, pd.Timestamp], Path]:
    """
    Build split-specific parquet files without loading the full feature table.

    The first pass validates global ordering and identifies the exact final
    event IDs in train and validation. The second pass adds split and
    eligibility flags and writes each chronological partition independently.
    """
    feature_path = Path(feature_path)
    output_dir = Path(output_dir)

    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature table: {feature_path}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Pass overwrite=True only after verifying the target."
        )
    if config.boundary_drop_events < max(DEFAULT_HORIZONS):
        raise ValueError(
            "boundary_drop_events does not cover the maximum label horizon."
        )

    parquet_file = pq.ParquetFile(feature_path)
    schema_columns = set(parquet_file.schema_arrow.names)
    missing = sorted(set(REQUIRED_SPLIT_COLUMNS) - schema_columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")

    boundaries = compute_time_boundaries(start, end, config=config)
    scan = _scan_split_boundaries(parquet_file, boundaries)
    undersized = {
        split: scan["counts"][split]
        for split in ("train", "validation")
        if scan["counts"][split] < config.boundary_drop_events
    }
    if undersized:
        raise ValueError(
            "Train/validation splits must each contain at least "
            f"{config.boundary_drop_events} observations for the boundary "
            f"embargo. Observed: {undersized}"
        )
    boundary_drop_start = {
        split: scan["last_event_id"][split] - config.boundary_drop_events + 1
        for split in ("train", "validation")
    }

    staging_dir = output_dir.with_name(
        f".{output_dir.name}.tmp-{uuid4().hex}"
    )
    staging_dir.mkdir(parents=True, exist_ok=False)

    writers: dict[str, pq.ParquetWriter] = {}
    summary_state = {
        split: _empty_summary_state()
        for split in VALID_SPLITS
    }

    try:
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_index)
            frame = table.to_pandas()
            _validate_feature_chunk(frame, row_group_index)
            split_values = _assign_split_values(frame["timestamp"], boundaries)

            for split in VALID_SPLITS:
                mask = split_values == split
                if not bool(mask.any()):
                    continue

                part = frame.loc[mask].copy()
                part["split"] = pd.Categorical(
                    [split] * len(part),
                    categories=list(VALID_SPLITS),
                    ordered=True,
                )
                if split in boundary_drop_start:
                    part["is_boundary_drop"] = (
                        part["event_id"].to_numpy(dtype=np.int64)
                        >= boundary_drop_start[split]
                    )
                else:
                    part["is_boundary_drop"] = False
                part["model_eligible"] = (
                    part["feature_complete"].astype(bool)
                    & ~part["is_boundary_drop"]
                )

                _update_summary_state(summary_state[split], part)
                arrow_part = pa.Table.from_pandas(
                    part,
                    preserve_index=False,
                )
                if split not in writers:
                    writers[split] = pq.ParquetWriter(
                        staging_dir / f"{split}.parquet",
                        arrow_part.schema,
                        compression=compression,
                        use_dictionary=["split"],
                    )
                writers[split].write_table(arrow_part)

        for writer in writers.values():
            writer.close()
        writers.clear()

        missing_partitions = [
            split
            for split in VALID_SPLITS
            if not (staging_dir / f"{split}.parquet").exists()
        ]
        if missing_partitions:
            raise ValueError(
                f"No rows were written for partitions: {missing_partitions}"
            )

        summary = _summary_frame(summary_state, boundaries)
        _validate_partition_counts(
            summary,
            scan,
            boundary_drop_events=config.boundary_drop_events,
        )
        manifest_path = staging_dir / "manifest.json"
        _write_manifest(
            manifest_path,
            feature_path=feature_path,
            config=config,
            boundaries=boundaries,
            scan=scan,
            summary=summary,
        )

        _replace_directory_transactionally(staging_dir, output_dir)
    except Exception:
        for writer in writers.values():
            writer.close()
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return summary, boundaries, output_dir / "manifest.json"


def _replace_directory_transactionally(
    staging_dir: Path,
    output_dir: Path,
) -> None:
    backup_dir: Path | None = None
    if output_dir.exists():
        backup_dir = output_dir.with_name(
            f".{output_dir.name}.backup-{uuid4().hex}"
        )
        os.replace(output_dir, backup_dir)

    try:
        os.replace(staging_dir, output_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    else:
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)


def _scan_split_boundaries(
    parquet_file: pq.ParquetFile,
    boundaries: dict[str, pd.Timestamp],
) -> dict[str, Any]:
    counts = {split: 0 for split in VALID_SPLITS}
    last_event_id: dict[str, int] = {}
    previous_event_id: int | None = None
    previous_timestamp: pd.Timestamp | None = None

    for row_group_index in range(parquet_file.num_row_groups):
        frame = parquet_file.read_row_group(
            row_group_index,
            columns=["event_id", "timestamp"],
        ).to_pandas()

        if frame.empty:
            continue
        event_ids = frame["event_id"].to_numpy(dtype=np.int64, copy=False)
        if event_ids[0] != (0 if previous_event_id is None else previous_event_id + 1):
            raise ValueError(
                "event_id is not globally contiguous at row group "
                f"{row_group_index}: first={event_ids[0]}, "
                f"previous={previous_event_id}."
            )
        if len(event_ids) > 1 and not bool(np.all(np.diff(event_ids) == 1)):
            raise ValueError(
                f"event_id is not contiguous inside row group {row_group_index}."
            )

        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                f"timestamp is not monotone in row group {row_group_index}."
            )
        if (
            previous_timestamp is not None
            and timestamps.iloc[0] < previous_timestamp
        ):
            raise ValueError(
                "timestamp is not globally monotone at row group "
                f"{row_group_index}."
            )

        split_values = _assign_split_values(timestamps, boundaries)
        for split in VALID_SPLITS:
            mask = split_values == split
            count = int(mask.sum())
            counts[split] += count
            if count:
                last_event_id[split] = int(event_ids[mask][-1])

        previous_event_id = int(event_ids[-1])
        previous_timestamp = timestamps.iloc[-1]

    if previous_event_id is None:
        raise ValueError("Feature table is empty.")
    if set(last_event_id) != set(VALID_SPLITS):
        raise ValueError(
            "Every chronological split must contain observations; "
            f"found {sorted(last_event_id)}."
        )

    return {
        "rows_total": previous_event_id + 1,
        "counts": counts,
        "last_event_id": last_event_id,
    }


def _assign_split_values(
    timestamps: pd.Series,
    boundaries: dict[str, pd.Timestamp],
) -> np.ndarray:
    values = pd.to_datetime(timestamps, utc=True)
    in_sample = (
        (values >= boundaries["sample_start"])
        & (values < boundaries["sample_end_exclusive"])
    )
    if not bool(in_sample.all()):
        bad_count = int((~in_sample).sum())
        raise ValueError(
            f"{bad_count} rows fall outside the declared sample window."
        )

    split_values = np.empty(len(values), dtype=object)
    train_mask = values < boundaries["train_end"]
    validation_mask = (
        (values >= boundaries["train_end"])
        & (values < boundaries["validation_end"])
    )
    test_mask = values >= boundaries["validation_end"]

    split_values[train_mask] = "train"
    split_values[validation_mask] = "validation"
    split_values[test_mask] = "test"
    return split_values


def _validate_feature_chunk(frame: pd.DataFrame, row_group_index: int) -> None:
    if frame.empty:
        return
    if frame["feature_complete"].isna().any():
        raise ValueError(
            f"feature_complete contains missing values in row group {row_group_index}."
        )
    for flag in ("feature_complete", "has_full_trade_lookback_1s"):
        if not pd.api.types.is_bool_dtype(frame[flag]):
            raise TypeError(
                f"{flag} must have boolean dtype in row group {row_group_index}."
            )
    finite_features = np.ones(len(frame), dtype=bool)
    for feature in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[feature]):
            raise TypeError(
                f"{feature} must be numeric in row group {row_group_index}."
            )
        finite_features &= np.isfinite(
            frame[feature].to_numpy(dtype=np.float64, copy=False)
        )
    expected_complete = (
        frame["has_full_trade_lookback_1s"].to_numpy(copy=False)
        & finite_features
    )
    if not np.array_equal(
        frame["feature_complete"].to_numpy(copy=False),
        expected_complete,
    ):
        raise ValueError(
            "feature_complete does not match finite feature availability and "
            f"trade lookback state in row group {row_group_index}."
        )
    for column in ("y_10", "y_20", "y_50"):
        if frame[column].isna().any():
            raise ValueError(
                f"{column} contains missing values in row group {row_group_index}."
            )
        values = set(frame[column].unique())
        if not values.issubset({-1, 0, 1}):
            raise ValueError(
                f"{column} contains invalid labels in row group "
                f"{row_group_index}: {sorted(values)}"
            )


def _empty_summary_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "rows_total": 0,
        "feature_complete_rows": 0,
        "boundary_drop_rows": 0,
        "model_eligible_rows": 0,
        "timestamp_start": None,
        "timestamp_end": None,
    }
    for horizon in DEFAULT_HORIZONS:
        for label_name in ("down", "unchanged", "up"):
            state[f"y_{horizon}_{label_name}"] = 0
    return state


def _update_summary_state(state: dict[str, Any], part: pd.DataFrame) -> None:
    state["rows_total"] += len(part)
    state["feature_complete_rows"] += int(part["feature_complete"].sum())
    state["boundary_drop_rows"] += int(part["is_boundary_drop"].sum())
    state["model_eligible_rows"] += int(part["model_eligible"].sum())
    timestamp_start = part["timestamp"].min()
    timestamp_end = part["timestamp"].max()
    if state["timestamp_start"] is None:
        state["timestamp_start"] = timestamp_start
    state["timestamp_end"] = timestamp_end

    label_names = {-1: "down", 0: "unchanged", 1: "up"}
    for horizon in DEFAULT_HORIZONS:
        counts = part[f"y_{horizon}"].value_counts()
        for label, label_name in label_names.items():
            state[f"y_{horizon}_{label_name}"] += int(counts.get(label, 0))


def _summary_frame(
    summary_state: dict[str, dict[str, Any]],
    boundaries: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    rows = [
        {"split": split, **summary_state[split]}
        for split in VALID_SPLITS
    ]
    for name, timestamp in (
        ("BOUNDARIES", boundaries["sample_start"]),
        ("TRAIN_END", boundaries["train_end"]),
        ("VALIDATION_END", boundaries["validation_end"]),
    ):
        row = {column: pd.NA for column in rows[0]}
        row.update(
            {
                "split": name,
                "timestamp_start": timestamp,
                "timestamp_end": (
                    boundaries["sample_end_exclusive"]
                    if name == "BOUNDARIES"
                    else timestamp
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_partition_counts(
    summary: pd.DataFrame,
    scan: dict[str, Any],
    *,
    boundary_drop_events: int,
) -> None:
    split_rows = summary.loc[summary["split"].isin(VALID_SPLITS)]
    observed_total = int(split_rows["rows_total"].sum())
    if observed_total != scan["rows_total"]:
        raise AssertionError(
            "Partition row counts do not reconstruct the source: "
            f"{observed_total} != {scan['rows_total']}."
        )
    for split in VALID_SPLITS:
        observed = int(
            split_rows.loc[split_rows["split"] == split, "rows_total"].iloc[0]
        )
        if observed != scan["counts"][split]:
            raise AssertionError(
                f"Partition count mismatch for {split}: "
                f"{observed} != {scan['counts'][split]}."
            )
    for split in ("train", "validation"):
        dropped = int(
            split_rows.loc[
                split_rows["split"] == split,
                "boundary_drop_rows",
            ].iloc[0]
        )
        if dropped != boundary_drop_events:
            raise AssertionError(
                f"Boundary-drop count mismatch for {split}: "
                f"{dropped} != {boundary_drop_events}."
            )


def _write_manifest(
    path: Path,
    *,
    feature_path: Path,
    config: ChronologicalSplitConfig,
    boundaries: dict[str, pd.Timestamp],
    scan: dict[str, Any],
    summary: pd.DataFrame,
) -> None:
    experiment_spec = ExperimentSpec(
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        boundary_drop_events=config.boundary_drop_events,
    )
    payload = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_feature_table": feature_path.name,
        "layout": "split_parquet",
        "config": asdict(config),
        "experiment_fingerprint": experiment_spec.fingerprint(),
        "boundaries": {
            key: value.isoformat()
            for key, value in boundaries.items()
        },
        "source_rows": scan["rows_total"],
        "partition_rows": {
            split: int(scan["counts"][split])
            for split in VALID_SPLITS
        },
        "summary": (
            summary.loc[summary["split"].isin(VALID_SPLITS)]
            .where(pd.notna(summary), None)
            .to_dict(orient="records")
        ),
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    try:
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
