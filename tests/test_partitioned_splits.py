from pathlib import Path

import numpy as np
import pandas as pd

from src.model_dataset_io import (
    ModelDatasetLocation,
    load_model_split,
    load_model_splits,
    model_dataset_sha256,
    model_split_row_count,
    resolve_model_dataset,
)
from src.partitioned_splits import build_partitioned_model_dataset
from src.protocol import DEFAULT_HORIZONS, FEATURE_COLUMNS
from src.split_config import build_model_dataset


def test_partitioned_splits_match_in_memory_logic(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    n_rows = 1_000
    feature_complete = np.arange(n_rows) >= 20
    frame = pd.DataFrame(
        {
            "event_id": np.arange(n_rows, dtype=np.int64),
            "timestamp": pd.date_range(
                "2024-03-01",
                periods=n_rows,
                freq="80s",
                tz="UTC",
            ),
            "feature_complete": feature_complete,
            "has_full_trade_lookback_1s": feature_complete,
            "y_10": rng.choice([-1, 0, 1], n_rows).astype(np.int8),
            "y_20": rng.choice([-1, 0, 1], n_rows).astype(np.int8),
            "y_50": rng.choice([-1, 0, 1], n_rows).astype(np.int8),
        }
    )
    for feature in FEATURE_COLUMNS:
        frame[feature] = rng.normal(size=n_rows).astype(np.float32)
    frame["mid_return_5"] = frame["mid_return_5"].astype(np.float64)
    frame["realized_vol_20"] = frame["realized_vol_20"].astype(np.float64)
    frame.loc[~feature_complete, "delta_bid_size"] = np.nan
    expected, _ = build_model_dataset(
        frame,
        "2024-03-01",
        "2024-03-01",
    )
    feature_path = tmp_path / "feature.parquet"
    frame.to_parquet(feature_path, index=False, row_group_size=113)
    build_partitioned_model_dataset(
        feature_path,
        tmp_path / "model",
        start="2024-03-01",
        end="2024-03-01",
    )

    location = resolve_model_dataset(tmp_path, "model")
    loaded = load_model_splits(
        location,
        ("train", "validation", "test"),
        tuple(expected.columns),
        eligible_only=False,
    )
    actual = pd.concat(loaded.values(), ignore_index=True)

    pd.testing.assert_frame_equal(
        actual.drop(columns="split"),
        expected.drop(columns="split"),
        check_dtype=True,
    )
    assert actual["split"].astype(str).tolist() == expected["split"].tolist()
    assert (
        actual.groupby("split", observed=True)["is_boundary_drop"].sum().to_dict()
        == {"train": 50, "validation": 50, "test": 0}
    )
    eligible_train = load_model_split(
        location,
        "train",
        ("event_id", "split", "model_eligible"),
        eligible_only=True,
    )
    expected_train_eligible = expected.loc[
        (expected["split"] == "train")
        & expected["model_eligible"],
        "event_id",
    ].reset_index(drop=True)
    pd.testing.assert_series_equal(
        eligible_train["event_id"],
        expected_train_eligible,
        check_names=False,
    )
    assert bool(eligible_train["model_eligible"].all())
    assert model_split_row_count(location, "train") == len(
        expected_train_eligible
    )
    for split in ("train", "validation"):
        partition = actual.loc[actual["split"].astype(str) == split]
        last_event_id = int(partition["event_id"].max())
        eligible_ids = partition.loc[
            partition["model_eligible"],
            "event_id",
        ].to_numpy()
        for horizon in DEFAULT_HORIZONS:
            assert bool(np.all(eligible_ids + horizon <= last_event_id))

    replacement_summary, _, replacement_manifest = (
        build_partitioned_model_dataset(
            feature_path,
            tmp_path / "model",
            start="2024-03-01",
            end="2024-03-01",
            overwrite=True,
        )
    )
    assert replacement_manifest.exists()
    assert int(
        replacement_summary.loc[
            replacement_summary["split"].isin(
                ["train", "validation", "test"]
            ),
            "rows_total",
        ].sum()
    ) == n_rows


def test_dataset_hash_cache_invalidates_after_file_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.parquet"
    path.write_bytes(b"first")
    location = ModelDatasetLocation(
        dataset_stem="model",
        layout="monolithic",
        path=path,
    )

    first_hash = model_dataset_sha256(location)
    assert model_dataset_sha256(location) == first_hash
    path.write_bytes(b"second")
    second_hash = model_dataset_sha256(location)

    assert second_hash != first_hash
