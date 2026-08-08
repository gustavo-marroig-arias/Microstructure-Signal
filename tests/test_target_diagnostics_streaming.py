from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.target_diagnostics import (
    streaming_target_diagnostics,
    target_diagnostics_by_split,
)
from src.model_dataset_io import ModelDatasetLocation
from src.split_config import build_model_dataset


def test_streaming_target_diagnostics_match_in_memory(tmp_path: Path) -> None:
    rng = np.random.default_rng(15)
    n_rows = 1_000
    feature_table = pd.DataFrame(
        {
            "event_id": np.arange(n_rows, dtype=np.int64),
            "timestamp": pd.date_range(
                "2024-03-01",
                periods=n_rows,
                freq="80s",
                tz="UTC",
            ),
            "feature_complete": np.arange(n_rows) >= 20,
            "y_10": rng.choice([-1, 0, 1], n_rows).astype(np.int8),
            "y_20": rng.choice([-1, 0, 1], n_rows).astype(np.int8),
            "y_50": rng.choice([-1, 0, 1], n_rows).astype(np.int8),
        }
    )
    model_dataset, _ = build_model_dataset(
        feature_table,
        "2024-03-01",
        "2024-03-01",
    )
    path = tmp_path / "model.parquet"
    model_dataset.to_parquet(path, index=False, row_group_size=73)
    location = ModelDatasetLocation(
        dataset_stem="model",
        layout="monolithic",
        path=path,
    )

    diagnostics, split_summary, total_rows = streaming_target_diagnostics(
        location,
        batch_size=37,
    )
    expected = target_diagnostics_by_split(
        model_dataset,
        eligible_only=True,
    )

    pd.testing.assert_frame_equal(diagnostics, expected, check_dtype=False)
    assert total_rows == len(model_dataset)
    expected_eligible = (
        model_dataset.groupby("split", sort=False)["model_eligible"]
        .sum()
        .astype(int)
        .to_dict()
    )
    actual_eligible = (
        split_summary.set_index("split")["model_eligible_rows"]
        .astype(int)
        .to_dict()
    )
    assert actual_eligible == expected_eligible


def test_streaming_target_diagnostics_reject_inconsistent_eligibility(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "event_id": np.arange(12, dtype=np.int64),
            "timestamp": pd.date_range(
                "2024-03-01",
                periods=12,
                freq="h",
                tz="UTC",
            ),
            "split": np.repeat(["train", "validation", "test"], 4),
            "feature_complete": True,
            "is_boundary_drop": False,
            "model_eligible": True,
            "y_10": 0,
            "y_20": 0,
            "y_50": 0,
        }
    )
    frame.loc[1, "model_eligible"] = False
    path = tmp_path / "bad_model.parquet"
    frame.to_parquet(path, index=False)
    location = ModelDatasetLocation(
        dataset_stem="bad_model",
        layout="monolithic",
        path=path,
    )

    with pytest.raises(ValueError, match="model_eligible must equal"):
        streaming_target_diagnostics(location, batch_size=2)
