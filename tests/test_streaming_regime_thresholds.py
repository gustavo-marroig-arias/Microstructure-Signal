from pathlib import Path

import pandas as pd

from scripts.run_regime_analysis import (
    exact_training_thresholds_and_majorities,
)
from src.model_dataset_io import ModelDatasetLocation


def test_exact_disk_backed_train_medians_and_tie_counts(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "model"
    dataset.mkdir()
    values = {
        "relative_spread": [1.0, 2.0, 2.0, 2.0, 3.0, 4.0],
        "realized_vol_20": [0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
        "trade_intensity_1s": [1, 2, 3, 4, 5, 6],
    }
    for split in ("train", "validation", "test"):
        frame = pd.DataFrame(
            {
                "split": split,
                "model_eligible": True,
                "y_10": [-1, 0, 0, 0, 1, 0],
                "y_20": [-1, 0, 0, 0, 1, 0],
                "y_50": [-1, 0, 0, 0, 1, 0],
                **values,
            }
        )
        frame.to_parquet(dataset / f"{split}.parquet", index=False)

    location = ModelDatasetLocation("model", "partitioned", dataset)
    temporary_root = tmp_path / "tmp"
    temporary_root.mkdir()
    thresholds, majorities = exact_training_thresholds_and_majorities(
        location,
        train_rows=6,
        batch_size=2,
        temporary_root=temporary_root,
    )
    by_feature = thresholds.set_index("feature")
    assert by_feature.loc["relative_spread", "threshold_value"] == 2.0
    assert by_feature.loc["relative_spread", "train_lower_n"] == 4
    assert by_feature.loc["realized_vol_20", "threshold_value"] == 0.0
    assert by_feature.loc["realized_vol_20", "train_lower_n"] == 4
    assert by_feature.loc["trade_intensity_1s", "threshold_value"] == 3.5
    assert by_feature.loc["trade_intensity_1s", "train_lower_n"] == 3
    assert majorities == {10: 0, 20: 0, 50: 0}
