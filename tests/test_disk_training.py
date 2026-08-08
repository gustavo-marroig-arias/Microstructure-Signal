from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.model_dataset_io import ModelDatasetLocation
from src.modeling.disk_training import prepare_training_data
from src.modeling.logistic_models import (
    LogisticModelConfig,
    fit_logistic_model,
    fit_prestandardized_logistic_model,
)


def test_disk_training_matches_incremental_scaler_and_row_order(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "model"
    dataset.mkdir()
    rng = np.random.default_rng(52)
    frames = {}
    for split, size in (("train", 303), ("validation", 60), ("test", 60)):
        frame = pd.DataFrame(
            {
                "split": split,
                "model_eligible": True,
                "y_10": np.resize(np.array([-1, 0, 1], dtype=np.int8), size),
                "a": rng.normal(size=size).astype(np.float32),
                "b": rng.normal(size=size).astype(np.float64),
            }
        )
        frame.to_parquet(dataset / f"{split}.parquet", index=False)
        frames[split] = frame
    location = ModelDatasetLocation("model", "partitioned", dataset)
    temporary_root = tmp_path / "tmp"

    with prepare_training_data(
        location,
        feature_columns=("a", "b"),
        horizons=(10,),
        batch_size=47,
        temporary_root=temporary_root,
    ) as prepared:
        expected_scaler = StandardScaler()
        for start in range(0, len(frames["train"]), 47):
            expected_scaler.partial_fit(
                frames["train"][["a", "b"]]
                .iloc[start : start + 47]
                .to_numpy(dtype=np.float64)
            )
        expected = expected_scaler.transform(
            frames["train"][["a", "b"]].to_numpy(dtype=np.float64)
        )
        np.testing.assert_allclose(
            prepared.scaler_mean,
            expected_scaler.mean_,
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            prepared.scaler_scale,
            expected_scaler.scale_,
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            prepared.features(),
            expected,
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(
            prepared.labels(10),
            frames["train"]["y_10"],
        )


def test_disk_training_matches_original_pipeline_probabilities(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "model"
    dataset.mkdir()
    rng = np.random.default_rng(117)
    train = pd.DataFrame(
        {
            "split": "train",
            "model_eligible": True,
            "y_10": np.resize(np.array([-1, 0, 1], dtype=np.int8), 600),
            "a": rng.normal(loc=2.0, scale=4.0, size=600),
            "b": rng.normal(loc=-5.0, scale=0.25, size=600),
        }
    )
    validation = pd.DataFrame(
        {
            "split": "validation",
            "model_eligible": True,
            "y_10": np.resize(np.array([-1, 0, 1], dtype=np.int8), 120),
            "a": rng.normal(loc=2.0, scale=4.0, size=120),
            "b": rng.normal(loc=-5.0, scale=0.25, size=120),
        }
    )
    test = validation.assign(split="test")
    train.to_parquet(dataset / "train.parquet", index=False)
    validation.to_parquet(dataset / "validation.parquet", index=False)
    test.to_parquet(dataset / "test.parquet", index=False)

    config = LogisticModelConfig(max_iter=1_000, tol=1e-10)
    original = fit_logistic_model(
        train,
        "y_10",
        ("a", "b"),
        config,
    )
    location = ModelDatasetLocation("model", "partitioned", dataset)

    with prepare_training_data(
        location,
        feature_columns=("a", "b"),
        horizons=(10,),
        batch_size=47,
        temporary_root=tmp_path / "tmp",
    ) as prepared:
        optimized = fit_prestandardized_logistic_model(
            prepared.features(),
            prepared.labels(10),
            config,
        )
        np.testing.assert_allclose(
            prepared.scaler_mean,
            original.named_steps["scaler"].mean_,
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            prepared.scaler_scale,
            original.named_steps["scaler"].scale_,
            rtol=1e-14,
            atol=1e-14,
        )
        standardized_validation = (
            validation[["a", "b"]].to_numpy(dtype=np.float64)
            - prepared.scaler_mean
        ) / prepared.scaler_scale
        np.testing.assert_allclose(
            optimized.predict_proba(standardized_validation),
            original.predict_proba(validation[["a", "b"]]),
            rtol=1e-10,
            atol=1e-12,
        )
