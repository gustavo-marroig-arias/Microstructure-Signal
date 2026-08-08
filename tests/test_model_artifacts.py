from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.logistic_models import (
    LogisticModelConfig,
    fit_logistic_model,
)
from src.modeling.model_artifacts import (
    artifact_from_pipeline,
    assert_pipeline_artifact_parity,
    load_frozen_logistic_artifact,
    save_frozen_logistic_artifact,
    validate_artifact_compatibility,
)


def test_frozen_artifact_round_trip_matches_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.normal(size=(600, 3)),
        columns=["a", "b", "c"],
    )
    frame["y"] = np.tile(np.array([-1, 0, 1]), 200)
    config = LogisticModelConfig(max_iter=500)
    pipeline = fit_logistic_model(
        frame,
        "y",
        ("a", "b", "c"),
        config,
    )
    artifact = artifact_from_pipeline(
        pipeline=pipeline,
        model_name="full_logistic",
        horizon=50,
        feature_names=("a", "b", "c"),
        transform_dtype=frame[["a", "b", "c"]].to_numpy().dtype.name,
        model_config=config.as_dict(),
        experiment_fingerprint="protocol",
        dataset_sha256="dataset",
        source_git_commit="commit",
        source_tree_sha256="source",
    )

    assert assert_pipeline_artifact_parity(
        pipeline=pipeline,
        artifact=artifact,
        sample=frame,
    ) <= 1e-12

    save_frozen_logistic_artifact(artifact, tmp_path / "model")
    loaded = load_frozen_logistic_artifact(tmp_path / "model")
    assert loaded.parameter_fingerprint() == artifact.parameter_fingerprint()
    assert loaded.source_git_commit == "commit"
    assert loaded.source_tree_sha256 == "source"
    assert assert_pipeline_artifact_parity(
        pipeline=pipeline,
        artifact=loaded,
        sample=frame,
    ) <= 1e-12


def test_artifact_compatibility_fails_on_feature_order() -> None:
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
    frame["y"] = np.tile(np.array([-1, 0, 1]), 100)
    config = LogisticModelConfig(max_iter=500)
    pipeline = fit_logistic_model(frame, "y", ("a", "b"), config)
    artifact = artifact_from_pipeline(
        pipeline=pipeline,
        model_name="model",
        horizon=10,
        feature_names=("a", "b"),
        transform_dtype=frame[["a", "b"]].to_numpy().dtype.name,
        model_config=config.as_dict(),
    )

    with pytest.raises(ValueError, match="feature order"):
        validate_artifact_compatibility(
            artifact,
            model_name="model",
            horizon=10,
            feature_names=("b", "a"),
        )

    with pytest.raises(ValueError, match="source-tree"):
        validate_artifact_compatibility(
            artifact,
            model_name="model",
            horizon=10,
            feature_names=("a", "b"),
            source_tree_sha256="different",
        )


def test_frozen_artifact_rejects_array_metadata_mismatch(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
    frame["y"] = np.tile(np.array([-1, 0, 1]), 100)
    config = LogisticModelConfig(max_iter=500)
    pipeline = fit_logistic_model(frame, "y", ("a", "b"), config)
    artifact = artifact_from_pipeline(
        pipeline=pipeline,
        model_name="model",
        horizon=20,
        feature_names=("a", "b"),
        transform_dtype=frame[["a", "b"]].to_numpy().dtype.name,
        model_config=config.as_dict(),
    )
    arrays_path, _ = save_frozen_logistic_artifact(
        artifact,
        tmp_path / "model",
    )
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_frozen_logistic_artifact(tmp_path / "model")


def test_float32_artifact_matches_standard_scaler_dtype_semantics() -> None:
    rng = np.random.default_rng(10)
    frame = pd.DataFrame(
        {
            "queue_imbalance": rng.uniform(-1.0, 1.0, 3_000).astype(
                np.float32
            )
        }
    )
    frame["y"] = np.tile(np.array([-1, 0, 1]), 1_000)
    config = LogisticModelConfig(max_iter=500)
    pipeline = fit_logistic_model(
        frame,
        "y",
        ("queue_imbalance",),
        config,
    )
    artifact = artifact_from_pipeline(
        pipeline=pipeline,
        model_name="queue_imbalance_logistic",
        horizon=50,
        feature_names=("queue_imbalance",),
        transform_dtype="float32",
        model_config=config.as_dict(),
    )

    assert artifact.transform_dtype == "float32"
    assert assert_pipeline_artifact_parity(
        pipeline=pipeline,
        artifact=artifact,
        sample=frame,
    ) <= 1e-12
