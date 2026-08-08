from pathlib import Path

import pandas as pd
import pytest

from src.horizon_checkpoints import HorizonCheckpointStore


def test_horizon_checkpoint_round_trip_and_identity_guard(
    tmp_path: Path,
) -> None:
    identity = {
        "dataset_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "model_config": {"solver": "lbfgs"},
    }
    store = HorizonCheckpointStore(tmp_path / "checkpoints", identity=identity)
    tables = {
        "aggregate": pd.DataFrame(
            [{"horizon": 10, "accuracy": 0.5}]
        ),
        "coefficients": pd.DataFrame(
            [{"horizon": 10, "coefficient": 0.25}]
        ),
    }
    store.save(
        10,
        tables=tables,
        artifact_parameter_fingerprint="model",
    )
    loaded = store.load(
        10,
        artifact_parameter_fingerprint="model",
    )
    for name in tables:
        pd.testing.assert_frame_equal(loaded[name], tables[name])

    incompatible = HorizonCheckpointStore(
        tmp_path / "checkpoints",
        identity={**identity, "source_tree_sha256": "c" * 64},
    )
    with pytest.raises(ValueError, match="run identity"):
        incompatible.load(
            10,
            artifact_parameter_fingerprint="model",
        )


def test_horizon_checkpoint_rejects_tampered_table(tmp_path: Path) -> None:
    store = HorizonCheckpointStore(
        tmp_path / "checkpoints",
        identity={"dataset": "dataset"},
    )
    store.save(
        50,
        tables={"aggregate": pd.DataFrame([{"value": 1}])},
        artifact_parameter_fingerprint="model",
    )
    table_path = store.path_for(50) / "aggregate.csv"
    table_path.write_text("value\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        store.load(
            50,
            artifact_parameter_fingerprint="model",
        )
