from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model_dataset_io import ModelDatasetLocation
from src.modeling.logistic_models import LogisticModelConfig
from src.modeling.offline_experiment import (
    run_checkpointed_logistic_experiment,
)


def _write_partitioned_dataset(root: Path) -> ModelDatasetLocation:
    rng = np.random.default_rng(91)
    sizes = {"train": 300, "validation": 120, "test": 90}
    offset = 0
    root.mkdir()
    for split, size in sizes.items():
        labels = np.tile(np.array([-1, 0, 1], dtype=np.int8), size // 3)
        frame = pd.DataFrame(
            {
                "event_id": np.arange(offset, offset + size),
                "split": split,
                "model_eligible": True,
                "y_10": labels,
                "a": rng.normal(size=size),
                "b": rng.normal(size=size),
            }
        )
        frame.to_parquet(root / f"{split}.parquet", index=False)
        offset += size
    return ModelDatasetLocation("model", "partitioned", root)


def test_offline_experiment_checkpoints_and_resumes(tmp_path: Path) -> None:
    location = _write_partitioned_dataset(tmp_path / "model")
    config = LogisticModelConfig(max_iter=500)
    common = {
        "location": location,
        "model_name": "model",
        "feature_columns": ("a", "b"),
        "horizons": (10,),
        "config": config,
        "experiment_fingerprint": "protocol",
        "dataset_sha256": "a" * 64,
        "source_state": {
            "git_commit": "commit",
            "git_dirty": False,
            "source_tree_sha256": "b" * 64,
        },
        "checkpoint_root": tmp_path / "checkpoints",
        "artifact_bases": {10: tmp_path / "artifacts" / "h10"},
        "results_dir": tmp_path / "results",
        "result_prefix": "model",
        "batch_size": 37,
        "temporary_root": tmp_path / "tmp",
    }
    first = run_checkpointed_logistic_experiment(
        **common,
        resume=False,
        overwrite=False,
    )
    resumed = run_checkpointed_logistic_experiment(
        **common,
        resume=True,
        overwrite=False,
    )
    pd.testing.assert_frame_equal(
        first.tables["aggregate"],
        resumed.tables["aggregate"],
        check_dtype=False,
    )
    assert (
        first.artifacts[10].parameter_fingerprint()
        == resumed.artifacts[10].parameter_fingerprint()
    )

    incompatible = {
        **common,
        "source_state": {
            **common["source_state"],
            "source_tree_sha256": "c" * 64,
        },
    }
    with pytest.raises(ValueError, match="source-tree SHA-256"):
        run_checkpointed_logistic_experiment(
            **incompatible,
            resume=True,
            overwrite=False,
        )
