from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.atomic_io import atomic_write_csv
from src.evaluation.streaming_metrics import TernaryMetricAccumulator
from src.horizon_checkpoints import HorizonCheckpointStore
from src.model_dataset_io import (
    ModelDatasetLocation,
    iter_model_split_batches,
)
from src.modeling.disk_training import (
    PreparedTrainingData,
    prepare_training_data,
)
from src.modeling.logistic_models import (
    LogisticModelConfig,
    coefficient_table_from_estimator,
    fit_prestandardized_logistic_model,
)
from src.modeling.model_artifacts import (
    FrozenLogisticArtifact,
    artifact_from_prestandardized_estimator,
    assert_prestandardized_artifact_parity,
    load_frozen_logistic_artifact,
    predict_from_artifact,
    save_frozen_logistic_artifact,
    validate_artifact_compatibility,
)


EVALUATION_TABLE_NAMES = (
    "aggregate",
    "class_proportions",
    "per_class",
    "confusion_counts",
    "confusion_true_normalized",
)
SCALER_FIT_METHOD = "StandardScaler.partial_fit"


@dataclass(frozen=True)
class OfflineExperimentResult:
    tables: dict[str, pd.DataFrame]
    nonzero: pd.DataFrame
    coefficients: pd.DataFrame
    artifacts: dict[int, FrozenLogisticArtifact]


def run_checkpointed_logistic_experiment(
    *,
    location: ModelDatasetLocation,
    model_name: str,
    feature_columns: Sequence[str],
    horizons: Sequence[int],
    config: LogisticModelConfig,
    experiment_fingerprint: str,
    dataset_sha256: str,
    source_state: dict[str, Any],
    checkpoint_root: Path,
    artifact_bases: dict[int, Path],
    results_dir: Path,
    result_prefix: str,
    batch_size: int,
    temporary_root: Path,
    resume: bool,
    overwrite: bool,
) -> OfflineExperimentResult:
    """Fit, validate, checkpoint, and publish one linear model per horizon."""
    requested_horizons = _validated_horizons(horizons)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if set(artifact_bases) != set(requested_horizons):
        raise ValueError("artifact_bases must match the requested horizons.")

    model_config = {
        **config.as_dict(),
        "scaler_fit_method": SCALER_FIT_METHOD,
        "scaler_batch_size": batch_size,
        "training_matrix_dtype": "float64",
    }
    identity = {
        "model_name": model_name,
        "feature_columns": list(feature_columns),
        "model_config": model_config,
        "experiment_fingerprint": experiment_fingerprint,
        "dataset_sha256": dataset_sha256,
        "source_tree_sha256": source_state["source_tree_sha256"],
    }
    store = HorizonCheckpointStore(checkpoint_root, identity=identity)
    collected: dict[int, dict[str, pd.DataFrame]] = {}
    artifacts: dict[int, FrozenLogisticArtifact] = {}
    fit_horizons = []

    for horizon in requested_horizons:
        artifact_base = artifact_bases[horizon]
        if resume and store.exists(horizon):
            artifact = load_frozen_logistic_artifact(artifact_base)
            _validate_resumed_artifact(
                artifact,
                model_name=model_name,
                horizon=horizon,
                feature_columns=feature_columns,
                model_config=model_config,
                experiment_fingerprint=experiment_fingerprint,
                dataset_sha256=dataset_sha256,
                source_state=source_state,
            )
            collected[horizon] = store.load(
                horizon,
                artifact_parameter_fingerprint=artifact.parameter_fingerprint(),
            )
            artifacts[horizon] = artifact
            print(
                f"Resumed completed {model_name} horizon {horizon} "
                f"from {store.path_for(horizon)}",
                flush=True,
            )
            continue

        if store.exists(horizon) and not overwrite:
            raise FileExistsError(
                f"Checkpoint exists for horizon {horizon}. "
                "Use --resume to reuse it or --overwrite to replace it."
            )
        if not overwrite and (
            artifact_base.with_suffix(".npz").exists()
            or artifact_base.with_suffix(".json").exists()
        ):
            raise FileExistsError(
                f"Frozen artifact already exists: {artifact_base}. "
                "Use --resume when a matching checkpoint exists, or "
                "--overwrite for a deliberate regeneration."
            )
        fit_horizons.append(horizon)

    if collected:
        _publish_results(
            collected,
            results_dir=results_dir,
            result_prefix=result_prefix,
        )

    if fit_horizons:
        with prepare_training_data(
            location,
            feature_columns=feature_columns,
            horizons=fit_horizons,
            batch_size=batch_size,
            temporary_root=temporary_root,
        ) as prepared:
            print(
                f"Prepared disk-backed train matrix: "
                f"{prepared.n_rows:,} x {len(prepared.feature_columns)}",
                flush=True,
            )
            for horizon in fit_horizons:
                artifact, horizon_tables = _fit_and_evaluate_horizon(
                    location=location,
                    prepared=prepared,
                    model_name=model_name,
                    feature_columns=feature_columns,
                    horizon=horizon,
                    config=config,
                    model_config=model_config,
                    experiment_fingerprint=experiment_fingerprint,
                    dataset_sha256=dataset_sha256,
                    source_state=source_state,
                    batch_size=batch_size,
                )
                artifact_base = artifact_bases[horizon]
                arrays_path, metadata_path = save_frozen_logistic_artifact(
                    artifact,
                    artifact_base,
                )
                print(f"Saved frozen model arrays: {arrays_path}", flush=True)
                print(
                    f"Saved frozen model metadata: {metadata_path}",
                    flush=True,
                )
                store.save(
                    horizon,
                    tables=horizon_tables,
                    artifact_parameter_fingerprint=(
                        artifact.parameter_fingerprint()
                    ),
                    overwrite=overwrite,
                )
                print(
                    f"Committed horizon checkpoint: "
                    f"{store.path_for(horizon)}",
                    flush=True,
                )
                artifacts[horizon] = artifact
                collected[horizon] = horizon_tables
                _publish_results(
                    collected,
                    results_dir=results_dir,
                    result_prefix=result_prefix,
                )

    return _combined_result(collected, artifacts)


def _fit_and_evaluate_horizon(
    *,
    location: ModelDatasetLocation,
    prepared: PreparedTrainingData,
    model_name: str,
    feature_columns: Sequence[str],
    horizon: int,
    config: LogisticModelConfig,
    model_config: dict[str, Any],
    experiment_fingerprint: str,
    dataset_sha256: str,
    source_state: dict[str, Any],
    batch_size: int,
) -> tuple[FrozenLogisticArtifact, dict[str, pd.DataFrame]]:
    label_col = f"y_{horizon}"
    print()
    print("-" * 80)
    print(f"Fitting {model_name} for horizon {horizon}")
    print("-" * 80)
    print(f"Train rows: {prepared.n_rows:,}")
    print(f"Features:   {len(feature_columns)}")

    standardized_features = prepared.features()
    train_labels = prepared.labels(horizon)
    model = fit_prestandardized_logistic_model(
        standardized_features,
        train_labels,
        config,
    )
    artifact = artifact_from_prestandardized_estimator(
        model=model,
        model_name=model_name,
        horizon=horizon,
        feature_names=tuple(feature_columns),
        scaler_mean=prepared.scaler_mean,
        scaler_scale=prepared.scaler_scale,
        model_config=model_config,
        experiment_fingerprint=experiment_fingerprint,
        dataset_sha256=dataset_sha256,
        source_git_commit=source_state["git_commit"],
        source_tree_sha256=source_state["source_tree_sha256"],
    )

    parity_sample = next(
        iter_model_split_batches(
            location,
            "validation",
            feature_columns,
            eligible_only=True,
            batch_size=min(batch_size, 10_000),
        )
    )
    parity_error = assert_prestandardized_artifact_parity(
        model=model,
        artifact=artifact,
        sample=parity_sample,
    )
    print(f"Frozen artifact parity max abs error: {parity_error:.3e}")

    train_accumulator = TernaryMetricAccumulator()
    for start in range(0, prepared.n_rows, batch_size):
        stop = min(start + batch_size, prepared.n_rows)
        prediction = model.predict(standardized_features[start:stop])
        train_accumulator.update(train_labels[start:stop], prediction)

    validation_accumulator = TernaryMetricAccumulator()
    validation_columns = (label_col, *feature_columns)
    for batch in iter_model_split_batches(
        location,
        "validation",
        validation_columns,
        eligible_only=True,
        batch_size=batch_size,
    ):
        prediction = predict_from_artifact(
            artifact,
            batch,
            chunk_size=batch_size,
        )
        validation_accumulator.update(batch[label_col], prediction)
    print(f"Validation rows: {validation_accumulator.n_obs:,}")

    split_tables = [
        train_accumulator.evaluation_tables(
            split="train",
            horizon=horizon,
            model_name=model_name,
        ),
        validation_accumulator.evaluation_tables(
            split="validation",
            horizon=horizon,
            model_name=model_name,
        ),
    ]
    horizon_tables = {
        name: pd.concat(
            [split_table[name] for split_table in split_tables],
            ignore_index=True,
        )
        for name in EVALUATION_TABLE_NAMES
    }
    horizon_tables["nonzero_subset"] = pd.concat(
        [
            train_accumulator.nonzero_table(
                split="train",
                horizon=horizon,
                model_name=model_name,
            ),
            validation_accumulator.nonzero_table(
                split="validation",
                horizon=horizon,
                model_name=model_name,
            ),
        ],
        ignore_index=True,
    )
    horizon_tables["coefficients"] = coefficient_table_from_estimator(
        model=model,
        feature_columns=feature_columns,
        horizon=horizon,
        model_name=model_name,
    )
    return artifact, horizon_tables


def _validate_resumed_artifact(
    artifact: FrozenLogisticArtifact,
    *,
    model_name: str,
    horizon: int,
    feature_columns: Sequence[str],
    model_config: dict[str, Any],
    experiment_fingerprint: str,
    dataset_sha256: str,
    source_state: dict[str, Any],
) -> None:
    validate_artifact_compatibility(
        artifact,
        model_name=model_name,
        horizon=horizon,
        feature_names=feature_columns,
        experiment_fingerprint=experiment_fingerprint,
        dataset_sha256=dataset_sha256,
        source_tree_sha256=source_state["source_tree_sha256"],
    )
    if artifact.model_config != model_config:
        raise ValueError(
            f"Incompatible frozen artifact for resume, horizon {horizon}: "
            "model configuration differs"
        )


def _publish_results(
    collected: dict[int, dict[str, pd.DataFrame]],
    *,
    results_dir: Path,
    result_prefix: str,
) -> None:
    combined = _combined_tables(collected)
    for name, table in combined.items():
        atomic_write_csv(
            results_dir / f"{result_prefix}_{name}.csv",
            table,
        )


def _combined_result(
    collected: dict[int, dict[str, pd.DataFrame]],
    artifacts: dict[int, FrozenLogisticArtifact],
) -> OfflineExperimentResult:
    combined = _combined_tables(collected)
    return OfflineExperimentResult(
        tables={
            name: combined[name]
            for name in EVALUATION_TABLE_NAMES
        },
        nonzero=combined["nonzero_subset"],
        coefficients=combined["coefficients"],
        artifacts=artifacts,
    )


def _combined_tables(
    collected: dict[int, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    if not collected:
        raise ValueError("No horizon checkpoints were collected.")
    names = set(next(iter(collected.values())))
    if any(set(tables) != names for tables in collected.values()):
        raise ValueError("Horizon checkpoints contain different table sets.")
    return {
        name: pd.concat(
            [collected[horizon][name] for horizon in sorted(collected)],
            ignore_index=True,
        )
        for name in sorted(names)
    }


def _validated_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in horizons)
    if not values:
        raise ValueError("At least one horizon is required.")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate horizons: {values}")
    if any(value <= 0 for value in values):
        raise ValueError("Horizons must be positive.")
    return values
