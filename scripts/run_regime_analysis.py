from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_naming import (  # noqa: E402
    frozen_model_artifact_stem,
    tagged_artifact_stem,
)
from src.atomic_io import atomic_write_csv  # noqa: E402
from src.data_loader import parse_date  # noqa: E402
from src.evaluation.probability_diagnostics import (  # noqa: E402
    frozen_threshold_mapping,
)
from src.evaluation.regime_analysis import (  # noqa: E402
    REGIME_SPECS,
    validate_regime_thresholds,
)
from src.evaluation.streaming_metrics import (  # noqa: E402
    TernaryMetricAccumulator,
)
from src.model_dataset_io import (  # noqa: E402
    iter_model_split_batches,
    model_dataset_sha256,
    model_split_row_count,
    resolve_model_dataset,
)
from src.modeling.model_artifacts import (  # noqa: E402
    FrozenLogisticArtifact,
    load_frozen_logistic_artifact,
    predict_from_artifact,
    predict_probabilities_from_artifact,
    validate_artifact_compatibility,
)
from src.protocol import (  # noqa: E402
    DEFAULT_HORIZONS,
    ExperimentSpec,
    FEATURE_COLUMNS,
    QUEUE_IMBALANCE_FEATURES,
    TERNARY_LABELS,
    validate_protocol_symbol,
)
from src.run_provenance import RunRecorder  # noqa: E402


HORIZONS = DEFAULT_HORIZONS
MODEL_ORDER = (
    "majority_baseline",
    "queue_imbalance_logistic",
    "full_logistic",
    "full_logistic_thresholded",
)
REGIME_FEATURE_COLUMNS = tuple(spec.feature for spec in REGIME_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen models by exact train-median regimes with bounded "
            "memory."
        )
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--artifact-tag", default=None)
    parser.add_argument(
        "--threshold-selection-metric",
        default="macro_f1",
        choices=["macro_f1", "balanced_accuracy"],
    )
    parser.add_argument(
        "--splits",
        default="validation,test",
        help="Comma-separated evaluation splits.",
    )
    parser.add_argument(
        "--batch-size",
        "--prediction-chunk-size",
        dest="batch_size",
        type=int,
        default=1_000_000,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def parse_splits(raw: str) -> tuple[str, ...]:
    splits = tuple(value.strip() for value in raw.split(",") if value.strip())
    allowed = {"validation", "test"}
    invalid = sorted(set(splits) - allowed)
    if invalid:
        raise ValueError(
            f"Unsupported splits: {invalid}. Allowed: {sorted(allowed)}"
        )
    if not splits:
        raise ValueError("At least one split must be requested.")
    if len(set(splits)) != len(splits):
        raise ValueError(f"Duplicate splits requested: {splits}")
    return splits


def load_artifacts(
    models_dir: Path,
    *,
    model_name: str,
    feature_names: tuple[str, ...],
    symbol: str,
    start: str,
    end: str,
    artifact_tag: str | None,
    dataset_sha256: str,
    experiment_fingerprint: str,
    source_tree_sha256: str | None = None,
) -> dict[int, FrozenLogisticArtifact]:
    artifacts = {}
    for horizon in HORIZONS:
        stem = frozen_model_artifact_stem(
            model_name,
            symbol,
            start,
            end,
            horizon,
            artifact_tag,
        )
        artifact = load_frozen_logistic_artifact(models_dir / stem)
        validate_artifact_compatibility(
            artifact,
            model_name=model_name,
            horizon=horizon,
            feature_names=feature_names,
            dataset_sha256=dataset_sha256,
            experiment_fingerprint=experiment_fingerprint,
            source_tree_sha256=source_tree_sha256,
        )
        artifacts[horizon] = artifact
    return artifacts


def load_frozen_thresholds(
    reports_dir: Path,
    *,
    symbol: str,
    start: str,
    end: str,
    selection_metric: str,
    dataset_sha256: str,
    experiment_fingerprint: str,
    artifacts: dict[int, FrozenLogisticArtifact],
    artifact_tag: str | None,
) -> dict[int, tuple[float, float]]:
    prefix = tagged_artifact_stem(
        "probability_diagnostics_full_logistic",
        symbol,
        start,
        end,
        artifact_tag,
    )
    path = reports_dir / f"{prefix}_best_thresholds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing threshold file: {path}")
    return frozen_threshold_mapping(
        pd.read_csv(path),
        selection_metric=selection_metric,
        horizons=HORIZONS,
        dataset_sha256=dataset_sha256,
        experiment_fingerprint=experiment_fingerprint,
        model_created_utc={
            horizon: artifact.created_utc
            for horizon, artifact in artifacts.items()
        },
        model_parameter_fingerprint={
            horizon: artifact.parameter_fingerprint()
            for horizon, artifact in artifacts.items()
        },
    )


def exact_training_thresholds_and_majorities(
    location,
    *,
    train_rows: int,
    batch_size: int,
    temporary_root: Path,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Compute exact train medians and majority labels with bounded memory."""
    with tempfile.TemporaryDirectory(
        dir=temporary_root,
        prefix="regime_train_",
    ) as directory:
        workdir = Path(directory)
        feature_arrays = {
            feature: np.memmap(
                workdir / f"{feature}.float64",
                mode="w+",
                dtype=np.float64,
                shape=(train_rows,),
            )
            for feature in REGIME_FEATURE_COLUMNS
        }
        label_counts = {
            horizon: np.zeros(3, dtype=np.int64)
            for horizon in HORIZONS
        }
        columns = (
            *(f"y_{horizon}" for horizon in HORIZONS),
            *REGIME_FEATURE_COLUMNS,
        )
        offset = 0
        for batch_number, batch in enumerate(
            iter_model_split_batches(
                location,
                "train",
                columns,
                eligible_only=True,
                batch_size=batch_size,
            ),
            start=1,
        ):
            stop = offset + len(batch)
            for feature, values in feature_arrays.items():
                batch_values = batch[feature].to_numpy(dtype=np.float64)
                if not np.isfinite(batch_values).all():
                    raise ValueError(
                        f"Non-finite train regime feature: {feature}"
                    )
                values[offset:stop] = batch_values
            for horizon in HORIZONS:
                labels = batch[f"y_{horizon}"].to_numpy(dtype=np.int8)
                if not np.isin(labels, np.asarray(TERNARY_LABELS)).all():
                    raise ValueError(
                        f"Unexpected train labels for horizon {horizon}."
                    )
                label_counts[horizon] += np.bincount(
                    labels + 1,
                    minlength=3,
                )
            offset = stop
            print(
                f"Train statistics batch {batch_number}: "
                f"{offset:,}/{train_rows:,}",
                flush=True,
            )
        if offset != train_rows:
            raise RuntimeError(
                f"Train row mismatch: wrote {offset}, expected {train_rows}."
            )

        rows = []
        for spec in REGIME_SPECS:
            values = feature_arrays[spec.feature]
            values.flush()
            left = (train_rows - 1) // 2
            right = train_rows // 2
            values.partition((left, right))
            threshold = float((values[left] + values[right]) / 2.0)
            lower_n = 0
            for start in range(0, train_rows, batch_size):
                chunk = values[start : start + batch_size]
                lower_n += int(np.count_nonzero(chunk <= threshold))
            upper_n = train_rows - lower_n
            rows.append(
                {
                    "regime_variable": spec.regime_variable,
                    "feature": spec.feature,
                    "threshold_source": "train_median",
                    "threshold_value": threshold,
                    "lower_regime": spec.lower_label,
                    "lower_rule": "<= train_median",
                    "upper_regime": spec.upper_label,
                    "upper_rule": "> train_median",
                    "train_n": train_rows,
                    "train_lower_n": lower_n,
                    "train_upper_n": upper_n,
                    "train_lower_fraction": lower_n / train_rows,
                    "train_upper_fraction": upper_n / train_rows,
                }
            )
        thresholds = pd.DataFrame(rows)
        validate_regime_thresholds(thresholds)
        majorities = {
            horizon: int(TERNARY_LABELS[int(np.argmax(counts))])
            for horizon, counts in label_counts.items()
        }
        return thresholds, majorities


def threshold_predictions(
    probabilities: np.ndarray,
    *,
    threshold_down: float,
    threshold_up: float,
) -> np.ndarray:
    p_down = probabilities[:, 0]
    p_up = probabilities[:, 2]
    predictions = np.zeros(len(probabilities), dtype=np.int8)
    predictions[(p_up >= threshold_up) & (p_up >= p_down)] = 1
    predictions[(p_down >= threshold_down) & (p_down > p_up)] = -1
    return predictions


def _regime_metadata(
    thresholds: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, object]]:
    metadata = {}
    for row in thresholds.to_dict(orient="records"):
        shared = {
            "regime_variable": row["regime_variable"],
            "feature": row["feature"],
            "threshold_source": row["threshold_source"],
            "threshold_value": row["threshold_value"],
        }
        metadata[(row["regime_variable"], row["lower_regime"])] = {
            **shared,
            "regime": row["lower_regime"],
            "threshold_rule": row["lower_rule"],
        }
        metadata[(row["regime_variable"], row["upper_regime"])] = {
            **shared,
            "regime": row["upper_regime"],
            "threshold_rule": row["upper_rule"],
        }
    return metadata


def _create_accumulators(
    splits: tuple[str, ...],
) -> dict[tuple[str, str, str, str, int], TernaryMetricAccumulator]:
    return {
        (
            split,
            spec.regime_variable,
            regime,
            model_name,
            horizon,
        ): TernaryMetricAccumulator()
        for split in splits
        for spec in REGIME_SPECS
        for regime in (spec.lower_label, spec.upper_label)
        for model_name in MODEL_ORDER
        for horizon in HORIZONS
    }


def _tables_from_accumulators(
    accumulators: dict[
        tuple[str, str, str, str, int],
        TernaryMetricAccumulator,
    ],
    *,
    regime_metadata: dict[tuple[str, str], dict[str, object]],
    dataset_hash: str,
    experiment_fingerprint: str,
    queue_artifacts: dict[int, FrozenLogisticArtifact],
    full_artifacts: dict[int, FrozenLogisticArtifact],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows = []
    nonzero_rows = []
    for key, accumulator in sorted(accumulators.items()):
        split, regime_variable, regime, model_name, horizon = key
        if accumulator.n_obs == 0:
            continue
        metadata = regime_metadata[(regime_variable, regime)]
        aggregate = accumulator.evaluation_tables(
            split=split,
            horizon=horizon,
            model_name=model_name,
        )["aggregate"].iloc[0].to_dict()
        nonzero = accumulator.nonzero_table(
            split=split,
            horizon=horizon,
            model_name=model_name,
        ).iloc[0].to_dict()
        if model_name == "queue_imbalance_logistic":
            artifact = queue_artifacts[horizon]
            created = artifact.created_utc
            parameter_fingerprint = artifact.parameter_fingerprint()
        elif model_name in {"full_logistic", "full_logistic_thresholded"}:
            artifact = full_artifacts[horizon]
            created = artifact.created_utc
            parameter_fingerprint = artifact.parameter_fingerprint()
        else:
            created = "fit_from_training_labels"
            parameter_fingerprint = "fit_from_training_labels"
        provenance = {
            "dataset_sha256": dataset_hash,
            "experiment_fingerprint": experiment_fingerprint,
            "model_created_utc": created,
            "model_parameter_fingerprint": parameter_fingerprint,
        }
        aggregate_rows.append({**metadata, **aggregate, **provenance})
        nonzero_rows.append({**metadata, **nonzero, **provenance})
    aggregate_table = pd.DataFrame(aggregate_rows).sort_values(
        ["split", "regime_variable", "regime", "horizon", "model"]
    )
    nonzero_table = pd.DataFrame(nonzero_rows).sort_values(
        ["split", "regime_variable", "regime", "horizon", "model"]
    )
    return aggregate_table, nonzero_table


def main() -> None:
    args = parse_args()
    validate_protocol_symbol(args.symbol)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    requested_splits = parse_splits(args.splits)

    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "outputs" / "models"
    results_dir = PROJECT_ROOT / "outputs" / "results"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    temporary_root = PROJECT_ROOT / "outputs" / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)

    dataset_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    prefix = tagged_artifact_stem(
        "final_test",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    location = resolve_model_dataset(processed_dir, dataset_stem)
    metadata_path = reports_dir / "run_metadata" / f"{prefix}_regimes.json"
    expected_outputs = [
        reports_dir / f"{prefix}_regime_thresholds.csv",
        results_dir / f"{prefix}_regime_aggregate.csv",
        results_dir / f"{prefix}_regime_nonzero_subset.csv",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Regime outputs already exist: {existing}. "
            "Pass --overwrite for a deliberate replacement."
        )

    with RunRecorder(
        metadata_path,
        project_root=PROJECT_ROOT,
        stage="regime_analysis",
        arguments=vars(args),
        require_clean_git=args.require_clean_git,
    ) as recorder:
        print("=" * 80, flush=True)
        print("REGIME ANALYSIS: EXACT TRAIN MEDIANS, STREAMED EVALUATION", flush=True)
        print("=" * 80, flush=True)
        dataset_hash = model_dataset_sha256(location)
        recorder.set_dataset_sha256(dataset_hash)
        experiment_fingerprint = ExperimentSpec().fingerprint()
        recorder.set_experiment_fingerprint(experiment_fingerprint)
        train_rows = model_split_row_count(location, "train", eligible_only=True)
        regime_thresholds, majorities = (
            exact_training_thresholds_and_majorities(
                location,
                train_rows=train_rows,
                batch_size=args.batch_size,
                temporary_root=temporary_root,
            )
        )
        print(regime_thresholds.to_string(index=False), flush=True)

        queue_artifacts = load_artifacts(
            models_dir,
            model_name="queue_imbalance_logistic",
            feature_names=QUEUE_IMBALANCE_FEATURES,
            symbol=args.symbol,
            start=start_str,
            end=end_str,
            artifact_tag=args.artifact_tag,
            dataset_sha256=dataset_hash,
            experiment_fingerprint=experiment_fingerprint,
            source_tree_sha256=recorder.source_state[
                "source_tree_sha256"
            ],
        )
        full_artifacts = load_artifacts(
            models_dir,
            model_name="full_logistic",
            feature_names=FEATURE_COLUMNS,
            symbol=args.symbol,
            start=start_str,
            end=end_str,
            artifact_tag=args.artifact_tag,
            dataset_sha256=dataset_hash,
            experiment_fingerprint=experiment_fingerprint,
            source_tree_sha256=recorder.source_state[
                "source_tree_sha256"
            ],
        )
        frozen_thresholds = load_frozen_thresholds(
            reports_dir,
            symbol=args.symbol,
            start=start_str,
            end=end_str,
            selection_metric=args.threshold_selection_metric,
            dataset_sha256=dataset_hash,
            experiment_fingerprint=experiment_fingerprint,
            artifacts=full_artifacts,
            artifact_tag=args.artifact_tag,
        )
        accumulators = _create_accumulators(requested_splits)
        threshold_rows = {
            row["feature"]: row
            for row in regime_thresholds.to_dict(orient="records")
        }

        columns = (
            *(f"y_{horizon}" for horizon in HORIZONS),
            *FEATURE_COLUMNS,
        )
        for split in requested_splits:
            expected_rows = model_split_row_count(
                location,
                split,
                eligible_only=True,
            )
            rows_seen = 0
            for batch_number, batch in enumerate(
                iter_model_split_batches(
                    location,
                    split,
                    columns,
                    eligible_only=True,
                    batch_size=args.batch_size,
                ),
                start=1,
            ):
                rows_seen += len(batch)
                regime_masks = {}
                for spec in REGIME_SPECS:
                    row = threshold_rows[spec.feature]
                    threshold = float(row["threshold_value"])
                    values = batch[spec.feature].to_numpy(copy=False)
                    regime_masks[
                        (spec.regime_variable, spec.lower_label)
                    ] = values <= threshold
                    regime_masks[
                        (spec.regime_variable, spec.upper_label)
                    ] = values > threshold

                for horizon in HORIZONS:
                    y_true = batch[f"y_{horizon}"].to_numpy(dtype=np.int8)
                    predictions = {
                        "majority_baseline": np.full(
                            len(batch),
                            majorities[horizon],
                            dtype=np.int8,
                        ),
                        "queue_imbalance_logistic": predict_from_artifact(
                            queue_artifacts[horizon],
                            batch,
                            chunk_size=args.batch_size,
                        ).astype(np.int8),
                    }
                    probabilities = predict_probabilities_from_artifact(
                        full_artifacts[horizon],
                        batch,
                        chunk_size=args.batch_size,
                    )
                    predictions["full_logistic"] = full_artifacts[
                        horizon
                    ].classes[np.argmax(probabilities, axis=1)].astype(np.int8)
                    threshold_down, threshold_up = frozen_thresholds[horizon]
                    predictions[
                        "full_logistic_thresholded"
                    ] = threshold_predictions(
                        probabilities,
                        threshold_down=threshold_down,
                        threshold_up=threshold_up,
                    )
                    for regime_key, mask in regime_masks.items():
                        if not bool(mask.any()):
                            continue
                        regime_variable, regime = regime_key
                        for model_name, y_pred in predictions.items():
                            accumulator_key = (
                                split,
                                regime_variable,
                                regime,
                                model_name,
                                horizon,
                            )
                            accumulators[accumulator_key].update(
                                y_true[mask],
                                y_pred[mask],
                            )
                print(
                    f"{split} batch {batch_number}: "
                    f"{rows_seen:,}/{expected_rows:,}",
                    flush=True,
                )
            if rows_seen != expected_rows:
                raise RuntimeError(
                    f"{split} row mismatch: {rows_seen} != {expected_rows}."
                )

        aggregate_table, nonzero_table = _tables_from_accumulators(
            accumulators,
            regime_metadata=_regime_metadata(regime_thresholds),
            dataset_hash=dataset_hash,
            experiment_fingerprint=experiment_fingerprint,
            queue_artifacts=queue_artifacts,
            full_artifacts=full_artifacts,
        )
        regime_thresholds["dataset_sha256"] = dataset_hash
        regime_thresholds["experiment_fingerprint"] = experiment_fingerprint
        thresholds_path = reports_dir / f"{prefix}_regime_thresholds.csv"
        aggregate_path = results_dir / f"{prefix}_regime_aggregate.csv"
        nonzero_path = results_dir / f"{prefix}_regime_nonzero_subset.csv"
        atomic_write_csv(thresholds_path, regime_thresholds)
        atomic_write_csv(aggregate_path, aggregate_table)
        atomic_write_csv(nonzero_path, nonzero_table)
        print(f"Saved regime thresholds: {thresholds_path}", flush=True)
        print(f"Saved regime aggregate metrics: {aggregate_path}", flush=True)
        print(f"Saved regime non-zero metrics: {nonzero_path}", flush=True)
        print(f"Run metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
