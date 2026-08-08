from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
from src.evaluation.metrics import concat_evaluation_tables  # noqa: E402
from src.evaluation.model_comparison import (  # noqa: E402
    signal_decay_table,
    validation_ranking_table,
)
from src.evaluation.probability_diagnostics import (  # noqa: E402
    frozen_threshold_mapping,
)
from src.evaluation.streaming_metrics import (  # noqa: E402
    DailyTernaryMetricAccumulator,
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
    coefficient_table_from_artifact,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen models on test with bounded memory."
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
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def load_model_artifacts(
    models_dir: Path,
    *,
    model_name: str,
    feature_names: tuple[str, ...],
    symbol: str,
    start: str,
    end: str,
    artifact_tag: str | None,
    experiment_fingerprint: str,
    dataset_sha256: str,
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
            experiment_fingerprint=experiment_fingerprint,
            dataset_sha256=dataset_sha256,
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
        raise FileNotFoundError(
            f"Missing threshold file: {path}. Run probability diagnostics first."
        )
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


def compute_deltas(
    aggregate: pd.DataFrame,
    model_name: str,
    reference_models: list[str],
) -> pd.DataFrame:
    metrics = ("accuracy", "macro_f1", "balanced_accuracy")
    rows = []
    for horizon, group in aggregate.groupby("horizon", sort=True):
        by_model = group.set_index("model")
        if model_name not in by_model.index:
            raise ValueError(f"Missing model {model_name} for horizon {horizon}")
        model_row = by_model.loc[model_name]
        for reference in reference_models:
            if reference not in by_model.index:
                raise ValueError(
                    f"Missing reference {reference} for horizon {horizon}"
                )
            reference_row = by_model.loc[reference]
            row = {
                "horizon": horizon,
                "model": model_name,
                "reference_model": reference,
            }
            for metric in metrics:
                row[f"{metric}_reference"] = float(reference_row[metric])
                row[f"{metric}_model"] = float(model_row[metric])
                row[f"{metric}_delta"] = float(
                    model_row[metric] - reference_row[metric]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _training_majority_counts(
    location,
    *,
    batch_size: int,
) -> tuple[dict[int, int], pd.DataFrame]:
    counts = {
        horizon: np.zeros(3, dtype=np.int64)
        for horizon in HORIZONS
    }
    columns = tuple(f"y_{horizon}" for horizon in HORIZONS)
    for batch in iter_model_split_batches(
        location,
        "train",
        columns,
        eligible_only=True,
        batch_size=batch_size,
    ):
        for horizon in HORIZONS:
            labels = batch[f"y_{horizon}"].to_numpy(dtype=np.int8)
            if not np.isin(labels, np.asarray(TERNARY_LABELS)).all():
                raise ValueError(f"Unexpected training labels for horizon {horizon}.")
            counts[horizon] += np.bincount(labels + 1, minlength=3)

    majority = {
        horizon: int(TERNARY_LABELS[int(np.argmax(class_counts))])
        for horizon, class_counts in counts.items()
    }
    rows = []
    for horizon in HORIZONS:
        class_counts = counts[horizon]
        total = int(class_counts.sum())
        rows.append(
            {
                "model": "majority_baseline",
                "horizon": horizon,
                "majority_class": majority[horizon],
                "train_count_down": int(class_counts[0]),
                "train_count_unchanged": int(class_counts[1]),
                "train_count_up": int(class_counts[2]),
                "train_prop_down": float(class_counts[0] / total),
                "train_prop_unchanged": float(class_counts[1] / total),
                "train_prop_up": float(class_counts[2] / total),
            }
        )
    return majority, pd.DataFrame(rows)


def _threshold_predictions(
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


def _metric_outputs(
    pooled: dict[tuple[str, int], TernaryMetricAccumulator],
    daily: dict[tuple[str, int], DailyTernaryMetricAccumulator],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    evaluations = []
    nonzero = []
    daily_tables = []
    for horizon in HORIZONS:
        for model_name in MODEL_ORDER:
            key = (model_name, horizon)
            evaluations.append(
                pooled[key].evaluation_tables(
                    split="test",
                    horizon=horizon,
                    model_name=model_name,
                )
            )
            nonzero.append(
                pooled[key].nonzero_table(
                    split="test",
                    horizon=horizon,
                    model_name=model_name,
                )
            )
            daily_tables.append(
                daily[key].table(
                    split="test",
                    horizon=horizon,
                    model_name=model_name,
                )
            )
    return (
        concat_evaluation_tables(evaluations),
        pd.concat(nonzero, ignore_index=True),
        pd.concat(daily_tables, ignore_index=True),
    )


def main() -> None:
    args = parse_args()
    validate_protocol_symbol(args.symbol)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")

    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "outputs" / "results"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    models_dir = PROJECT_ROOT / "outputs" / "models"

    dataset_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    output_prefix = tagged_artifact_stem(
        "final_test",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    location = resolve_model_dataset(processed_dir, dataset_stem)
    metadata_path = reports_dir / "run_metadata" / f"{output_prefix}.json"
    expected_outputs = [
        *(
            results_dir / f"{output_prefix}_{name}.csv"
            for name in (
                "aggregate",
                "class_proportions",
                "per_class",
                "confusion_counts",
                "confusion_true_normalized",
                "nonzero_subset",
                "daily_blocks",
                "majority_fit_summary",
                "logistic_coefficients",
            )
        ),
        *(
            reports_dir / f"{output_prefix}_{name}.csv"
            for name in (
                "frozen_thresholds",
                "ranking",
                "signal_decay",
                "full_vs_baselines_deltas",
                "thresholded_deltas",
            )
        ),
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Final-test outputs already exist: {existing}. "
            "Pass --overwrite for a deliberate replacement."
        )

    with RunRecorder(
        metadata_path,
        project_root=PROJECT_ROOT,
        stage="final_test_evaluation",
        arguments=vars(args),
        require_clean_git=args.require_clean_git,
    ) as recorder:
        print("=" * 80)
        print("FINAL TEST EVALUATION: BOUNDED-MEMORY PASS")
        print("=" * 80)
        dataset_hash = model_dataset_sha256(location)
        recorder.set_dataset_sha256(dataset_hash)
        experiment_fingerprint = ExperimentSpec().fingerprint()
        recorder.set_experiment_fingerprint(experiment_fingerprint)

        queue_artifacts = load_model_artifacts(
            models_dir,
            model_name="queue_imbalance_logistic",
            feature_names=QUEUE_IMBALANCE_FEATURES,
            symbol=args.symbol,
            start=start_str,
            end=end_str,
            artifact_tag=args.artifact_tag,
            experiment_fingerprint=experiment_fingerprint,
            dataset_sha256=dataset_hash,
            source_tree_sha256=recorder.source_state[
                "source_tree_sha256"
            ],
        )
        full_artifacts = load_model_artifacts(
            models_dir,
            model_name="full_logistic",
            feature_names=FEATURE_COLUMNS,
            symbol=args.symbol,
            start=start_str,
            end=end_str,
            artifact_tag=args.artifact_tag,
            experiment_fingerprint=experiment_fingerprint,
            dataset_sha256=dataset_hash,
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
        majority, fit_summary = _training_majority_counts(
            location,
            batch_size=args.batch_size,
        )

        pooled = {
            (model_name, horizon): TernaryMetricAccumulator()
            for model_name in MODEL_ORDER
            for horizon in HORIZONS
        }
        daily = {
            (model_name, horizon): DailyTernaryMetricAccumulator()
            for model_name in MODEL_ORDER
            for horizon in HORIZONS
        }
        test_rows = model_split_row_count(location, "test", eligible_only=True)
        print(f"Model-eligible test rows: {test_rows:,}")

        timestamp_min = None
        timestamp_max = None
        columns = (
            "timestamp",
            *(f"y_{horizon}" for horizon in HORIZONS),
            *FEATURE_COLUMNS,
        )
        rows_seen = 0
        for batch_number, batch in enumerate(
            iter_model_split_batches(
                location,
                "test",
                columns,
                eligible_only=True,
                batch_size=args.batch_size,
            ),
            start=1,
        ):
            rows_seen += len(batch)
            batch_min = batch["timestamp"].min()
            batch_max = batch["timestamp"].max()
            timestamp_min = (
                batch_min if timestamp_min is None else min(timestamp_min, batch_min)
            )
            timestamp_max = (
                batch_max if timestamp_max is None else max(timestamp_max, batch_max)
            )
            for horizon in HORIZONS:
                y_true = batch[f"y_{horizon}"].to_numpy(dtype=np.int8)
                predictions = {
                    "majority_baseline": np.full(
                        len(batch),
                        majority[horizon],
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
                predictions["full_logistic"] = full_artifacts[horizon].classes[
                    np.argmax(probabilities, axis=1)
                ].astype(np.int8)
                threshold_down, threshold_up = frozen_thresholds[horizon]
                predictions["full_logistic_thresholded"] = _threshold_predictions(
                    probabilities,
                    threshold_down=threshold_down,
                    threshold_up=threshold_up,
                )
                for model_name, y_pred in predictions.items():
                    key = (model_name, horizon)
                    pooled[key].update(y_true, y_pred)
                    daily[key].update(batch["timestamp"], y_true, y_pred)
            print(
                f"Processed test batch {batch_number}: {rows_seen:,}/{test_rows:,}",
                flush=True,
            )

        if rows_seen != test_rows:
            raise RuntimeError(
                f"Test row-count mismatch: streamed {rows_seen}, expected {test_rows}."
            )
        print(f"Test timestamp range: {timestamp_min} -> {timestamp_max}")

        test_tables, nonzero_table, daily_table = _metric_outputs(pooled, daily)
        test_comparison = test_tables["aggregate"].sort_values(
            ["horizon", "model"]
        ).reset_index(drop=True)
        ranking = validation_ranking_table(test_comparison)
        decay = signal_decay_table(test_comparison)
        full_deltas = compute_deltas(
            test_comparison,
            "full_logistic",
            ["majority_baseline", "queue_imbalance_logistic"],
        )
        thresholded_deltas = compute_deltas(
            test_comparison,
            "full_logistic_thresholded",
            [
                "majority_baseline",
                "queue_imbalance_logistic",
                "full_logistic",
            ],
        )
        coefficients = pd.concat(
            [
                *(
                    coefficient_table_from_artifact(queue_artifacts[horizon])
                    for horizon in HORIZONS
                ),
                *(
                    coefficient_table_from_artifact(full_artifacts[horizon])
                    for horizon in HORIZONS
                ),
            ],
            ignore_index=True,
        )
        threshold_table = pd.DataFrame(
            [
                {
                    "model": "full_logistic_thresholded",
                    "horizon": horizon,
                    "threshold_down": frozen_thresholds[horizon][0],
                    "threshold_up": frozen_thresholds[horizon][1],
                    "threshold_source": (
                        "validation_selected_"
                        f"{args.threshold_selection_metric}"
                    ),
                    "dataset_sha256": full_artifacts[horizon].dataset_sha256,
                    "experiment_fingerprint": (
                        full_artifacts[horizon].experiment_fingerprint
                    ),
                    "model_created_utc": full_artifacts[horizon].created_utc,
                    "model_parameter_fingerprint": (
                        full_artifacts[horizon].parameter_fingerprint()
                    ),
                }
                for horizon in HORIZONS
            ]
        )

        for name, table in test_tables.items():
            atomic_write_csv(
                results_dir / f"{output_prefix}_{name}.csv",
                table,
            )
        atomic_write_csv(
            results_dir / f"{output_prefix}_nonzero_subset.csv",
            nonzero_table,
        )
        atomic_write_csv(
            results_dir / f"{output_prefix}_daily_blocks.csv",
            daily_table,
        )
        atomic_write_csv(
            results_dir / f"{output_prefix}_majority_fit_summary.csv",
            fit_summary,
        )
        atomic_write_csv(
            results_dir / f"{output_prefix}_logistic_coefficients.csv",
            coefficients,
        )
        atomic_write_csv(
            reports_dir / f"{output_prefix}_frozen_thresholds.csv",
            threshold_table,
        )
        atomic_write_csv(
            reports_dir / f"{output_prefix}_ranking.csv",
            ranking,
        )
        atomic_write_csv(
            reports_dir / f"{output_prefix}_signal_decay.csv",
            decay,
        )
        atomic_write_csv(
            reports_dir / f"{output_prefix}_full_vs_baselines_deltas.csv",
            full_deltas,
        )
        atomic_write_csv(
            reports_dir / f"{output_prefix}_thresholded_deltas.csv",
            thresholded_deltas,
        )

        print()
        print("Final test aggregate comparison:")
        print(test_comparison.to_string(index=False))
        print(f"Run metadata: {metadata_path}")
        print("Done. Final test evaluation completed.")


if __name__ == "__main__":
    main()
