from __future__ import annotations

import argparse
import gc
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
    DEFAULT_THRESHOLD_GRID,
    auc_average_precision_table,
    best_threshold_confusion_tables,
    best_thresholds_table,
    lift_tables,
    probability_by_true_class_table,
    threshold_grid_results,
)
from src.horizon_checkpoints import HorizonCheckpointStore  # noqa: E402
from src.model_dataset_io import (  # noqa: E402
    iter_model_split_batches,
    model_dataset_sha256,
    model_split_row_count,
    resolve_model_dataset,
)
from src.modeling.model_artifacts import (  # noqa: E402
    load_frozen_logistic_artifact,
    predict_probabilities_from_artifact,
    validate_artifact_compatibility,
)
from src.protocol import (  # noqa: E402
    DEFAULT_HORIZONS,
    ExperimentSpec,
    FEATURE_COLUMNS,
    validate_protocol_symbol,
)
from src.run_provenance import RunRecorder  # noqa: E402


HORIZONS = DEFAULT_HORIZONS
MODEL_NAME = "full_logistic"
TABLE_TO_SUFFIX = {
    "auc_ap": "auc_ap",
    "probability_by_true_class": "probability_by_true_class",
    "lift": "lift",
    "threshold_grid": "threshold_grid",
    "threshold_nonzero": "threshold_nonzero",
    "best_thresholds": "best_thresholds",
    "best_threshold_confusion_counts": "best_threshold_confusion_counts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact validation probability diagnostics with disk-backed "
            "per-horizon arrays."
        )
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--artifact-tag", default=None)
    parser.add_argument(
        "--batch-size",
        "--prediction-chunk-size",
        dest="batch_size",
        type=int,
        default=1_000_000,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def _compute_horizon_tables(
    *,
    location,
    artifact,
    horizon: int,
    validation_rows: int,
    batch_size: int,
    temporary_root: Path,
    dataset_hash: str,
    experiment_fingerprint: str,
) -> dict[str, pd.DataFrame]:
    with tempfile.TemporaryDirectory(
        dir=temporary_root,
        prefix=f"probability_h{horizon}_",
    ) as directory:
        workdir = Path(directory)
        labels = np.memmap(
            workdir / "labels.int8",
            mode="w+",
            dtype=np.int8,
            shape=(validation_rows,),
        )
        probabilities = np.memmap(
            workdir / "probabilities.float64",
            mode="w+",
            dtype=np.float64,
            shape=(validation_rows, 3),
        )
        offset = 0
        columns = (f"y_{horizon}", *FEATURE_COLUMNS)
        for batch_number, batch in enumerate(
            iter_model_split_batches(
                location,
                "validation",
                columns,
                eligible_only=True,
                batch_size=batch_size,
            ),
            start=1,
        ):
            stop = offset + len(batch)
            labels[offset:stop] = batch[f"y_{horizon}"].to_numpy(
                dtype=np.int8
            )
            probabilities[offset:stop] = predict_probabilities_from_artifact(
                artifact,
                batch,
                chunk_size=batch_size,
            )
            offset = stop
            print(
                f"h={horizon} probability batch {batch_number}: "
                f"{offset:,}/{validation_rows:,}",
                flush=True,
            )
        if offset != validation_rows:
            raise RuntimeError(
                f"Validation row mismatch: wrote {offset}, "
                f"expected {validation_rows}."
            )
        labels.flush()
        probabilities.flush()

        y_validation = pd.Series(labels, copy=False)
        probability_frame = pd.DataFrame(
            probabilities,
            columns=["proba_-1", "proba_0", "proba_1"],
            copy=False,
        )
        auc_ap = auc_average_precision_table(
            y_validation,
            probability_frame,
            horizon,
            "validation",
            MODEL_NAME,
        )
        probability_summary = probability_by_true_class_table(
            y_validation,
            probability_frame,
            horizon,
            "validation",
            MODEL_NAME,
        )
        lift = lift_tables(
            y_validation,
            probability_frame,
            horizon,
            "validation",
            MODEL_NAME,
            n_bins=10,
        )
        threshold_grid, threshold_nonzero = threshold_grid_results(
            y_validation,
            probability_frame,
            horizon,
            "validation",
            "full_logistic_thresholded",
            threshold_grid=DEFAULT_THRESHOLD_GRID,
        )
        best_thresholds = best_thresholds_table(threshold_grid)
        best_thresholds["dataset_sha256"] = dataset_hash
        best_thresholds["experiment_fingerprint"] = experiment_fingerprint
        best_thresholds["model_created_utc"] = artifact.created_utc
        best_thresholds["model_parameter_fingerprint"] = (
            artifact.parameter_fingerprint()
        )
        best_confusion = best_threshold_confusion_tables(
            y_validation,
            probability_frame,
            best_thresholds,
            "validation",
            "full_logistic_thresholded",
            selection_metric="macro_f1",
        )

        return {
            "auc_ap": auc_ap,
            "probability_by_true_class": probability_summary,
            "lift": lift,
            "threshold_grid": threshold_grid,
            "threshold_nonzero": threshold_nonzero,
            "best_thresholds": best_thresholds,
            "best_threshold_confusion_counts": best_confusion,
        }


def _publish(
    collected: dict[int, dict[str, pd.DataFrame]],
    *,
    reports_dir: Path,
    prefix: str,
) -> None:
    for table_name, suffix in TABLE_TO_SUFFIX.items():
        combined = pd.concat(
            [
                collected[horizon][table_name]
                for horizon in sorted(collected)
            ],
            ignore_index=True,
        )
        atomic_write_csv(reports_dir / f"{prefix}_{suffix}.csv", combined)


def main() -> None:
    args = parse_args()
    validate_protocol_symbol(args.symbol)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")

    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    models_dir = PROJECT_ROOT / "outputs" / "models"
    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"
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
        "probability_diagnostics_full_logistic",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    location = resolve_model_dataset(processed_dir, dataset_stem)
    metadata_path = reports_dir / "run_metadata" / f"{prefix}.json"
    expected_outputs = [
        reports_dir / f"{prefix}_{suffix}.csv"
        for suffix in TABLE_TO_SUFFIX.values()
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"Probability diagnostic outputs already exist: {existing}. "
            "Use --resume or --overwrite."
        )

    with RunRecorder(
        metadata_path,
        project_root=PROJECT_ROOT,
        stage="probability_diagnostics",
        arguments=vars(args),
        require_clean_git=args.require_clean_git,
    ) as recorder:
        print("=" * 80)
        print("PROBABILITY DIAGNOSTICS: EXACT DISK-BACKED PASS")
        print("=" * 80)
        dataset_hash = model_dataset_sha256(location)
        recorder.set_dataset_sha256(dataset_hash)
        experiment_fingerprint = ExperimentSpec().fingerprint()
        recorder.set_experiment_fingerprint(experiment_fingerprint)
        validation_rows = model_split_row_count(
            location,
            "validation",
            eligible_only=True,
        )
        print(f"Validation rows: {validation_rows:,}")

        identity = {
            "stage": "probability_diagnostics",
            "dataset_sha256": dataset_hash,
            "experiment_fingerprint": experiment_fingerprint,
            "source_tree_sha256": recorder.source_state["source_tree_sha256"],
            "threshold_grid": list(DEFAULT_THRESHOLD_GRID),
            "n_lift_bins": 10,
        }
        checkpoint_store = HorizonCheckpointStore(
            checkpoints_dir / prefix,
            identity=identity,
        )
        collected = {}

        for horizon in HORIZONS:
            stem = frozen_model_artifact_stem(
                MODEL_NAME,
                args.symbol,
                start_str,
                end_str,
                horizon,
                args.artifact_tag,
            )
            artifact = load_frozen_logistic_artifact(models_dir / stem)
            validate_artifact_compatibility(
                artifact,
                model_name=MODEL_NAME,
                horizon=horizon,
                feature_names=FEATURE_COLUMNS,
                experiment_fingerprint=experiment_fingerprint,
                dataset_sha256=dataset_hash,
                source_tree_sha256=recorder.source_state[
                    "source_tree_sha256"
                ],
            )
            parameter_fingerprint = artifact.parameter_fingerprint()
            if args.resume and checkpoint_store.exists(horizon):
                tables = checkpoint_store.load(
                    horizon,
                    artifact_parameter_fingerprint=parameter_fingerprint,
                )
                print(f"Resumed probability diagnostics h={horizon}.")
            else:
                if checkpoint_store.exists(horizon) and not args.overwrite:
                    raise FileExistsError(
                        f"Checkpoint exists for horizon {horizon}. "
                        "Use --resume or --overwrite."
                    )
                tables = _compute_horizon_tables(
                    location=location,
                    artifact=artifact,
                    horizon=horizon,
                    validation_rows=validation_rows,
                    batch_size=args.batch_size,
                    temporary_root=temporary_root,
                    dataset_hash=dataset_hash,
                    experiment_fingerprint=experiment_fingerprint,
                )
                checkpoint_store.save(
                    horizon,
                    tables=tables,
                    artifact_parameter_fingerprint=parameter_fingerprint,
                    overwrite=args.overwrite,
                )
                print(f"Committed probability checkpoint h={horizon}.")
            collected[horizon] = tables
            _publish(collected, reports_dir=reports_dir, prefix=prefix)
            print(tables["auc_ap"].to_string(index=False))
            print(tables["best_thresholds"].to_string(index=False))
            gc.collect()

        print(f"Run metadata: {metadata_path}")
        print("Done. Probability diagnostics completed.")


if __name__ == "__main__":
    main()
