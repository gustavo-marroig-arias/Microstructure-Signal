from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
from src.model_dataset_io import (  # noqa: E402
    iter_model_split_batches,
    model_dataset_sha256,
    model_split_row_count,
    resolve_model_dataset,
)
from src.modeling.model_artifacts import (  # noqa: E402
    FrozenLogisticArtifact,
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
DIAGNOSTIC_COLUMNS = (
    "relative_spread",
    "realized_vol_20",
    "trade_intensity_1s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream compact test-only probabilities and predictions to Parquet."
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
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def load_thresholds(
    reports_dir: Path,
    *,
    symbol: str,
    start: str,
    end: str,
    artifact_tag: str | None,
    selection_metric: str,
    dataset_sha256: str,
    experiment_fingerprint: str,
    artifacts: dict[int, FrozenLogisticArtifact],
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


def main() -> None:
    args = parse_args()
    validate_protocol_symbol(args.symbol)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")

    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "outputs" / "models"
    results_dir = PROJECT_ROOT / "outputs" / "results"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"

    dataset_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    output_prefix = tagged_artifact_stem(
        "compact_test_predictions",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    output_path = results_dir / f"{output_prefix}.parquet"
    summary_path = reports_dir / f"{output_prefix}_summary.csv"
    metadata_path = reports_dir / "run_metadata" / f"{output_prefix}.json"
    existing = [path for path in (output_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Compact outputs already exist: {existing}. Pass --overwrite "
            "for a deliberate atomic replacement."
        )

    location = resolve_model_dataset(processed_dir, dataset_stem)
    with RunRecorder(
        metadata_path,
        project_root=PROJECT_ROOT,
        stage="compact_test_predictions",
        arguments=vars(args),
        require_clean_git=args.require_clean_git,
    ) as recorder:
        dataset_hash = model_dataset_sha256(location)
        recorder.set_dataset_sha256(dataset_hash)
        experiment_fingerprint = ExperimentSpec().fingerprint()
        recorder.set_experiment_fingerprint(experiment_fingerprint)

        artifacts = {}
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
            artifacts[horizon] = artifact
        thresholds = load_thresholds(
            reports_dir,
            symbol=args.symbol,
            start=start_str,
            end=end_str,
            artifact_tag=args.artifact_tag,
            selection_metric=args.threshold_selection_metric,
            dataset_sha256=dataset_hash,
            experiment_fingerprint=experiment_fingerprint,
            artifacts=artifacts,
        )

        test_rows = model_split_row_count(location, "test", eligible_only=True)
        print("=" * 80, flush=True)
        print("COMPACT TEST PREDICTIONS: STREAMING EXPORT", flush=True)
        print("=" * 80, flush=True)
        print(f"Test rows: {test_rows:,}", flush=True)

        results_dir.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_name(
            f".{output_path.name}.tmp-{uuid4().hex}"
        )
        writer: pq.ParquetWriter | None = None
        rows_seen = 0
        prediction_counts = {
            horizon: {
                "argmax": np.zeros(3, dtype=np.int64),
                "thresholded": np.zeros(3, dtype=np.int64),
            }
            for horizon in HORIZONS
        }
        columns = (
            "event_id",
            "timestamp",
            *(f"y_{horizon}" for horizon in HORIZONS),
            *FEATURE_COLUMNS,
        )
        try:
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
                output = batch.loc[
                    :,
                    ["event_id", "timestamp", *DIAGNOSTIC_COLUMNS],
                ].copy()
                output["relative_spread"] = output["relative_spread"].astype(
                    np.float32
                )
                output["realized_vol_20"] = output["realized_vol_20"].astype(
                    np.float32
                )
                output["trade_intensity_1s"] = output[
                    "trade_intensity_1s"
                ].astype(np.int32)

                for horizon in HORIZONS:
                    artifact = artifacts[horizon]
                    probabilities = predict_probabilities_from_artifact(
                        artifact,
                        batch,
                        chunk_size=args.batch_size,
                    )
                    argmax = artifact.classes[
                        np.argmax(probabilities, axis=1)
                    ].astype(np.int8)
                    threshold_down, threshold_up = thresholds[horizon]
                    thresholded = threshold_predictions(
                        probabilities,
                        threshold_down=threshold_down,
                        threshold_up=threshold_up,
                    )
                    output[f"y_true_h{horizon}"] = batch[
                        f"y_{horizon}"
                    ].to_numpy(dtype=np.int8)
                    output[f"y_pred_argmax_h{horizon}"] = argmax
                    output[f"y_pred_thresholded_h{horizon}"] = thresholded
                    output[f"proba_down_h{horizon}"] = probabilities[
                        :, 0
                    ].astype(np.float32)
                    output[f"proba_unchanged_h{horizon}"] = probabilities[
                        :, 1
                    ].astype(np.float32)
                    output[f"proba_up_h{horizon}"] = probabilities[:, 2].astype(
                        np.float32
                    )
                    prediction_counts[horizon]["argmax"] += np.bincount(
                        argmax + 1,
                        minlength=3,
                    )
                    prediction_counts[horizon]["thresholded"] += np.bincount(
                        thresholded + 1,
                        minlength=3,
                    )

                arrow_table = pa.Table.from_pandas(
                    output,
                    preserve_index=False,
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_output,
                        arrow_table.schema,
                        compression=args.compression,
                    )
                writer.write_table(arrow_table)
                rows_seen += len(output)
                print(
                    f"Wrote batch {batch_number}: {rows_seen:,}/{test_rows:,}",
                    flush=True,
                )

            if writer is None:
                raise RuntimeError("No compact prediction rows were written.")
            writer.close()
            writer = None
            if rows_seen != test_rows:
                raise RuntimeError(
                    f"Row-count mismatch: wrote {rows_seen}, expected {test_rows}."
                )
            os.replace(temporary_output, output_path)
        finally:
            if writer is not None:
                writer.close()
            temporary_output.unlink(missing_ok=True)

        summary_rows = []
        for horizon in HORIZONS:
            artifact = artifacts[horizon]
            threshold_down, threshold_up = thresholds[horizon]
            argmax_counts = prediction_counts[horizon]["argmax"]
            thresholded_counts = prediction_counts[horizon]["thresholded"]
            summary_rows.append(
                {
                    "horizon": horizon,
                    "threshold_selection_metric": (
                        args.threshold_selection_metric
                    ),
                    "threshold_down": threshold_down,
                    "threshold_up": threshold_up,
                    "dataset_sha256": dataset_hash,
                    "experiment_fingerprint": experiment_fingerprint,
                    "model_created_utc": artifact.created_utc,
                    "model_parameter_fingerprint": (
                        artifact.parameter_fingerprint()
                    ),
                    "argmax_pred_down_fraction": argmax_counts[0] / test_rows,
                    "argmax_pred_unchanged_fraction": (
                        argmax_counts[1] / test_rows
                    ),
                    "argmax_pred_up_fraction": argmax_counts[2] / test_rows,
                    "thresholded_pred_down_fraction": (
                        thresholded_counts[0] / test_rows
                    ),
                    "thresholded_pred_unchanged_fraction": (
                        thresholded_counts[1] / test_rows
                    ),
                    "thresholded_pred_up_fraction": (
                        thresholded_counts[2] / test_rows
                    ),
                }
            )
        atomic_write_csv(summary_path, pd.DataFrame(summary_rows))

        observed_rows = pq.ParquetFile(output_path).metadata.num_rows
        if observed_rows != test_rows:
            raise RuntimeError(
                "Written Parquet metadata row count differs from the test split."
            )
        print(f"Saved compact predictions: {output_path}", flush=True)
        print(f"Saved compact summary: {summary_path}", flush=True)
        print(f"Run metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
