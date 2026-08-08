from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_naming import tagged_artifact_stem  # noqa: E402
from src.atomic_io import atomic_write_csv  # noqa: E402
from src.data_loader import parse_date  # noqa: E402
from src.evaluation.metrics import concat_evaluation_tables  # noqa: E402
from src.evaluation.streaming_metrics import (  # noqa: E402
    TernaryMetricAccumulator,
)
from src.model_dataset_io import (  # noqa: E402
    iter_model_split_batches,
    model_dataset_sha256,
    model_split_row_count,
    resolve_model_dataset,
)
from src.protocol import (  # noqa: E402
    DEFAULT_HORIZONS,
    ExperimentSpec,
    TERNARY_LABELS,
    validate_protocol_symbol,
)
from src.run_provenance import RunRecorder  # noqa: E402


HORIZONS = DEFAULT_HORIZONS
MODEL_NAME = "majority_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the train-only majority baseline with bounded memory."
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--artifact-tag", default=None)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


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
    dataset_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    prefix = tagged_artifact_stem(
        MODEL_NAME,
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    location = resolve_model_dataset(processed_dir, dataset_stem)
    metadata_path = reports_dir / "run_metadata" / f"{prefix}.json"
    expected_outputs = [
        results_dir / f"{prefix}_{name}.csv"
        for name in (
            "aggregate",
            "class_proportions",
            "per_class",
            "confusion_counts",
            "confusion_true_normalized",
            "nonzero_subset",
            "fit_summary",
        )
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Majority outputs already exist: {existing}. "
            "Pass --overwrite for a deliberate replacement."
        )

    with RunRecorder(
        metadata_path,
        project_root=PROJECT_ROOT,
        stage=MODEL_NAME,
        arguments=vars(args),
        require_clean_git=args.require_clean_git,
    ) as recorder:
        print("=" * 80)
        print("BASELINE: MAJORITY CLASS")
        print("=" * 80)
        dataset_hash = model_dataset_sha256(location)
        recorder.set_dataset_sha256(dataset_hash)
        experiment_fingerprint = ExperimentSpec().fingerprint()
        recorder.set_experiment_fingerprint(experiment_fingerprint)
        columns = tuple(f"y_{horizon}" for horizon in HORIZONS)

        counts = {
            horizon: np.zeros(3, dtype=np.int64)
            for horizon in HORIZONS
        }
        for batch in iter_model_split_batches(
            location,
            "train",
            columns,
            eligible_only=True,
            batch_size=args.batch_size,
        ):
            for horizon in HORIZONS:
                labels = batch[f"y_{horizon}"].to_numpy(dtype=np.int8)
                if not np.isin(labels, np.asarray(TERNARY_LABELS)).all():
                    raise ValueError(
                        f"Unexpected train labels for horizon {horizon}."
                    )
                counts[horizon] += np.bincount(labels + 1, minlength=3)
        majorities = {
            horizon: int(TERNARY_LABELS[int(np.argmax(class_counts))])
            for horizon, class_counts in counts.items()
        }

        accumulators = {
            (split, horizon): TernaryMetricAccumulator()
            for split in ("train", "validation")
            for horizon in HORIZONS
        }
        for split in ("train", "validation"):
            expected_rows = model_split_row_count(
                location,
                split,
                eligible_only=True,
            )
            rows_seen = 0
            for batch in iter_model_split_batches(
                location,
                split,
                columns,
                eligible_only=True,
                batch_size=args.batch_size,
            ):
                rows_seen += len(batch)
                for horizon in HORIZONS:
                    y_true = batch[f"y_{horizon}"].to_numpy(dtype=np.int8)
                    y_pred = np.full(
                        len(batch),
                        majorities[horizon],
                        dtype=np.int8,
                    )
                    accumulators[(split, horizon)].update(y_true, y_pred)
            if rows_seen != expected_rows:
                raise RuntimeError(
                    f"{split} row mismatch: {rows_seen} != {expected_rows}."
                )

        evaluations = []
        nonzero = []
        for horizon in HORIZONS:
            for split in ("train", "validation"):
                accumulator = accumulators[(split, horizon)]
                evaluations.append(
                    accumulator.evaluation_tables(
                        split=split,
                        horizon=horizon,
                        model_name=MODEL_NAME,
                    )
                )
                nonzero.append(
                    accumulator.nonzero_table(
                        split=split,
                        horizon=horizon,
                        model_name=MODEL_NAME,
                    )
                )
        tables = concat_evaluation_tables(evaluations)
        nonzero_table = pd.concat(nonzero, ignore_index=True)
        fit_rows = []
        for horizon in HORIZONS:
            class_counts = counts[horizon]
            total = int(class_counts.sum())
            fit_rows.append(
                {
                    "horizon": horizon,
                    "majority_class": majorities[horizon],
                    "train_count_down": int(class_counts[0]),
                    "train_count_unchanged": int(class_counts[1]),
                    "train_count_up": int(class_counts[2]),
                    "train_prop_down": float(class_counts[0] / total),
                    "train_prop_unchanged": float(class_counts[1] / total),
                    "train_prop_up": float(class_counts[2] / total),
                }
            )
        fit_summary = pd.DataFrame(fit_rows)

        for name, table in tables.items():
            atomic_write_csv(
                results_dir / f"{prefix}_{name}.csv",
                table,
            )
        atomic_write_csv(
            results_dir / f"{prefix}_nonzero_subset.csv",
            nonzero_table,
        )
        atomic_write_csv(
            results_dir / f"{prefix}_fit_summary.csv",
            fit_summary,
        )
        print("Aggregate metrics:")
        print(tables["aggregate"].to_string(index=False))
        print(f"Run metadata: {metadata_path}")
        print("Done. Majority baseline completed.")


if __name__ == "__main__":
    main()
