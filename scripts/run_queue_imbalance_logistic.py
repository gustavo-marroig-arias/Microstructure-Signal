from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_naming import (  # noqa: E402
    frozen_model_artifact_stem,
    tagged_artifact_stem,
)
from src.data_loader import parse_date  # noqa: E402
from src.model_dataset_io import (  # noqa: E402
    model_dataset_sha256,
    model_split_row_count,
    resolve_model_dataset,
)
from src.modeling.logistic_models import LogisticModelConfig  # noqa: E402
from src.modeling.offline_experiment import (  # noqa: E402
    run_checkpointed_logistic_experiment,
)
from src.protocol import (  # noqa: E402
    DEFAULT_HORIZONS,
    ExperimentSpec,
    QUEUE_IMBALANCE_FEATURES,
    validate_protocol_symbol,
)
from src.run_provenance import RunRecorder  # noqa: E402


HORIZONS = DEFAULT_HORIZONS
MODEL_NAME = "queue_imbalance_logistic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the queue-imbalance logistic baseline with bounded-memory "
            "validation and transactional per-horizon checkpoints."
        )
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--artifact-tag", default=None)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--solver",
        default="lbfgs",
        choices=["lbfgs", "saga", "sag", "newton-cg"],
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--allow-nonconvergence", action="store_true")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(HORIZONS),
        choices=list(HORIZONS),
    )
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--resume", action="store_true")
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
    models_dir = PROJECT_ROOT / "outputs" / "models"
    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    temporary_dir = PROJECT_ROOT / "outputs" / "tmp"

    dataset_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    result_prefix = tagged_artifact_stem(
        MODEL_NAME,
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    location = resolve_model_dataset(processed_dir, dataset_stem)
    metadata_path = reports_dir / "run_metadata" / f"{result_prefix}.json"

    with RunRecorder(
        metadata_path,
        project_root=PROJECT_ROOT,
        stage=MODEL_NAME,
        arguments=vars(args),
        require_clean_git=args.require_clean_git,
    ) as recorder:
        print("=" * 80)
        print("BASELINE: QUEUE-IMBALANCE LOGISTIC REGRESSION")
        print("=" * 80)
        print("Hashing model dataset for provenance...", flush=True)
        dataset_hash = model_dataset_sha256(location)
        recorder.set_dataset_sha256(dataset_hash)
        experiment_fingerprint = ExperimentSpec().fingerprint()
        recorder.set_experiment_fingerprint(experiment_fingerprint)

        train_rows = model_split_row_count(
            location,
            "train",
            eligible_only=True,
        )
        print(f"Model-eligible train rows: {train_rows:,}")
        print(
            "Train scaling uses a disk-backed matrix; validation is read in "
            "ordered batches and is never concatenated with train."
        )

        config = LogisticModelConfig(
            C=args.C,
            max_iter=args.max_iter,
            solver=args.solver,
            n_jobs=args.n_jobs,
            tol=args.tol,
            fail_on_nonconvergence=not args.allow_nonconvergence,
        )
        artifact_bases = {
            horizon: models_dir
            / frozen_model_artifact_stem(
                MODEL_NAME,
                args.symbol,
                start_str,
                end_str,
                horizon,
                args.artifact_tag,
            )
            for horizon in args.horizons
        }
        result = run_checkpointed_logistic_experiment(
            location=location,
            model_name=MODEL_NAME,
            feature_columns=QUEUE_IMBALANCE_FEATURES,
            horizons=args.horizons,
            config=config,
            experiment_fingerprint=experiment_fingerprint,
            dataset_sha256=dataset_hash,
            source_state=recorder.source_state,
            checkpoint_root=checkpoints_dir / result_prefix,
            artifact_bases=artifact_bases,
            results_dir=results_dir,
            result_prefix=result_prefix,
            batch_size=args.batch_size,
            temporary_root=temporary_dir,
            resume=args.resume,
            overwrite=args.overwrite,
        )

        print()
        print("Aggregate metrics:")
        print(result.tables["aggregate"].to_string(index=False))
        print()
        print("Non-zero subset metrics:")
        print(result.nonzero.to_string(index=False))
        print()
        print(f"Run metadata: {metadata_path}")
        print("Done. Queue-imbalance logistic models completed.")


if __name__ == "__main__":
    main()
