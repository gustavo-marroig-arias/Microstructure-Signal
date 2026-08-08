from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import parse_date  # noqa: E402
from src.artifact_naming import tagged_artifact_stem  # noqa: E402
from src.partitioned_splits import build_partitioned_model_dataset  # noqa: E402
from src.model_dataset_io import (  # noqa: E402
    model_dataset_sha256,
    resolve_model_dataset,
)
from src.protocol import ExperimentSpec, validate_protocol_symbol  # noqa: E402
from src.run_provenance import RunRecorder  # noqa: E402

from src.split_config import (  # noqa: E402
    build_model_dataset,
    save_model_dataset,
    save_split_summary,
    split_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chronological train/validation/test split dataset."
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Protocol symbol. Must be BTCUSDT.",
    )
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help=(
            "Optional filename tag inserted after the artifact kind, e.g. "
            "'v3_fixed_window_features'."
        ),
    )
    parser.add_argument(
        "--storage-layout",
        choices=["partitioned", "monolithic"],
        default="partitioned",
        help=(
            "Write split-specific parquet files by default. The monolithic "
            "layout is retained only for backward-compatible small runs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing partitioned output directory.",
    )
    parser.add_argument("--require-clean-git", action="store_true")

    return parser.parse_args()


def run(args: argparse.Namespace, recorder: RunRecorder) -> None:
    validate_protocol_symbol(args.symbol)
    
    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"

    feature_stem = tagged_artifact_stem(
        "feature_table",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    feature_path = processed_dir / f"{feature_stem}.parquet"

    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature table: {feature_path}")

    print("=" * 80)
    print("STEP 7: BUILD CHRONOLOGICAL SPLITS")
    print("=" * 80)

    model_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    summary_stem = tagged_artifact_stem(
        "split_summary",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    model_path = processed_dir / f"{model_stem}.parquet"
    model_dir = processed_dir / model_stem
    summary_path = reports_dir / f"{summary_stem}.csv"

    if args.storage_layout == "partitioned":
        print(f"Streaming feature table: {feature_path}")
        print()
        print("Building split-specific model dataset...")
        summary, boundaries, manifest_path = build_partitioned_model_dataset(
            feature_path,
            model_dir,
            start=start_str,
            end=end_str,
            overwrite=args.overwrite,
        )
        print(f"Saved partition manifest: {manifest_path}")
    else:
        print(f"Reading feature table: {feature_path}")
        feature_table = pd.read_parquet(feature_path)

        print()
        print("Feature table:")
        print(f"Rows: {len(feature_table):,}")
        print(
            "Timestamp range: "
            f"{feature_table['timestamp'].min()} -> "
            f"{feature_table['timestamp'].max()}"
        )
        print()
        print("Building monolithic model dataset...")
        model_dataset, boundaries = build_model_dataset(
            feature_table,
            start=start_str,
            end=end_str,
        )
        summary = split_summary(model_dataset, boundaries)
        save_model_dataset(model_dataset, model_path)

    print()
    print("Split boundaries:")
    print(f"Sample start:         {boundaries['sample_start']}")
    print(f"Train end:            {boundaries['train_end']}")
    print(f"Validation end:       {boundaries['validation_end']}")
    print(f"Sample end exclusive: {boundaries['sample_end_exclusive']}")

    print()
    print("Split summary:")
    print(summary.to_string(index=False))

    print()
    save_split_summary(summary, summary_path)
    location = resolve_model_dataset(processed_dir, model_stem)
    recorder.set_dataset_sha256(model_dataset_sha256(location))

    print()
    print("Done. Chronological split dataset built.")


def main() -> None:
    args = parse_args()
    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")
    model_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    metadata_path = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / "run_metadata"
        / f"{model_stem}.json"
    )
    with RunRecorder(
        metadata_path,
        project_root=PROJECT_ROOT,
        stage="build_splits",
        arguments=vars(args),
        experiment_fingerprint=ExperimentSpec().fingerprint(),
        require_clean_git=args.require_clean_git,
    ) as recorder:
        run(args, recorder)
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
