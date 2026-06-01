from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import parse_date # noqa: E402
from src.artifact_naming import tagged_artifact_stem  # noqa: E402

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
            "'v2_float64_features'."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.symbol != "BTCUSDT":
        raise ValueError("Protocol violation: symbol must remain BTCUSDT.")
    
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

    print(f"Reading feature table: {feature_path}")
    feature_table = pd.read_parquet(feature_path)

    print()
    print("Feature table:")
    print(f"Rows: {len(feature_table):,}")
    print(f"Timestamp range: {feature_table['timestamp'].min()} → {feature_table['timestamp'].max()}")

    print()
    print("Building model dataset with chronological split...")
    model_dataset, boundaries = build_model_dataset(
        feature_table,
        start=start_str,
        end=end_str,
    )

    summary = split_summary(model_dataset, boundaries)

    print()
    print("Split boundaries:")
    print(f"Sample start:         {boundaries['sample_start']}")
    print(f"Train end:            {boundaries['train_end']}")
    print(f"Validation end:       {boundaries['validation_end']}")
    print(f"Sample end exclusive: {boundaries['sample_end_exclusive']}")

    print()
    print("Split summary:")
    print(summary.to_string(index=False))

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
    summary_path = reports_dir / f"{summary_stem}.csv"

    print()
    save_model_dataset(model_dataset, model_path)
    save_split_summary(summary, summary_path)

    print()
    print("Done. Chronological split dataset built.")


if __name__ == "__main__":
    main()
