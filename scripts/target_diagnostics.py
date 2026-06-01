from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.target_diagnostics import (  # noqa: E402
    build_dataset_manifest,
    save_dataset_manifest,
    save_target_diagnostics,
    target_diagnostics_by_split,
)

from src.data_loader import parse_date # noqa: E402
from src.artifact_naming import tagged_artifact_stem  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze dataset version and compute target diagnostics."
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

    dataset_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    dataset_path = processed_dir / f"{dataset_stem}.parquet"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing model dataset: {dataset_path}")
    
    print("=" * 80)
    print("STEP 8: DATASET FREEZE + TARGET DIAGNOSTICS")
    print("=" * 80)

    print(f"Reading model dataset: {dataset_path}")
    data = pd.read_parquet(dataset_path)

    print()
    print("Dataset summary:")
    print(f"Rows: {len(data):,}")
    print(f"Timestamp range: {data['timestamp'].min()} → {data['timestamp'].max()}")
    print()
    print(data.groupby("split")[["feature_complete", "is_boundary_drop", "model_eligible"]].sum())

    print()
    print("Building dataset manifest...")
    manifest = build_dataset_manifest(
        data=data,
        dataset_path=dataset_path,
        start=start_str,
        end=end_str,
        symbol=args.symbol,
    )

    manifest_stem = tagged_artifact_stem(
        "dataset_manifest",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    manifest_path = reports_dir / f"{manifest_stem}.json"
    save_dataset_manifest(manifest, manifest_path)

    print()
    print("Computing target diagnostics on model-eligible rows...")
    diagnostics = target_diagnostics_by_split(data, eligible_only=True)

    diagnostics_stem = tagged_artifact_stem(
        "target_diagnostics",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    diagnostics_path = reports_dir / f"{diagnostics_stem}.csv"
    save_target_diagnostics(diagnostics, diagnostics_path)

    print()
    print("Target diagnostics:")
    print(diagnostics.to_string(index=False))

    print()
    print("Done. Dataset version frozen and target diagnostics created.")


if __name__ == "__main__":
    main()

 
