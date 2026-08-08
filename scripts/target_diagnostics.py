from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.target_diagnostics import (  # noqa: E402
    build_streaming_dataset_manifest,
    save_dataset_manifest,
    save_target_diagnostics,
    streaming_target_diagnostics,
)

from src.data_loader import parse_date  # noqa: E402
from src.artifact_naming import tagged_artifact_stem  # noqa: E402
from src.model_dataset_io import (  # noqa: E402
    model_dataset_sha256,
    resolve_model_dataset,
)
from src.protocol import validate_protocol_symbol  # noqa: E402


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
            "'v3_fixed_window_features'."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    validate_protocol_symbol(args.symbol)
    
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
    location = resolve_model_dataset(processed_dir, dataset_stem)
    
    print("=" * 80)
    print("STEP 8: DATASET FREEZE + TARGET DIAGNOSTICS")
    print("=" * 80)

    print(f"Reading model dataset metadata from: {location.path}")
    diagnostics, split_summary, total_rows = streaming_target_diagnostics(
        location,
    )
    dataset_hash = model_dataset_sha256(location)

    print()
    print("Dataset summary:")
    print(f"Rows: {total_rows:,}")
    print(split_summary.to_string(index=False))

    print()
    print("Building dataset manifest...")
    manifest = build_streaming_dataset_manifest(
        split_summary=split_summary,
        total_rows=total_rows,
        dataset_path=location.path,
        dataset_sha256=dataset_hash,
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

 
