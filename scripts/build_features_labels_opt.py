from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_naming import tagged_artifact_stem  # noqa: E402
from src.feature_builder_opt_float_64 import (  # noqa: E402
    build_feature_table,
    feature_summary,
    label_distribution,
    save_feature_summary,
    save_feature_table,
    save_label_distribution,
)
from src.streaming_features import build_feature_table_streaming  # noqa: E402
from src.data_loader import parse_date  # noqa: E402
from src.protocol import ExperimentSpec, validate_protocol_symbol  # noqa: E402
from src.run_provenance import RunRecorder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build memory-lean labels and features with float64 "
            "log-return-derived columns."
        )
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
        "--execution-mode",
        choices=["streaming", "in-memory"],
        default="streaming",
        help=(
            "Use bounded-memory quote/trade chunks by default. In-memory mode "
            "is retained for small parity checks."
        ),
    )
    parser.add_argument(
        "--event-chunk-size",
        type=int,
        default=1_000_000,
        help="Core quote events per streaming chunk.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing feature table.",
    )
    parser.add_argument("--require-clean-git", action="store_true")

    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    validate_protocol_symbol(args.symbol)
    
    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")

    interim_dir = PROJECT_ROOT / "data" / "interim"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"

    events_path = processed_dir / f"quote_events_{args.symbol}_{start_str}_to_{end_str}.parquet"
    trades_path = interim_dir / f"raw_trades_{args.symbol}_{start_str}_to_{end_str}.parquet"

    if not events_path.exists():
        raise FileNotFoundError(f"Missing quote events file: {events_path}")

    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades file: {trades_path}")

    feature_stem = tagged_artifact_stem(
        "feature_table",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    label_dist_stem = tagged_artifact_stem(
        "label_distribution",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    feature_summary_stem = tagged_artifact_stem(
        "feature_summary",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    feature_path = processed_dir / f"{feature_stem}.parquet"
    label_dist_path = reports_dir / f"{label_dist_stem}.csv"
    feature_summary_path = reports_dir / f"{feature_summary_stem}.csv"

    print("=" * 80)
    print("STEP 6: BUILD MEMORY-LEAN LABELS AND FLOAT64 LOG-RETURN FEATURES")
    print("=" * 80)

    event_columns = [
        "event_id",
        "timestamp",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
    ]

    trade_columns = [
        "timestamp",
        "quantity",
        "buyer_is_maker",
    ]

    if args.execution_mode == "streaming":
        print(f"Streaming quote events: {events_path}")
        print(f"Streaming trade windows: {trades_path}")
        report = build_feature_table_streaming(
            events_path,
            trades_path,
            feature_path,
            event_chunk_size=args.event_chunk_size,
            compression=args.compression,
            overwrite=args.overwrite,
        )
        label_dist = report.label_distribution()
        feat_summary = report.feature_summary()
        print()
        print(f"Feature rows: {report.rows_written:,}")
        print(f"Complete feature rows: {report.feature_complete_rows:,}")
        print(
            "Incomplete feature rows: "
            f"{report.rows_written - report.feature_complete_rows:,}"
        )
    else:
        if feature_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {feature_path}. Pass --overwrite "
                "only after verifying the target."
            )
        print(f"Reading quote events: {events_path}")
        events = pd.read_parquet(events_path, columns=event_columns)
        print(f"Reading trades:       {trades_path}")
        trades = pd.read_parquet(trades_path, columns=trade_columns)
        print()
        print("Building in-memory parity path...")
        feature_table = build_feature_table(events, trades)
        label_dist = label_distribution(feature_table)
        feat_summary = feature_summary(feature_table)
        save_feature_table(feature_table, feature_path)

    print()
    print("Label distribution:")
    print(label_dist.to_string(index=False))

    print()
    print("Feature summary:")
    print(feat_summary.to_string(index=False))

    print()
    save_label_distribution(label_dist, label_dist_path)
    save_feature_summary(feat_summary, feature_summary_path)

    print()
    print("Note: no separate labeled_events parquet was saved in the lean pipeline.")
    print("Labels y_10, y_20, y_50 are stored directly in the feature_table.")
    print()
    print("Done. Memory-lean labels and float64 log-return features built.")


def main() -> None:
    args = parse_args()
    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")
    feature_stem = tagged_artifact_stem(
        "feature_table",
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
        / f"{feature_stem}.json"
    )
    with RunRecorder(
        metadata_path,
        project_root=PROJECT_ROOT,
        stage="build_features_labels",
        arguments=vars(args),
        experiment_fingerprint=ExperimentSpec().fingerprint(),
        require_clean_git=args.require_clean_git,
    ):
        run(args)
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
