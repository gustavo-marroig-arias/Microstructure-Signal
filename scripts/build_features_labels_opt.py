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
from src.data_loader import parse_date # noqa: E402


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

    interim_dir = PROJECT_ROOT / "data" / "interim"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"

    events_path = processed_dir / f"quote_events_{args.symbol}_{start_str}_to_{end_str}.parquet"
    trades_path = interim_dir / f"raw_trades_{args.symbol}_{start_str}_to_{end_str}.parquet"

    if not events_path.exists():
        raise FileNotFoundError(f"Missing quote events file: {events_path}")

    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades file: {trades_path}")

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

    print(f"Reading quote events: {events_path}")
    events = pd.read_parquet(events_path, columns=event_columns)

    print(f"Reading trades:       {trades_path}")
    trades = pd.read_parquet(trades_path, columns=trade_columns)

    print()
    print("Event table:")
    print(f"Rows: {len(events):,}")
    print(f"Timestamp range: {events['timestamp'].min()} → {events['timestamp'].max()}")

    print()
    print("Trade table:")
    print(f"Rows: {len(trades):,}")
    print(f"Timestamp range: {trades['timestamp'].min()} → {trades['timestamp'].max()}")

    print()
    print("Building memory-lean feature table with float64 log-return features...")
    feature_table = build_feature_table(events, trades)

    print()
    print("Feature table:")
    print(f"Rows: {len(feature_table):,}")
    print(f"Complete feature rows: {int(feature_table['feature_complete'].sum()):,}")
    print(f"Incomplete feature rows: {int((~feature_table['feature_complete']).sum()):,}")

    label_dist = label_distribution(feature_table)
    feat_summary = feature_summary(feature_table)

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

    print()
    print("Label distribution:")
    print(label_dist.to_string(index=False))

    print()
    print("Feature summary:")
    print(feat_summary.to_string(index=False))

    print()
    save_feature_table(feature_table, feature_path)
    save_label_distribution(label_dist, label_dist_path)
    save_feature_summary(feat_summary, feature_summary_path)

    print()
    print("Note: no separate labeled_events parquet was saved in the lean pipeline.")
    print("Labels y_10, y_20, y_50 are stored directly in the feature_table.")
    print()
    print("Done. Memory-lean labels and float64 log-return features built.")


if __name__ == "__main__":
    main()
