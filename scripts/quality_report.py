from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import parse_date  # noqa: E402
from src.quality_checks import combined_quality_report, save_quality_report  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create protocol-level raw data quality report."
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.symbol != "BTCUSDT":
        raise ValueError("Protocol violation: symbol must remain BTCUSDT.")
    
    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")

    interim_dir = PROJECT_ROOT / "data" / "interim"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"

    quotes_path = interim_dir / f"raw_quotes_{args.symbol}_{start_str}_to_{end_str}.parquet"
    trades_path = interim_dir / f"raw_trades_{args.symbol}_{start_str}_to_{end_str}.parquet"

    if not quotes_path.exists():
        raise FileNotFoundError(f"Missing quotes parquet: {quotes_path}")

    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades parquet: {trades_path}")

    print("=" * 80)
    print("STEP 3: RAW DATA QUALITY REPORT")
    print("=" * 80)
    print(f"Reading quotes: {quotes_path}")
    quotes = pd.read_parquet(quotes_path)

    print(f"Reading trades: {trades_path}")
    trades = pd.read_parquet(trades_path)

    print()
    print("Building quality report...")
    report = combined_quality_report(quotes, trades)

    output_path = reports_dir / f"raw_quality_report_{args.symbol}_{start_str}_to_{end_str}.csv"
    save_quality_report(report, output_path)

    print()
    # print("Full Quality report:")
    # print(report.to_string(index=False))
    print("Quality report summary:")
    summary_cols = [
        "utc_day",
        "quote_rows",
        "trade_rows",
        "max_quote_gap_seconds",
        "max_trade_gap_seconds",
        "invalid_quote_count",
        "invalid_trade_count",
        "keep_day",
        "drop_reason",
    ]
    print(report[summary_cols].to_string(index=False))

    print()
    print("Overall Summary:")
    print(f"Days checked: {len(report)}")
    print(f"Days kept:    {int(report['keep_day'].sum())}")
    print(f"Days dropped: {int((~report['keep_day']).sum())}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()