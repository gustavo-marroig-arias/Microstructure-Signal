from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import parse_date  # noqa: E402
from src.quality_checks import (  # noqa: E402
    combined_quality_report_from_parquet,
    save_quality_report,
)
from src.protocol import validate_protocol_symbol  # noqa: E402

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

    validate_protocol_symbol(args.symbol)
    
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
    print(f"Streaming quotes: {quotes_path}")
    print(f"Streaming trades: {trades_path}")

    print()
    print("Building quality report...")
    report = combined_quality_report_from_parquet(
        quotes_path,
        trades_path,
        expected_start=start_str,
        expected_end=end_str,
    )

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
