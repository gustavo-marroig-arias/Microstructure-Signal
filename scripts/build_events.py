from __future__ import annotations

import argparse
from pathlib import Path
import sys
import gc
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import parse_date  # noqa: E402
from src.event_builder_opt import (  # noqa: E402
    build_quote_events,
    build_quote_events_streaming,
    save_event_summary,
    save_quote_events,
)
from src.protocol import ExperimentSpec, validate_protocol_symbol  # noqa: E402
from src.run_provenance import RunRecorder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build distinct top-of-book quote-event stream."
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
        "--execution-mode",
        choices=["streaming", "in-memory"],
        default="streaming",
        help="Process parquet row groups by default.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing event table.",
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

    quotes_path = interim_dir / f"raw_quotes_{args.symbol}_{start_str}_to_{end_str}.parquet"

    quality_report_path = reports_dir / f"raw_quality_report_{args.symbol}_{start_str}_to_{end_str}.csv"

    if not quality_report_path.exists():
        raise FileNotFoundError(
            f"Missing quality report: {quality_report_path}. "
            "Run scripts/quality_report.py before building events."
        )

    quality_report = pd.read_csv(quality_report_path)

    if not quality_report["keep_day"].all():
        bad_days = quality_report.loc[~quality_report["keep_day"], ["utc_day", "drop_reason"]]
        raise ValueError(
            "Quality report contains dropped days. Refusing to build events.\n"
            f"{bad_days.to_string(index=False)}"
        )

    if not quotes_path.exists():
        raise FileNotFoundError(f"Missing quotes parquet: {quotes_path}")

    events_path = (
        processed_dir
        / f"quote_events_{args.symbol}_{start_str}_to_{end_str}.parquet"
    )
    summary_path = (
        reports_dir
        / f"quote_event_summary_{args.symbol}_{start_str}_to_{end_str}.csv"
    )

    print("=" * 80)
    print("STEP 4: BUILD DISTINCT QUOTE EVENTS")
    print("=" * 80)
    print(f"Quote source: {quotes_path}")

    quote_columns = [
    "timestamp",
    "update_id",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
    ]

    if args.execution_mode == "streaming":
        print("Building quote events from parquet row groups...")
        summary = build_quote_events_streaming(
            quotes_path,
            events_path,
            compression=args.compression,
            overwrite=args.overwrite,
        )
        events = None
    else:
        if events_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {events_path}. Pass --overwrite "
                "only after verifying the target."
            )
        quotes = pd.read_parquet(quotes_path, columns=quote_columns)
        print()
        print("Raw quote table:")
        print(f"Rows: {len(quotes):,}")
        print(
            f"Timestamp range: {quotes['timestamp'].min()} -> "
            f"{quotes['timestamp'].max()}"
        )
        print()
        print("Building in-memory parity path...")
        events, summary = build_quote_events(quotes)
        del quotes
        gc.collect()
        save_quote_events(events, events_path)

    print()
    print("Event-building summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if events is not None:
        print()
        print("Event table preview:")
        preview_cols = [
            "event_id",
            "timestamp",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
            "midprice",
            "relative_spread",
        ]
        print(events[preview_cols].head(10).to_string(index=False))

    save_event_summary(summary, summary_path)

    print()
    print("Done. Distinct quote-event stream built.")


def main() -> None:
    args = parse_args()
    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")
    prefix = (
        f"build_events_{args.symbol}_{start_str}_to_{end_str}"
    )
    metadata_path = (
        PROJECT_ROOT / "outputs" / "reports" / "run_metadata" / f"{prefix}.json"
    )
    with RunRecorder(
        metadata_path,
        project_root=PROJECT_ROOT,
        stage="build_events",
        arguments=vars(args),
        experiment_fingerprint=ExperimentSpec().fingerprint(),
        require_clean_git=args.require_clean_git,
    ):
        run(args)
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
