from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allows this script to import from src/ when run from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    BinanceRawConfig,
    download_daily_data,
    load_raw_aggtrades,
    load_raw_bookticker,
    save_interim_raw_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and load Binance BTCUSDT futures raw quote/trade data."
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format, inclusive.",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYY-MM-DD format, inclusive.",
    )

    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Binance symbol. Protocol default is BTCUSDT.",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download raw zip files before loading.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = BinanceRawConfig(
        symbol=args.symbol,
        raw_dir=PROJECT_ROOT / "data" / "raw",
        interim_dir=PROJECT_ROOT / "data" / "interim",
    )

    print("=" * 80)
    print("STEP 2: RAW DATA LOADING")
    print("=" * 80)
    print(f"Symbol: {config.symbol}")
    print(f"Start:  {args.start}")
    print(f"End:    {args.end}")
    print()

    if config.symbol != "BTCUSDT":
        raise ValueError("Protocol violation: symbol must remain BTCUSDT.")

    if args.download:
        print("Downloading bookTicker files...")
        download_daily_data("bookTicker", args.start, args.end, config=config)

        print()
        print("Downloading aggTrades files...")
        download_daily_data("aggTrades", args.start, args.end, config=config)

    print()
    print("Loading local bookTicker files...")
    quotes = load_raw_bookticker(args.start, args.end, config=config)

    print()
    print("Loading local aggTrades files...")
    trades = load_raw_aggtrades(args.start, args.end, config=config)

    print()
    print("Quotes table preview:")
    print(quotes.head())
    print()
    quotes.info()

    print()
    print("Trades table preview:")
    print(trades.head())
    print()
    trades.info()

    print()
    print("Basic timestamp coverage:")
    print(f"Quotes: {quotes['timestamp'].min()} → {quotes['timestamp'].max()}")
    print(f"Trades: {trades['timestamp'].min()} → {trades['timestamp'].max()}")

    print()
    save_interim_raw_tables(quotes, trades, args.start, args.end, config=config)

    print()
    print("Done. Step 2 is complete for this date range.")


if __name__ == "__main__":
    main()