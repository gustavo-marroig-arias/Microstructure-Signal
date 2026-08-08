from __future__ import annotations 

import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

# Pipeline Note ########################################################
# Choose symbol and date range -> 
# -> Build Binance daily file URLs ->
# -> Download .zip files ->                          
# -> Read zipped CSV files ->                                
# -> Standardize column names ->
# -> Convert prices, quantities, IDs, Timestamps ->
# -> Save cleaned raw tables as parquet
########################################################################
# Example usage:
# config = BinanceRawConfig(symbol="BTCUSDT")
# quotes, trades = build_raw_dataset("2024-03-01", "2024-03-01", config)
########################################################################

BINANCE_BASE_URL = "https://data.binance.vision/data/futures/um/daily"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Final URLs will look like this pattern: 
# https://data.binance.vision/data/futures/um/daily/{data_type}/{symbol}/{filename}

@dataclass(frozen=True)                       # can't modify the config after creating it
class BinanceRawConfig:
    symbol: str = "BTCUSDT"
    raw_dir: Path = DATA_DIR / "raw"           # where raw zip files go
    interim_dir: Path = DATA_DIR / "interim"  # where cleaned parquet files go


DEFAULT_BINANCE_RAW_CONFIG = BinanceRawConfig()

# HELPER: date parser
def parse_date(value: str | date | datetime) -> date:
    """
    Converts a supported date input into a date object.

    Intentionally does not support MM/DD/YYYY because it is ambiguous with
    DD/MM/YYYY.
    """
    SUPPORTED_DATE_FORMATS = (
        "%Y-%m-%d",   # 2024-03-01
        "%Y/%m/%d",   # 2024/03/01
        "%d-%m-%Y",   # 01-03-2024
        "%d/%m/%Y",   # 01/03/2024
        "%Y%m%d",     # 20240301
    )

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if not isinstance(value, str):
        raise TypeError(
            f"Expected str, date, or datetime, got {type(value).__name__}"
        )

    value = value.strip()

    for fmt in SUPPORTED_DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    raise ValueError(
        f"Invalid date format: {value!r}. "
        "Use YYYY-MM-DD, YYYY/MM/DD, DD-MM-YYYY, DD/MM/YYYY, or YYYYMMDD. "
        "MM/DD/YYYY is intentionally not supported because it is ambiguous."
    )

# HELPER: date range
def date_range_inclusive(start: str | date | datetime, 
                         end: str | date | datetime) -> Iterable[date]:
    """
    Generator: Yields dates from start to end inclusive.

    Example:
        start="2024-03-01", end="2024-03-03"
        yields 2024-03-01, 2024-03-02, 2024-03-03
    """
    start_date = parse_date(start)
    end_date = parse_date(end)
    
    if end_date < start_date:
        raise ValueError("end date must be >= start date")
    
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days = 1)

# HELPER: Builds Download URL str for one file
def binance_daily_url(data_type: str, symbol: str, day: date) -> str:
    """
    Builds the Binance public-data URL for a daily futures file.

    data_type should be:
        - "bookTicker"
        - "aggTrades"
    """
    day_str = day.strftime("%Y-%m-%d")
    filename = f"{symbol}-{data_type}-{day_str}.zip"
    return f"{BINANCE_BASE_URL}/{data_type}/{symbol}/{filename}"


def filter_to_utc_date_range(
    df: pd.DataFrame,
    start: str | date | datetime,
    end: str | date | datetime,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Keeps rows whose timestamp falls inside the requested UTC date range.

    The range is inclusive by date, but exclusive of the next UTC day:
        start 2024-03-01, end 2024-03-03
        keeps timestamps >= 2024-03-01 00:00:00 UTC
        and timestamps <  2024-03-04 00:00:00 UTC
    """
    start_ts = pd.Timestamp(parse_date(start)).tz_localize("UTC")
    end_exclusive_ts = pd.Timestamp(parse_date(end) + timedelta(days=1)).tz_localize("UTC")

    timestamps = pd.to_datetime(df[timestamp_col], utc=True)

    mask = (timestamps >= start_ts) & (timestamps < end_exclusive_ts)

    return df.loc[mask].copy()


# HELPER: File downloader
def download_file(url: str, output_path: Path, timeout: int = 60) -> bool:
    """
    Downloads one file if it does not already exist.

    Returns:
        True if file exists after this function.
        False if Binance returned 404 / unavailable or the download failed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already exists: {output_path}")
        return True
    
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    # Remove any old partial download.
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"Downloading: {url}")

    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            if response.status_code == 404:
                print(f"Missing file: {url}")
                return False
            
            # raise an error for other bad responses
            response.raise_for_status()  

            # progress bar
            total = int(response.headers.get("content-length", 0))

            with open(tmp_path, "wb") as f:
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=output_path.name,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))
            
            # Only after the full download succeeds do we move it into place.
            tmp_path.replace(output_path)
            return True

    except (requests.RequestException, OSError) as exc:
        print(f"Download failed for {url}")
        print(f"Reason: {exc}")

        if tmp_path.exists():
            tmp_path.unlink()

        return False


def download_daily_data(
        data_type: str,
        start: str | date | datetime,
        end: str | date | datetime,
        config: BinanceRawConfig = DEFAULT_BINANCE_RAW_CONFIG,
) -> list[Path]:
    """
    Downloads daily Binance files for one data type.
    Provides the list of files available locally.

    data_type:
        - "bookTicker"
        - "aggTrades"
    """
    if data_type not in {"bookTicker", "aggTrades"}:
        raise ValueError("data_type must be either 'bookTicker' or 'aggTrades'")
    
    downloaded_paths: list[Path] = []

    for day in date_range_inclusive(start, end):
        day_str = day.strftime("%Y-%m-%d")
        filename = f"{config.symbol}-{data_type}-{day_str}.zip"
        output_path = config.raw_dir / data_type / filename
        url = binance_daily_url(data_type, config.symbol, day)

        ok = download_file(url, output_path)

        if ok:
            downloaded_paths.append(output_path)
    
    return downloaded_paths

# HELPER: check whether zipped CSV has a header
def _zip_has_header(zip_path: Path) -> bool:
    """
    Checks whether the first line inside the zipped CSV looks like a header.

    Binance headerless rows usually begin with a numeric ID.
    Header rows begin with a column name.
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found inside {zip_path}")

        with zf.open(csv_names[0]) as f:
            first_line = f.readline().decode("utf-8-sig").strip()

    first_field = first_line.split(",", 1)[0].strip().strip('"').strip("'")

    try:
        int(first_field)
        return False
    except ValueError:
        return True

def read_bookticker_zip(zip_path: Path) -> pd.DataFrame:
    """
    Reads one Binance futures bookTicker zip file.

    Expected historical archive columns:
        update_id
        best_bid_price
        best_bid_qty
        best_ask_price
        best_ask_qty
        transaction_time
        event_time

    We standardize them to names used by this project:
        update_id
        bid_price
        bid_size
        ask_price
        ask_size
        transaction_time
        event_time
        timestamp
        raw_row_number
    """

    names = [
        "update_id",
        "bid_price",
        "bid_size",
        "ask_price",
        "ask_size",
        "transaction_time",
        "event_time",
    ]

    has_header = _zip_has_header(zip_path)

    if has_header:
        df = pd.read_csv(zip_path, compression="zip")
        rename_map = {
            "best_bid_price": "bid_price",
            "best_bid_qty": "bid_size",
            "best_ask_price": "ask_price",
            "best_ask_qty": "ask_size",
        }
        df = df.rename(columns=rename_map)
    else:
        df = pd.read_csv(zip_path, compression="zip", header=None, names=names)
    
    required = [
        "update_id",
        "bid_price",
        "bid_size",
        "ask_price",
        "ask_size",
        "transaction_time",
        "event_time",
    ]

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{zip_path} missing columns: {missing}")
    
    df = df[required].copy()

    numeric_cols = [
        "update_id",
        "bid_price",
        "bid_size",
        "ask_price",
        "ask_size",
        "transaction_time",
        "event_time",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Protocol says use exchange timestamps.
    # For bookTicker, event_time is Binance event time. We use it as our quote timestamp.
    df["timestamp"] = pd.to_datetime(df["event_time"], unit="ms", utc=True)

    # Keeps original ordering within each file as a fallback tie-breaker.
    df["raw_row_number"] = range(len(df))

    return df      

def read_aggtrades_zip(zip_path: Path) -> pd.DataFrame:
    """
    Reads one Binance futures aggTrades zip file.

    Expected futures archive columns:
        agg_trade_id
        price
        quantity
        first_trade_id
        last_trade_id
        timestamp
        buyer_is_maker

    We standardize them to:
        agg_trade_id
        price
        quantity
        first_trade_id
        last_trade_id
        trade_time
        timestamp
        buyer_is_maker
        raw_row_number
    """
    names = [
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "trade_time",
        "buyer_is_maker",
    ]

    has_header = _zip_has_header(zip_path)

    if has_header:
        df = pd.read_csv(zip_path, compression="zip")

        # Clean possible whitespace / BOM characters from header names.
        df.columns = [str(col).strip().lstrip("\ufeff") for col in df.columns]

        rename_map = {
            # Binance archive-style names
            "agg_trade_id": "agg_trade_id",
            "price": "price",
            "quantity": "quantity",
            "first_trade_id": "first_trade_id",
            "last_trade_id": "last_trade_id",
            "transact_time": "trade_time",
            "is_buyer_maker": "buyer_is_maker",

            # Alternative human-readable names
            "Aggregate tradeId": "agg_trade_id",
            "Price": "price",
            "Quantity": "quantity",
            "First tradeId": "first_trade_id",
            "Last tradeId": "last_trade_id",
            "Timestamp": "trade_time",
            "Transact time": "trade_time",
            "Transact Time": "trade_time",
            "Was the buyer the maker": "buyer_is_maker",

            # Already-standardized names
            "trade_time": "trade_time",
            "buyer_is_maker": "buyer_is_maker",
        }

        df = df.rename(columns=rename_map)
    else:
        df = pd.read_csv(zip_path, compression="zip", header=None, names=names)

    required = [
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "trade_time",
        "buyer_is_maker",
    ]

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{zip_path} missing columns: {missing}")
    
    df = df[required].copy()

    numeric_cols = [
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "trade_time",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["buyer_is_maker"].dtype != bool:
        df["buyer_is_maker"] = (
            df["buyer_is_maker"]
            .astype(str).str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    # Protocol says use exchange timestamps.
    df["timestamp"] = pd.to_datetime(df["trade_time"], unit="ms", utc=True)

    df["raw_row_number"] = range(len(df))

    return df 

# HELPER: many zips reader
def load_many_zips(paths: list[Path], reader_func) -> pd.DataFrame:
    """
    Reads many zip files and concatenates them.
    """
    if not paths:
        raise ValueError("No paths provided")
    
    frames = []

    for path in paths:
        print(f"Reading: {path}")
        frame = reader_func(path)
        frame["source_file"] = path.name
        frames.append(frame)
    
    return pd.concat(frames, ignore_index=True)


def load_raw_bookticker(
        start: str | date | datetime,
        end: str | date | datetime,
        config: BinanceRawConfig = DEFAULT_BINANCE_RAW_CONFIG,
) -> pd.DataFrame:
    """
    Loads downloaded bookTicker zip files between start and end.
    """
    paths = []
    missing_paths = []

    for day in date_range_inclusive(start, end):
        day_str = day.strftime("%Y-%m-%d")
        path = config.raw_dir / "bookTicker" / f"{config.symbol}-bookTicker-{day_str}.zip"
        if path.exists():
            paths.append(path)
        else:
            missing_paths.append(path)

    if missing_paths:
        formatted = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Requested bookTicker sample is incomplete. Missing daily files:\n"
            f"{formatted}"
        )

    quotes = load_many_zips(paths, read_bookticker_zip)
    return filter_to_utc_date_range(quotes, start, end)

def load_raw_aggtrades(
        start: str | date | datetime,
        end: str | date | datetime,
        config: BinanceRawConfig = DEFAULT_BINANCE_RAW_CONFIG,
) -> pd.DataFrame:
    """
    Loads downloaded aggTrades zip files between start and end.
    """
    paths = []
    missing_paths = []

    for day in date_range_inclusive(start, end):
        day_str = day.strftime("%Y-%m-%d")
        path = config.raw_dir / "aggTrades" / f"{config.symbol}-aggTrades-{day_str}.zip"
        if path.exists():
            paths.append(path)
        else:
            missing_paths.append(path)

    if missing_paths:
        formatted = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Requested aggTrades sample is incomplete. Missing daily files:\n"
            f"{formatted}"
        )

    trades = load_many_zips(paths, read_aggtrades_zip)
    return filter_to_utc_date_range(trades, start, end)


def save_interim_raw_tables(
        quotes: pd.DataFrame,
        trades: pd.DataFrame,
        start: str | date | datetime,
        end: str | date | datetime,
        config: BinanceRawConfig = DEFAULT_BINANCE_RAW_CONFIG,
) -> tuple[Path, Path]:
    """
    Saves standardized raw quote/trade tables as parquet.
    """
    config.interim_dir.mkdir(parents=True, exist_ok=True)

    start_str = parse_date(start).strftime("%Y-%m-%d")
    end_str = parse_date(end).strftime("%Y-%m-%d")

    quotes_path = config.interim_dir / f"raw_quotes_{config.symbol}_{start_str}_to_{end_str}.parquet"
    trades_path = config.interim_dir / f"raw_trades_{config.symbol}_{start_str}_to_{end_str}.parquet"

    quotes.to_parquet(quotes_path, index=False)
    trades.to_parquet(trades_path, index=False)

    print(f"Saved quotes: {quotes_path}")
    print(f"Saved trades: {trades_path}")

    return quotes_path, trades_path


# WRAPPER:
def build_raw_dataset(
    start: str | date | datetime,
    end: str | date | datetime,
    config: BinanceRawConfig = DEFAULT_BINANCE_RAW_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    download_daily_data("bookTicker", start, end, config)
    download_daily_data("aggTrades", start, end, config)

    quotes = load_raw_bookticker(start, end, config)
    trades = load_raw_aggtrades(start, end, config)

    save_interim_raw_tables(quotes, trades, start, end, config)

    return quotes, trades


# # TEST: just a safe small testing block
# if __name__ == "__main__":
#     config = BinanceRawConfig(symbol="BTCUSDT")

#     print("Testing aggTrades...")
#     download_daily_data("aggTrades", "2024-03-01", "2024-03-01", config)
#     trades = load_raw_aggtrades("2024-03-01", "2024-03-01", config)

#     print(trades.head())
#     print(trades.dtypes)

#     print("Testing bookTicker...")
#     download_daily_data("bookTicker", "2024-03-01", "2024-03-01", config)
#     quotes = load_raw_bookticker("2024-03-01", "2024-03-01", config)

#     print(quotes.head())
#     print(quotes.dtypes)

#     print("Testing parquet save...")
#     save_interim_raw_tables(
#         quotes=quotes,
#         trades=trades,
#         start="2024-03-01",
#         end="2024-03-01",
#         config=config,
#     )
