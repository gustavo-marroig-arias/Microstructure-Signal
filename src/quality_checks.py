from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Note: The main goal is to decide whether a day of data
# is clean enough for research.

# Small configuration object to include quality rules:
# - If the largest time gap between consecutive rows is more than 60 seconds,
# flag that day as having a feed gap violation.
@dataclass(frozen=True)
class QualityThresholds:
    max_allowed_gap_seconds: float = 60.0


def add_utc_day(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Adds a UTC calendar day column.

    This project uses full UTC days, not local days.
    """
    out = df.copy()
    out["utc_day"] = out[timestamp_col].dt.floor("D")
    return out


def max_gap_seconds(df: pd.DataFrame, timestamp_col: str = "timestamp") -> float:
    """
    Computes maximum timestamp gap in seconds.

    Assumes timestamps are pandas datetime64[ns, UTC].
    """
    if len(df) <= 1:
        return float("nan")

    timestamps = df[timestamp_col].sort_values()
    gaps = timestamps.diff().dt.total_seconds()
    return float(gaps.max())


def quote_quality_by_day(
        quotes: pd.DataFrame,
        thresholds: QualityThresholds = QualityThresholds(),
) -> pd.DataFrame:
    """
    Produces quote-feed quality metrics by UTC day.

    Protocol rules:
    - invalid if bid_price > ask_price
    - invalid if bid_price <= 0
    - invalid if ask_price <= 0
    - invalid if bid_size <= 0
    - invalid if ask_size <= 0
    - locked quotes (bid_price == ask_price) are allowed but logged
    - material outage if max gap > 60 seconds
    """ 
    required = [
        "timestamp",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
    ]

    missing = sorted(set(required) - set(quotes.columns))
    if missing:
        raise ValueError(f"quotes missing required columns: {missing}")
    
    q = add_utc_day(quotes)

    rows = []

    for day, g in q.groupby("utc_day", sort=True):
        crossed = g["bid_price"] > g["ask_price"]
        locked = g["bid_price"] == g["ask_price"]

        non_positive_bid_price = g["bid_price"] <= 0
        non_positive_ask_price = g["ask_price"] <= 0
        non_positive_bid_size = g["bid_size"] <= 0
        non_positive_ask_size = g["ask_size"] <= 0       
        
        invalid = (
            crossed
            | non_positive_bid_price
            | non_positive_ask_price
            | non_positive_bid_size
            | non_positive_ask_size
        )

        max_gap = max_gap_seconds(g)

        rows.append(
            {
                "utc_day": day,
                "quote_rows": len(g),
                "quote_start": g["timestamp"].min(),
                "quote_end": g["timestamp"].max(),
                "max_quote_gap_seconds": max_gap,
                "crossed_quote_count": int(crossed.sum()),
                "locked_quote_count": int(locked.sum()),
                "locked_quote_fraction": float(locked.mean()),
                "non_positive_bid_price_count": int(non_positive_bid_price.sum()),
                "non_positive_ask_price_count": int(non_positive_ask_price.sum()),
                "non_positive_bid_size_count": int(non_positive_bid_size.sum()),
                "non_positive_ask_size_count": int(non_positive_ask_size.sum()),
                "invalid_quote_count": int(invalid.sum()),
                "quote_gap_violation": bool(max_gap > thresholds.max_allowed_gap_seconds),       
            }
        )
    
    return pd.DataFrame(rows)


def trade_quality_by_day(
        trades: pd.DataFrame,
        thresholds: QualityThresholds = QualityThresholds(),
) -> pd.DataFrame:
    """
    Produces trade-feed quality metrics by UTC day.

    Protocol requires checking gaps > 60 seconds in the trade feed.
    We also sanity-check trade price and quantity.
    """
    required = [
        "timestamp",
        "price",
        "quantity",
        "buyer_is_maker",
    ]

    missing = sorted(set(required) - set(trades.columns))
    if missing:
        raise ValueError(f"trades missing required columns: {missing}")  
    
    t = add_utc_day(trades)

    rows = []

    for day, g in t.groupby("utc_day", sort=True):
        non_positive_price = g["price"] <= 0
        non_positive_quantity = g["quantity"] <= 0
        missing_buyer_is_maker = g["buyer_is_maker"].isna()

        invalid_trade = (
            non_positive_price
            | non_positive_quantity
            | missing_buyer_is_maker
        )

        max_gap = max_gap_seconds(g)

        rows.append(
            {
                "utc_day": day,
                "trade_rows": len(g),
                "trade_start": g["timestamp"].min(),
                "trade_end": g["timestamp"].max(),
                "max_trade_gap_seconds": max_gap,
                "non_positive_trade_price_count": int(non_positive_price.sum()),
                "non_positive_trade_quantity_count": int(non_positive_quantity.sum()),
                "missing_buyer_is_maker_count": int(missing_buyer_is_maker.sum()),
                "invalid_trade_count": int(invalid_trade.sum()),
                "trade_gap_violation": bool(max_gap > thresholds.max_allowed_gap_seconds),
            }
        )

    return pd.DataFrame(rows)


def combined_quality_report(
    quotes: pd.DataFrame,
    trades: pd.DataFrame, 
    thresholds: QualityThresholds = QualityThresholds(),
) -> pd.DataFrame:
    """
    Combines quote and trade quality metrics into one per-day report.
    """
    quote_report = quote_quality_by_day(quotes, thresholds=thresholds)
    trade_report = trade_quality_by_day(trades, thresholds=thresholds)

    report = quote_report.merge(
        trade_report,
        on = "utc_day",
        how = "outer",
        validate="one_to_one",
    )

    report = report.sort_values("utc_day").reset_index(drop=True)

    report["has_quote_data"] = (report["quote_rows"].fillna(0) > 0).astype(bool)
    report["has_trade_data"] = (report["trade_rows"].fillna(0) > 0).astype(bool)

    # if we do not know wether the feed had a gap, treat it as violation
    report["quote_gap_violation"] = [
    True if pd.isna(x) else bool(x)
    for x in report["quote_gap_violation"]
    ]

    report["trade_gap_violation"] = [
        True if pd.isna(x) else bool(x)
        for x in report["trade_gap_violation"]
    ]

    report["invalid_quote_count"] = report["invalid_quote_count"].fillna(0).astype(int)
    report["invalid_trade_count"] = report["invalid_trade_count"].fillna(0).astype(int)

    # Keep day rules
    report["keep_day"] = (
        report["has_quote_data"]
        & report["has_trade_data"]
        & ~report["quote_gap_violation"]
        & ~report["trade_gap_violation"]
        & (report["invalid_quote_count"] == 0)
        & (report["invalid_trade_count"] == 0)
    )

    report["drop_reason"] = ""

    report.loc[~report["has_quote_data"], "drop_reason"] += "missing_quote_data;"
    report.loc[~report["has_trade_data"], "drop_reason"] += "missing_trade_data;"
    report.loc[report["quote_gap_violation"], "drop_reason"] += "quote_gap_gt_60s;"
    report.loc[report["trade_gap_violation"], "drop_reason"] += "trade_gap_gt_60s;"
    report.loc[report["invalid_quote_count"] > 0, "drop_reason"] += "invalid_quotes;"
    report.loc[report["invalid_trade_count"] > 0, "drop_reason"] += "invalid_trades;"

    report.loc[report["keep_day"], "drop_reason"] = "keep"

    return report


def save_quality_report(report: pd.DataFrame, output_path: Path) -> None:
    """
    Saves the quality report as CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"Saved quality report: {output_path}")