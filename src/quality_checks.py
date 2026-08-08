from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Note: The main goal is to decide whether a day of data
# is clean enough for research.

# Small configuration object to include quality rules:
# - If the largest time gap between consecutive rows is more than 60 seconds,
# flag that day as having a feed gap violation.
@dataclass(frozen=True)
class QualityThresholds:
    max_allowed_gap_seconds: float = 60.0
    max_allowed_day_edge_gap_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_allowed_gap_seconds <= 0:
            raise ValueError("max_allowed_gap_seconds must be positive.")
        if self.max_allowed_day_edge_gap_seconds <= 0:
            raise ValueError(
                "max_allowed_day_edge_gap_seconds must be positive."
            )


DEFAULT_QUALITY_THRESHOLDS = QualityThresholds()


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
        thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
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
        non_finite = ~np.isfinite(
            g[["bid_price", "ask_price", "bid_size", "ask_size"]]
        ).all(axis=1)
        
        invalid = (
            crossed
            | non_positive_bid_price
            | non_positive_ask_price
            | non_positive_bid_size
            | non_positive_ask_size
            | non_finite
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
        thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
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
        non_finite = ~np.isfinite(g[["price", "quantity"]]).all(axis=1)

        invalid_trade = (
            non_positive_price
            | non_positive_quantity
            | missing_buyer_is_maker
            | non_finite
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
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
    *,
    expected_start: str | None = None,
    expected_end: str | None = None,
) -> pd.DataFrame:
    """
    Combines quote and trade quality metrics into one per-day report.
    """
    quote_report = quote_quality_by_day(quotes, thresholds=thresholds)
    trade_report = trade_quality_by_day(trades, thresholds=thresholds)

    return combine_daily_quality_reports(
        quote_report,
        trade_report,
        thresholds=thresholds,
        expected_start=expected_start,
        expected_end=expected_end,
    )


def combine_daily_quality_reports(
    quote_report: pd.DataFrame,
    trade_report: pd.DataFrame,
    *,
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
    expected_start: str | None = None,
    expected_end: str | None = None,
) -> pd.DataFrame:
    report = quote_report.merge(
        trade_report,
        on = "utc_day",
        how = "outer",
        validate="one_to_one",
    )

    if (expected_start is None) != (expected_end is None):
        raise ValueError(
            "expected_start and expected_end must be provided together."
        )
    if expected_start is not None and expected_end is not None:
        expected_days = pd.date_range(
            pd.Timestamp(expected_start, tz="UTC"),
            pd.Timestamp(expected_end, tz="UTC"),
            freq="D",
        )
        if expected_days.empty:
            raise ValueError("Expected UTC day range must not be empty.")
        expected_frame = pd.DataFrame({"utc_day": expected_days})
        report = expected_frame.merge(
            report,
            on="utc_day",
            how="left",
            validate="one_to_one",
        )

    report = report.sort_values("utc_day").reset_index(drop=True)

    report["has_quote_data"] = (report["quote_rows"].fillna(0) > 0).astype(bool)
    report["has_trade_data"] = (report["trade_rows"].fillna(0) > 0).astype(bool)
    next_day = report["utc_day"] + pd.Timedelta(days=1)
    report["quote_start_delay_seconds"] = (
        report["quote_start"] - report["utc_day"]
    ).dt.total_seconds()
    report["quote_end_early_seconds"] = (
        next_day - report["quote_end"]
    ).dt.total_seconds()
    report["trade_start_delay_seconds"] = (
        report["trade_start"] - report["utc_day"]
    ).dt.total_seconds()
    report["trade_end_early_seconds"] = (
        next_day - report["trade_end"]
    ).dt.total_seconds()

    edge_limit = thresholds.max_allowed_day_edge_gap_seconds
    report["quote_edge_gap_violation"] = (
        ~report["has_quote_data"]
        | (report["quote_start_delay_seconds"] > edge_limit)
        | (report["quote_end_early_seconds"] > edge_limit)
    )
    report["trade_edge_gap_violation"] = (
        ~report["has_trade_data"]
        | (report["trade_start_delay_seconds"] > edge_limit)
        | (report["trade_end_early_seconds"] > edge_limit)
    )

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
        & ~report["quote_edge_gap_violation"]
        & ~report["trade_edge_gap_violation"]
        & (report["invalid_quote_count"] == 0)
        & (report["invalid_trade_count"] == 0)
    )

    report["drop_reason"] = ""

    report.loc[~report["has_quote_data"], "drop_reason"] += "missing_quote_data;"
    report.loc[~report["has_trade_data"], "drop_reason"] += "missing_trade_data;"
    report.loc[report["quote_gap_violation"], "drop_reason"] += "quote_gap_gt_60s;"
    report.loc[report["trade_gap_violation"], "drop_reason"] += "trade_gap_gt_60s;"
    report.loc[
        report["quote_edge_gap_violation"],
        "drop_reason",
    ] += "quote_day_edge_gap;"
    report.loc[
        report["trade_edge_gap_violation"],
        "drop_reason",
    ] += "trade_day_edge_gap;"
    report.loc[report["invalid_quote_count"] > 0, "drop_reason"] += "invalid_quotes;"
    report.loc[report["invalid_trade_count"] > 0, "drop_reason"] += "invalid_trades;"

    report.loc[report["keep_day"], "drop_reason"] = "keep"

    return report


def combined_quality_report_from_parquet(
    quotes_path: Path,
    trades_path: Path,
    *,
    expected_start: str,
    expected_end: str,
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
    batch_size: int = 1_000_000,
) -> pd.DataFrame:
    """Compute daily quality statistics without materializing raw tables."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    quote_report = _stream_quality_report(
        Path(quotes_path),
        kind="quote",
        thresholds=thresholds,
        batch_size=batch_size,
    )
    trade_report = _stream_quality_report(
        Path(trades_path),
        kind="trade",
        thresholds=thresholds,
        batch_size=batch_size,
    )
    return combine_daily_quality_reports(
        quote_report,
        trade_report,
        thresholds=thresholds,
        expected_start=expected_start,
        expected_end=expected_end,
    )


def _stream_quality_report(
    path: Path,
    *,
    kind: str,
    thresholds: QualityThresholds,
    batch_size: int,
) -> pd.DataFrame:
    if kind == "quote":
        columns = [
            "timestamp",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
        ]
    elif kind == "trade":
        columns = ["timestamp", "price", "quantity", "buyer_is_maker"]
    else:
        raise ValueError(f"Unsupported quality stream kind: {kind}")

    parquet_file = pq.ParquetFile(path)
    missing = sorted(set(columns) - set(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"{kind} parquet is missing columns: {missing}")

    states: dict[pd.Timestamp, dict] = {}
    previous_timestamp: pd.Timestamp | None = None
    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=batch_size,
    ):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        if not timestamps.is_monotonic_increasing:
            raise ValueError(f"{kind} timestamps are not monotone.")
        if (
            previous_timestamp is not None
            and timestamps.iloc[0] < previous_timestamp
        ):
            raise ValueError(f"{kind} timestamps go backwards across batches.")

        frame = frame.copy()
        frame["timestamp"] = timestamps
        frame["utc_day"] = timestamps.dt.floor("D")
        for day, part in frame.groupby("utc_day", sort=False):
            state = states.setdefault(day, _empty_quality_state(kind))
            part_timestamps = part["timestamp"]
            state["rows"] += len(part)
            if state["start"] is None:
                state["start"] = part_timestamps.iloc[0]
            if state["last_timestamp"] is not None:
                boundary_gap = (
                    part_timestamps.iloc[0] - state["last_timestamp"]
                ).total_seconds()
                state["max_gap_seconds"] = max(
                    state["max_gap_seconds"],
                    float(boundary_gap),
                )
            if len(part_timestamps) > 1:
                internal_gap = float(
                    part_timestamps.diff().dt.total_seconds().max()
                )
                state["max_gap_seconds"] = max(
                    state["max_gap_seconds"],
                    internal_gap,
                )
            state["last_timestamp"] = part_timestamps.iloc[-1]
            state["end"] = part_timestamps.iloc[-1]
            _update_quality_counts(state, part, kind)

        previous_timestamp = timestamps.iloc[-1]

    rows = [
        _quality_state_row(day, state, kind, thresholds)
        for day, state in sorted(states.items())
    ]
    return pd.DataFrame(rows)


def _empty_quality_state(kind: str) -> dict:
    state = {
        "rows": 0,
        "start": None,
        "end": None,
        "last_timestamp": None,
        "max_gap_seconds": 0.0,
        "invalid_rows": 0,
    }
    if kind == "quote":
        state.update(
            {
                "crossed": 0,
                "locked": 0,
                "non_positive_bid_price": 0,
                "non_positive_ask_price": 0,
                "non_positive_bid_size": 0,
                "non_positive_ask_size": 0,
            }
        )
    else:
        state.update(
            {
                "non_positive_price": 0,
                "non_positive_quantity": 0,
                "missing_buyer_is_maker": 0,
            }
        )
    return state


def _update_quality_counts(state: dict, part: pd.DataFrame, kind: str) -> None:
    if kind == "quote":
        crossed = part["bid_price"] > part["ask_price"]
        state["crossed"] += int(crossed.sum())
        state["locked"] += int(
            (part["bid_price"] == part["ask_price"]).sum()
        )
        invalid = crossed.copy()
        for key, column in (
            ("non_positive_bid_price", "bid_price"),
            ("non_positive_ask_price", "ask_price"),
            ("non_positive_bid_size", "bid_size"),
            ("non_positive_ask_size", "ask_size"),
        ):
            non_positive = part[column] <= 0
            state[key] += int(non_positive.sum())
            invalid |= non_positive
        invalid |= ~np.isfinite(
            part[["bid_price", "ask_price", "bid_size", "ask_size"]]
        ).all(axis=1)
    else:
        non_positive_price = part["price"] <= 0
        non_positive_quantity = part["quantity"] <= 0
        missing_buyer_is_maker = part["buyer_is_maker"].isna()
        state["non_positive_price"] += int(non_positive_price.sum())
        state["non_positive_quantity"] += int(non_positive_quantity.sum())
        state["missing_buyer_is_maker"] += int(missing_buyer_is_maker.sum())
        invalid = (
            non_positive_price
            | non_positive_quantity
            | missing_buyer_is_maker
            | ~np.isfinite(part[["price", "quantity"]]).all(axis=1)
        )
    state["invalid_rows"] += int(invalid.sum())


def _quality_state_row(
    day: pd.Timestamp,
    state: dict,
    kind: str,
    thresholds: QualityThresholds,
) -> dict:
    max_gap = (
        state["max_gap_seconds"] if state["rows"] > 1 else float("nan")
    )
    if kind == "quote":
        return {
            "utc_day": day,
            "quote_rows": state["rows"],
            "quote_start": state["start"],
            "quote_end": state["end"],
            "max_quote_gap_seconds": max_gap,
            "crossed_quote_count": state["crossed"],
            "locked_quote_count": state["locked"],
            "locked_quote_fraction": state["locked"] / state["rows"],
            "non_positive_bid_price_count": state["non_positive_bid_price"],
            "non_positive_ask_price_count": state["non_positive_ask_price"],
            "non_positive_bid_size_count": state["non_positive_bid_size"],
            "non_positive_ask_size_count": state["non_positive_ask_size"],
            "invalid_quote_count": state["invalid_rows"],
            "quote_gap_violation": bool(
                max_gap > thresholds.max_allowed_gap_seconds
            ),
        }

    return {
        "utc_day": day,
        "trade_rows": state["rows"],
        "trade_start": state["start"],
        "trade_end": state["end"],
        "max_trade_gap_seconds": max_gap,
        "non_positive_trade_price_count": state["non_positive_price"],
        "non_positive_trade_quantity_count": state["non_positive_quantity"],
        "missing_buyer_is_maker_count": state["missing_buyer_is_maker"],
        "invalid_trade_count": state["invalid_rows"],
        "trade_gap_violation": bool(
            max_gap > thresholds.max_allowed_gap_seconds
        ),
    }


def save_quality_report(report: pd.DataFrame, output_path: Path) -> None:
    """
    Saves the quality report as CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"Saved quality report: {output_path}")
