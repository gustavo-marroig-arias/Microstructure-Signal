from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


QUOTE_STATE_COLS = [
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
]


EVENT_OUTPUT_COLS = [
    "event_id",
    "timestamp",
    "update_id",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
    "midprice",
    "relative_spread",
]


def validate_quote_columns(quotes: pd.DataFrame) -> None:
    """
    Checks that the quote table has the columns needed to build events.

    Memory-optimized version only requires columns used downstream.
    """
    required = [
        "timestamp",
        "update_id",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
    ]

    missing = sorted(set(required) - set(quotes.columns))

    if missing:
        raise ValueError(f"quotes missing required columns: {missing}")


def validate_quote_order(quotes: pd.DataFrame) -> None:
    """
    Checks that quotes are already in deterministic event order.

    Required order:
    - timestamp nondecreasing
    - update_id nondecreasing within equal timestamps
    """
    timestamp_ns = quotes["timestamp"].astype("int64").to_numpy(copy=False)
    update_id = quotes["update_id"].to_numpy(copy=False)

    if len(quotes) <= 1:
        return

    timestamp_goes_back = timestamp_ns[1:] < timestamp_ns[:-1]

    if timestamp_goes_back.any():
        raise ValueError(
            "Quote timestamps are not monotonic increasing. "
            "The memory-optimized event builder requires pre-ordered input."
        )

    same_timestamp = timestamp_ns[1:] == timestamp_ns[:-1]
    update_id_goes_back_within_same_timestamp = (
        same_timestamp & (update_id[1:] < update_id[:-1])
    )

    if update_id_goes_back_within_same_timestamp.any():
        raise ValueError(
            "Quotes have equal timestamps but non-monotonic update_id order. "
            "The memory-optimized event builder requires deterministic pre-ordered input."
        )


def valid_quote_mask(quotes: pd.DataFrame) -> np.ndarray:
    """
    Returns a NumPy boolean mask for protocol-valid quote rows.

    Invalid rows:
    - bid_price > ask_price
    - bid_price <= 0
    - ask_price <= 0
    - bid_size <= 0
    - ask_size <= 0
    """
    bid_price = quotes["bid_price"].to_numpy(copy=False)
    ask_price = quotes["ask_price"].to_numpy(copy=False)
    bid_size = quotes["bid_size"].to_numpy(copy=False)
    ask_size = quotes["ask_size"].to_numpy(copy=False)

    return (
        (bid_price <= ask_price)
        & (bid_price > 0)
        & (ask_price > 0)
        & (bid_size > 0)
        & (ask_size > 0)
    )


def consecutive_duplicate_state_mask_after_valid_filter(
    quotes: pd.DataFrame,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Returns True for valid rows that duplicate the previous valid quote state.
    Invalid rows are ignored when determining consecutiveness.
    """
    n = len(quotes)
    duplicate = np.zeros(n, dtype=bool)

    if n <= 1:
        return duplicate

    bid_price = quotes["bid_price"].to_numpy(copy=False)
    ask_price = quotes["ask_price"].to_numpy(copy=False)
    bid_size = quotes["bid_size"].to_numpy(copy=False)
    ask_size = quotes["ask_size"].to_numpy(copy=False)

    # Fast memory-light path for clean data.
    if valid.all():
        duplicate[1:] = (
            (bid_price[1:] == bid_price[:-1])
            & (ask_price[1:] == ask_price[:-1])
            & (bid_size[1:] == bid_size[:-1])
            & (ask_size[1:] == ask_size[:-1])
        )
        return duplicate

    # General defensive path when invalid rows exist.
    valid_idx = np.flatnonzero(valid)

    if len(valid_idx) <= 1:
        return duplicate

    curr = valid_idx[1:]
    prev = valid_idx[:-1]

    duplicate[curr] = (
        (bid_price[curr] == bid_price[prev])
        & (ask_price[curr] == ask_price[prev])
        & (bid_size[curr] == bid_size[prev])
        & (ask_size[curr] == ask_size[prev])
    )

    return duplicate


def add_event_fields(events: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - event_id
    - midprice
    - relative_spread
    """
    # Keep RangeIndex compact.
    events = events.reset_index(drop=True)

    events.insert(0, "event_id", np.arange(len(events), dtype=np.int64))

    events["midprice"] = (events["bid_price"] + events["ask_price"]) / 2.0
    events["relative_spread"] = (
        (events["ask_price"] - events["bid_price"]) / events["midprice"]
    )

    return events[EVENT_OUTPUT_COLS]


def build_quote_events(quotes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Memory-optimized event-building pipeline.
    """
    validate_quote_columns(quotes)
    validate_quote_order(quotes)

    n_raw = len(quotes)

    valid = valid_quote_mask(quotes)
    n_clean = int(valid.sum())
    n_invalid = n_raw - n_clean

    if n_invalid > 0:
        print(f"Warning: removing {n_invalid:,} invalid quote rows.")

    duplicate = consecutive_duplicate_state_mask_after_valid_filter(quotes, valid)
    n_duplicates = int((valid & duplicate).sum())

    keep = valid & ~duplicate

    event_base = quotes.loc[
        keep,
        [
            "timestamp",
            "update_id",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
        ],
    ].copy()

    events = add_event_fields(event_base)

    summary = {
        "raw_quote_rows": n_raw,
        "clean_quote_rows": n_clean,
        "invalid_quote_rows_removed": n_invalid,
        "distinct_quote_events": int(len(events)),
        "duplicate_rows_collapsed": n_duplicates,
        "collapse_fraction": n_duplicates / n_clean if n_clean > 0 else float("nan"),
        "first_event_timestamp": events["timestamp"].min(),
        "last_event_timestamp": events["timestamp"].max(),
    }

    return events, summary


def save_quote_events(events: pd.DataFrame, output_path: Path) -> None:
    """
    Saves quote events as parquet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output_path, index=False)
    print(f"Saved quote events: {output_path}")


def save_event_summary(summary: dict, output_path: Path) -> None:
    """
    Saves event-building summary as CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_path, index=False)

    print(f"Saved event summary: {output_path}")