from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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
    if quotes["timestamp"].isna().any():
        raise ValueError("Quote timestamps contain missing values.")
    if quotes["update_id"].isna().any():
        raise ValueError("Quote update_id contains missing values.")

    timestamp_ns = quotes["timestamp"].astype("int64").to_numpy(copy=False)
    update_id = quotes["update_id"].to_numpy(copy=False)
    if not np.issubdtype(update_id.dtype, np.number):
        raise TypeError("Quote update_id must be numeric.")
    if not bool(np.isfinite(update_id).all()):
        raise ValueError("Quote update_id must be finite.")

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
        np.isfinite(bid_price)
        & np.isfinite(ask_price)
        & np.isfinite(bid_size)
        & np.isfinite(ask_size)
        & (bid_price <= ask_price)
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

    bid_price = events["bid_price"].to_numpy(dtype=np.float64, copy=False)
    ask_price = events["ask_price"].to_numpy(dtype=np.float64, copy=False)
    events["midprice"] = (bid_price + ask_price) * 0.5
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
    if events.empty:
        raise ValueError("No valid quote events were produced.")

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


def build_quote_events_streaming(
    quotes_path: Path,
    output_path: Path,
    *,
    compression: str = "zstd",
    overwrite: bool = False,
) -> dict:
    """Build quote events row group by row group with exact boundary state."""
    quotes_path = Path(quotes_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Pass overwrite=True only after verifying the target."
        )

    parquet_file = pq.ParquetFile(quotes_path)
    required = [
        "timestamp",
        "update_id",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
    ]
    missing = sorted(set(required) - set(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"quotes missing required columns: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp-{uuid4().hex}"
    )
    writer: pq.ParquetWriter | None = None
    previous_timestamp_ns: int | None = None
    previous_update_id: int | None = None
    previous_valid_state: tuple[float, float, float, float] | None = None
    next_event_id = 0
    raw_rows = 0
    clean_rows = 0
    invalid_rows = 0
    duplicate_rows = 0
    first_event_timestamp: pd.Timestamp | None = None
    last_event_timestamp: pd.Timestamp | None = None

    try:
        for row_group_index in range(parquet_file.num_row_groups):
            quotes = parquet_file.read_row_group(
                row_group_index,
                columns=required,
            ).to_pandas()
            if quotes.empty:
                continue
            validate_quote_columns(quotes)
            validate_quote_order(quotes)
            timestamp_ns = quotes["timestamp"].astype("int64").to_numpy(
                copy=False
            )
            update_ids = quotes["update_id"].to_numpy(copy=False)
            if previous_timestamp_ns is not None:
                if timestamp_ns[0] < previous_timestamp_ns:
                    raise ValueError(
                        "Quote timestamp order breaks across row groups."
                    )
                if (
                    timestamp_ns[0] == previous_timestamp_ns
                    and update_ids[0] < previous_update_id
                ):
                    raise ValueError(
                        "Quote update_id order breaks across row groups."
                    )

            valid = valid_quote_mask(quotes)
            valid_indices = np.flatnonzero(valid)
            duplicate = np.zeros(len(quotes), dtype=bool)
            if len(valid_indices):
                state_values = quotes.loc[
                    valid,
                    QUOTE_STATE_COLS,
                ].to_numpy(copy=False)
                if previous_valid_state is not None:
                    duplicate[valid_indices[0]] = bool(
                        np.array_equal(
                            state_values[0],
                            np.asarray(previous_valid_state),
                        )
                    )
                if len(valid_indices) > 1:
                    duplicate[valid_indices[1:]] = np.all(
                        state_values[1:] == state_values[:-1],
                        axis=1,
                    )
                previous_valid_state = tuple(
                    float(value) for value in state_values[-1]
                )

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
            if not event_base.empty:
                event_base.reset_index(drop=True, inplace=True)
                event_base.insert(
                    0,
                    "event_id",
                    np.arange(
                        next_event_id,
                        next_event_id + len(event_base),
                        dtype=np.int64,
                    ),
                )
                bid_price = event_base["bid_price"].to_numpy(
                    dtype=np.float64,
                    copy=False,
                )
                ask_price = event_base["ask_price"].to_numpy(
                    dtype=np.float64,
                    copy=False,
                )
                event_base["midprice"] = (bid_price + ask_price) * 0.5
                event_base["relative_spread"] = (
                    event_base["ask_price"] - event_base["bid_price"]
                ) / event_base["midprice"]
                events = event_base.loc[:, EVENT_OUTPUT_COLS]
                table = pa.Table.from_pandas(events, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_path,
                        table.schema,
                        compression=compression,
                    )
                writer.write_table(table)
                if first_event_timestamp is None:
                    first_event_timestamp = pd.Timestamp(
                        events["timestamp"].iloc[0]
                    )
                last_event_timestamp = pd.Timestamp(events["timestamp"].iloc[-1])
                next_event_id += len(events)

            raw_rows += len(quotes)
            clean_rows += int(valid.sum())
            invalid_rows += int((~valid).sum())
            duplicate_rows += int((valid & duplicate).sum())
            previous_timestamp_ns = int(timestamp_ns[-1])
            previous_update_id = int(update_ids[-1])

        if writer is None:
            raise ValueError("No valid quote events were produced.")
        writer.close()
        writer = None
        os.replace(temporary_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "raw_quote_rows": raw_rows,
        "clean_quote_rows": clean_rows,
        "invalid_quote_rows_removed": invalid_rows,
        "distinct_quote_events": next_event_id,
        "duplicate_rows_collapsed": duplicate_rows,
        "collapse_fraction": (
            duplicate_rows / clean_rows if clean_rows else float("nan")
        ),
        "first_event_timestamp": first_event_timestamp,
        "last_event_timestamp": last_event_timestamp,
    }
