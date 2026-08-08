from pathlib import Path

import numpy as np
import pandas as pd

from src.event_builder_opt import (
    add_event_fields,
    build_quote_events,
    build_quote_events_streaming,
)


def test_streaming_events_match_in_memory_across_row_groups(
    tmp_path: Path,
) -> None:
    timestamps = pd.to_datetime(
        [
            "2024-03-01T00:00:00.000Z",
            "2024-03-01T00:00:00.001Z",
            "2024-03-01T00:00:00.002Z",
            "2024-03-01T00:00:00.003Z",
            "2024-03-01T00:00:00.004Z",
            "2024-03-01T00:00:00.005Z",
        ]
    )
    quotes = pd.DataFrame(
        {
            "timestamp": timestamps,
            "update_id": np.arange(6, dtype=np.int64),
            "bid_price": [100, 100, 100, 102, 101, 101],
            "ask_price": [101, 101, 101, 101, 102, 102],
            "bid_size": [1, 1, 1, 1, 2, 2],
            "ask_size": [2, 2, 2, 2, 1, 1],
        }
    )
    expected, expected_summary = build_quote_events(quotes)
    source = tmp_path / "quotes.parquet"
    output = tmp_path / "events.parquet"
    quotes.to_parquet(source, index=False, row_group_size=2)

    actual_summary = build_quote_events_streaming(source, output)
    actual = pd.read_parquet(output)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual_summary == expected_summary


def test_event_builder_rejects_nonfinite_quote_state() -> None:
    quotes = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-03-01T00:00:00Z",
                    "2024-03-01T00:00:01Z",
                ]
            ),
            "update_id": np.arange(2, dtype=np.int64),
            "bid_price": [np.inf, 100.0],
            "ask_price": [np.inf, 101.0],
            "bid_size": [1.0, 1.0],
            "ask_size": [1.0, 1.0],
        }
    )

    events, summary = build_quote_events(quotes)

    assert len(events) == 1
    assert summary["invalid_quote_rows_removed"] == 1
    assert np.isfinite(events["midprice"]).all()


def test_midprice_addition_is_float64_even_for_float32_quotes() -> None:
    event_base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-03-01T00:00:00Z"]),
            "update_id": np.array([1], dtype=np.int64),
            "bid_price": np.array([100_000_000], dtype=np.float32),
            "ask_price": np.array([100_000_008], dtype=np.float32),
            "bid_size": np.array([1], dtype=np.float32),
            "ask_size": np.array([1], dtype=np.float32),
        }
    )

    events = add_event_fields(event_base)

    assert events["midprice"].dtype == np.dtype("float64")
    assert events.loc[0, "midprice"] == 100_000_004.0
