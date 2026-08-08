import numpy as np
import pandas as pd

from src.quality_checks import (
    combined_quality_report,
    combined_quality_report_from_parquet,
)


def test_expected_day_missing_from_both_feeds_is_reported() -> None:
    quotes = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-03-01T00:00:01Z", "2024-03-01T23:59:59Z"]
            ),
            "bid_price": [100.0, 100.0],
            "ask_price": [101.0, 101.0],
            "bid_size": [1.0, 1.0],
            "ask_size": [1.0, 1.0],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-03-01T00:00:01Z", "2024-03-01T23:59:59Z"]
            ),
            "price": [100.5, 100.5],
            "quantity": [1.0, 1.0],
            "buyer_is_maker": [True, False],
        }
    )

    report = combined_quality_report(
        quotes,
        trades,
        expected_start="2024-03-01",
        expected_end="2024-03-02",
    )
    missing_day = report.loc[
        report["utc_day"] == pd.Timestamp("2024-03-02", tz="UTC")
    ].iloc[0]
    assert not missing_day["has_quote_data"]
    assert not missing_day["has_trade_data"]
    assert not missing_day["keep_day"]
    assert "missing_quote_data" in missing_day["drop_reason"]
    assert "missing_trade_data" in missing_day["drop_reason"]


def test_streaming_quality_matches_in_memory_and_keeps_boundary_gap(
    tmp_path,
) -> None:
    timestamps = pd.to_datetime(
        [
            "2024-03-01T00:00:01Z",
            "2024-03-01T00:00:02Z",
            "2024-03-01T00:01:10Z",
            "2024-03-01T00:01:11Z",
        ]
    )
    quotes = pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid_price": [100.0] * 4,
            "ask_price": [101.0] * 4,
            "bid_size": [1.0] * 4,
            "ask_size": [1.0] * 4,
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": [100.5] * 4,
            "quantity": [1.0] * 4,
            "buyer_is_maker": [True, False, True, False],
        }
    )
    quotes_path = tmp_path / "quotes.parquet"
    trades_path = tmp_path / "trades.parquet"
    quotes.to_parquet(quotes_path, index=False, row_group_size=2)
    trades.to_parquet(trades_path, index=False, row_group_size=2)

    expected = combined_quality_report(
        quotes,
        trades,
        expected_start="2024-03-01",
        expected_end="2024-03-01",
    )
    actual = combined_quality_report_from_parquet(
        quotes_path,
        trades_path,
        expected_start="2024-03-01",
        expected_end="2024-03-01",
        batch_size=2,
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.loc[0, "max_quote_gap_seconds"] == 68.0


def test_streaming_quality_counts_invalid_rows_once(tmp_path) -> None:
    timestamps = pd.to_datetime(
        ["2024-03-01T00:00:01Z", "2024-03-01T23:59:59Z"]
    )
    quotes = pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid_price": [np.nan, 100.0],
            "ask_price": [-1.0, 101.0],
            "bid_size": [0.0, 1.0],
            "ask_size": [1.0, 1.0],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": [np.inf, 100.5],
            "quantity": [0.0, 1.0],
            "buyer_is_maker": [None, False],
        }
    )
    quotes_path = tmp_path / "quotes.parquet"
    trades_path = tmp_path / "trades.parquet"
    quotes.to_parquet(quotes_path, index=False, row_group_size=1)
    trades.to_parquet(trades_path, index=False, row_group_size=1)

    expected = combined_quality_report(
        quotes,
        trades,
        expected_start="2024-03-01",
        expected_end="2024-03-01",
    )
    actual = combined_quality_report_from_parquet(
        quotes_path,
        trades_path,
        expected_start="2024-03-01",
        expected_end="2024-03-01",
        batch_size=1,
    )

    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.loc[0, "invalid_quote_count"] == 1
    assert actual.loc[0, "invalid_trade_count"] == 1
