from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_builder_opt_float_64 import (
    build_feature_table,
    build_quote_feature_arrays,
)
from src.streaming_features import build_feature_table_streaming


def test_streaming_features_match_in_memory_across_boundaries(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(123)
    n_events = 523
    base = pd.Timestamp("2024-03-01T00:00:01Z")
    timestamps = base + pd.to_timedelta(
        np.arange(n_events) * 100,
        unit="ms",
    )
    midprice = 60_000 + np.cumsum(
        rng.choice([-0.5, 0.0, 0.5], n_events)
    )
    events = pd.DataFrame(
        {
            "event_id": np.arange(n_events, dtype=np.int64),
            "timestamp": timestamps,
            "bid_price": midprice - 0.5,
            "ask_price": midprice + 0.5,
            "bid_size": rng.uniform(0.1, 8.0, n_events).astype(np.float32),
            "ask_size": rng.uniform(0.1, 8.0, n_events).astype(np.float32),
        }
    )
    n_trades = 1_500
    trades = pd.DataFrame(
        {
            "timestamp": (
                base
                - pd.Timedelta(seconds=2)
                + pd.to_timedelta(np.arange(n_trades) * 40, unit="ms")
            ),
            "quantity": rng.uniform(0.001, 2.0, n_trades).astype(np.float32),
            "buyer_is_maker": rng.choice([True, False], n_trades),
        }
    )

    expected = build_feature_table(events.copy(), trades.copy())
    events_path = tmp_path / "events.parquet"
    trades_path = tmp_path / "trades.parquet"
    output_path = tmp_path / "features.parquet"
    events.to_parquet(events_path, index=False, row_group_size=71)
    trades.to_parquet(trades_path, index=False, row_group_size=103)
    report = build_feature_table_streaming(
        events_path,
        trades_path,
        output_path,
        event_chunk_size=97,
    )
    actual = pd.read_parquet(output_path)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert report.rows_written == len(expected)
    assert report.feature_complete_rows == int(
        expected["feature_complete"].sum()
    )


def test_realized_volatility_matches_direct_window_definition() -> None:
    midprice = np.array(
        [100.0 + 0.1 * index + (-1) ** index * 0.02 for index in range(80)],
        dtype=np.float64,
    )
    features = build_quote_feature_arrays(
        bid_price=midprice - 0.01,
        ask_price=midprice + 0.01,
        bid_size=np.linspace(1.0, 2.0, len(midprice)),
        ask_size=np.linspace(2.0, 1.0, len(midprice)),
        midprice=midprice,
        n_keep=len(midprice),
    )

    log_mid = np.log(midprice)
    for event_id in range(20, len(midprice)):
        direct = np.sqrt(
            np.square(
                np.diff(log_mid[event_id - 20 : event_id + 1])
            ).sum(dtype=np.float64)
        )
        assert np.isclose(
            features["realized_vol_20"][event_id],
            direct,
            rtol=0.0,
            atol=1e-18,
        )
