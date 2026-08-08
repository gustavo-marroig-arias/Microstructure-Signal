import numpy as np
import pandas as pd

from src.feature_builder_opt_float_64 import build_trade_flow_feature_arrays


def test_trade_window_is_open_left_and_open_at_event_time() -> None:
    event_time = pd.Timestamp("2024-03-01T00:00:02Z")
    trades = pd.DataFrame(
        {
            "timestamp": [
                event_time - pd.Timedelta(seconds=1),
                event_time - pd.Timedelta(milliseconds=999),
                event_time - pd.Timedelta(milliseconds=1),
                event_time,
            ],
            "quantity": [10.0, 1.0, 3.0, 100.0],
            "buyer_is_maker": [False, False, True, False],
        }
    )
    features, complete = build_trade_flow_feature_arrays(
        pd.Series([event_time]),
        trades,
        n_keep=1,
    )

    assert features["trade_intensity_1s"][0] == 2
    assert np.isclose(features["signed_trade_count_imbalance_1s"][0], 0.0)
    assert np.isclose(
        features["signed_trade_volume_imbalance_1s"][0],
        (1.0 - 3.0) / (1.0 + 3.0),
    )
    assert bool(complete[0])
