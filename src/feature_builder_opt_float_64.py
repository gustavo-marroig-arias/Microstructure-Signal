"""
Memory-lean feature builder with float64 log-return-derived features.

The feature table keeps the high-volume quote and trade-flow features compact while
preserving float64 precision for the log-return path used by `mid_return_5` and
`realized_vol_20`.
"""

from __future__ import annotations

from pathlib import Path

import gc
import numpy as np
import pandas as pd


DEFAULT_HORIZONS = (10, 20, 50)

FEATURE_COLUMNS = [
    "relative_spread",
    "queue_imbalance",
    "log_bid_size",
    "log_ask_size",
    "delta_bid_size",
    "delta_ask_size",
    "mid_return_5",
    "realized_vol_20",
    "trade_intensity_1s",
    "signed_trade_count_imbalance_1s",
    "signed_trade_volume_imbalance_1s",
]


def validate_events(events: pd.DataFrame) -> None:
    required = [
        "event_id",
        "timestamp",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
    ]

    missing = sorted(set(required) - set(events.columns))
    if missing:
        raise ValueError(f"events missing required columns: {missing}")

    if not events["event_id"].is_monotonic_increasing:
        raise ValueError("event_id must be monotonic increasing.")

    if not events["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamp must be monotonic increasing.")

    if events["event_id"].iloc[0] != 0:
        raise ValueError("event_id must start at 0.")

    if events["event_id"].iloc[-1] != len(events) - 1:
        raise ValueError("event_id must end at len(events) - 1.")

    if not (events["bid_price"] > 0).all():
        raise ValueError("bid_price must be positive.")

    if not (events["ask_price"] > 0).all():
        raise ValueError("ask_price must be positive.")

    if not (events["bid_size"] > 0).all():
        raise ValueError("bid_size must be positive.")

    if not (events["ask_size"] > 0).all():
        raise ValueError("ask_size must be positive.")

    if not (events["ask_price"] >= events["bid_price"]).all():
        raise ValueError("ask_price must be >= bid_price.")


def validate_trades(trades: pd.DataFrame) -> None:
    required = [
        "timestamp",
        "quantity",
        "buyer_is_maker",
    ]

    missing = sorted(set(required) - set(trades.columns))
    if missing:
        raise ValueError(f"trades missing required columns: {missing}")

    if not trades["timestamp"].is_monotonic_increasing:
        trades.sort_values("timestamp", kind="mergesort", inplace=True)
        trades.reset_index(drop=True, inplace=True)

    if not (trades["quantity"] > 0).all():
        raise ValueError("trade quantity must be positive.")

    if trades["buyer_is_maker"].isna().any():
        raise ValueError("buyer_is_maker contains missing values.")


def _timestamp_to_int64_ns(timestamp_series: pd.Series) -> np.ndarray:
    return timestamp_series.astype("int64").to_numpy(copy=False)


def build_labels_from_midprice(
    midprice: np.ndarray,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> tuple[dict[str, np.ndarray], int]:
    """
    Builds y_h labels directly as int8 arrays.

    Does not materialize future_midprice_h or midprice_change_h columns.
    """
    max_horizon = max(horizons)

    if len(midprice) <= max_horizon:
        raise ValueError("Not enough events to build labels.")

    n_keep = len(midprice) - max_horizon

    labels: dict[str, np.ndarray] = {}

    current_mid = midprice[:n_keep]

    for h in horizons:
        future_mid = midprice[h : h + n_keep]
        change = future_mid - current_mid
        labels[f"y_{h}"] = np.sign(change).astype(np.int8)

    return labels, n_keep


def build_quote_feature_arrays(
    bid_price: np.ndarray,
    ask_price: np.ndarray,
    bid_size: np.ndarray,
    ask_size: np.ndarray,
    midprice: np.ndarray,
    n_keep: int,
) -> dict[str, np.ndarray]:
    """
    Builds quote-state features as compact NumPy arrays.

    Float64 log-return-derived variant:
    - only uses the first n_keep rows
    - keeps most quote features as float32 for memory efficiency
    - stores mid_return_5 and realized_vol_20 as float64
    - avoids unnecessary full-length temporary arrays
    - computes realized_vol_20 with cumulative sums on float64 log returns
    """
    features: dict[str, np.ndarray] = {}

    bid_price_k = bid_price[:n_keep]
    ask_price_k = ask_price[:n_keep]
    bid_size_k = bid_size[:n_keep]
    ask_size_k = ask_size[:n_keep]
    midprice_k = midprice[:n_keep]

    # ------------------------------------------------------------------
    # relative_spread = (ask_price - bid_price) / midprice
    # ------------------------------------------------------------------
    relative_spread = np.empty(n_keep, dtype=np.float32)
    np.subtract(
        ask_price_k,
        bid_price_k,
        out=relative_spread,
        casting="unsafe",
    )
    np.divide(
        relative_spread,
        midprice_k,
        out=relative_spread,
        casting="unsafe",
    )
    features["relative_spread"] = relative_spread

    # ------------------------------------------------------------------
    # queue_imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    # ------------------------------------------------------------------
    queue_imbalance = np.empty(n_keep, dtype=np.float32)
    queue_denominator = np.empty(n_keep, dtype=np.float32)

    np.subtract(
        bid_size_k,
        ask_size_k,
        out=queue_imbalance,
        casting="unsafe",
    )
    np.add(
        bid_size_k,
        ask_size_k,
        out=queue_denominator,
        casting="unsafe",
    )
    np.divide(
        queue_imbalance,
        queue_denominator,
        out=queue_imbalance,
        casting="unsafe",
    )

    features["queue_imbalance"] = queue_imbalance
    del queue_denominator

    # ------------------------------------------------------------------
    # log sizes
    # ------------------------------------------------------------------
    log_bid_size = np.empty(n_keep, dtype=np.float32)
    log_ask_size = np.empty(n_keep, dtype=np.float32)

    np.log1p(
        bid_size_k,
        out=log_bid_size,
        casting="unsafe",
    )
    np.log1p(
        ask_size_k,
        out=log_ask_size,
        casting="unsafe",
    )

    features["log_bid_size"] = log_bid_size
    features["log_ask_size"] = log_ask_size

    # ------------------------------------------------------------------
    # size deltas
    # ------------------------------------------------------------------
    delta_bid_size = np.empty(n_keep, dtype=np.float32)
    delta_ask_size = np.empty(n_keep, dtype=np.float32)

    delta_bid_size[:] = np.nan
    delta_ask_size[:] = np.nan

    if n_keep > 1:
        np.subtract(
            bid_size[1:n_keep],
            bid_size[: n_keep - 1],
            out=delta_bid_size[1:],
            casting="unsafe",
        )
        np.subtract(
            ask_size[1:n_keep],
            ask_size[: n_keep - 1],
            out=delta_ask_size[1:],
            casting="unsafe",
        )

    features["delta_bid_size"] = delta_bid_size
    features["delta_ask_size"] = delta_ask_size

    # ------------------------------------------------------------------
    # log midprice, used by mid_return_5 and realized_vol_20
    # ------------------------------------------------------------------
    log_mid = np.empty(n_keep, dtype=np.float64)

    np.log(
        midprice_k,
        out=log_mid,
    )

    # ------------------------------------------------------------------
    # mid_return_5 = log_mid[t] - log_mid[t - 5]
    # ------------------------------------------------------------------
    mid_return_5 = np.empty(n_keep, dtype=np.float64)
    mid_return_5[:] = np.nan

    if n_keep > 5:
        np.subtract(
            log_mid[5:],
            log_mid[:-5],
            out=mid_return_5[5:],
        )

    features["mid_return_5"] = mid_return_5

    # ------------------------------------------------------------------
    # realized_vol_20
    #
    # Same definition as:
    # one_event_log_return = log_mid.diff()
    # sqrt(sum(one_event_log_return^2 over last 20 returns))
    #
    # First valid row is index 20.
    # ------------------------------------------------------------------
    realized_vol_20 = np.empty(n_keep, dtype=np.float64)
    realized_vol_20[:] = np.nan

    if n_keep > 20:
        squared_returns = np.empty(n_keep - 1, dtype=np.float64)

        np.subtract(
            log_mid[1:],
            log_mid[:-1],
            out=squared_returns,
        )

        squared_returns *= squared_returns

        cumulative = np.empty(n_keep, dtype=np.float64)
        cumulative[0] = 0.0

        np.cumsum(
            squared_returns,
            dtype=np.float64,
            out=cumulative[1:],
        )

        np.subtract(
            cumulative[20:],
            cumulative[:-20],
            out=realized_vol_20[20:],
        )

        np.sqrt(
            realized_vol_20[20:],
            out=realized_vol_20[20:],
        )

        del squared_returns, cumulative

    features["realized_vol_20"] = realized_vol_20

    del log_mid

    return features

def build_trade_flow_feature_arrays(
    event_timestamps: pd.Series,
    trades: pd.DataFrame,
    n_keep: int,
    window_ms: int = 1000,
    chunk_size: int = 5_000_000,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Builds 1-second trade-flow features as arrays.

    Window:
        (event_timestamp - 1 second, event_timestamp)

    Trades exactly at event_timestamp are excluded.
    If no trades are in the window, imbalances are set to zero.
    """
    validate_trades(trades)

    event_time_ns = _timestamp_to_int64_ns(event_timestamps.iloc[:n_keep])

    trade_time_ns = _timestamp_to_int64_ns(trades["timestamp"])

    quantity = trades["quantity"].astype("float64").to_numpy(copy=False)

    # buyer_is_maker == False => buyer aggressor => +1
    # buyer_is_maker == True  => seller aggressor => -1
    trade_sign = np.where(trades["buyer_is_maker"].to_numpy(copy=False), -1.0, 1.0)

    signed_count = trade_sign
    signed_volume = trade_sign * quantity

    n_trades = len(trades)

    prefix_count = np.arange(n_trades + 1, dtype=np.float64)
    prefix_signed_count = np.concatenate([[0.0], np.cumsum(signed_count)])
    prefix_volume = np.concatenate([[0.0], np.cumsum(quantity)])
    prefix_signed_volume = np.concatenate([[0.0], np.cumsum(signed_volume)])

    trade_intensity = np.empty(n_keep, dtype=np.int32)
    signed_count_imbalance = np.empty(n_keep, dtype=np.float32)
    signed_volume_imbalance = np.empty(n_keep, dtype=np.float32)

    window_ns = int(window_ms * 1_000_000)

    for start in range(0, n_keep, chunk_size):
        stop = min(start + chunk_size, n_keep)

        event_chunk = event_time_ns[start:stop]
        lower_bound = event_chunk - window_ns

        left_idx = np.searchsorted(trade_time_ns, lower_bound, side="right")
        right_idx = np.searchsorted(trade_time_ns, event_chunk, side="left")

        count = prefix_count[right_idx] - prefix_count[left_idx]
        signed_count_sum = prefix_signed_count[right_idx] - prefix_signed_count[left_idx]

        total_volume = prefix_volume[right_idx] - prefix_volume[left_idx]
        signed_volume_sum = prefix_signed_volume[right_idx] - prefix_signed_volume[left_idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            count_imb = signed_count_sum / count
            volume_imb = signed_volume_sum / total_volume

        count_imb = np.where(count > 0, count_imb, 0.0)
        volume_imb = np.where(total_volume > 0, volume_imb, 0.0)

        count_imb = np.clip(count_imb, -1.0, 1.0)
        volume_imb = np.clip(volume_imb, -1.0, 1.0)

        trade_intensity[start:stop] = count.astype(np.int32)
        signed_count_imbalance[start:stop] = count_imb.astype(np.float32)
        signed_volume_imbalance[start:stop] = volume_imb.astype(np.float32)

        print(f"Processed trade-flow features for events {start:,} to {stop:,}")

    first_trade_ts = trades["timestamp"].min()
    lookback = pd.Timedelta(milliseconds=window_ms)

    has_full_trade_lookback = (
        event_timestamps.iloc[:n_keep].reset_index(drop=True) - lookback >= first_trade_ts
    ).to_numpy(dtype=bool)

    features = {
        "trade_intensity_1s": trade_intensity,
        "signed_trade_count_imbalance_1s": signed_count_imbalance,
        "signed_trade_volume_imbalance_1s": signed_volume_imbalance,
    }

    return features, has_full_trade_lookback


def build_feature_table(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    trade_lookback_ms: int = 1000,
) -> pd.DataFrame:
    """
    Builds a memory-lean model feature table with float64 log-return features.

    Output columns:
    - event_id
    - timestamp
    - 11 feature columns
    - y_10, y_20, y_50
    - has_full_trade_lookback_1s
    - feature_complete

    This deliberately does not store future_midprice or midprice_change columns.
    """
    validate_events(events)

    print("Preparing NumPy arrays...")

    event_id = events["event_id"].to_numpy(copy=False)
    timestamp = events["timestamp"]

    bid_price = events["bid_price"].to_numpy(copy=False)
    ask_price = events["ask_price"].to_numpy(copy=False)
    bid_size = events["bid_size"].to_numpy(copy=False)
    ask_size = events["ask_size"].to_numpy(copy=False)

    midprice = ((bid_price + ask_price) / 2.0).astype(np.float64)

    print("Building lean labels...")
    labels, n_keep = build_labels_from_midprice(midprice, horizons=horizons)

    print(f"Rows before horizon drop: {len(events):,}")
    print(f"Rows after horizon drop:  {n_keep:,}")
    print(f"Rows dropped:            {len(events) - n_keep:,}")

    print("Building quote features...")
    quote_features = build_quote_feature_arrays(
        bid_price=bid_price,
        ask_price=ask_price,
        bid_size=bid_size,
        ask_size=ask_size,
        midprice=midprice,
        n_keep=n_keep,
    )

    print("Building trade-flow features...")
    trade_features, has_full_trade_lookback = build_trade_flow_feature_arrays(
        event_timestamps=timestamp,
        trades=trades,
        n_keep=n_keep,
        window_ms=trade_lookback_ms,
    )

    print("Combining output table...")

    data = {
        "event_id": event_id[:n_keep],
        "timestamp": timestamp.iloc[:n_keep].reset_index(drop=True),
    }

    data.update(quote_features)
    data.update(trade_features)
    data.update(labels)

    data["has_full_trade_lookback_1s"] = has_full_trade_lookback

    feature_complete = has_full_trade_lookback.copy()

    for col in FEATURE_COLUMNS:
        values = data[col]
        if np.issubdtype(values.dtype, np.floating):
            feature_complete &= ~np.isnan(values)

    data["feature_complete"] = feature_complete

    out = pd.DataFrame(data)

    # Release large references as early as possible.
    del quote_features, trade_features, labels
    gc.collect()

    return out


def label_distribution(
    feature_table: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    rows = []

    for h in horizons:
        label_col = f"y_{h}"

        counts = feature_table[label_col].value_counts(dropna=False).sort_index()
        proportions = feature_table[label_col].value_counts(
            dropna=False,
            normalize=True,
        ).sort_index()

        for label_value in [-1, 0, 1]:
            rows.append(
                {
                    "horizon": h,
                    "label": label_value,
                    "count": int(counts.get(label_value, 0)),
                    "proportion": float(proportions.get(label_value, 0.0)),
                }
            )

    return pd.DataFrame(rows)


def feature_summary(feature_table: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in FEATURE_COLUMNS:
        s = feature_table[col]

        rows.append(
            {
                "feature": col,
                "missing_count": int(s.isna().sum()),
                "missing_fraction": float(s.isna().mean()),
                "min": float(s.min(skipna=True)),
                "max": float(s.max(skipna=True)),
                "mean": float(s.mean(skipna=True)),
                "std": float(s.std(skipna=True)),
            }
        )

    return pd.DataFrame(rows)


def save_feature_table(feature_table: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_parquet(output_path, index=False)
    print(f"Saved feature table: {output_path}")


def save_feature_summary(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved feature summary: {output_path}")


def save_label_distribution(distribution: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(output_path, index=False)
    print(f"Saved label distribution: {output_path}")
