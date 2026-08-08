from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.feature_builder_opt_float_64 import (
    FEATURE_COLUMNS,
    TradeFlowIndex,
    build_feature_table,
)
from src.protocol import DEFAULT_HORIZONS


EVENT_COLUMNS = (
    "event_id",
    "timestamp",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
)
TRADE_COLUMNS = ("timestamp", "quantity", "buyer_is_maker")


@dataclass
class RunningFeatureStat:
    nonmissing_count: int = 0
    missing_count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = np.inf
    maximum: float = -np.inf

    def update(self, values: pd.Series) -> None:
        array = values.to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(array)
        self.missing_count += int((~finite).sum())
        observed = array[finite]
        if observed.size == 0:
            return

        chunk_count = int(observed.size)
        chunk_mean = float(observed.mean(dtype=np.float64))
        chunk_m2 = float(
            np.square(observed - chunk_mean, dtype=np.float64).sum(
                dtype=np.float64
            )
        )
        if self.nonmissing_count == 0:
            self.mean = chunk_mean
            self.m2 = chunk_m2
        else:
            delta = chunk_mean - self.mean
            total = self.nonmissing_count + chunk_count
            self.mean += delta * chunk_count / total
            self.m2 += (
                chunk_m2
                + delta * delta
                * self.nonmissing_count
                * chunk_count
                / total
            )
        self.nonmissing_count += chunk_count
        self.minimum = min(self.minimum, float(observed.min()))
        self.maximum = max(self.maximum, float(observed.max()))

    def as_row(self, feature: str) -> dict:
        total = self.nonmissing_count + self.missing_count
        sample_std = (
            np.sqrt(self.m2 / (self.nonmissing_count - 1))
            if self.nonmissing_count > 1
            else np.nan
        )
        return {
            "feature": feature,
            "missing_count": self.missing_count,
            "missing_fraction": (
                self.missing_count / total if total else np.nan
            ),
            "min": self.minimum if self.nonmissing_count else np.nan,
            "max": self.maximum if self.nonmissing_count else np.nan,
            "mean": self.mean if self.nonmissing_count else np.nan,
            "std": float(sample_std),
        }


@dataclass
class StreamingFeatureReport:
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    rows_written: int = 0
    feature_complete_rows: int = 0
    feature_stats: dict[str, RunningFeatureStat] = field(
        default_factory=lambda: {
            feature: RunningFeatureStat()
            for feature in FEATURE_COLUMNS
        }
    )
    label_counts: dict[int, dict[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label_counts:
            self.label_counts = {
                horizon: {-1: 0, 0: 0, 1: 0}
                for horizon in self.horizons
            }

    def update(self, frame: pd.DataFrame) -> None:
        self.rows_written += len(frame)
        self.feature_complete_rows += int(frame["feature_complete"].sum())
        for feature, stats in self.feature_stats.items():
            stats.update(frame[feature])
        for horizon in self.horizons:
            counts = frame[f"y_{horizon}"].value_counts()
            for label in (-1, 0, 1):
                self.label_counts[horizon][label] += int(counts.get(label, 0))

    def feature_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                self.feature_stats[feature].as_row(feature)
                for feature in FEATURE_COLUMNS
            ]
        )

    def label_distribution(self) -> pd.DataFrame:
        rows = []
        for horizon in self.horizons:
            total = sum(self.label_counts[horizon].values())
            for label in (-1, 0, 1):
                count = self.label_counts[horizon][label]
                rows.append(
                    {
                        "horizon": horizon,
                        "label": label,
                        "count": count,
                        "proportion": count / total if total else np.nan,
                    }
                )
        return pd.DataFrame(rows)


def build_feature_table_streaming(
    events_path: Path,
    trades_path: Path,
    output_path: Path,
    *,
    event_chunk_size: int = 1_000_000,
    trade_lookback_ms: int = 1_000,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    compression: str = "zstd",
    overwrite: bool = False,
) -> StreamingFeatureReport:
    """
    Build an exact feature table using bounded quote-event and trade memory.

    Each core event chunk is read with 20 prior events for quote lookbacks and
    the maximum horizon of future events for labels. Its exact trade-time
    window is loaded independently. Only core rows are emitted, so overlapping
    context is never duplicated.
    """
    events_path = Path(events_path)
    trades_path = Path(trades_path)
    output_path = Path(output_path)
    if event_chunk_size <= 0:
        raise ValueError("event_chunk_size must be positive.")
    if trade_lookback_ms <= 0:
        raise ValueError("trade_lookback_ms must be positive.")
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizons must be positive, unique, and sorted.")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Pass overwrite=True only after verifying the target."
        )

    events_file = pq.ParquetFile(events_path)
    available_event_columns = set(events_file.schema_arrow.names)
    missing_events = sorted(set(EVENT_COLUMNS) - available_event_columns)
    if missing_events:
        raise ValueError(f"Quote events are missing columns: {missing_events}")
    total_events = events_file.metadata.num_rows
    max_horizon = max(horizons)
    if total_events <= max_horizon:
        raise ValueError("Not enough quote events to construct labels.")

    trades_file = pq.ParquetFile(trades_path)
    available_trade_columns = set(trades_file.schema_arrow.names)
    missing_trades = sorted(set(TRADE_COLUMNS) - available_trade_columns)
    if missing_trades:
        raise ValueError(f"Trades are missing columns: {missing_trades}")
    first_trade_timestamp = _validate_trade_stream(trades_file)

    event_dataset = ds.dataset(events_path, format="parquet")
    trade_dataset = ds.dataset(trades_path, format="parquet")
    output_rows = total_events - max_horizon
    history_events = 20
    report = StreamingFeatureReport(horizons=horizons)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp-{uuid4().hex}"
    )
    writer: pq.ParquetWriter | None = None

    try:
        for core_start in range(0, output_rows, event_chunk_size):
            core_stop = min(core_start + event_chunk_size, output_rows)
            extended_start = max(0, core_start - history_events)
            extended_stop = core_stop + max_horizon
            events = _read_event_range(
                event_dataset,
                extended_start,
                extended_stop,
            )
            emitted_event_timestamps = events.loc[
                events["event_id"] < core_stop,
                "timestamp",
            ]
            trades = _read_trade_window(
                trade_dataset,
                first_event_timestamp=emitted_event_timestamps.iloc[0],
                last_event_timestamp=emitted_event_timestamps.iloc[-1],
                lookback_ms=trade_lookback_ms,
            )
            trade_index = (
                TradeFlowIndex.from_trades(
                    trades,
                    first_trade_timestamp=first_trade_timestamp,
                )
                if not trades.empty
                else TradeFlowIndex.empty(first_trade_timestamp)
            )

            feature_context = build_feature_table(
                events,
                trades=None,
                horizons=horizons,
                trade_lookback_ms=trade_lookback_ms,
                trade_index=trade_index,
                require_zero_based_event_ids=False,
            )
            emit_mask = (
                (feature_context["event_id"] >= core_start)
                & (feature_context["event_id"] < core_stop)
            )
            emit = feature_context.loc[emit_mask].reset_index(drop=True)
            expected_rows = core_stop - core_start
            if len(emit) != expected_rows:
                raise AssertionError(
                    f"Chunk emitted {len(emit)} rows; expected {expected_rows}."
                )
            event_ids = emit["event_id"].to_numpy(dtype=np.int64, copy=False)
            if (
                event_ids[0] != core_start
                or event_ids[-1] != core_stop - 1
                or (
                    len(event_ids) > 1
                    and not bool(np.all(np.diff(event_ids) == 1))
                )
            ):
                raise AssertionError("Emitted event IDs are not exact and contiguous.")

            report.update(emit)
            table = pa.Table.from_pandas(emit, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    table.schema,
                    compression=compression,
                )
            writer.write_table(table)
            print(
                f"Wrote feature rows {core_start:,} to {core_stop:,}",
                flush=True,
            )

        if writer is None:
            raise AssertionError("No feature rows were written.")
        writer.close()
        writer = None
        if report.rows_written != output_rows:
            raise AssertionError(
                f"Wrote {report.rows_written} rows; expected {output_rows}."
            )
        os.replace(temporary_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise

    return report


def _validate_trade_stream(trades_file: pq.ParquetFile) -> pd.Timestamp:
    """Validate chronology and types without retaining the complete trade table."""
    first_timestamp: pd.Timestamp | None = None
    previous_timestamp: pd.Timestamp | None = None
    rows_seen = 0

    for batch in trades_file.iter_batches(
        columns=list(TRADE_COLUMNS),
        batch_size=1_000_000,
    ):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        rows_seen += len(frame)
        if frame["timestamp"].isna().any():
            raise ValueError("Trade timestamp contains missing values.")
        if not frame["timestamp"].is_monotonic_increasing:
            raise ValueError("Trade timestamps must be globally monotone.")
        batch_first = pd.Timestamp(frame["timestamp"].iloc[0])
        batch_last = pd.Timestamp(frame["timestamp"].iloc[-1])
        if previous_timestamp is not None and batch_first < previous_timestamp:
            raise ValueError(
                "Trade timestamps decrease across Parquet batch boundaries."
            )

        quantity = frame["quantity"].to_numpy(dtype=np.float64, copy=False)
        if (
            not bool(np.isfinite(quantity).all())
            or not bool((quantity > 0).all())
        ):
            raise ValueError("Trade quantity must be finite and positive.")
        if frame["buyer_is_maker"].isna().any():
            raise ValueError("buyer_is_maker contains missing values.")
        if not pd.api.types.is_bool_dtype(frame["buyer_is_maker"]):
            raise TypeError("buyer_is_maker must have boolean dtype.")

        if first_timestamp is None:
            first_timestamp = batch_first
        previous_timestamp = batch_last

    if rows_seen == 0 or first_timestamp is None:
        raise ValueError("Trade dataset must not be empty.")
    return first_timestamp


def _read_trade_window(
    trade_dataset: ds.Dataset,
    *,
    first_event_timestamp: pd.Timestamp,
    last_event_timestamp: pd.Timestamp,
    lookback_ms: int,
) -> pd.DataFrame:
    lower_exclusive = (
        pd.Timestamp(first_event_timestamp)
        - pd.Timedelta(milliseconds=lookback_ms)
    )
    upper_exclusive = pd.Timestamp(last_event_timestamp)
    table = trade_dataset.to_table(
        columns=list(TRADE_COLUMNS),
        filter=(
            (ds.field("timestamp") > lower_exclusive)
            & (ds.field("timestamp") < upper_exclusive)
        ),
    )
    frame = table.to_pandas()
    if not frame.empty and not frame["timestamp"].is_monotonic_increasing:
        frame = frame.sort_values("timestamp", kind="mergesort").reset_index(
            drop=True
        )
    return frame


def _read_event_range(
    event_dataset: ds.Dataset,
    start_event_id: int,
    stop_event_id: int,
) -> pd.DataFrame:
    table = event_dataset.to_table(
        columns=list(EVENT_COLUMNS),
        filter=(
            (ds.field("event_id") >= start_event_id)
            & (ds.field("event_id") < stop_event_id)
        ),
    )
    frame = table.to_pandas()
    expected_rows = stop_event_id - start_event_id
    if len(frame) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} events in [{start_event_id}, "
            f"{stop_event_id}), found {len(frame)}."
        )
    event_ids = frame["event_id"].to_numpy(dtype=np.int64, copy=False)
    if event_ids[0] != start_event_id or event_ids[-1] != stop_event_id - 1:
        raise ValueError("Source event IDs do not match the requested range.")
    return frame
