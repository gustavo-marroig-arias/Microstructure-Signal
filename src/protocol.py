from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


PROTOCOL_SYMBOL = "BTCUSDT"
TERNARY_LABELS = (-1, 0, 1)
DEFAULT_HORIZONS = (10, 20, 50)
FEATURE_COLUMNS = (
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
)
QUEUE_IMBALANCE_FEATURES = ("queue_imbalance",)


@dataclass(frozen=True)
class ExperimentSpec:
    """Immutable definition of the protocol choices that affect an experiment."""

    symbol: str = PROTOCOL_SYMBOL
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    boundary_drop_events: int = max(DEFAULT_HORIZONS)
    trade_lookback_ms: int = 1_000

    def __post_init__(self) -> None:
        validate_protocol_symbol(self.symbol)

        if not self.horizons:
            raise ValueError("horizons must not be empty.")
        if any(h <= 0 for h in self.horizons):
            raise ValueError(f"horizons must be positive: {self.horizons}")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError(f"horizons must be unique: {self.horizons}")
        if tuple(sorted(self.horizons)) != self.horizons:
            raise ValueError(f"horizons must be sorted: {self.horizons}")

        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty.")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must be unique.")

        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be strictly between 0 and 1.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between 0 and 1.")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError(
                "train_fraction + validation_fraction must be strictly below 1."
            )
        if self.boundary_drop_events < max(self.horizons):
            raise ValueError(
                "boundary_drop_events must cover the maximum label horizon."
            )
        if self.trade_lookback_ms <= 0:
            raise ValueError("trade_lookback_ms must be positive.")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def validate_protocol_symbol(symbol: str) -> None:
    if symbol != PROTOCOL_SYMBOL:
        raise ValueError(
            f"Protocol violation: symbol must be {PROTOCOL_SYMBOL}, got {symbol!r}."
        )
