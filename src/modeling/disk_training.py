from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Iterator, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.model_dataset_io import (
    ModelDatasetLocation,
    iter_model_split_batches,
    model_split_row_count,
)
from src.protocol import TERNARY_LABELS


@dataclass(frozen=True)
class PreparedTrainingData:
    """Disk-backed standardized train matrix and ordered horizon labels."""

    feature_path: Path
    label_path: Path
    n_rows: int
    feature_columns: tuple[str, ...]
    horizons: tuple[int, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    batch_size: int

    def features(self) -> np.memmap:
        return np.memmap(
            self.feature_path,
            mode="r",
            dtype=np.float64,
            shape=(self.n_rows, len(self.feature_columns)),
        )

    def labels(self, horizon: int) -> np.ndarray:
        if horizon not in self.horizons:
            raise ValueError(f"Horizon {horizon} was not prepared.")
        matrix = np.memmap(
            self.label_path,
            mode="r",
            dtype=np.int8,
            shape=(self.n_rows, len(self.horizons)),
        )
        return matrix[:, self.horizons.index(horizon)]


@contextmanager
def prepare_training_data(
    location: ModelDatasetLocation,
    *,
    feature_columns: Sequence[str],
    horizons: Sequence[int],
    batch_size: int,
    temporary_root: Path,
) -> Iterator[PreparedTrainingData]:
    """Build one standardized float64 train matrix without a full DataFrame."""
    features = tuple(feature_columns)
    requested_horizons = tuple(int(value) for value in horizons)
    if not features:
        raise ValueError("feature_columns must not be empty.")
    if len(set(features)) != len(features):
        raise ValueError("feature_columns must be unique.")
    if not requested_horizons:
        raise ValueError("At least one horizon must be prepared.")
    if len(set(requested_horizons)) != len(requested_horizons):
        raise ValueError("Prepared horizons must be unique.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    temporary_root = Path(temporary_root)
    temporary_root.mkdir(parents=True, exist_ok=True)
    train_rows = model_split_row_count(
        location,
        "train",
        eligible_only=True,
    )
    with tempfile.TemporaryDirectory(
        dir=temporary_root,
        prefix="standardized_train_",
    ) as directory:
        workdir = Path(directory)
        label_path = workdir / "labels.int8"
        feature_path = workdir / "features.float64"
        labels = np.memmap(
            label_path,
            mode="w+",
            dtype=np.int8,
            shape=(train_rows, len(requested_horizons)),
        )
        label_counts = {
            horizon: np.zeros(3, dtype=np.int64)
            for horizon in requested_horizons
        }
        scaler = StandardScaler(copy=False)
        columns = (
            *(f"y_{horizon}" for horizon in requested_horizons),
            *features,
        )
        offset = 0
        for batch_number, batch in enumerate(
            iter_model_split_batches(
                location,
                "train",
                columns,
                eligible_only=True,
                batch_size=batch_size,
            ),
            start=1,
        ):
            stop = offset + len(batch)
            values = batch.loc[:, list(features)].to_numpy(
                dtype=np.float64,
                copy=True,
            )
            if not np.isfinite(values).all():
                raise ValueError("Training features contain non-finite values.")
            scaler.partial_fit(values)
            for index, horizon in enumerate(requested_horizons):
                batch_labels = batch[f"y_{horizon}"].to_numpy(dtype=np.int8)
                if not np.isin(
                    batch_labels,
                    np.asarray(TERNARY_LABELS),
                ).all():
                    raise ValueError(
                        f"Unexpected training labels for horizon {horizon}."
                    )
                labels[offset:stop, index] = batch_labels
                label_counts[horizon] += np.bincount(
                    batch_labels + 1,
                    minlength=3,
                )
            offset = stop
            print(
                f"Scaler pass batch {batch_number}: "
                f"{offset:,}/{train_rows:,}",
                flush=True,
            )
        if offset != train_rows:
            raise RuntimeError(
                f"Scaler row mismatch: {offset} != {train_rows}."
            )
        for horizon, counts in label_counts.items():
            if bool((counts == 0).any()):
                raise ValueError(
                    f"All ternary classes are required for horizon {horizon}: "
                    f"counts={counts.tolist()}."
                )
        labels.flush()

        standardized = np.memmap(
            feature_path,
            mode="w+",
            dtype=np.float64,
            shape=(train_rows, len(features)),
        )
        offset = 0
        for batch_number, batch in enumerate(
            iter_model_split_batches(
                location,
                "train",
                features,
                eligible_only=True,
                batch_size=batch_size,
            ),
            start=1,
        ):
            stop = offset + len(batch)
            values = batch.loc[:, list(features)].to_numpy(
                dtype=np.float64,
                copy=True,
            )
            scaler.transform(values, copy=False)
            standardized[offset:stop] = values
            offset = stop
            print(
                f"Standardization pass batch {batch_number}: "
                f"{offset:,}/{train_rows:,}",
                flush=True,
            )
        if offset != train_rows:
            raise RuntimeError(
                f"Standardization row mismatch: {offset} != {train_rows}."
            )
        standardized.flush()
        del standardized, labels

        prepared = PreparedTrainingData(
            feature_path=feature_path,
            label_path=label_path,
            n_rows=train_rows,
            feature_columns=features,
            horizons=requested_horizons,
            scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
            scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
            batch_size=batch_size,
        )
        yield prepared
