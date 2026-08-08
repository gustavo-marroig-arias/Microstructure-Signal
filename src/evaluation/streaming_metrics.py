from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.metrics import TERNARY_LABEL_NAMES
from src.protocol import TERNARY_LABELS


LABEL_ARRAY = np.asarray(TERNARY_LABELS, dtype=np.int8)


def confusion_from_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return a fixed-order {-1, 0, 1} confusion matrix without sklearn copies."""
    true = np.asarray(y_true, dtype=np.int8)
    pred = np.asarray(y_pred, dtype=np.int8)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(true) == 0:
        raise ValueError("Cannot accumulate an empty prediction batch.")
    if not np.isin(true, LABEL_ARRAY).all():
        raise ValueError("y_true contains labels outside {-1, 0, 1}.")
    if not np.isin(pred, LABEL_ARRAY).all():
        raise ValueError("y_pred contains labels outside {-1, 0, 1}.")

    encoded = (true.astype(np.int64) + 1) * 3 + (
        pred.astype(np.int64) + 1
    )
    return np.bincount(encoded, minlength=9).reshape(3, 3)


@dataclass
class TernaryMetricAccumulator:
    """Exact sufficient statistics for all hard-prediction project metrics."""

    confusion: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.int64)
    )

    def update(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
    ) -> None:
        self.confusion += confusion_from_predictions(y_true, y_pred)

    def merge(self, other: "TernaryMetricAccumulator") -> None:
        self.confusion += _validated_confusion(other.confusion)

    @property
    def n_obs(self) -> int:
        return int(self.confusion.sum())

    def evaluation_tables(
        self,
        *,
        split: str,
        horizon: int,
        model_name: str,
    ) -> dict[str, pd.DataFrame]:
        return evaluation_tables_from_confusion(
            self.confusion,
            split=split,
            horizon=horizon,
            model_name=model_name,
        )

    def nonzero_table(
        self,
        *,
        split: str,
        horizon: int,
        model_name: str,
    ) -> pd.DataFrame:
        return nonzero_table_from_confusion(
            self.confusion,
            split=split,
            horizon=horizon,
            model_name=model_name,
        )


@dataclass
class DailyTernaryMetricAccumulator:
    """Exact hard-prediction metrics grouped by UTC day."""

    by_day: dict[pd.Timestamp, TernaryMetricAccumulator] = field(
        default_factory=dict
    )

    def update(
        self,
        timestamps: pd.Series,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
    ) -> None:
        ts = pd.Series(timestamps).reset_index(drop=True)
        true = np.asarray(y_true, dtype=np.int8)
        pred = np.asarray(y_pred, dtype=np.int8)
        if len(ts) != len(true) or len(true) != len(pred):
            raise ValueError("timestamps, y_true, and y_pred lengths differ.")
        if ts.isna().any():
            raise ValueError("timestamps contain missing values.")

        days = ts.dt.floor("D")
        for day in days.drop_duplicates().sort_values():
            mask = (days == day).to_numpy()
            accumulator = self.by_day.setdefault(
                pd.Timestamp(day),
                TernaryMetricAccumulator(),
            )
            accumulator.update(true[mask], pred[mask])

    def table(
        self,
        *,
        split: str,
        horizon: int,
        model_name: str,
    ) -> pd.DataFrame:
        if not self.by_day:
            raise ValueError("No daily observations were accumulated.")
        rows = []
        for day, accumulator in sorted(self.by_day.items()):
            row = _aggregate_row(
                accumulator.confusion,
                split=split,
                horizon=horizon,
                model_name=model_name,
            )
            row["utc_day"] = day
            rows.append(row)
        return pd.DataFrame(rows)


def evaluation_tables_from_confusion(
    confusion: np.ndarray,
    *,
    split: str,
    horizon: int,
    model_name: str,
) -> dict[str, pd.DataFrame]:
    matrix = _validated_confusion(confusion)
    if int(matrix.sum()) == 0:
        raise ValueError("Cannot evaluate an empty confusion matrix.")

    support = matrix.sum(axis=1)
    precision, recall, f1 = _class_metrics(matrix)
    n_obs = int(matrix.sum())

    class_rows = []
    per_class_rows = []
    count_rows = []
    normalized_rows = []
    for true_index, true_label in enumerate(TERNARY_LABELS):
        class_rows.append(
            {
                "model": model_name,
                "split": split,
                "horizon": horizon,
                "class_label": true_label,
                "class_name": TERNARY_LABEL_NAMES[true_label],
                "count": int(support[true_index]),
                "proportion": float(support[true_index] / n_obs),
            }
        )
        per_class_rows.append(
            {
                "model": model_name,
                "split": split,
                "horizon": horizon,
                "class_label": true_label,
                "class_name": TERNARY_LABEL_NAMES[true_label],
                "precision": float(precision[true_index]),
                "recall": float(recall[true_index]),
                "f1": float(f1[true_index]),
                "support": int(support[true_index]),
            }
        )
        for pred_index, pred_label in enumerate(TERNARY_LABELS):
            shared = {
                "model": model_name,
                "split": split,
                "horizon": horizon,
                "true_label": true_label,
                "true_name": TERNARY_LABEL_NAMES[true_label],
                "pred_label": pred_label,
                "pred_name": TERNARY_LABEL_NAMES[pred_label],
            }
            count_rows.append(
                {
                    "model": shared["model"],
                    "split": shared["split"],
                    "horizon": shared["horizon"],
                    "normalize": "none",
                    "true_label": shared["true_label"],
                    "true_name": shared["true_name"],
                    "pred_label": shared["pred_label"],
                    "pred_name": shared["pred_name"],
                    "value": int(matrix[true_index, pred_index]),
                }
            )
            normalized_rows.append(
                {
                    "model": shared["model"],
                    "split": shared["split"],
                    "horizon": shared["horizon"],
                    "normalize": "true",
                    "true_label": shared["true_label"],
                    "true_name": shared["true_name"],
                    "pred_label": shared["pred_label"],
                    "pred_name": shared["pred_name"],
                    "value": (
                        float(matrix[true_index, pred_index] / support[true_index])
                        if support[true_index] > 0
                        else 0.0
                    ),
                }
            )

    return {
        "aggregate": pd.DataFrame(
            [
                _aggregate_row(
                    matrix,
                    split=split,
                    horizon=horizon,
                    model_name=model_name,
                )
            ]
        ),
        "class_proportions": pd.DataFrame(class_rows),
        "per_class": pd.DataFrame(per_class_rows),
        "confusion_counts": pd.DataFrame(count_rows),
        "confusion_true_normalized": pd.DataFrame(normalized_rows),
    }


def nonzero_table_from_confusion(
    confusion: np.ndarray,
    *,
    split: str,
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    matrix = _validated_confusion(confusion).copy()
    matrix[1, :] = 0
    n_nonzero = int(matrix.sum())
    base = {
        "model": model_name,
        "split": split,
        "horizon": horizon,
        "n_nonzero_obs": n_nonzero,
    }
    if n_nonzero == 0:
        return pd.DataFrame(
            [
                {
                    **base,
                    "nonzero_accuracy": np.nan,
                    "nonzero_macro_f1": np.nan,
                    "nonzero_balanced_accuracy": np.nan,
                    "predicted_zero_on_nonzero_fraction": np.nan,
                }
            ]
        )

    support = matrix.sum(axis=1)
    _, recall, f1 = _class_metrics(matrix)
    directional = recall[[0, 2]][support[[0, 2]] > 0]
    return pd.DataFrame(
        [
            {
                **base,
                "nonzero_accuracy": float(np.trace(matrix) / n_nonzero),
                "nonzero_macro_f1": float(f1.mean()),
                "nonzero_balanced_accuracy": float(directional.mean()),
                "predicted_zero_on_nonzero_fraction": float(
                    matrix[:, 1].sum() / n_nonzero
                ),
            }
        ]
    )


def _aggregate_row(
    confusion: np.ndarray,
    *,
    split: str,
    horizon: int,
    model_name: str,
) -> dict[str, int | float | str]:
    matrix = _validated_confusion(confusion)
    n_obs = int(matrix.sum())
    if n_obs == 0:
        raise ValueError("Cannot evaluate an empty confusion matrix.")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    _, recall, f1 = _class_metrics(matrix)
    observed_recalls = recall[support > 0]
    return {
        "model": model_name,
        "split": split,
        "horizon": horizon,
        "n_obs": n_obs,
        "accuracy": float(np.trace(matrix) / n_obs),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(observed_recalls.mean()),
        "pred_down_fraction": float(predicted[0] / n_obs),
        "pred_unchanged_fraction": float(predicted[1] / n_obs),
        "pred_up_fraction": float(predicted[2] / n_obs),
        "true_down_fraction": float(support[0] / n_obs),
        "true_unchanged_fraction": float(support[1] / n_obs),
        "true_up_fraction": float(support[2] / n_obs),
    }


def _class_metrics(
    confusion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = _validated_confusion(confusion)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(np.float64)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(3, dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(3, dtype=np.float64),
        where=predicted > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(3, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return precision, recall, f1


def _validated_confusion(confusion: np.ndarray) -> np.ndarray:
    matrix = np.asarray(confusion)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 confusion matrix, got {matrix.shape}.")
    if not np.issubdtype(matrix.dtype, np.integer):
        raise TypeError("Confusion counts must be integers.")
    if (matrix < 0).any():
        raise ValueError("Confusion counts must be non-negative.")
    return matrix.astype(np.int64, copy=False)
