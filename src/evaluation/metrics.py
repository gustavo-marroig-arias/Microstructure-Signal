from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


TERNARY_LABELS = [-1, 0, 1]
TERNARY_LABEL_NAMES = {
    -1: "down",
    0: "unchanged",
    1: "up",
}


def validate_labels(y_true: Iterable, y_pred: Iterable | None = None) -> None:
    """
    Checks that labels are in {-1, 0, +1}.
    """
    y_true_arr = pd.Series(y_true)

    observed_true = set(y_true_arr.dropna().unique())

    if not observed_true.issubset(set(TERNARY_LABELS)):
        raise ValueError(f"Unexpected y_true labels: {observed_true}")

    if y_pred is not None:
        y_pred_arr = pd.Series(y_pred)

        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError("y_true and y_pred must have the same length.")

        observed_pred = set(y_pred_arr.dropna().unique())

        if not observed_pred.issubset(set(TERNARY_LABELS)):
            raise ValueError(f"Unexpected y_pred labels: {observed_pred}")


def class_proportion_table(
        y_true: Iterable,
        split: str,
        horizon: int,
        model_name: str,
) -> pd.DataFrame:
    """
    Returns class counts and proportions for the true labels.
    """
    validate_labels(y_true)

    y = pd.Series(y_true)
    n = len(y)

    rows = []

    for label in TERNARY_LABELS:
        count = int((y == label).sum())
        proportion = count / n if n > 0 else np.nan

        rows.append(
            {
                "model": model_name,
                "split": split,
                "horizon": horizon,
                "class_label": label,
                "class_name": TERNARY_LABEL_NAMES[label],
                "count": count,
                "proportion": proportion,
            }
        )

    return pd.DataFrame(rows)


def confusion_matrix_table(
    y_true: Iterable,
    y_pred: Iterable,
    split: str,
    horizon: int,
    model_name: str,
    normalize: str | None = None,
) -> pd.DataFrame:
    """
    Returns a tidy confusion matrix table.

    normalize:
    - None: raw counts
    - "true": rows sum to 1
    - "pred": columns sum to 1
    - "all": full matrix sums to 1
    """
    validate_labels(y_true, y_pred)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=TERNARY_LABELS,
        normalize=normalize,
    )

    rows = []

    for i, true_label in enumerate(TERNARY_LABELS):
        for j, pred_label in enumerate(TERNARY_LABELS):
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "horizon": horizon,
                    "normalize": normalize if normalize is not None else "none",
                    "true_label": true_label,
                    "true_name": TERNARY_LABEL_NAMES[true_label],
                    "pred_label": pred_label,
                    "pred_name": TERNARY_LABEL_NAMES[pred_label],
                    "value": matrix[i, j],
                }
            )

    return pd.DataFrame(rows)


def per_class_metric_table(
    y_true: Iterable,
    y_pred: Iterable,
    split: str,
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    """
    Returns precision, recall, F1, and support for each class.
    """
    validate_labels(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=TERNARY_LABELS,
        zero_division=0,
    )

    rows = []

    for label, p, r, f, s in zip(
        TERNARY_LABELS,
        precision,
        recall,
        f1,
        support,
        strict=True,
    ):
        rows.append(
            {
                "model": model_name,
                "split": split,
                "horizon": horizon,
                "class_label": label,
                "class_name": TERNARY_LABEL_NAMES[label],
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
        )

    return pd.DataFrame(rows)


def aggregate_metric_row(
    y_true: Iterable,
    y_pred: Iterable,
    split: str,
    horizon: int,
    model_name: str,
) -> dict:
    """
    Returns aggregate classification metrics for the ternary task.
    """
    validate_labels(y_true, y_pred)

    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    return {
        "model": model_name,
        "split": split,
        "horizon": horizon,
        "n_obs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=TERNARY_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "pred_down_fraction": float((y_pred == -1).mean()),
        "pred_unchanged_fraction": float((y_pred == 0).mean()),
        "pred_up_fraction": float((y_pred == 1).mean()),
        "true_down_fraction": float((y_true == -1).mean()),
        "true_unchanged_fraction": float((y_true == 0).mean()),
        "true_up_fraction": float((y_true == 1).mean()),
    }


def evaluate_ternary_predictions(
    y_true: Iterable,
    y_pred: Iterable,
    split: str,
    horizon: int,
    model_name: str,
) -> dict[str, pd.DataFrame]:
    """
    Evaluation wrapper: evaluates ternary predictions and returns multiple tidy tables.
    """
    aggregate = pd.DataFrame(
        [
            aggregate_metric_row(
                y_true=y_true,
                y_pred=y_pred,
                split=split,
                horizon=horizon,
                model_name=model_name,
            )
        ]
    )

    class_props = class_proportion_table(
        y_true=y_true,
        split=split,
        horizon=horizon,
        model_name=model_name,
    )

    per_class = per_class_metric_table(
        y_true=y_true,
        y_pred=y_pred,
        split=split,
        horizon=horizon,
        model_name=model_name,
    )

    confusion_counts = confusion_matrix_table(
        y_true=y_true,
        y_pred=y_pred,
        split=split,
        horizon=horizon,
        model_name=model_name,
        normalize=None,
    )

    confusion_true_norm = confusion_matrix_table(
        y_true=y_true,
        y_pred=y_pred,
        split=split,
        horizon=horizon,
        model_name=model_name,
        normalize="true",
    )

    return {
        "aggregate": aggregate,
        "class_proportions": class_props,
        "per_class": per_class,
        "confusion_counts": confusion_counts,
        "confusion_true_normalized": confusion_true_norm,
    }


def evaluate_nonzero_subset(
    y_true: Iterable,
    y_pred: Iterable,
    split: str,
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    """
    Secondary robustness metric on rows where the true label is non-zero.

    Important:
    - This is not the headline task.
    - It answers: when the future midprice actually moved, how often did the
      model choose the correct direction?

    Implementation note:
    - Convert y_true and y_pred to aligned position-based Series.
    - Do not preserve external DataFrame indices, because y_true may come from
      a filtered DataFrame while y_pred may be a fresh NumPy array.
    """
    validate_labels(y_true, y_pred)

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length. "
            f"Got {len(y_true)} and {len(y_pred)}."
        )

    mask = y_true != 0

    if int(mask.sum()) == 0:
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "split": split,
                    "horizon": horizon,
                    "n_nonzero_obs": 0,
                    "nonzero_accuracy": np.nan,
                    "nonzero_macro_f1": np.nan,
                    "nonzero_balanced_accuracy": np.nan,
                    "predicted_zero_on_nonzero_fraction": np.nan,
                }
            ]
        )

    y_true_nz = y_true.loc[mask].reset_index(drop=True)
    y_pred_nz = y_pred.loc[mask].reset_index(drop=True)

    recall_down = ((y_true_nz == -1) & (y_pred_nz == -1)).sum() / (y_true_nz == -1).sum()
    recall_up = ((y_true_nz == 1) & (y_pred_nz == 1)).sum() / (y_true_nz == 1).sum()
    nonzero_balanced_accuracy = 0.5 * (recall_down + recall_up)

    return pd.DataFrame(
        [
            {
                "model": model_name,
                "split": split,
                "horizon": horizon,
                "n_nonzero_obs": int(len(y_true_nz)),
                "nonzero_accuracy": float(accuracy_score(y_true_nz, y_pred_nz)),
                "nonzero_macro_f1": float(
                    f1_score(
                        y_true_nz,
                        y_pred_nz,
                        labels=TERNARY_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "nonzero_balanced_accuracy": float(nonzero_balanced_accuracy),
                "predicted_zero_on_nonzero_fraction": float((y_pred_nz == 0).mean()),
            }
        ]
    )


def evaluate_daily_blocks(
    data: pd.DataFrame,
    y_pred: Iterable,
    split: str,
    horizon: int,
    model_name: str,
    timestamp_col: str = "timestamp",
    label_col: str | None = None,
) -> pd.DataFrame:
    """
    Computes aggregate metrics by UTC day.

    Memory-lean implementation:
    - does not copy the full input DataFrame
    - only uses timestamp, true label, and prediction arrays
    """
    if label_col is None:
        label_col = f"y_{horizon}"

    required = [timestamp_col, label_col]

    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"data missing required columns: {missing}")

    timestamps = data[timestamp_col].reset_index(drop=True)
    y_true = data[label_col].astype(int).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} and {len(y_pred)}."
        )

    utc_day = timestamps.dt.floor("D")

    rows = []

    for day in utc_day.drop_duplicates().sort_values():
        mask = utc_day == day

        row = aggregate_metric_row(
            y_true=y_true.loc[mask],
            y_pred=y_pred.loc[mask],
            split=split,
            horizon=horizon,
            model_name=model_name,
        )
        row["utc_day"] = day
        rows.append(row)

    return pd.DataFrame(rows)


def concat_evaluation_tables(results: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """
    Concatenates evaluation output dictionaries returned by evaluate_ternary_predictions.
    """
    keys = [
        "aggregate",
        "class_proportions",
        "per_class",
        "confusion_counts",
        "confusion_true_normalized",
    ]

    combined = {}

    for key in keys:
        combined[key] = pd.concat([r[key] for r in results], ignore_index=True)

    return combined


def save_evaluation_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    prefix: str,
) -> None:
    """
    Saves evaluation tables as CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        path = output_dir / f"{prefix}_{name}.csv"
        table.to_csv(path, index=False)
        print(f"Saved {name}: {path}")