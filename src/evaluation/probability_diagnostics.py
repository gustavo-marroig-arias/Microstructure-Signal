from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.metrics import (
    TERNARY_LABELS,
    confusion_matrix_table,
)


DEFAULT_THRESHOLD_GRID = (
    0.10, 0.125, 0.15, 0.175, 0.20,
    0.225, 0.25, 0.275, 0.30,
    0.325, 0.35, 0.375, 0.40,
    0.45, 0.50,
)


def validate_probability_inputs(
    y_true: pd.Series,
    probabilities: pd.DataFrame,
) -> None:
    """
    Validates probability diagnostic inputs.
    """
    y_true = pd.Series(y_true)
    if y_true.empty:
        raise ValueError("y_true must not be empty.")
    if y_true.isna().any():
        raise ValueError("y_true contains missing values.")

    observed = set(y_true.unique())

    if not observed.issubset(set(TERNARY_LABELS)):
        raise ValueError(f"Unexpected labels: {observed}")

    required_probability_cols = [f"proba_{label}" for label in TERNARY_LABELS]
    missing = sorted(set(required_probability_cols) - set(probabilities.columns))

    if missing:
        raise ValueError(f"Missing probability columns: {missing}")

    if len(y_true) != len(probabilities):
        raise ValueError(
            f"y_true and probabilities must have same length. "
            f"Got {len(y_true)} and {len(probabilities)}."
        )

    p = probabilities[required_probability_cols]

    if p.isna().any().any():
        bad_cols = p.columns[p.isna().any()].tolist()
        raise ValueError(f"Probability columns contain NaNs: {bad_cols}")

    if not np.isfinite(p.to_numpy()).all():
        raise ValueError("Probability columns contain non-finite values.")

    if ((p < 0) | (p > 1)).any().any():
        raise ValueError("Probability values must be between 0 and 1.")

    row_sums = p.sum(axis=1)

    if not np.allclose(row_sums.to_numpy(), 1.0, atol=1e-6):
        raise ValueError("Probability rows do not sum approximately to 1.")


def auc_average_precision_table(
    y_true: pd.Series,
    probabilities: pd.DataFrame,
    horizon: int,
    split: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Computes one-vs-rest ROC AUC and average precision for each class.

    This answers:
    - does P(up) rank true-up rows above other rows?
    - does P(down) rank true-down rows above other rows?
    - does P(unchanged) rank unchanged rows above other rows?
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    probabilities = probabilities.reset_index(drop=True)

    validate_probability_inputs(y_true, probabilities)

    rows = []

    for class_label in TERNARY_LABELS:
        y_binary = (y_true == class_label).astype(int)
        score = probabilities[f"proba_{class_label}"]

        base_rate = float(y_binary.mean())

        if y_binary.nunique() < 2:
            roc_auc = np.nan
            avg_precision = np.nan
        else:
            roc_auc = float(roc_auc_score(y_binary, score))
            avg_precision = float(average_precision_score(y_binary, score))

        rows.append(
            {
                "model": model_name,
                "split": split,
                "horizon": horizon,
                "class_label": class_label,
                "base_rate": base_rate,
                "roc_auc_ovr": roc_auc,
                "average_precision_ovr": avg_precision,
            }
        )

    return pd.DataFrame(rows)


def probability_by_true_class_table(
    y_true: pd.Series,
    probabilities: pd.DataFrame,
    horizon: int,
    split: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Summarizes predicted probabilities by true class.

    Example question:
    - On true up rows, is P(up) higher than on other rows?
    - On true down rows, is P(down) higher than on other rows?
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    probabilities = probabilities.reset_index(drop=True)

    validate_probability_inputs(y_true, probabilities)

    rows = []

    for true_label in TERNARY_LABELS:
        mask = y_true == true_label
        n = int(mask.sum())

        for proba_label in TERNARY_LABELS:
            values = probabilities.loc[mask, f"proba_{proba_label}"]

            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "horizon": horizon,
                    "true_label": true_label,
                    "probability_class": proba_label,
                    "n_obs": n,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "q01": float(values.quantile(0.01)),
                    "q05": float(values.quantile(0.05)),
                    "q10": float(values.quantile(0.10)),
                    "q25": float(values.quantile(0.25)),
                    "q50": float(values.quantile(0.50)),
                    "q75": float(values.quantile(0.75)),
                    "q90": float(values.quantile(0.90)),
                    "q95": float(values.quantile(0.95)),
                    "q99": float(values.quantile(0.99)),
                }
            )

    return pd.DataFrame(rows)


def lift_table_for_class(
    y_true: pd.Series,
    score: pd.Series,
    target_label: int,
    horizon: int,
    split: str,
    model_name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Computes lift table for one class probability.

    Rows are sorted by predicted probability descending and divided into bins.
    Bin 1 is the highest-score bin.

    For example:
    - target_label = +1 and score = P(up)
    - top bin tells us how often true up occurs among the highest P(up) rows
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    score = pd.Series(score).reset_index(drop=True)

    if len(y_true) != len(score):
        raise ValueError("y_true and score must have same length.")

    target = (y_true == target_label).astype(int).to_numpy()
    score_array = score.to_numpy()

    base_rate = float(target.mean())

    order = np.argsort(-score_array)
    bins = np.array_split(order, n_bins)

    rows = []

    for bin_number, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue

        bin_target_rate = float(target[idx].mean())
        lift = bin_target_rate / base_rate if base_rate > 0 else np.nan

        rows.append(
            {
                "model": model_name,
                "split": split,
                "horizon": horizon,
                "target_label": target_label,
                "bin": bin_number,
                "n_obs": int(len(idx)),
                "score_min": float(score_array[idx].min()),
                "score_max": float(score_array[idx].max()),
                "base_rate": base_rate,
                "bin_target_rate": bin_target_rate,
                "lift": lift,
            }
        )

    return pd.DataFrame(rows)


def lift_tables(
    y_true: pd.Series,
    probabilities: pd.DataFrame,
    horizon: int,
    split: str,
    model_name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Computes lift tables for down and up probabilities.

    We focus on down and up because the practical question is whether the model
    can rank non-zero directional events, even when argmax is conservative.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    probabilities = probabilities.reset_index(drop=True)

    validate_probability_inputs(y_true, probabilities)

    frames = []

    for target_label in [-1, 1]:
        frames.append(
            lift_table_for_class(
                y_true=y_true,
                score=probabilities[f"proba_{target_label}"],
                target_label=target_label,
                horizon=horizon,
                split=split,
                model_name=model_name,
                n_bins=n_bins,
            )
        )

    return pd.concat(frames, ignore_index=True)


def threshold_predict(
    probabilities: pd.DataFrame,
    threshold_down: float,
    threshold_up: float,
) -> np.ndarray:
    """
    Converts class probabilities into thresholded hard predictions.

    Rule:
    - predict +1 if P(up) >= threshold_up and P(up) >= P(down)
    - predict -1 if P(down) >= threshold_down and P(down) > P(up)
    - otherwise predict 0

    This is a secondary decision-rule variant, not the primary argmax result.
    """
    p_down = probabilities["proba_-1"].to_numpy()
    p_up = probabilities["proba_1"].to_numpy()

    y_pred = np.zeros(len(probabilities), dtype=int)

    up_signal = (p_up >= threshold_up) & (p_up >= p_down)
    down_signal = (p_down >= threshold_down) & (p_down > p_up)

    y_pred[up_signal] = 1
    y_pred[down_signal] = -1

    return y_pred


def threshold_grid_results(
    y_true: pd.Series,
    probabilities: pd.DataFrame,
    horizon: int,
    split: str,
    model_name: str,
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates a small pre-declared grid of threshold rules on validation.

    Returns:
    - aggregate threshold metrics
    - non-zero subset threshold metrics
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    probabilities = probabilities.reset_index(drop=True)

    validate_probability_inputs(y_true, probabilities)

    thresholds = _validate_threshold_grid(threshold_grid)
    y_array = y_true.to_numpy(dtype=np.int8, copy=False)
    true_indices = y_array + 1
    p_down = probabilities["proba_-1"].to_numpy(dtype=np.float64, copy=False)
    p_up = probabilities["proba_1"].to_numpy(dtype=np.float64, copy=False)

    base_confusion = np.zeros((3, 3), dtype=np.int64)
    base_confusion[:, 1] = np.bincount(true_indices, minlength=3)

    down_candidate = p_down > p_up
    up_candidate = p_up >= p_down
    down_contributions = {
        threshold: _signal_contribution(
            true_indices,
            down_candidate & (p_down >= threshold),
            prediction_index=0,
        )
        for threshold in thresholds
    }
    up_contributions = {
        threshold: _signal_contribution(
            true_indices,
            up_candidate & (p_up >= threshold),
            prediction_index=2,
        )
        for threshold in thresholds
    }

    aggregate_rows = []
    nonzero_rows = []
    for threshold_down in thresholds:
        for threshold_up in thresholds:
            confusion = (
                base_confusion
                + down_contributions[threshold_down]
                + up_contributions[threshold_up]
            )
            row = _aggregate_row_from_confusion(
                confusion,
                split=split,
                horizon=horizon,
                model_name=model_name,
            )
            row["threshold_down"] = threshold_down
            row["threshold_up"] = threshold_up
            aggregate_rows.append(row)

            nonzero = _nonzero_row_from_confusion(
                confusion,
                split=split,
                horizon=horizon,
                model_name=model_name,
            )
            nonzero["threshold_down"] = threshold_down
            nonzero["threshold_up"] = threshold_up
            nonzero_rows.append(nonzero)

    return pd.DataFrame(aggregate_rows), pd.DataFrame(nonzero_rows)


def _validate_threshold_grid(
    threshold_grid: tuple[float, ...],
) -> tuple[float, ...]:
    if not threshold_grid:
        raise ValueError("threshold_grid must not be empty.")
    thresholds = tuple(float(value) for value in threshold_grid)
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("threshold_grid values must be unique.")
    if any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("threshold_grid values must be between 0 and 1.")
    return thresholds


def _signal_contribution(
    true_indices: np.ndarray,
    signal_mask: np.ndarray,
    *,
    prediction_index: int,
) -> np.ndarray:
    counts = np.bincount(true_indices[signal_mask], minlength=3)
    contribution = np.zeros((3, 3), dtype=np.int64)
    contribution[:, 1] -= counts
    contribution[:, prediction_index] += counts
    return contribution


def _aggregate_row_from_confusion(
    confusion: np.ndarray,
    *,
    split: str,
    horizon: int,
    model_name: str,
) -> dict:
    n_obs = int(confusion.sum())
    if n_obs <= 0:
        raise ValueError("Cannot evaluate an empty confusion matrix.")

    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
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

    return {
        "model": model_name,
        "split": split,
        "horizon": horizon,
        "n_obs": n_obs,
        "accuracy": float(true_positive.sum() / n_obs),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "pred_down_fraction": float(predicted[0] / n_obs),
        "pred_unchanged_fraction": float(predicted[1] / n_obs),
        "pred_up_fraction": float(predicted[2] / n_obs),
        "true_down_fraction": float(support[0] / n_obs),
        "true_unchanged_fraction": float(support[1] / n_obs),
        "true_up_fraction": float(support[2] / n_obs),
    }


def _nonzero_row_from_confusion(
    confusion: np.ndarray,
    *,
    split: str,
    horizon: int,
    model_name: str,
) -> dict:
    nonzero_confusion = confusion.copy()
    nonzero_confusion[1, :] = 0
    n_nonzero = int(nonzero_confusion.sum())
    if n_nonzero == 0:
        return {
            "model": model_name,
            "split": split,
            "horizon": horizon,
            "n_nonzero_obs": 0,
            "nonzero_accuracy": np.nan,
            "nonzero_macro_f1": np.nan,
            "nonzero_balanced_accuracy": np.nan,
            "predicted_zero_on_nonzero_fraction": np.nan,
        }

    support = nonzero_confusion.sum(axis=1)
    predicted = nonzero_confusion.sum(axis=0)
    true_positive = np.diag(nonzero_confusion).astype(np.float64)
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
    directional_support = support[[0, 2]] > 0
    directional_recall = recall[[0, 2]][directional_support]

    return {
        "model": model_name,
        "split": split,
        "horizon": horizon,
        "n_nonzero_obs": n_nonzero,
        "nonzero_accuracy": float(true_positive.sum() / n_nonzero),
        "nonzero_macro_f1": float(f1.mean()),
        "nonzero_balanced_accuracy": float(directional_recall.mean()),
        "predicted_zero_on_nonzero_fraction": float(
            nonzero_confusion[:, 1].sum() / n_nonzero
        ),
    }


def best_thresholds_table(
    threshold_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Selects best threshold settings by validation macro F1 and balanced accuracy.

    This is still validation-only. The selected rule must be frozen before test.
    """
    rows = []

    for h, g in threshold_results.groupby("horizon", sort=True):
        best_macro = g.sort_values(
            [
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
                "threshold_down",
                "threshold_up",
            ],
            ascending=[False, False, False, True, True],
        ).iloc[0]

        best_balanced = g.sort_values(
            [
                "balanced_accuracy",
                "macro_f1",
                "accuracy",
                "threshold_down",
                "threshold_up",
            ],
            ascending=[False, False, False, True, True],
        ).iloc[0]

        for selection_metric, row in [
            ("macro_f1", best_macro),
            ("balanced_accuracy", best_balanced),
        ]:
            rows.append(
                {
                    "horizon": h,
                    "selection_metric": selection_metric,
                    "threshold_down": float(row["threshold_down"]),
                    "threshold_up": float(row["threshold_up"]),
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "balanced_accuracy": float(row["balanced_accuracy"]),
                    "pred_down_fraction": float(row["pred_down_fraction"]),
                    "pred_unchanged_fraction": float(row["pred_unchanged_fraction"]),
                    "pred_up_fraction": float(row["pred_up_fraction"]),
                    "true_down_fraction": float(row["true_down_fraction"]),
                    "true_unchanged_fraction": float(row["true_unchanged_fraction"]),
                    "true_up_fraction": float(row["true_up_fraction"]),
                }
            )

    return pd.DataFrame(rows)


def frozen_threshold_mapping(
    table: pd.DataFrame,
    *,
    selection_metric: str,
    horizons: tuple[int, ...],
    dataset_sha256: str,
    experiment_fingerprint: str,
    model_created_utc: dict[int, str],
    model_parameter_fingerprint: dict[int, str],
) -> dict[int, tuple[float, float]]:
    """Validate validation-selected thresholds against exact frozen models."""
    required = {
        "horizon",
        "selection_metric",
        "threshold_down",
        "threshold_up",
        "dataset_sha256",
        "experiment_fingerprint",
        "model_created_utc",
        "model_parameter_fingerprint",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Frozen threshold table is missing columns: {missing}")

    selected = table.loc[table["selection_metric"] == selection_metric].copy()
    if selected.empty:
        raise ValueError(
            f"No thresholds found for selection_metric={selection_metric!r}."
        )
    if selected["horizon"].duplicated().any():
        duplicated = sorted(
            selected.loc[selected["horizon"].duplicated(), "horizon"]
            .astype(int)
            .unique()
        )
        raise ValueError(f"Duplicate frozen thresholds for horizons: {duplicated}")

    selected["horizon"] = selected["horizon"].astype(int)
    observed_horizons = set(selected["horizon"])
    expected_horizons = set(horizons)
    if observed_horizons != expected_horizons:
        raise ValueError(
            "Frozen threshold horizons differ from the protocol: "
            f"observed={sorted(observed_horizons)}, "
            f"expected={sorted(expected_horizons)}"
        )
    if set(model_created_utc) != expected_horizons:
        raise ValueError("Frozen model creation IDs are incomplete.")
    if set(model_parameter_fingerprint) != expected_horizons:
        raise ValueError("Frozen model parameter fingerprints are incomplete.")

    mapping = {}
    for _, row in selected.iterrows():
        horizon = int(row["horizon"])
        threshold_down = float(row["threshold_down"])
        threshold_up = float(row["threshold_up"])
        if (
            not np.isfinite(threshold_down)
            or not np.isfinite(threshold_up)
            or not 0.0 <= threshold_down <= 1.0
            or not 0.0 <= threshold_up <= 1.0
        ):
            raise ValueError(
                f"Invalid frozen thresholds for horizon {horizon}: "
                f"down={threshold_down}, up={threshold_up}"
            )
        if str(row["dataset_sha256"]) != dataset_sha256:
            raise ValueError(
                f"Threshold dataset hash mismatch for horizon {horizon}."
            )
        if str(row["experiment_fingerprint"]) != experiment_fingerprint:
            raise ValueError(
                f"Threshold protocol fingerprint mismatch for horizon {horizon}."
            )
        if str(row["model_created_utc"]) != model_created_utc[horizon]:
            raise ValueError(
                f"Threshold model identity mismatch for horizon {horizon}."
            )
        if (
            str(row["model_parameter_fingerprint"])
            != model_parameter_fingerprint[horizon]
        ):
            raise ValueError(
                "Threshold model parameter fingerprint mismatch for "
                f"horizon {horizon}."
            )
        mapping[horizon] = (threshold_down, threshold_up)

    return mapping


def best_threshold_confusion_tables(
    y_true: pd.Series,
    probabilities: pd.DataFrame,
    best_thresholds: pd.DataFrame,
    split: str,
    model_name: str,
    selection_metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Builds confusion matrices for selected threshold rules.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    probabilities = probabilities.reset_index(drop=True)

    frames = []

    selected = best_thresholds.loc[
        best_thresholds["selection_metric"] == selection_metric
    ]

    for _, row in selected.iterrows():
        h = int(row["horizon"])

        y_pred = threshold_predict(
            probabilities=probabilities,
            threshold_down=float(row["threshold_down"]),
            threshold_up=float(row["threshold_up"]),
        )

        confusion = confusion_matrix_table(
            y_true=y_true,
            y_pred=y_pred,
            split=split,
            horizon=h,
            model_name=model_name,
            normalize=None,
        )

        confusion["selection_metric"] = selection_metric
        confusion["threshold_down"] = float(row["threshold_down"])
        confusion["threshold_up"] = float(row["threshold_up"])

        frames.append(confusion)

    return pd.concat(frames, ignore_index=True)


def save_table(table: pd.DataFrame, output_path: Path) -> None:
    """
    Saves table to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
