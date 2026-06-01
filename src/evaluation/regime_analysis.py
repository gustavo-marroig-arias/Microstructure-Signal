from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TERNARY_LABELS = (-1, 0, 1)


@dataclass(frozen=True)
class RegimeSpec:
    regime_variable: str
    feature: str
    lower_label: str
    upper_label: str


REGIME_SPECS = (
    RegimeSpec(
        regime_variable="spread",
        feature="relative_spread",
        lower_label="tight_spread",
        upper_label="wide_spread",
    ),
    RegimeSpec(
        regime_variable="realized_volatility",
        feature="realized_vol_20",
        lower_label="low_realized_volatility",
        upper_label="high_realized_volatility",
    ),
    RegimeSpec(
        regime_variable="trade_intensity",
        feature="trade_intensity_1s",
        lower_label="low_trade_intensity",
        upper_label="high_trade_intensity",
    ),
)


def validate_regime_columns(data: pd.DataFrame) -> None:
    missing = sorted({spec.feature for spec in REGIME_SPECS} - set(data.columns))

    if missing:
        raise ValueError(f"Missing regime feature columns: {missing}")


def training_regime_thresholds(train: pd.DataFrame) -> pd.DataFrame:
    """
    Computes regime cutoffs from model-eligible training rows only.

    Protocol:
    - tight/low regime: feature <= train median
    - wide/high regime: feature > train median

    The <= / > convention makes ties deterministic and avoids dropping rows that
    are exactly equal to the train median.
    """
    validate_regime_columns(train)

    rows = []

    for spec in REGIME_SPECS:
        values = train[spec.feature]
        threshold = float(values.median())

        lower_n = int((values <= threshold).sum())
        upper_n = int((values > threshold).sum())
        total_n = int(len(values))

        rows.append(
            {
                "regime_variable": spec.regime_variable,
                "feature": spec.feature,
                "threshold_source": "train_median",
                "threshold_value": threshold,
                "lower_regime": spec.lower_label,
                "lower_rule": "<= train_median",
                "upper_regime": spec.upper_label,
                "upper_rule": "> train_median",
                "train_n": total_n,
                "train_lower_n": lower_n,
                "train_upper_n": upper_n,
                "train_lower_fraction": lower_n / total_n if total_n > 0 else np.nan,
                "train_upper_fraction": upper_n / total_n if total_n > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_idx = y_true.astype(np.int8, copy=False) + 1
    pred_idx = y_pred.astype(np.int8, copy=False) + 1

    if (
        (true_idx < 0).any()
        or (true_idx > 2).any()
        or (pred_idx < 0).any()
        or (pred_idx > 2).any()
    ):
        raise ValueError("Labels must be in {-1, 0, 1}.")

    return np.bincount(
        true_idx * 3 + pred_idx,
        minlength=9,
    ).reshape(3, 3)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=float)
    np.divide(
        numerator,
        denominator,
        out=out,
        where=denominator != 0,
    )
    return out


def aggregate_metric_row_fast(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
    horizon: int,
    model_name: str,
) -> dict:
    n_obs = int(len(y_true))

    if n_obs == 0:
        raise ValueError("Cannot compute aggregate metrics on an empty slice.")

    counts = _confusion_counts(y_true, y_pred)
    row_sum = counts.sum(axis=1).astype(float)
    col_sum = counts.sum(axis=0).astype(float)
    diag = np.diag(counts).astype(float)

    precision = _safe_divide(diag, col_sum)
    recall = _safe_divide(diag, row_sum)

    f1_denominator = precision + recall
    f1 = np.zeros_like(f1_denominator, dtype=float)
    np.divide(
        2.0 * precision * recall,
        f1_denominator,
        out=f1,
        where=f1_denominator != 0,
    )

    pred_counts = counts.sum(axis=0)
    true_counts = counts.sum(axis=1)

    return {
        "model": model_name,
        "split": split,
        "horizon": horizon,
        "n_obs": n_obs,
        "accuracy": float(diag.sum() / n_obs),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "pred_down_fraction": float(pred_counts[0] / n_obs),
        "pred_unchanged_fraction": float(pred_counts[1] / n_obs),
        "pred_up_fraction": float(pred_counts[2] / n_obs),
        "true_down_fraction": float(true_counts[0] / n_obs),
        "true_unchanged_fraction": float(true_counts[1] / n_obs),
        "true_up_fraction": float(true_counts[2] / n_obs),
    }


def nonzero_metric_row_fast(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
    horizon: int,
    model_name: str,
) -> dict:
    nonzero_mask = y_true != 0
    n_nonzero = int(nonzero_mask.sum())

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

    y_true_nz = y_true[nonzero_mask]
    y_pred_nz = y_pred[nonzero_mask]

    aggregate = aggregate_metric_row_fast(
        y_true=y_true_nz,
        y_pred=y_pred_nz,
        split=split,
        horizon=horizon,
        model_name=model_name,
    )

    down_mask = y_true_nz == -1
    up_mask = y_true_nz == 1

    recall_down = (
        float(((y_pred_nz == -1) & down_mask).sum() / down_mask.sum())
        if int(down_mask.sum()) > 0
        else np.nan
    )
    recall_up = (
        float(((y_pred_nz == 1) & up_mask).sum() / up_mask.sum())
        if int(up_mask.sum()) > 0
        else np.nan
    )

    return {
        "model": model_name,
        "split": split,
        "horizon": horizon,
        "n_nonzero_obs": n_nonzero,
        "nonzero_accuracy": aggregate["accuracy"],
        "nonzero_macro_f1": aggregate["macro_f1"],
        "nonzero_balanced_accuracy": float(np.nanmean([recall_down, recall_up])),
        "predicted_zero_on_nonzero_fraction": float((y_pred_nz == 0).mean()),
    }


def _add_regime_metadata(
    row: dict,
    *,
    regime_variable: str,
    feature: str,
    regime: str,
    threshold_source: str,
    threshold_value: float,
    threshold_rule: str,
) -> dict:
    return {
        "regime_variable": regime_variable,
        "feature": feature,
        "regime": regime,
        "threshold_source": threshold_source,
        "threshold_value": threshold_value,
        "threshold_rule": threshold_rule,
        **row,
    }


def evaluate_regime_predictions(
    *,
    data: pd.DataFrame,
    y_pred: np.ndarray,
    split: str,
    horizon: int,
    model_name: str,
    thresholds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates predictions by pre-defined regime slices.

    Regime thresholds must be computed from train and then passed unchanged to
    validation/test. This function does not estimate thresholds from `data`.
    """
    validate_regime_columns(data)

    label_col = f"y_{horizon}"
    if label_col not in data.columns:
        raise ValueError(f"Missing label column: {label_col}")

    y_true_all = data[label_col].astype(np.int8).to_numpy(copy=False)
    y_pred_all = np.asarray(y_pred, dtype=np.int8)

    if len(y_true_all) != len(y_pred_all):
        raise ValueError(
            f"data and y_pred must have same length. "
            f"Got {len(y_true_all)} and {len(y_pred_all)}."
        )

    aggregate_rows = []
    nonzero_rows = []

    for _, threshold_row in thresholds.iterrows():
        feature = str(threshold_row["feature"])
        threshold = float(threshold_row["threshold_value"])

        values = data[feature].to_numpy(copy=False)

        slices = (
            (
                str(threshold_row["lower_regime"]),
                str(threshold_row["lower_rule"]),
                values <= threshold,
            ),
            (
                str(threshold_row["upper_regime"]),
                str(threshold_row["upper_rule"]),
                values > threshold,
            ),
        )

        for regime_label, threshold_rule, mask in slices:
            n_slice = int(mask.sum())

            if n_slice == 0:
                continue

            y_true = y_true_all[mask]
            y_pred_slice = y_pred_all[mask]

            metadata = {
                "regime_variable": str(threshold_row["regime_variable"]),
                "feature": feature,
                "regime": regime_label,
                "threshold_source": str(threshold_row["threshold_source"]),
                "threshold_value": threshold,
                "threshold_rule": threshold_rule,
            }

            aggregate_rows.append(
                _add_regime_metadata(
                    aggregate_metric_row_fast(
                        y_true=y_true,
                        y_pred=y_pred_slice,
                        split=split,
                        horizon=horizon,
                        model_name=model_name,
                    ),
                    **metadata,
                )
            )

            nonzero_rows.append(
                _add_regime_metadata(
                    nonzero_metric_row_fast(
                        y_true=y_true,
                        y_pred=y_pred_slice,
                        split=split,
                        horizon=horizon,
                        model_name=model_name,
                    ),
                    **metadata,
                )
            )

    return pd.DataFrame(aggregate_rows), pd.DataFrame(nonzero_rows)
