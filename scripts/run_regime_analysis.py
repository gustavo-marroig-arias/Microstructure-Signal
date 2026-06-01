from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import parse_date  # noqa: E402
from src.artifact_naming import tagged_artifact_stem  # noqa: E402
from src.evaluation.regime_analysis import (  # noqa: E402
    evaluate_regime_predictions,
    training_regime_thresholds,
)
from src.modeling.majority_baseline import MajorityClassPredictor  # noqa: E402


HORIZONS = (10, 20, 50)

QUEUE_IMBALANCE_FEATURES = ["queue_imbalance"]

FULL_FEATURE_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen final models by train-median liquidity regimes. "
            "Uses saved final-test coefficients rather than refitting models."
        )
    )

    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help=(
            "Optional filename tag inserted after the artifact kind, e.g. "
            "'v2_float64_features'."
        ),
    )

    parser.add_argument(
        "--threshold-selection-metric",
        default="macro_f1",
        choices=["macro_f1", "balanced_accuracy"],
        help="Which validation-selected full-logistic threshold rule to apply.",
    )
    parser.add_argument(
        "--splits",
        default="validation,test",
        help="Comma-separated evaluation splits. Default: validation,test.",
    )
    parser.add_argument(
        "--prediction-chunk-size",
        type=int,
        default=2_000_000,
        help="Rows per frozen-coefficient prediction chunk.",
    )

    return parser.parse_args()


def parse_splits(raw: str) -> tuple[str, ...]:
    splits = tuple(x.strip() for x in raw.split(",") if x.strip())

    allowed = {"validation", "test"}
    invalid = sorted(set(splits) - allowed)

    if invalid:
        raise ValueError(f"Unsupported splits: {invalid}. Allowed: {sorted(allowed)}")

    if not splits:
        raise ValueError("At least one split must be requested.")

    return splits


def load_model_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing model dataset: {path}")

    columns = [
        "split",
        "model_eligible",
        "y_10",
        "y_20",
        "y_50",
    ] + FULL_FEATURE_COLUMNS

    data = pd.read_parquet(path, columns=columns)

    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"model dataset missing required columns: {missing}")

    return data


def load_frozen_thresholds(
    reports_dir: Path,
    symbol: str,
    start: str,
    end: str,
    selection_metric: str,
    artifact_tag: str | None = None,
) -> dict[int, tuple[float, float]]:
    prefix = tagged_artifact_stem(
        "probability_diagnostics_full_logistic",
        symbol,
        start,
        end,
        artifact_tag,
    )
    path = reports_dir / f"{prefix}_best_thresholds.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing threshold file: {path}\n"
            "Run scripts/run_probability_diagnostics.py before regime analysis."
        )

    table = pd.read_csv(path)
    selected = table.loc[table["selection_metric"] == selection_metric].copy()

    if selected.empty:
        raise ValueError(f"No thresholds found for selection_metric={selection_metric}")

    thresholds: dict[int, tuple[float, float]] = {}

    for _, row in selected.iterrows():
        thresholds[int(row["horizon"])] = (
            float(row["threshold_down"]),
            float(row["threshold_up"]),
        )

    missing_horizons = sorted(set(HORIZONS) - set(thresholds))
    if missing_horizons:
        raise ValueError(f"Missing thresholds for horizons: {missing_horizons}")

    return thresholds


def load_frozen_coefficients(results_dir: Path, prefix: str) -> pd.DataFrame:
    path = results_dir / f"{prefix}_logistic_coefficients.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen coefficient file: {path}\n"
            "Run scripts/run_final_test_evaluation.py before regime analysis."
        )

    coefficients = pd.read_csv(path)

    required = {
        "model",
        "horizon",
        "class_label",
        "feature",
        "standardized_coefficient",
        "intercept",
    }
    missing = sorted(required - set(coefficients.columns))

    if missing:
        raise ValueError(f"coefficient file missing columns: {missing}")

    return coefficients


def train_scaler_stats(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recomputes StandardScaler statistics on model-eligible train rows.

    The saved coefficient table stores coefficients in standardized-feature
    space, so prediction reconstruction needs train-only means and population
    standard deviations.
    """
    means = train[feature_columns].mean().to_numpy(dtype=np.float64)
    scales = train[feature_columns].std(ddof=0).to_numpy(dtype=np.float64)
    scales = np.where(scales == 0.0, 1.0, scales)

    return means, scales


def coefficient_arrays(
    coefficients: pd.DataFrame,
    *,
    model_name: str,
    horizon: int,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = coefficients.loc[
        (coefficients["model"] == model_name)
        & (coefficients["horizon"] == horizon)
    ].copy()

    if subset.empty:
        raise ValueError(f"Missing coefficients for {model_name}, horizon {horizon}")

    class_labels = np.array(sorted(subset["class_label"].unique()), dtype=np.int8)

    if not np.array_equal(class_labels, np.array([-1, 0, 1], dtype=np.int8)):
        raise ValueError(
            f"Expected class labels [-1, 0, 1], got {class_labels.tolist()}"
        )

    coef = np.empty((len(class_labels), len(feature_columns)), dtype=np.float64)
    intercept = np.empty(len(class_labels), dtype=np.float64)

    for i, class_label in enumerate(class_labels):
        class_rows = subset.loc[subset["class_label"] == class_label]
        by_feature = class_rows.set_index("feature")

        missing_features = sorted(set(feature_columns) - set(by_feature.index))
        if missing_features:
            raise ValueError(
                f"Missing coefficient features for {model_name}, h={horizon}: "
                f"{missing_features}"
            )

        coef[i, :] = by_feature.loc[
            feature_columns,
            "standardized_coefficient",
        ].to_numpy(dtype=np.float64)
        intercept[i] = float(class_rows["intercept"].iloc[0])

    return class_labels, coef, intercept


def predict_from_frozen_coefficients(
    *,
    data: pd.DataFrame,
    feature_columns: list[str],
    means: np.ndarray,
    scales: np.ndarray,
    class_labels: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
    chunk_size: int,
    threshold_down: float | None = None,
    threshold_up: float | None = None,
) -> np.ndarray:
    n = len(data)
    y_pred = np.empty(n, dtype=np.int8)

    if len(feature_columns) != len(means) or len(feature_columns) != len(scales):
        raise ValueError("feature_columns, means, and scales must have same length.")

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)

        x = data.iloc[start:stop][feature_columns].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        x -= means
        x /= scales

        logits = x @ coef.T
        logits += intercept

        if threshold_down is None or threshold_up is None:
            y_pred[start:stop] = class_labels[np.argmax(logits, axis=1)]
            continue

        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        p_down = probabilities[:, 0]
        p_up = probabilities[:, 2]

        chunk_pred = np.zeros(stop - start, dtype=np.int8)
        up_signal = (p_up >= threshold_up) & (p_up >= p_down)
        down_signal = (p_down >= threshold_down) & (p_down > p_up)

        chunk_pred[up_signal] = 1
        chunk_pred[down_signal] = -1

        y_pred[start:stop] = chunk_pred

    return y_pred


def append_regime_results(
    *,
    aggregate_frames: list[pd.DataFrame],
    nonzero_frames: list[pd.DataFrame],
    split_frames: dict[str, pd.DataFrame],
    y_pred_by_split: dict[str, np.ndarray],
    horizon: int,
    model_name: str,
    regime_thresholds: pd.DataFrame,
) -> None:
    for split_name, split_df in split_frames.items():
        aggregate, nonzero = evaluate_regime_predictions(
            data=split_df,
            y_pred=y_pred_by_split[split_name],
            split=split_name,
            horizon=horizon,
            model_name=model_name,
            thresholds=regime_thresholds,
        )

        aggregate_frames.append(aggregate)
        nonzero_frames.append(nonzero)


def main() -> None:
    args = parse_args()

    if args.symbol != "BTCUSDT":
        raise ValueError("Protocol violation: symbol must remain BTCUSDT.")

    requested_splits = parse_splits(args.splits)

    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "outputs" / "results"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"

    dataset_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    dataset_path = processed_dir / f"{dataset_stem}.parquet"
    prefix = tagged_artifact_stem(
        "final_test",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )

    print("=" * 80, flush=True)
    print("REGIME ANALYSIS: TRAIN-MEDIAN CUTS APPLIED TO VALIDATION/TEST", flush=True)
    print("=" * 80, flush=True)
    print(f"Reading model dataset: {dataset_path}", flush=True)

    data = load_model_dataset(dataset_path)
    eligible_mask = data["model_eligible"].astype(bool)

    train = data.loc[eligible_mask & (data["split"] == "train")].copy()
    split_frames = {
        split_name: data.loc[eligible_mask & (data["split"] == split_name)].copy()
        for split_name in requested_splits
    }

    del data
    gc.collect()

    if train.empty:
        raise ValueError("No model-eligible train rows.")

    for split_name, split_df in split_frames.items():
        if split_df.empty:
            raise ValueError(f"No model-eligible rows for split: {split_name}")

    print()
    print("Eligible rows:", flush=True)
    print(f"train: {len(train):,}", flush=True)
    for split_name, split_df in split_frames.items():
        print(f"{split_name}: {len(split_df):,}", flush=True)

    print()
    print("Computing train-median regime thresholds...", flush=True)
    regime_thresholds = training_regime_thresholds(train)
    print(regime_thresholds.to_string(index=False), flush=True)

    print()
    print("Computing train-only scaler statistics...", flush=True)
    full_means, full_scales = train_scaler_stats(train, FULL_FEATURE_COLUMNS)
    queue_means, queue_scales = train_scaler_stats(train, QUEUE_IMBALANCE_FEATURES)

    frozen_thresholds = load_frozen_thresholds(
        reports_dir=reports_dir,
        symbol=args.symbol,
        start=start_str,
        end=end_str,
        selection_metric=args.threshold_selection_metric,
        artifact_tag=args.artifact_tag,
    )
    frozen_coefficients = load_frozen_coefficients(results_dir, prefix)

    aggregate_frames: list[pd.DataFrame] = []
    nonzero_frames: list[pd.DataFrame] = []

    print()
    print("Evaluating majority baseline by regime...", flush=True)
    for h in HORIZONS:
        label_col = f"y_{h}"
        model = MajorityClassPredictor()
        model.fit(train[label_col])

        y_pred_by_split = {
            split_name: model.predict(len(split_df)).astype(np.int8)
            for split_name, split_df in split_frames.items()
        }

        append_regime_results(
            aggregate_frames=aggregate_frames,
            nonzero_frames=nonzero_frames,
            split_frames=split_frames,
            y_pred_by_split=y_pred_by_split,
            horizon=h,
            model_name="majority_baseline",
            regime_thresholds=regime_thresholds,
        )

    print()
    print("Evaluating frozen queue-imbalance logistic by regime...", flush=True)
    for h in HORIZONS:
        print(f"Predicting queue_imbalance_logistic h={h}", flush=True)
        class_labels, coef, intercept = coefficient_arrays(
            frozen_coefficients,
            model_name="queue_imbalance_logistic",
            horizon=h,
            feature_columns=QUEUE_IMBALANCE_FEATURES,
        )

        y_pred_by_split = {
            split_name: predict_from_frozen_coefficients(
                data=split_df,
                feature_columns=QUEUE_IMBALANCE_FEATURES,
                means=queue_means,
                scales=queue_scales,
                class_labels=class_labels,
                coef=coef,
                intercept=intercept,
                chunk_size=args.prediction_chunk_size,
            )
            for split_name, split_df in split_frames.items()
        }

        append_regime_results(
            aggregate_frames=aggregate_frames,
            nonzero_frames=nonzero_frames,
            split_frames=split_frames,
            y_pred_by_split=y_pred_by_split,
            horizon=h,
            model_name="queue_imbalance_logistic",
            regime_thresholds=regime_thresholds,
        )

        del y_pred_by_split
        gc.collect()

    print()
    print("Evaluating frozen full logistic argmax and thresholded models by regime...", flush=True)
    for h in HORIZONS:
        print(f"Predicting full_logistic h={h}", flush=True)
        class_labels, coef, intercept = coefficient_arrays(
            frozen_coefficients,
            model_name="full_logistic",
            horizon=h,
            feature_columns=FULL_FEATURE_COLUMNS,
        )

        threshold_down, threshold_up = frozen_thresholds[h]

        argmax_pred_by_split: dict[str, np.ndarray] = {}
        thresholded_pred_by_split: dict[str, np.ndarray] = {}

        for split_name, split_df in split_frames.items():
            print(f"  split={split_name}: argmax", flush=True)
            argmax_pred_by_split[split_name] = predict_from_frozen_coefficients(
                data=split_df,
                feature_columns=FULL_FEATURE_COLUMNS,
                means=full_means,
                scales=full_scales,
                class_labels=class_labels,
                coef=coef,
                intercept=intercept,
                chunk_size=args.prediction_chunk_size,
            )

            print(f"  split={split_name}: thresholded", flush=True)
            thresholded_pred_by_split[split_name] = predict_from_frozen_coefficients(
                data=split_df,
                feature_columns=FULL_FEATURE_COLUMNS,
                means=full_means,
                scales=full_scales,
                class_labels=class_labels,
                coef=coef,
                intercept=intercept,
                chunk_size=args.prediction_chunk_size,
                threshold_down=threshold_down,
                threshold_up=threshold_up,
            )

        append_regime_results(
            aggregate_frames=aggregate_frames,
            nonzero_frames=nonzero_frames,
            split_frames=split_frames,
            y_pred_by_split=argmax_pred_by_split,
            horizon=h,
            model_name="full_logistic",
            regime_thresholds=regime_thresholds,
        )

        append_regime_results(
            aggregate_frames=aggregate_frames,
            nonzero_frames=nonzero_frames,
            split_frames=split_frames,
            y_pred_by_split=thresholded_pred_by_split,
            horizon=h,
            model_name="full_logistic_thresholded",
            regime_thresholds=regime_thresholds,
        )

        del argmax_pred_by_split, thresholded_pred_by_split
        gc.collect()

    aggregate_table = pd.concat(aggregate_frames, ignore_index=True).sort_values(
        ["split", "regime_variable", "regime", "horizon", "model"]
    )
    nonzero_table = pd.concat(nonzero_frames, ignore_index=True).sort_values(
        ["split", "regime_variable", "regime", "horizon", "model"]
    )

    thresholds_path = reports_dir / f"{prefix}_regime_thresholds.csv"
    aggregate_path = results_dir / f"{prefix}_regime_aggregate.csv"
    nonzero_path = results_dir / f"{prefix}_regime_nonzero_subset.csv"

    regime_thresholds.to_csv(thresholds_path, index=False)
    aggregate_table.to_csv(aggregate_path, index=False)
    nonzero_table.to_csv(nonzero_path, index=False)

    print()
    print(f"Saved regime thresholds: {thresholds_path}", flush=True)
    print(f"Saved regime aggregate metrics: {aggregate_path}", flush=True)
    print(f"Saved regime non-zero metrics: {nonzero_path}", flush=True)

    print()
    print("Test split full-logistic-thresholded summary:", flush=True)
    summary = aggregate_table.loc[
        (aggregate_table["split"] == "test")
        & (aggregate_table["model"] == "full_logistic_thresholded"),
        [
            "regime_variable",
            "regime",
            "horizon",
            "n_obs",
            "macro_f1",
            "balanced_accuracy",
            "true_down_fraction",
            "true_unchanged_fraction",
            "true_up_fraction",
        ],
    ]
    print(summary.to_string(index=False), flush=True)

    print()
    print("Done. Regime analysis completed.", flush=True)


if __name__ == "__main__":
    main()
