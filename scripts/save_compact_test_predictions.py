from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_naming import tagged_artifact_stem  # noqa: E402
from src.data_loader import parse_date  # noqa: E402


HORIZONS = (10, 20, 50)
MODEL_NAME = "full_logistic"

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

DIAGNOSTIC_COLUMNS = [
    "relative_spread",
    "realized_vol_20",
    "trade_intensity_1s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save compact test-only full-logistic probabilities and predictions "
            "from frozen final-test coefficients. This does not refit models."
        )
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help="Optional filename tag, e.g. 'v2_float64_features'.",
    )
    parser.add_argument(
        "--threshold-selection-metric",
        default="macro_f1",
        choices=["macro_f1", "balanced_accuracy"],
        help="Which validation-selected threshold rule to apply.",
    )
    parser.add_argument(
        "--prediction-chunk-size",
        type=int,
        default=2_000_000,
        help="Rows per prediction chunk.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec.",
    )
    return parser.parse_args()


def load_thresholds(
    reports_dir: Path,
    symbol: str,
    start: str,
    end: str,
    artifact_tag: str | None,
    selection_metric: str,
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
        raise FileNotFoundError(f"Missing threshold file: {path}")

    table = pd.read_csv(path)
    selected = table.loc[table["selection_metric"] == selection_metric]

    thresholds: dict[int, tuple[float, float]] = {}
    for _, row in selected.iterrows():
        thresholds[int(row["horizon"])] = (
            float(row["threshold_down"]),
            float(row["threshold_up"]),
        )

    missing = sorted(set(HORIZONS) - set(thresholds))
    if missing:
        raise ValueError(f"Missing thresholds for horizons: {missing}")

    return thresholds


def load_coefficients(results_dir: Path, prefix: str) -> pd.DataFrame:
    path = results_dir / f"{prefix}_logistic_coefficients.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing final-test coefficient file: {path}")

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
        raise ValueError(f"Coefficient file missing columns: {missing}")

    return coefficients


def coefficient_arrays(
    coefficients: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = coefficients.loc[
        (coefficients["model"] == MODEL_NAME)
        & (coefficients["horizon"] == horizon)
    ].copy()

    if subset.empty:
        raise ValueError(f"Missing {MODEL_NAME} coefficients for h={horizon}")

    class_labels = np.array(sorted(subset["class_label"].unique()), dtype=np.int8)
    expected_labels = np.array([-1, 0, 1], dtype=np.int8)
    if not np.array_equal(class_labels, expected_labels):
        raise ValueError(f"Expected class labels {-1, 0, 1}, got {class_labels}")

    coef = np.empty((len(class_labels), len(FEATURE_COLUMNS)), dtype=np.float64)
    intercept = np.empty(len(class_labels), dtype=np.float64)

    for i, class_label in enumerate(class_labels):
        class_rows = subset.loc[subset["class_label"] == class_label]
        by_feature = class_rows.set_index("feature")

        missing = sorted(set(FEATURE_COLUMNS) - set(by_feature.index))
        if missing:
            raise ValueError(f"Missing coefficient features for h={horizon}: {missing}")

        coef[i, :] = by_feature.loc[
            FEATURE_COLUMNS,
            "standardized_coefficient",
        ].to_numpy(dtype=np.float64)
        intercept[i] = float(class_rows["intercept"].iloc[0])

    return class_labels, coef, intercept


def train_scaler_stats(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    columns = ["split", "model_eligible", *FEATURE_COLUMNS]
    print(f"Reading train features for scaler stats: {dataset_path}", flush=True)
    data = pd.read_parquet(dataset_path, columns=columns)

    mask = data["model_eligible"].astype(bool) & (data["split"] == "train")
    train = data.loc[mask, FEATURE_COLUMNS]

    if train.empty:
        raise ValueError("No model-eligible train rows.")

    scaler = StandardScaler().fit(train)
    means = scaler.mean_.astype(np.float64, copy=False)
    scales = scaler.scale_.astype(np.float64, copy=False)

    del data, train
    gc.collect()

    return means, scales


def load_test_frame(dataset_path: Path) -> pd.DataFrame:
    columns = [
        "event_id",
        "timestamp",
        "split",
        "model_eligible",
        *FEATURE_COLUMNS,
        *(f"y_{h}" for h in HORIZONS),
    ]
    print(f"Reading model-eligible test rows: {dataset_path}", flush=True)
    data = pd.read_parquet(dataset_path, columns=columns)

    mask = data["model_eligible"].astype(bool) & (data["split"] == "test")
    test = data.loc[mask].copy()

    del data
    gc.collect()

    if test.empty:
        raise ValueError("No model-eligible test rows.")

    return test


def probabilities_from_coefficients(
    *,
    test: pd.DataFrame,
    means: np.ndarray,
    scales: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    n_rows = len(test)
    probabilities = np.empty((n_rows, 3), dtype=np.float32)

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)

        x = test.iloc[start:stop][FEATURE_COLUMNS].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        x -= means
        x /= scales

        logits = x @ coef.T
        logits += intercept
        logits -= logits.max(axis=1, keepdims=True)

        exp_logits = np.exp(logits)
        chunk_probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probabilities[start:stop, :] = chunk_probabilities.astype(np.float32)

    return probabilities


def threshold_predict(
    probabilities: np.ndarray,
    threshold_down: float,
    threshold_up: float,
) -> np.ndarray:
    p_down = probabilities[:, 0]
    p_up = probabilities[:, 2]

    y_pred = np.zeros(len(probabilities), dtype=np.int8)
    up_signal = (p_up >= threshold_up) & (p_up >= p_down)
    down_signal = (p_down >= threshold_down) & (p_down > p_up)

    y_pred[up_signal] = 1
    y_pred[down_signal] = -1

    return y_pred


def main() -> None:
    args = parse_args()

    if args.symbol != "BTCUSDT":
        raise ValueError("Protocol violation: symbol must remain BTCUSDT.")

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
    final_prefix = tagged_artifact_stem(
        "final_test",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    output_prefix = tagged_artifact_stem(
        "compact_test_predictions",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )

    dataset_path = processed_dir / f"{dataset_stem}.parquet"
    output_path = results_dir / f"{output_prefix}.parquet"
    summary_path = reports_dir / f"{output_prefix}_summary.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing model dataset: {dataset_path}")

    print("=" * 80, flush=True)
    print("COMPACT TEST-ONLY PROBABILITIES AND PREDICTIONS", flush=True)
    print("=" * 80, flush=True)
    print("This uses frozen final-test coefficients and does not refit models.", flush=True)

    thresholds = load_thresholds(
        reports_dir=reports_dir,
        symbol=args.symbol,
        start=start_str,
        end=end_str,
        artifact_tag=args.artifact_tag,
        selection_metric=args.threshold_selection_metric,
    )
    coefficients = load_coefficients(results_dir, final_prefix)
    means, scales = train_scaler_stats(dataset_path)
    test = load_test_frame(dataset_path)

    print(f"Test rows: {len(test):,}", flush=True)

    output = test[
        ["event_id", "timestamp", *DIAGNOSTIC_COLUMNS]
    ].copy()
    output["relative_spread"] = output["relative_spread"].astype(np.float32)
    output["realized_vol_20"] = output["realized_vol_20"].astype(np.float32)
    output["trade_intensity_1s"] = output["trade_intensity_1s"].astype(np.int32)

    summary_rows = []

    for h in HORIZONS:
        print(f"Predicting h={h}", flush=True)
        class_labels, coef, intercept = coefficient_arrays(coefficients, h)
        probabilities = probabilities_from_coefficients(
            test=test,
            means=means,
            scales=scales,
            coef=coef,
            intercept=intercept,
            chunk_size=args.prediction_chunk_size,
        )

        y_pred_argmax = class_labels[np.argmax(probabilities, axis=1)].astype(np.int8)
        threshold_down, threshold_up = thresholds[h]
        y_pred_thresholded = threshold_predict(
            probabilities,
            threshold_down=threshold_down,
            threshold_up=threshold_up,
        )

        output[f"y_true_h{h}"] = test[f"y_{h}"].to_numpy(dtype=np.int8)
        output[f"y_pred_argmax_h{h}"] = y_pred_argmax
        output[f"y_pred_thresholded_h{h}"] = y_pred_thresholded
        output[f"proba_down_h{h}"] = probabilities[:, 0]
        output[f"proba_unchanged_h{h}"] = probabilities[:, 1]
        output[f"proba_up_h{h}"] = probabilities[:, 2]

        summary_rows.append(
            {
                "horizon": h,
                "threshold_selection_metric": args.threshold_selection_metric,
                "threshold_down": threshold_down,
                "threshold_up": threshold_up,
                "argmax_pred_down_fraction": float((y_pred_argmax == -1).mean()),
                "argmax_pred_unchanged_fraction": float((y_pred_argmax == 0).mean()),
                "argmax_pred_up_fraction": float((y_pred_argmax == 1).mean()),
                "thresholded_pred_down_fraction": float(
                    (y_pred_thresholded == -1).mean()
                ),
                "thresholded_pred_unchanged_fraction": float(
                    (y_pred_thresholded == 0).mean()
                ),
                "thresholded_pred_up_fraction": float(
                    (y_pred_thresholded == 1).mean()
                ),
            }
        )

        del probabilities, y_pred_argmax, y_pred_thresholded
        gc.collect()

    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing compact parquet: {output_path}", flush=True)
    output.to_parquet(output_path, index=False, compression=args.compression)

    summary = pd.DataFrame(summary_rows)
    summary.insert(0, "rows", len(output))
    summary.insert(0, "output_path", str(output_path.relative_to(PROJECT_ROOT)))
    summary.insert(0, "artifact_tag", args.artifact_tag or "")
    summary.to_csv(summary_path, index=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved compact predictions: {output_path}", flush=True)
    print(f"File size: {file_size_mb:,.1f} MB", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
