from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.probability_diagnostics import (  # noqa: E402
    auc_average_precision_table,
    best_threshold_confusion_tables,
    best_thresholds_table,
    lift_tables,
    probability_by_true_class_table,
    save_table,
    threshold_grid_results,
)
from src.modeling.logistic_models import (  # noqa: E402
    LogisticModelConfig,
    fit_logistic_model,
    predict_logistic_probabilities,
)
from src.data_loader import parse_date # noqa: E402
from src.artifact_naming import tagged_artifact_stem  # noqa: E402

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run probability diagnostics and validation threshold grid for full logistic."
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Protocol symbol. Must be BTCUSDT.",
    )
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help=(
            "Optional filename tag inserted after the artifact kind, e.g. "
            "'v2_float64_features'."
        ),
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse L2 regularization strength.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum optimizer iterations.",
    )

    parser.add_argument(
        "--solver",
        default="saga",
        choices=["lbfgs", "saga", "sag", "newton-cg"],
        help="Logistic regression solver.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of CPU jobs for supported solvers.",
    )

    return parser.parse_args()


def load_model_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing model dataset: {path}")

    data = pd.read_parquet(path)

    required = [
        "event_id",
        "timestamp",
        "split",
        "model_eligible",
        "y_10",
        "y_20",
        "y_50",
    ] + FEATURE_COLUMNS

    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"model dataset missing required columns: {missing}")

    return data


def main() -> None:
    args = parse_args()

    if args.symbol != "BTCUSDT":
        raise ValueError("Protocol violation: symbol must remain BTCUSDT.")
    
    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"

    dataset_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    dataset_path = processed_dir / f"{dataset_stem}.parquet"

    print("=" * 80)
    print("STEP 13.5: PROBABILITY DIAGNOSTICS + VALIDATION THRESHOLD GRID")
    print("=" * 80)

    print(f"Reading model dataset: {dataset_path}")
    data = load_model_dataset(dataset_path)

    eligible = data.loc[data["model_eligible"]].copy()

    train = eligible.loc[eligible["split"] == "train"].copy()
    validation = eligible.loc[eligible["split"] == "validation"].copy()

    if train.empty:
        raise ValueError("No model-eligible train rows.")

    if validation.empty:
        raise ValueError("No model-eligible validation rows.")

    print()
    print("Eligible rows:")
    print(eligible["split"].value_counts().sort_index())

    config = LogisticModelConfig(
        C=args.C,
        max_iter=args.max_iter,
        solver=args.solver,
        n_jobs=args.n_jobs,
    )

    print()
    print("Model config:")
    print(config)

    auc_ap_frames = []
    probability_summary_frames = []
    lift_frames = []
    threshold_grid_frames = []
    threshold_nonzero_frames = []
    best_threshold_frames = []
    best_confusion_frames = []

    for h in HORIZONS:
        label_col = f"y_{h}"

        print()
        print("-" * 80)
        print(f"Probability diagnostics for horizon {h}")
        print("-" * 80)
        print(f"Train rows:      {len(train):,}")
        print(f"Validation rows: {len(validation):,}")

        model = fit_logistic_model(
            train=train,
            label_col=label_col,
            feature_columns=FEATURE_COLUMNS,
            config=config,
        )

        probabilities = predict_logistic_probabilities(
            model=model,
            data=validation,
            feature_columns=FEATURE_COLUMNS,
        )

        y_validation = validation[label_col].astype(int).reset_index(drop=True)

        auc_ap = auc_average_precision_table(
            y_true=y_validation,
            probabilities=probabilities,
            horizon=h,
            split="validation",
            model_name=MODEL_NAME,
        )

        probability_summary = probability_by_true_class_table(
            y_true=y_validation,
            probabilities=probabilities,
            horizon=h,
            split="validation",
            model_name=MODEL_NAME,
        )

        lift = lift_tables(
            y_true=y_validation,
            probabilities=probabilities,
            horizon=h,
            split="validation",
            model_name=MODEL_NAME,
            n_bins=10,
        )

        threshold_grid, threshold_nonzero = threshold_grid_results(
            y_true=y_validation,
            probabilities=probabilities,
            horizon=h,
            split="validation",
            model_name="full_logistic_thresholded",
        )

        best_thresholds = best_thresholds_table(threshold_grid)

        best_confusion = best_threshold_confusion_tables(
            y_true=y_validation,
            probabilities=probabilities,
            best_thresholds=best_thresholds,
            split="validation",
            model_name="full_logistic_thresholded",
            selection_metric="macro_f1",
        )

        auc_ap_frames.append(auc_ap)
        probability_summary_frames.append(probability_summary)
        lift_frames.append(lift)
        threshold_grid_frames.append(threshold_grid)
        threshold_nonzero_frames.append(threshold_nonzero)
        best_threshold_frames.append(best_thresholds)
        best_confusion_frames.append(best_confusion)

        print()
        print("AUC / AP:")
        print(auc_ap.to_string(index=False))

        print()
        print("Best thresholds:")
        print(best_thresholds.to_string(index=False))

    auc_ap_all = pd.concat(auc_ap_frames, ignore_index=True)
    probability_summary_all = pd.concat(probability_summary_frames, ignore_index=True)
    lift_all = pd.concat(lift_frames, ignore_index=True)
    threshold_grid_all = pd.concat(threshold_grid_frames, ignore_index=True)
    threshold_nonzero_all = pd.concat(threshold_nonzero_frames, ignore_index=True)
    best_thresholds_all = pd.concat(best_threshold_frames, ignore_index=True)
    best_confusion_all = pd.concat(best_confusion_frames, ignore_index=True)

    prefix = tagged_artifact_stem(
        "probability_diagnostics_full_logistic",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )

    print()
    save_table(auc_ap_all, reports_dir / f"{prefix}_auc_ap.csv")
    save_table(probability_summary_all, reports_dir / f"{prefix}_probability_by_true_class.csv")
    save_table(lift_all, reports_dir / f"{prefix}_lift.csv")
    save_table(threshold_grid_all, reports_dir / f"{prefix}_threshold_grid.csv")
    save_table(threshold_nonzero_all, reports_dir / f"{prefix}_threshold_nonzero.csv")
    save_table(best_thresholds_all, reports_dir / f"{prefix}_best_thresholds.csv")
    save_table(best_confusion_all, reports_dir / f"{prefix}_best_threshold_confusion_counts.csv")

    print()
    print("Done. Probability diagnostics and validation threshold grid completed.")


if __name__ == "__main__":
    main()
