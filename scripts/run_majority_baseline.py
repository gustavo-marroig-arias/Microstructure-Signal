from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    concat_evaluation_tables,
    evaluate_nonzero_subset,
    evaluate_ternary_predictions,
    save_evaluation_tables,
)
from src.modeling.majority_baseline import MajorityClassPredictor  # noqa: E402
from src.data_loader import parse_date # noqa: E402
from src.artifact_naming import tagged_artifact_stem  # noqa: E402


HORIZONS = (10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Baseline A: majority-class predictor."
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
        "--save-predictions",
        action="store_true",
        help="Save row-level train/validation predictions. Can be very large.",
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
    ]

    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"model dataset missing required columns: {missing}")

    return data


def run_majority_baseline(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    save_predictions: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Runs majority-class baseline for each horizon.

    Important:
    - fit majority class on train only
    - evaluate on train and validation only
    - use model_eligible rows only
    """
    if data["model_eligible"].isna().any():
        raise ValueError("model_eligible contains missing values.")

    expected_splits = {"train", "validation", "test"}
    observed_splits = set(data["split"].dropna().unique())

    if observed_splits != expected_splits:
        raise ValueError(f"Expected splits {expected_splits}, observed {observed_splits}")

    for col in ["y_10", "y_20", "y_50"]:
        if data[col].isna().any():
            raise ValueError(f"{col} contains missing labels.")

    eligible = data.loc[data["model_eligible"]].copy()

    train = eligible.loc[eligible["split"] == "train"].copy()
    validation = eligible.loc[eligible["split"] == "validation"].copy()

    if train.empty:
        raise ValueError("No model-eligible train rows.")

    if validation.empty:
        raise ValueError("No model-eligible validation rows.")

    eval_results = []
    nonzero_results = []
    prediction_frames = []
    fit_rows = []

    for h in horizons:
        label_col = f"y_{h}"

        model = MajorityClassPredictor()
        model.fit(train[label_col])

        fit_rows.append(
            {
                "horizon": h,
                "majority_class": model.majority_class_,
                "train_count_down": model.class_counts_[-1],
                "train_count_unchanged": model.class_counts_[0],
                "train_count_up": model.class_counts_[1],
                "train_prop_down": model.class_proportions_[-1],
                "train_prop_unchanged": model.class_proportions_[0],
                "train_prop_up": model.class_proportions_[1],
            }
        )

        for split_name, split_df in [
            ("train", train),
            ("validation", validation),
        ]:
            y_true = split_df[label_col].astype(int)
            y_pred = model.predict(len(split_df))

            eval_result = evaluate_ternary_predictions(
                y_true=y_true,
                y_pred=y_pred,
                split=split_name,
                horizon=h,
                model_name="majority_baseline",
            )
            eval_results.append(eval_result)

            nonzero = evaluate_nonzero_subset(
                y_true=y_true,
                y_pred=y_pred,
                split=split_name,
                horizon=h,
                model_name="majority_baseline",
            )
            nonzero_results.append(nonzero)
            
            if save_predictions:
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "event_id": split_df["event_id"].to_numpy(),
                            "timestamp": split_df["timestamp"].to_numpy(),
                            "split": split_name,
                            "horizon": h,
                            "y_true": y_true.to_numpy(),
                            "y_pred": y_pred,
                            "model": "majority_baseline",
                        }
                    )
                )
    
    tables = concat_evaluation_tables(eval_results)
    nonzero_table = pd.concat(nonzero_results, ignore_index=True)
    predictions = None
    if save_predictions:
        predictions = pd.concat(prediction_frames, ignore_index=True)

    fit_summary = pd.DataFrame(fit_rows)

    return tables, nonzero_table, predictions, fit_summary


def main() -> None:
    args = parse_args()

    if args.symbol != "BTCUSDT":
        raise ValueError("Protocol violation: symbol must remain BTCUSDT.")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "outputs" / "results"
    
    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")

    dataset_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    dataset_path = processed_dir / f"{dataset_stem}.parquet"

    print("=" * 80)
    print("STEP 10: BASELINE A — MAJORITY-CLASS PREDICTOR")
    print("=" * 80)

    print(f"Reading model dataset: {dataset_path}")
    data = load_model_dataset(dataset_path)

    print()
    print("Dataset rows:")
    print(f"Total rows:          {len(data):,}")
    print(f"Model-eligible rows: {int(data['model_eligible'].sum()):,}")
    print()
    print("Eligible rows by split:")
    print(data.loc[data["model_eligible"], "split"].value_counts().sort_index())

    print()
    print("Running majority baseline on train and validation...")
    tables, nonzero_table, predictions, fit_summary = run_majority_baseline(
    data=data,
    save_predictions=args.save_predictions,
    )

    prefix = tagged_artifact_stem(
        "majority_baseline",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )

    print()
    print("Fit summary:")
    print(fit_summary.to_string(index=False))

    print()
    print("Aggregate metrics:")
    print(tables["aggregate"].to_string(index=False))

    print()
    print("Non-zero subset metrics:")
    print(nonzero_table.to_string(index=False))

    print()
    save_evaluation_tables(
        tables=tables,
        output_dir=results_dir,
        prefix=prefix,
    )

    nonzero_path = results_dir / f"{prefix}_nonzero_subset.csv"
    nonzero_table.to_csv(nonzero_path, index=False)
    print(f"Saved nonzero subset metrics: {nonzero_path}")

    if predictions is not None:
        predictions_path = results_dir / f"{prefix}_predictions.parquet"
        predictions.to_parquet(predictions_path, index=False)
        print(f"Saved predictions: {predictions_path}")

    fit_summary_path = results_dir / f"{prefix}_fit_summary.csv"
    fit_summary.to_csv(fit_summary_path, index=False)
    print(f"Saved fit summary: {fit_summary_path}")

    print()
    print("Done. Majority baseline completed.")


if __name__ == "__main__":
    main()
