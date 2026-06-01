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
from src.modeling.logistic_models import (  # noqa: E402
    LogisticModelConfig,
    coefficient_table,
    fit_logistic_model,
    predict_logistic_model,
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
        description="Run primary model: full-feature multinomial logistic regression."
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
        "--max-iter",
        type=int,
        default=100,
        help="Maximum optimizer iterations.",
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse L2 regularization strength.",
    )

    parser.add_argument(
        "--solver",
        default="saga",
        choices=["lbfgs", "saga", "sag", "newton-cg"],
        help="Logistic regression solver. Use saga for large datasets.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of CPU jobs for supported solvers.",
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
    ] + FEATURE_COLUMNS

    missing = sorted(set(required) - set(data.columns))

    if missing:
        raise ValueError(f"model dataset missing required columns: {missing}")

    return data


def run_full_logistic(
        data: pd.DataFrame,
        config: LogisticModelConfig,
        horizons: tuple[int, ...] = HORIZONS,
        save_predictions: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Runs full-feature logistic regression for each horizon.

    Rules:
    - model_eligible rows only
    - fit on train only
    - evaluate on train and validation only
    - StandardScaler is fitted on train only through the pipeline
    - no class weights
    - no test-set evaluation here
    """
    eligible = data.loc[data["model_eligible"]]

    train = eligible.loc[eligible["split"] == "train"]
    validation = eligible.loc[eligible["split"] == "validation"]

    if train.empty:
        raise ValueError("No model-eligible train rows.")

    if validation.empty:
        raise ValueError("No model-eligible validation rows.")   
    
    eval_results = []
    nonzero_results = []
    coefficient_frames = []
    prediction_frames = []

    for h in horizons:
        label_col = f"y_{h}"

        print()
        print("-" * 80)
        print(f"Fitting {MODEL_NAME} for horizon {h}")
        print("-" * 80)
        print(f"Train rows:      {len(train):,}")
        print(f"Validation rows: {len(validation):,}")
        print(f"Features:        {len(FEATURE_COLUMNS)}")

        model = fit_logistic_model(
            train=train,
            label_col=label_col,
            feature_columns=FEATURE_COLUMNS,
            config=config,
        )

        coefficient_frames.append(
            coefficient_table(
                model=model,
                feature_columns=FEATURE_COLUMNS,
                horizon=h,
                model_name=MODEL_NAME,
            )
        )

        for split_name, split_df in [
            ("train", train),
            ("validation", validation),
        ]:
            y_true = split_df[label_col].astype(int)

            y_pred = predict_logistic_model(
                model=model,
                data=split_df,
                feature_columns=FEATURE_COLUMNS,
            )

            eval_result = evaluate_ternary_predictions(
                y_true=y_true,
                y_pred=y_pred,
                split=split_name,
                horizon=h,
                model_name=MODEL_NAME,
            )
            eval_results.append(eval_result)

            nonzero = evaluate_nonzero_subset(
                y_true=y_true,
                y_pred=y_pred,
                split=split_name,
                horizon=h,
                model_name=MODEL_NAME,
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
                            "model": MODEL_NAME,
                        }
                    )
                )

    tables = concat_evaluation_tables(eval_results)
    nonzero_table = pd.concat(nonzero_results, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)

    predictions = None
    if save_predictions:
        predictions = pd.concat(prediction_frames, ignore_index=True)

    return tables, nonzero_table, coefficients, predictions


def main() -> None:
    args = parse_args()

    if args.symbol != "BTCUSDT":
        raise ValueError("Protocol violation: symbol must remain BTCUSDT.")
    
    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "outputs" / "results"

    dataset_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    dataset_path = processed_dir / f"{dataset_stem}.parquet"

    print("=" * 80)
    print("STEP 12: PRIMARY MODEL — FULL-FEATURE LOGISTIC REGRESSION")
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

    config = LogisticModelConfig(
        C=args.C,
        max_iter=args.max_iter,
        solver=args.solver,
        n_jobs=args.n_jobs,
    )

    print()
    print("Model config:")
    print(config)

    print()
    print("Feature columns:")
    for col in FEATURE_COLUMNS:
        print(f"- {col}")

    print()
    print("Running full-feature logistic regression on train and validation...")

    tables, nonzero_table, coefficients, predictions = run_full_logistic(
        data=data,
        config=config,
        save_predictions=args.save_predictions,
    )

    prefix = tagged_artifact_stem(
        MODEL_NAME,
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )

    print()
    print("Aggregate metrics:")
    print(tables["aggregate"].to_string(index=False))

    print()
    print("Non-zero subset metrics:")
    print(nonzero_table.to_string(index=False))

    print()
    print("Standardized coefficients:")
    print(coefficients.to_string(index=False))

    print()
    save_evaluation_tables(
        tables=tables,
        output_dir=results_dir,
        prefix=prefix,
    )

    nonzero_path = results_dir / f"{prefix}_nonzero_subset.csv"
    nonzero_table.to_csv(nonzero_path, index=False)
    print(f"Saved nonzero subset metrics: {nonzero_path}")

    coefficients_path = results_dir / f"{prefix}_coefficients.csv"
    coefficients.to_csv(coefficients_path, index=False)
    print(f"Saved coefficients: {coefficients_path}")

    if predictions is not None:
        predictions_path = results_dir / f"{prefix}_predictions.parquet"
        predictions.to_parquet(predictions_path, index=False)
        print(f"Saved predictions: {predictions_path}")

    print()
    print("Done. Full-feature logistic model completed.")


if __name__ == "__main__":
    main()
