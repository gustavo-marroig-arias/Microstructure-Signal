from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    concat_evaluation_tables,
    evaluate_daily_blocks,
    evaluate_nonzero_subset,
    evaluate_ternary_predictions,
    save_evaluation_tables,
)
from src.evaluation.model_comparison import (  # noqa: E402
    signal_decay_table,
    validation_ranking_table,
)
from src.evaluation.probability_diagnostics import threshold_predict  # noqa: E402
from src.modeling.majority_baseline import MajorityClassPredictor  # noqa: E402
from src.modeling.logistic_models import (  # noqa: E402
    LogisticModelConfig,
    coefficient_table,
    fit_logistic_model,
    predict_logistic_model,
    predict_logistic_probabilities,
)
from src.data_loader import parse_date # noqa: E402
from src.artifact_naming import tagged_artifact_stem  # noqa: E402


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
        description="Run final test evaluation for the frozen model comparison."
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

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--solver",
        default="saga",
        choices=["lbfgs", "saga", "sag", "newton-cg"],
    )
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument(
        "--threshold-selection-metric",
        default="macro_f1",
        choices=["macro_f1", "balanced_accuracy"],
        help="Which validation-selected threshold rule to apply to test.",
    )

    return parser.parse_args()


def load_model_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing model dataset: {path}")

    columns = [
        "event_id",
        "timestamp",
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
    """
    Reads validation-selected thresholds from the probability diagnostics report.

    Returns:
        {horizon: (threshold_down, threshold_up)}
    """
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
            "Run scripts/run_probability_diagnostics.py before final test evaluation."
        )

    table = pd.read_csv(path)

    selected = table.loc[table["selection_metric"] == selection_metric].copy()

    if selected.empty:
        raise ValueError(f"No thresholds found for selection_metric={selection_metric}")

    thresholds: dict[int, tuple[float, float]] = {}

    for _, row in selected.iterrows():
        h = int(row["horizon"])
        thresholds[h] = (
            float(row["threshold_down"]),
            float(row["threshold_up"]),
        )

    missing_horizons = sorted(set(HORIZONS) - set(thresholds))
    if missing_horizons:
        raise ValueError(f"Missing thresholds for horizons: {missing_horizons}")

    return thresholds


def argmax_from_probabilities(probabilities: pd.DataFrame) -> np.ndarray:
    """
    Converts probability columns proba_-1, proba_0, proba_1 into argmax labels.
    """
    proba_matrix = probabilities[["proba_-1", "proba_0", "proba_1"]].to_numpy()
    label_array = np.array([-1, 0, 1], dtype=int)
    return label_array[np.argmax(proba_matrix, axis=1)]


def add_evaluation_outputs(
    eval_results: list[dict[str, pd.DataFrame]],
    nonzero_results: list[pd.DataFrame],
    daily_results: list[pd.DataFrame],
    data: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    horizon: int,
) -> None:
    """
    Adds pooled, non-zero, and daily test evaluations for one prediction vector.
    """
    eval_results.append(
        evaluate_ternary_predictions(
            y_true=y_true,
            y_pred=y_pred,
            split="test",
            horizon=horizon,
            model_name=model_name,
        )
    )

    nonzero_results.append(
        evaluate_nonzero_subset(
            y_true=y_true,
            y_pred=y_pred,
            split="test",
            horizon=horizon,
            model_name=model_name,
        )
    )

    daily_results.append(
        evaluate_daily_blocks(
            data=data,
            y_pred=y_pred,
            split="test",
            horizon=horizon,
            model_name=model_name,
            label_col=f"y_{horizon}",
        )
    )


def evaluate_majority_on_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[list[dict[str, pd.DataFrame]], list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame]:
    eval_results = []
    nonzero_results = []
    daily_results = []
    fit_rows = []

    model_name = "majority_baseline"

    for h in HORIZONS:
        label_col = f"y_{h}"

        model = MajorityClassPredictor()
        model.fit(train[label_col])

        fit_rows.append(
            {
                "model": model_name,
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

        y_true = test[label_col].astype(int)
        y_pred = model.predict(len(test))

        add_evaluation_outputs(
            eval_results=eval_results,
            nonzero_results=nonzero_results,
            daily_results=daily_results,
            data=test,
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_name,
            horizon=h,
        )

    return eval_results, nonzero_results, daily_results, pd.DataFrame(fit_rows)


def evaluate_queue_imbalance_on_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: LogisticModelConfig,
) -> tuple[list[dict[str, pd.DataFrame]], list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame]:
    eval_results = []
    nonzero_results = []
    daily_results = []
    coefficient_frames = []

    model_name = "queue_imbalance_logistic"

    for h in HORIZONS:
        label_col = f"y_{h}"

        print()
        print("-" * 80)
        print(f"Fitting {model_name} for final test evaluation, horizon {h}")
        print("-" * 80)

        model = fit_logistic_model(
            train=train,
            label_col=label_col,
            feature_columns=QUEUE_IMBALANCE_FEATURES,
            config=config,
        )

        coefficient_frames.append(
            coefficient_table(
                model=model,
                feature_columns=QUEUE_IMBALANCE_FEATURES,
                horizon=h,
                model_name=model_name,
            )
        )

        y_true = test[label_col].astype(int)

        y_pred = predict_logistic_model(
            model=model,
            data=test,
            feature_columns=QUEUE_IMBALANCE_FEATURES,
        )

        add_evaluation_outputs(
            eval_results=eval_results,
            nonzero_results=nonzero_results,
            daily_results=daily_results,
            data=test,
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_name,
            horizon=h,
        )

        del model, y_pred
        gc.collect()

    coefficients = pd.concat(coefficient_frames, ignore_index=True)

    return eval_results, nonzero_results, daily_results, coefficients


def evaluate_full_logistic_and_thresholded_on_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: LogisticModelConfig,
    frozen_thresholds: dict[int, tuple[float, float]],
    threshold_selection_metric: str,
) -> tuple[list[dict[str, pd.DataFrame]], list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """
    Evaluates:
    - full_logistic argmax, primary model
    - full_logistic_thresholded, secondary validation-selected decision rule
    """
    eval_results = []
    nonzero_results = []
    daily_results = []
    coefficient_frames = []
    threshold_rows = []

    for h in HORIZONS:
        label_col = f"y_{h}"

        print()
        print("-" * 80)
        print(f"Fitting full_logistic for final test evaluation, horizon {h}")
        print("-" * 80)
        print(f"Train rows: {len(train):,}")
        print(f"Test rows:  {len(test):,}")

        model = fit_logistic_model(
            train=train,
            label_col=label_col,
            feature_columns=FULL_FEATURE_COLUMNS,
            config=config,
        )

        coefficient_frames.append(
            coefficient_table(
                model=model,
                feature_columns=FULL_FEATURE_COLUMNS,
                horizon=h,
                model_name="full_logistic",
            )
        )

        y_true = test[label_col].astype(int)

        print("Computing test probabilities...")
        probabilities = predict_logistic_probabilities(
            model=model,
            data=test,
            feature_columns=FULL_FEATURE_COLUMNS,
        )

        # Primary argmax model
        y_pred_argmax = argmax_from_probabilities(probabilities)

        add_evaluation_outputs(
            eval_results=eval_results,
            nonzero_results=nonzero_results,
            daily_results=daily_results,
            data=test,
            y_true=y_true,
            y_pred=y_pred_argmax,
            model_name="full_logistic",
            horizon=h,
        )

        # Secondary thresholded model
        threshold_down, threshold_up = frozen_thresholds[h]

        threshold_rows.append(
            {
                "model": "full_logistic_thresholded",
                "horizon": h,
                "threshold_down": threshold_down,
                "threshold_up": threshold_up,
                "threshold_source": f"validation_selected_{threshold_selection_metric}",
            }
        )

        y_pred_thresholded = threshold_predict(
            probabilities=probabilities,
            threshold_down=threshold_down,
            threshold_up=threshold_up,
        )

        add_evaluation_outputs(
            eval_results=eval_results,
            nonzero_results=nonzero_results,
            daily_results=daily_results,
            data=test,
            y_true=y_true,
            y_pred=y_pred_thresholded,
            model_name="full_logistic_thresholded",
            horizon=h,
        )

        del model, probabilities, y_pred_argmax, y_pred_thresholded
        gc.collect()

    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    threshold_table = pd.DataFrame(threshold_rows)

    return eval_results, nonzero_results, daily_results, coefficients, threshold_table


def compute_deltas(
    aggregate: pd.DataFrame,
    model_name: str,
    reference_models: list[str],
) -> pd.DataFrame:
    """
    Computes metric deltas for one model versus reference models by horizon.
    """
    metrics = ["accuracy", "macro_f1", "balanced_accuracy"]

    rows = []

    for h, g in aggregate.groupby("horizon", sort=True):
        by_model = g.set_index("model")

        if model_name not in by_model.index:
            raise ValueError(f"Missing model {model_name} for horizon {h}")

        model_row = by_model.loc[model_name]

        for reference in reference_models:
            if reference not in by_model.index:
                raise ValueError(f"Missing reference {reference} for horizon {h}")

            ref_row = by_model.loc[reference]

            row = {
                "horizon": h,
                "model": model_name,
                "reference_model": reference,
            }

            for metric in metrics:
                row[f"{metric}_reference"] = float(ref_row[metric])
                row[f"{metric}_model"] = float(model_row[metric])
                row[f"{metric}_delta"] = float(model_row[metric] - ref_row[metric])

            rows.append(row)

    return pd.DataFrame(rows)


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
    dataset_path = processed_dir / f"{dataset_stem}.parquet"

    print("=" * 80)
    print("STEP 14: FINAL TEST EVALUATION")
    print("=" * 80)

    print(f"Reading model dataset: {dataset_path}")
    data = load_model_dataset(dataset_path)

    print()
    print("Dataset rows:")
    print(f"Total rows:          {len(data):,}")
    print(f"Model-eligible rows: {int(data['model_eligible'].sum()):,}")

    eligible_mask = data["model_eligible"].astype(bool)

    train = data.loc[eligible_mask & (data["split"] == "train")].copy()
    test = data.loc[eligible_mask & (data["split"] == "test")].copy()

    del data
    gc.collect()

    if train.empty:
        raise ValueError("No model-eligible train rows.")

    if test.empty:
        raise ValueError("No model-eligible test rows.")

    print()
    print("Eligible rows used:")
    print(f"Train: {len(train):,}")
    print(f"Test:  {len(test):,}")

    print()
    print("Test timestamp range:")
    print(f"{test['timestamp'].min()} → {test['timestamp'].max()}")

    frozen_thresholds = load_frozen_thresholds(
        reports_dir=reports_dir,
        symbol=args.symbol,
        start=start_str,
        end=end_str,
        selection_metric=args.threshold_selection_metric,
        artifact_tag=args.artifact_tag,
    )

    print()
    print("Frozen threshold rule:")
    for h, (td, tu) in frozen_thresholds.items():
        print(f"h={h}: threshold_down={td}, threshold_up={tu}")

    config = LogisticModelConfig(
        C=args.C,
        max_iter=args.max_iter,
        solver=args.solver,
        n_jobs=args.n_jobs,
    )

    print()
    print("Logistic config:")
    print(config)

    all_eval_results = []
    all_nonzero_results = []
    all_daily_results = []
    all_coefficients = []
    fit_summaries = []

    print()
    print("Evaluating majority baseline on test...")
    eval_results, nonzero_results, daily_results, fit_summary = evaluate_majority_on_test(
        train=train,
        test=test,
    )
    all_eval_results.extend(eval_results)
    all_nonzero_results.extend(nonzero_results)
    all_daily_results.extend(daily_results)
    fit_summaries.append(fit_summary)

    print()
    print("Evaluating queue-imbalance-only logistic on test...")
    eval_results, nonzero_results, daily_results, coefficients = evaluate_queue_imbalance_on_test(
        train=train,
        test=test,
        config=config,
    )
    all_eval_results.extend(eval_results)
    all_nonzero_results.extend(nonzero_results)
    all_daily_results.extend(daily_results)
    all_coefficients.append(coefficients)

    print()
    print("Evaluating full logistic argmax and thresholded models on test...")
    (
        eval_results,
        nonzero_results,
        daily_results,
        coefficients,
        threshold_table,
    ) = evaluate_full_logistic_and_thresholded_on_test(
        train=train,
        test=test,
        config=config,
        frozen_thresholds=frozen_thresholds,
        threshold_selection_metric=args.threshold_selection_metric,
    )
    all_eval_results.extend(eval_results)
    all_nonzero_results.extend(nonzero_results)
    all_daily_results.extend(daily_results)
    all_coefficients.append(coefficients)

    test_tables = concat_evaluation_tables(all_eval_results)
    nonzero_table = pd.concat(all_nonzero_results, ignore_index=True)
    daily_table = pd.concat(all_daily_results, ignore_index=True)
    fit_summary_table = pd.concat(fit_summaries, ignore_index=True)
    coefficient_table_all = pd.concat(all_coefficients, ignore_index=True)

    test_comparison = test_tables["aggregate"].sort_values(
        ["horizon", "model"]
    ).reset_index(drop=True)

    ranking = validation_ranking_table(test_comparison)
    decay = signal_decay_table(test_comparison)

    full_deltas = compute_deltas(
        aggregate=test_comparison,
        model_name="full_logistic",
        reference_models=["majority_baseline", "queue_imbalance_logistic"],
    )

    thresholded_deltas = compute_deltas(
        aggregate=test_comparison,
        model_name="full_logistic_thresholded",
        reference_models=[
            "majority_baseline",
            "queue_imbalance_logistic",
            "full_logistic",
        ],
    )

    prefix = tagged_artifact_stem(
        "final_test",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )

    print()
    print("Final test aggregate comparison:")
    print(test_comparison.to_string(index=False))

    print()
    print("Final test ranking:")
    print(ranking.to_string(index=False))

    print()
    print("Full logistic deltas:")
    print(full_deltas.to_string(index=False))

    print()
    print("Thresholded full logistic deltas:")
    print(thresholded_deltas.to_string(index=False))

    print()
    print("Final test non-zero subset:")
    print(nonzero_table.to_string(index=False))

    print()
    save_evaluation_tables(
        tables=test_tables,
        output_dir=results_dir,
        prefix=prefix,
    )

    nonzero_path = results_dir / f"{prefix}_nonzero_subset.csv"
    nonzero_table.to_csv(nonzero_path, index=False)
    print(f"Saved non-zero subset metrics: {nonzero_path}")

    daily_path = results_dir / f"{prefix}_daily_blocks.csv"
    daily_table.to_csv(daily_path, index=False)
    print(f"Saved daily block metrics: {daily_path}")

    fit_summary_path = results_dir / f"{prefix}_majority_fit_summary.csv"
    fit_summary_table.to_csv(fit_summary_path, index=False)
    print(f"Saved majority fit summary: {fit_summary_path}")

    coefficients_path = results_dir / f"{prefix}_logistic_coefficients.csv"
    coefficient_table_all.to_csv(coefficients_path, index=False)
    print(f"Saved logistic coefficients: {coefficients_path}")

    thresholds_path = reports_dir / f"{prefix}_frozen_thresholds.csv"
    threshold_table.to_csv(thresholds_path, index=False)
    print(f"Saved frozen thresholds: {thresholds_path}")

    ranking_path = reports_dir / f"{prefix}_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    print(f"Saved final test ranking: {ranking_path}")

    decay_path = reports_dir / f"{prefix}_signal_decay.csv"
    decay.to_csv(decay_path, index=False)
    print(f"Saved final test signal decay: {decay_path}")

    full_deltas_path = reports_dir / f"{prefix}_full_vs_baselines_deltas.csv"
    full_deltas.to_csv(full_deltas_path, index=False)
    print(f"Saved full logistic deltas: {full_deltas_path}")

    thresholded_deltas_path = reports_dir / f"{prefix}_thresholded_deltas.csv"
    thresholded_deltas.to_csv(thresholded_deltas_path, index=False)
    print(f"Saved thresholded deltas: {thresholded_deltas_path}")

    print()
    print("Done. Final test evaluation completed.")


if __name__ == "__main__":
    main()
