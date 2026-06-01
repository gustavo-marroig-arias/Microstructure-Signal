from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.artifact_naming import tagged_artifact_stem


MODEL_PREFIXES = {
    "majority_baseline": "majority_baseline",
    "queue_imbalance_logistic": "queue_imbalance_logistic",
    "full_logistic": "full_logistic",
}


PRIMARY_METRICS = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
]


def result_prefix(
    model_name: str,
    symbol: str,
    start: str,
    end: str,
    artifact_tag: str | None = None,
) -> str:
    """
    Builds the filename prefix used by the model scripts.
    """
    if model_name not in MODEL_PREFIXES:
        raise ValueError(f"Unknown model name: {model_name}")

    return tagged_artifact_stem(
        MODEL_PREFIXES[model_name],
        symbol,
        start,
        end,
        artifact_tag,
    )


def read_model_result_table(
        results_dir: Path,
        model_name: str,
        symbol: str,
        start: str,
        end: str,
        table_name: str,
        artifact_tag: str | None = None,
) -> pd.DataFrame:
    """
    Reads one result CSV for one model.
    """
    prefix = result_prefix(model_name, symbol, start, end, artifact_tag)
    path = results_dir / f"{prefix}_{table_name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    return pd.read_csv(path)


def load_all_aggregate_results(
    results_dir: Path,
    symbol: str,
    start: str,
    end: str,
    model_names: list[str] | None = None,
    artifact_tag: str | None = None,
) -> pd.DataFrame:
    """
    Loads aggregate metric CSVs for all models.
    """
    if model_names is None:
        model_names = list(MODEL_PREFIXES.keys())

    frames = []

    for model_name in model_names:
        frame = read_model_result_table(
            results_dir=results_dir,
            model_name=model_name,
            symbol=symbol,
            start=start,
            end=end,
            table_name="aggregate",
            artifact_tag=artifact_tag,
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def load_all_nonzero_results(
    results_dir: Path,
    symbol: str,
    start: str,
    end: str,
    model_names: list[str] | None = None,
    artifact_tag: str | None = None,
) -> pd.DataFrame:
    """
    Loads non-zero subset metric CSVs for all models.
    """
    if model_names is None:
        model_names = list(MODEL_PREFIXES.keys())

    frames = []

    for model_name in model_names:
        frame = read_model_result_table(
            results_dir=results_dir,
            model_name=model_name,
            symbol=symbol,
            start=start,
            end=end,
            table_name="nonzero_subset",
            artifact_tag=artifact_tag,
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def validation_comparison_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    """
    Returns validation-only comparison table for the primary ternary task.
    """
    required = [
        "model",
        "split",
        "horizon",
        "n_obs",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "pred_down_fraction",
        "pred_unchanged_fraction",
        "pred_up_fraction",
        "true_down_fraction",
        "true_unchanged_fraction",
        "true_up_fraction",
    ]

    missing = sorted(set(required) - set(aggregate.columns))
    if missing:
        raise ValueError(f"aggregate results missing columns: {missing}")

    out = (
        aggregate.loc[aggregate["split"] == "validation", required]
        .sort_values(["horizon", "model"])
        .reset_index(drop=True)
    )

    return out


def train_validation_comparison_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    """
    Returns train + validation comparison table.
    """
    required = [
        "model",
        "split",
        "horizon",
        "n_obs",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "pred_down_fraction",
        "pred_unchanged_fraction",
        "pred_up_fraction",
        "true_down_fraction",
        "true_unchanged_fraction",
        "true_up_fraction",
    ]
    missing = sorted(set(required) - set(aggregate.columns))
    if missing:
        raise ValueError(f"aggregate results missing columns: {missing}")

    out = (
        aggregate[required]
        .sort_values(["horizon", "model", "split"])
        .reset_index(drop=True)
    )

    return out


def validation_nonzero_comparison_table(nonzero: pd.DataFrame) -> pd.DataFrame:
    """
    Returns validation-only comparison for the secondary non-zero subset.
    """
    required = [
        "model",
        "split",
        "horizon",
        "n_nonzero_obs",
        "nonzero_accuracy",
        "nonzero_macro_f1",
        "nonzero_balanced_accuracy",
        "predicted_zero_on_nonzero_fraction",
    ]

    missing = sorted(set(required) - set(nonzero.columns))
    if missing:
        raise ValueError(f"nonzero results missing columns: {missing}")

    out = (
        nonzero.loc[nonzero["split"] == "validation", required]
        .sort_values(["horizon", "model"])
        .reset_index(drop=True)
    )

    return out


def validation_delta_vs_baselines(validation_comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Computes full_logistic validation deltas versus both baselines.

    Deltas are computed within each horizon.
    """
    rows = []

    for h, g in validation_comparison.groupby("horizon", sort=True):
        by_model = g.set_index("model")

        if "full_logistic" not in by_model.index:
            raise ValueError(f"Missing full_logistic for horizon {h}")

        full = by_model.loc["full_logistic"]

        for baseline in ["majority_baseline", "queue_imbalance_logistic"]:
            if baseline not in by_model.index:
                raise ValueError(f"Missing baseline {baseline} for horizon {h}")

            base = by_model.loc[baseline]

            row = {
                "horizon": h,
                "baseline": baseline,
                "model": "full_logistic",
            }

            for metric in PRIMARY_METRICS:
                row[f"{metric}_baseline"] = float(base[metric])
                row[f"{metric}_full"] = float(full[metric])
                row[f"{metric}_delta"] = float(full[metric] - base[metric])

            rows.append(row)

    return pd.DataFrame(rows)


def signal_decay_table(validation_comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a simple signal-decay table by model across horizons.

    This reports metrics at h=10, h=20, h=50 and changes from h=10 to h=50.
    """
    rows = []

    for model_name, g in validation_comparison.groupby("model", sort=True):
        by_horizon = g.set_index("horizon")

        row = {"model": model_name}

        for h in [10, 20, 50]:
            if h not in by_horizon.index:
                raise ValueError(f"Missing horizon {h} for model {model_name}")

            for metric in PRIMARY_METRICS:
                row[f"{metric}_h{h}"] = float(by_horizon.loc[h, metric])

        for metric in PRIMARY_METRICS:
            row[f"{metric}_h50_minus_h10"] = (
                row[f"{metric}_h50"] - row[f"{metric}_h10"]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def validation_ranking_table(validation_comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks models within each horizon by macro F1 and balanced accuracy.
    """
    out = validation_comparison.copy()

    out["rank_macro_f1"] = (
        out.groupby("horizon")["macro_f1"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    out["rank_balanced_accuracy"] = (
        out.groupby("horizon")["balanced_accuracy"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    out = out.sort_values(
        ["horizon", "rank_macro_f1", "rank_balanced_accuracy", "model"]
    ).reset_index(drop=True)

    return out


def save_comparison_table(table: pd.DataFrame, output_path: Path) -> None:
    """
    Saves a comparison table as CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
