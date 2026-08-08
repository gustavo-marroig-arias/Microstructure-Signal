from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_naming import tagged_artifact_stem  # noqa: E402
from src.data_loader import parse_date  # noqa: E402
from src.protocol import DEFAULT_HORIZONS, validate_protocol_symbol  # noqa: E402


HORIZONS = DEFAULT_HORIZONS
MODEL_LABELS = {
    "majority_baseline": "Majority",
    "queue_imbalance_logistic": "Queue only",
    "full_logistic": "Full argmax",
    "full_logistic_thresholded": "Full thresholded",
}
MODEL_STYLES = {
    "majority_baseline": {
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "o",
        "zorder": 2,
    },
    "queue_imbalance_logistic": {
        "color": "#ff7f0e",
        "linestyle": "--",
        "marker": "x",
        "zorder": 3,
    },
    "full_logistic": {
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "s",
        "zorder": 2,
    },
    "full_logistic_thresholded": {
        "color": "#d62728",
        "linestyle": "-",
        "marker": "o",
        "zorder": 2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create compact plots from frozen result CSVs."
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help="Optional filename tag, e.g. 'v3_fixed_window_features'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional plot output directory. Default: outputs/reports/plots.",
    )
    parser.add_argument(
        "--include-regime-plot",
        action="store_true",
        help=(
            "Generate the regime plot after provenance-validated regime tables "
            "have been produced from complete frozen artifacts."
        ),
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def plot_target_drift(target_diag: pd.DataFrame, output_path: Path) -> None:
    data = target_diag.loc[target_diag["eligible_only"]].copy()
    split_order = ["train", "validation", "test"]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    for split in split_order:
        split_df = data.loc[data["split"] == split].sort_values("horizon")
        ax.plot(
            split_df["horizon"],
            split_df["nonzero_fraction"],
            marker="o",
            linewidth=2.0,
            label=split,
        )

    ax.set_title("Target Drift: Nonzero Label Fraction")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Nonzero fraction")
    ax.set_xticks(HORIZONS)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_path)


def plot_horizon_performance(aggregate: pd.DataFrame, output_path: Path) -> None:
    data = aggregate.loc[aggregate["split"] == "test"].copy()
    model_order = [
        "majority_baseline",
        "queue_imbalance_logistic",
        "full_logistic",
        "full_logistic_thresholded",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)

    for ax, metric, title in [
        (axes[0], "macro_f1", "Macro F1"),
        (axes[1], "balanced_accuracy", "Balanced Accuracy"),
    ]:
        for model in model_order:
            model_df = data.loc[data["model"] == model].sort_values("horizon")
            ax.plot(
                model_df["horizon"],
                model_df[metric],
                linewidth=2.0,
                label=MODEL_LABELS[model],
                **MODEL_STYLES[model],
            )

        ax.set_title(title)
        ax.set_xlabel("Horizon")
        ax.set_xticks(HORIZONS)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Final-Test Performance by Horizon", y=1.03)
    save_figure(fig, output_path)


def plot_thresholded_vs_argmax_recall(per_class: pd.DataFrame, output_path: Path) -> None:
    data = per_class.loc[
        (per_class["split"] == "test")
        & (per_class["model"].isin(["full_logistic", "full_logistic_thresholded"]))
        & (per_class["class_label"].isin([-1, 1]))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    for ax, class_label, class_name in [
        (axes[0], -1, "Down Recall"),
        (axes[1], 1, "Up Recall"),
    ]:
        class_df = data.loc[data["class_label"] == class_label]
        for model in ["full_logistic", "full_logistic_thresholded"]:
            model_df = class_df.loc[class_df["model"] == model].sort_values("horizon")
            ax.plot(
                model_df["horizon"],
                model_df["recall"],
                linewidth=2.0,
                label=MODEL_LABELS[model],
                **MODEL_STYLES[model],
            )

        ax.set_title(class_name)
        ax.set_xlabel("Horizon")
        ax.set_xticks(HORIZONS)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Recall")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Thresholded vs Argmax Directional Recall", y=1.03)
    save_figure(fig, output_path)


def plot_regime_performance(regime: pd.DataFrame, output_path: Path) -> None:
    required_provenance = {
        "dataset_sha256",
        "experiment_fingerprint",
        "model_created_utc",
        "model_parameter_fingerprint",
    }
    missing = sorted(required_provenance - set(regime.columns))
    if missing:
        raise ValueError(
            "Regime table lacks required frozen-artifact provenance: "
            f"{missing}"
        )
    data = regime.loc[
        (regime["split"] == "test")
        & (regime["model"] == "full_logistic_thresholded")
    ].copy()

    regime_variables = ["spread", "realized_volatility", "trade_intensity"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)

    for ax, regime_variable in zip(axes, regime_variables, strict=True):
        subset = data.loc[data["regime_variable"] == regime_variable]
        for regime_name, regime_df in subset.groupby("regime", sort=True):
            regime_df = regime_df.sort_values("horizon")
            ax.plot(
                regime_df["horizon"],
                regime_df["balanced_accuracy"],
                marker="o",
                linewidth=2.0,
                label=regime_name.replace("_", " "),
            )

        ax.set_title(regime_variable.replace("_", " ").title())
        ax.set_xlabel("Horizon")
        ax.set_xticks(HORIZONS)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("Balanced accuracy")
    fig.suptitle(
        "Final-Test Regime Performance: Thresholded Full Model\n"
        "Cutoffs fixed at training-set medians",
        y=1.08,
    )
    save_figure(fig, output_path)


def plot_nonzero_performance(nonzero: pd.DataFrame, output_path: Path) -> None:
    data = nonzero.loc[nonzero["split"] == "test"].copy()
    model_order = [
        "majority_baseline",
        "queue_imbalance_logistic",
        "full_logistic",
        "full_logistic_thresholded",
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for model in model_order:
        model_df = data.loc[data["model"] == model].sort_values("horizon")
        ax.plot(
            model_df["horizon"],
            model_df["nonzero_balanced_accuracy"],
            linewidth=2.0,
            label=MODEL_LABELS[model],
            **MODEL_STYLES[model],
        )

    ax.set_title("Nonzero Subset Performance")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Nonzero balanced accuracy")
    ax.set_xticks(HORIZONS)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_path)


def main() -> None:
    args = parse_args()

    validate_protocol_symbol(args.symbol)

    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")

    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    results_dir = PROJECT_ROOT / "outputs" / "results"
    plot_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else reports_dir / "plots"
    )

    target_prefix = tagged_artifact_stem(
        "target_diagnostics",
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
    plot_prefix = tagged_artifact_stem(
        "plot",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )

    target_diag = pd.read_csv(require_file(reports_dir / f"{target_prefix}.csv"))
    aggregate = pd.read_csv(require_file(results_dir / f"{final_prefix}_aggregate.csv"))
    per_class = pd.read_csv(require_file(results_dir / f"{final_prefix}_per_class.csv"))
    nonzero = pd.read_csv(require_file(results_dir / f"{final_prefix}_nonzero_subset.csv"))

    generated = [
        (
            "target_drift",
            plot_dir / f"{plot_prefix}_target_drift.png",
            lambda path: plot_target_drift(target_diag, path),
        ),
        (
            "horizon_performance",
            plot_dir / f"{plot_prefix}_horizon_performance.png",
            lambda path: plot_horizon_performance(aggregate, path),
        ),
        (
            "thresholded_vs_argmax_recall",
            plot_dir / f"{plot_prefix}_thresholded_vs_argmax_recall.png",
            lambda path: plot_thresholded_vs_argmax_recall(per_class, path),
        ),
        (
            "nonzero_performance",
            plot_dir / f"{plot_prefix}_nonzero_performance.png",
            lambda path: plot_nonzero_performance(nonzero, path),
        ),
    ]
    if args.include_regime_plot:
        regime = pd.read_csv(
            require_file(results_dir / f"{final_prefix}_regime_aggregate.csv")
        )
        generated.append(
            (
                "regime_performance",
                plot_dir / f"{plot_prefix}_regime_performance.png",
                lambda path: plot_regime_performance(regime, path),
            )
        )

    manifest_rows = []
    for plot_name, path, plotter in generated:
        plotter(path)
        manifest_rows.append(
            {
                "plot": plot_name,
                "path": str(path.relative_to(PROJECT_ROOT)),
            }
        )

    manifest_path = plot_dir / f"{plot_prefix}_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved plot manifest: {manifest_path}")


if __name__ == "__main__":
    main()
