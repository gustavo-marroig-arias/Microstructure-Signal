from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_naming import (  # noqa: E402
    frozen_model_artifact_stem,
    tagged_artifact_stem,
)
from src.data_loader import parse_date  # noqa: E402
from src.model_dataset_io import (  # noqa: E402
    iter_model_split_batches,
    model_dataset_sha256,
    resolve_model_dataset,
)
from src.modeling.model_artifacts import (  # noqa: E402
    load_frozen_logistic_artifact,
)
from src.protocol import (  # noqa: E402
    DEFAULT_HORIZONS,
    ExperimentSpec,
    FEATURE_COLUMNS,
    validate_protocol_symbol,
)


LABELS = (-1, 0, 1)
MODEL_FEATURES = tuple(FEATURE_COLUMNS)
FINAL_MODELS = (
    "majority_baseline",
    "queue_imbalance_logistic",
    "full_logistic",
    "full_logistic_thresholded",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently reconcile one completed integration run across "
            "features, splits, models, thresholds, regimes, and compact outputs."
        )
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--artifact-tag", required=True)
    parser.add_argument(
        "--threshold-selection-metric",
        default="macro_f1",
        choices=["macro_f1", "balanced_accuracy"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000_000,
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_close(
    actual: float,
    expected: float,
    name: str,
    *,
    atol: float = 1e-12,
) -> float:
    error = abs(float(actual) - float(expected))
    require(error <= atol, f"{name}: error {error} exceeds {atol}")
    return error


def confusion_matrix_from_rows(rows: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for row in rows.itertuples(index=False):
        true_index = LABELS.index(int(row.true_label))
        pred_index = LABELS.index(int(row.pred_label))
        matrix[true_index, pred_index] = int(row.value)
    require(
        len(rows) == 9 and bool((matrix >= 0).all()),
        "Confusion table must contain one nonnegative 3x3 matrix.",
    )
    return matrix


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, object]:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = safe_ratio(true_positive, predicted)
    recall = safe_ratio(true_positive, support)
    f1 = safe_ratio(2.0 * precision * recall, precision + recall)
    n_obs = int(matrix.sum())
    return {
        "n_obs": n_obs,
        "accuracy": float(true_positive.sum() / n_obs),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "pred_fractions": predicted / n_obs,
        "true_fractions": support / n_obs,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def nonzero_metrics_from_confusion(matrix: np.ndarray) -> dict[str, float | int]:
    subset = matrix[[0, 2], :]
    n_obs = int(subset.sum())
    true_positive = np.array(
        [subset[0, 0], 0, subset[1, 2]],
        dtype=np.float64,
    )
    support = np.array(
        [subset[0].sum(), 0, subset[1].sum()],
        dtype=np.float64,
    )
    predicted = subset.sum(axis=0).astype(np.float64)
    precision = safe_ratio(true_positive, predicted)
    recall = safe_ratio(true_positive, support)
    f1 = safe_ratio(2.0 * precision * recall, precision + recall)
    directional_recall = recall[[0, 2]]
    return {
        "n_nonzero_obs": n_obs,
        "nonzero_accuracy": float(true_positive.sum() / n_obs),
        "nonzero_macro_f1": float(f1.mean()),
        "nonzero_balanced_accuracy": float(directional_recall.mean()),
        "predicted_zero_on_nonzero_fraction": float(predicted[1] / n_obs),
    }


def reconcile_final_metrics(
    aggregate: pd.DataFrame,
    per_class: pd.DataFrame,
    nonzero: pd.DataFrame,
    confusion: pd.DataFrame,
) -> float:
    max_error = 0.0
    for model in FINAL_MODELS:
        for horizon in DEFAULT_HORIZONS:
            key = (aggregate["model"] == model) & (
                aggregate["horizon"] == horizon
            )
            aggregate_row = aggregate.loc[key].squeeze()
            require(
                isinstance(aggregate_row, pd.Series),
                f"Missing aggregate row for {model}, h={horizon}.",
            )
            confusion_rows = confusion.loc[
                (confusion["model"] == model)
                & (confusion["horizon"] == horizon)
                & (confusion["normalize"] == "none")
            ]
            matrix = confusion_matrix_from_rows(confusion_rows)
            rebuilt = metrics_from_confusion(matrix)

            for field in ("n_obs", "accuracy", "macro_f1", "balanced_accuracy"):
                max_error = max(
                    max_error,
                    assert_close(
                        aggregate_row[field],
                        rebuilt[field],
                        f"{model} h={horizon} {field}",
                    ),
                )
            for index, name in enumerate(("down", "unchanged", "up")):
                max_error = max(
                    max_error,
                    assert_close(
                        aggregate_row[f"pred_{name}_fraction"],
                        rebuilt["pred_fractions"][index],
                        f"{model} h={horizon} pred_{name}_fraction",
                    ),
                    assert_close(
                        aggregate_row[f"true_{name}_fraction"],
                        rebuilt["true_fractions"][index],
                        f"{model} h={horizon} true_{name}_fraction",
                    ),
                )

            class_rows = per_class.loc[
                (per_class["model"] == model)
                & (per_class["horizon"] == horizon)
            ].sort_values("class_label")
            require(len(class_rows) == 3, "Expected three per-class rows.")
            for field in ("precision", "recall", "f1", "support"):
                expected = np.asarray(rebuilt[field])
                actual = class_rows[field].to_numpy()
                error = float(np.max(np.abs(actual - expected)))
                require(
                    error <= 1e-12,
                    f"{model} h={horizon} per-class {field}: {error}",
                )
                max_error = max(max_error, error)

            rebuilt_nonzero = nonzero_metrics_from_confusion(matrix)
            nonzero_row = nonzero.loc[
                (nonzero["model"] == model)
                & (nonzero["horizon"] == horizon)
            ].squeeze()
            require(
                isinstance(nonzero_row, pd.Series),
                f"Missing nonzero row for {model}, h={horizon}.",
            )
            for field, expected in rebuilt_nonzero.items():
                max_error = max(
                    max_error,
                    assert_close(
                        nonzero_row[field],
                        expected,
                        f"{model} h={horizon} {field}",
                    ),
                )
    return max_error


def scan_split_invariants(location, batch_size: int) -> dict[str, dict[str, int]]:
    summaries: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        rows = 0
        complete_rows = 0
        boundary_rows = 0
        eligible_rows = 0
        split_max_event_id = -1
        eligible_max_event_id = -1
        for batch in iter_model_split_batches(
            location,
            split,
            (
                "event_id",
                "split",
                "feature_complete",
                "is_boundary_drop",
                "model_eligible",
            ),
            eligible_only=False,
            batch_size=batch_size,
        ):
            feature_complete = batch["feature_complete"].to_numpy(dtype=bool)
            boundary = batch["is_boundary_drop"].to_numpy(dtype=bool)
            eligible = batch["model_eligible"].to_numpy(dtype=bool)
            require(
                bool(np.array_equal(eligible, feature_complete & ~boundary)),
                f"Eligibility formula mismatch in {split}.",
            )
            event_ids = batch["event_id"].to_numpy(dtype=np.int64)
            rows += len(batch)
            complete_rows += int(feature_complete.sum())
            boundary_rows += int(boundary.sum())
            eligible_rows += int(eligible.sum())
            split_max_event_id = max(split_max_event_id, int(event_ids[-1]))
            if bool(eligible.any()):
                eligible_max_event_id = max(
                    eligible_max_event_id,
                    int(event_ids[eligible][-1]),
                )

        expected_boundary = 50 if split in {"train", "validation"} else 0
        require(
            boundary_rows == expected_boundary,
            f"{split} boundary rows {boundary_rows}, expected {expected_boundary}.",
        )
        if split in {"train", "validation"}:
            require(
                eligible_max_event_id + max(DEFAULT_HORIZONS)
                <= split_max_event_id,
                f"{split} maximum-horizon embargo failed.",
            )
        summaries[split] = {
            "rows": rows,
            "feature_complete_rows": complete_rows,
            "boundary_rows": boundary_rows,
            "eligible_rows": eligible_rows,
        }
    return summaries


def compact_confusions(
    path: Path,
    *,
    batch_size: int,
) -> dict[tuple[str, int], np.ndarray]:
    result = {
        (model, horizon): np.zeros((3, 3), dtype=np.int64)
        for model in ("full_logistic", "full_logistic_thresholded")
        for horizon in DEFAULT_HORIZONS
    }
    columns = []
    for horizon in DEFAULT_HORIZONS:
        columns.extend(
            [
                f"y_true_h{horizon}",
                f"y_pred_argmax_h{horizon}",
                f"y_pred_thresholded_h{horizon}",
            ]
        )
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        frame = batch.to_pandas()
        for horizon in DEFAULT_HORIZONS:
            y_true = frame[f"y_true_h{horizon}"].to_numpy(dtype=np.int8)
            for model, suffix in (
                ("full_logistic", "argmax"),
                ("full_logistic_thresholded", "thresholded"),
            ):
                y_pred = frame[f"y_pred_{suffix}_h{horizon}"].to_numpy(
                    dtype=np.int8
                )
                encoded = (y_true + 1) * 3 + (y_pred + 1)
                result[(model, horizon)] += np.bincount(
                    encoded,
                    minlength=9,
                ).reshape(3, 3)
    return result


def main() -> None:
    args = parse_args()
    validate_protocol_symbol(args.symbol)
    require(args.batch_size > 0, "batch-size must be positive.")
    start = parse_date(args.start).strftime("%Y-%m-%d")
    end = parse_date(args.end).strftime("%Y-%m-%d")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "outputs" / "results"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    models_dir = PROJECT_ROOT / "outputs" / "models"

    feature_stem = tagged_artifact_stem(
        "feature_table", args.symbol, start, end, args.artifact_tag
    )
    dataset_stem = tagged_artifact_stem(
        "model_dataset", args.symbol, start, end, args.artifact_tag
    )
    final_stem = tagged_artifact_stem(
        "final_test", args.symbol, start, end, args.artifact_tag
    )
    compact_stem = tagged_artifact_stem(
        "compact_test_predictions",
        args.symbol,
        start,
        end,
        args.artifact_tag,
    )
    probability_stem = tagged_artifact_stem(
        "probability_diagnostics_full_logistic",
        args.symbol,
        start,
        end,
        args.artifact_tag,
    )
    events_path = (
        processed_dir
        / f"quote_events_{args.symbol}_{start}_to_{end}.parquet"
    )
    feature_path = processed_dir / f"{feature_stem}.parquet"
    location = resolve_model_dataset(processed_dir, dataset_stem)

    event_rows = int(pq.ParquetFile(events_path).metadata.num_rows)
    feature_file = pq.ParquetFile(feature_path)
    feature_rows = int(feature_file.metadata.num_rows)
    require(
        event_rows - feature_rows == max(DEFAULT_HORIZONS),
        "Feature row count must equal event rows minus the maximum horizon.",
    )
    schema = feature_file.schema_arrow
    require(schema.field("mid_return_5").type == "double", "mid_return_5 not float64.")
    require(
        schema.field("realized_vol_20").type == "double",
        "realized_vol_20 not float64.",
    )

    split_summary = scan_split_invariants(location, args.batch_size)
    require(
        sum(value["rows"] for value in split_summary.values()) == feature_rows,
        "Split rows do not sum to feature rows.",
    )
    dataset_hash = model_dataset_sha256(location)
    experiment_fingerprint = ExperimentSpec().fingerprint()

    artifacts = {}
    for horizon in DEFAULT_HORIZONS:
        stem = frozen_model_artifact_stem(
            "full_logistic",
            args.symbol,
            start,
            end,
            horizon,
            args.artifact_tag,
        )
        artifact = load_frozen_logistic_artifact(models_dir / stem)
        require(artifact.converged, f"h={horizon} artifact is not converged.")
        require(
            max(artifact.n_iter) < int(artifact.model_config["max_iter"]),
            f"h={horizon} exhausted max_iter.",
        )
        require(
            artifact.dataset_sha256 == dataset_hash,
            f"h={horizon} dataset hash mismatch.",
        )
        require(
            artifact.experiment_fingerprint == experiment_fingerprint,
            f"h={horizon} experiment fingerprint mismatch.",
        )
        require(
            tuple(artifact.feature_names) == MODEL_FEATURES,
            f"h={horizon} feature order mismatch.",
        )
        artifacts[horizon] = artifact

    aggregate = pd.read_csv(results_dir / f"{final_stem}_aggregate.csv")
    per_class = pd.read_csv(results_dir / f"{final_stem}_per_class.csv")
    nonzero = pd.read_csv(results_dir / f"{final_stem}_nonzero_subset.csv")
    confusion = pd.read_csv(results_dir / f"{final_stem}_confusion_counts.csv")
    final_metric_max_error = reconcile_final_metrics(
        aggregate,
        per_class,
        nonzero,
        confusion,
    )

    best = pd.read_csv(
        reports_dir / f"{probability_stem}_best_thresholds.csv"
    )
    frozen = pd.read_csv(
        reports_dir / f"{final_stem}_frozen_thresholds.csv"
    )
    for horizon, artifact in artifacts.items():
        selected = best.loc[
            (best["horizon"] == horizon)
            & (best["selection_metric"] == args.threshold_selection_metric)
        ].squeeze()
        frozen_row = frozen.loc[frozen["horizon"] == horizon].squeeze()
        require(
            isinstance(selected, pd.Series)
            and isinstance(frozen_row, pd.Series),
            f"Threshold row missing for h={horizon}.",
        )
        for field in (
            "threshold_down",
            "threshold_up",
            "dataset_sha256",
            "experiment_fingerprint",
            "model_created_utc",
            "model_parameter_fingerprint",
        ):
            require(
                selected[field] == frozen_row[field],
                f"Frozen threshold provenance mismatch h={horizon}: {field}.",
            )
        require(
            frozen_row["dataset_sha256"] == dataset_hash,
            f"Frozen threshold dataset mismatch h={horizon}.",
        )
        require(
            frozen_row["model_parameter_fingerprint"]
            == artifact.parameter_fingerprint(),
            f"Frozen threshold model mismatch h={horizon}.",
        )

    regime = pd.read_csv(
        results_dir / f"{final_stem}_regime_aggregate.csv"
    )
    regime_thresholds = pd.read_csv(
        reports_dir / f"{final_stem}_regime_thresholds.csv"
    )
    require(
        bool((regime_thresholds["threshold_source"] == "train_median").all()),
        "Regime thresholds are not all train medians.",
    )
    regime_accuracy_max_error = 0.0
    test_n = split_summary["test"]["eligible_rows"]
    for regime_variable in regime["regime_variable"].unique():
        for model in FINAL_MODELS:
            for horizon in DEFAULT_HORIZONS:
                rows = regime.loc[
                    (regime["regime_variable"] == regime_variable)
                    & (regime["model"] == model)
                    & (regime["split"] == "test")
                    & (regime["horizon"] == horizon)
                ]
                require(
                    len(rows) == 2 and int(rows["n_obs"].sum()) == test_n,
                    f"Regime partition mismatch: {regime_variable}, "
                    f"{model}, h={horizon}.",
                )
                rebuilt_accuracy = float(
                    np.average(rows["accuracy"], weights=rows["n_obs"])
                )
                pooled = float(
                    aggregate.loc[
                        (aggregate["model"] == model)
                        & (aggregate["horizon"] == horizon),
                        "accuracy",
                    ].iloc[0]
                )
                regime_accuracy_max_error = max(
                    regime_accuracy_max_error,
                    assert_close(
                        rebuilt_accuracy,
                        pooled,
                        f"Regime accuracy {regime_variable}, {model}, h={horizon}",
                    ),
                )

    compact_path = results_dir / f"{compact_stem}.parquet"
    require(
        int(pq.ParquetFile(compact_path).metadata.num_rows) == test_n,
        "Compact prediction row count does not match eligible test rows.",
    )
    compact_matrices = compact_confusions(
        compact_path,
        batch_size=args.batch_size,
    )
    for (model, horizon), compact_matrix in compact_matrices.items():
        final_matrix = confusion_matrix_from_rows(
            confusion.loc[
                (confusion["model"] == model)
                & (confusion["horizon"] == horizon)
                & (confusion["normalize"] == "none")
            ]
        )
        require(
            bool(np.array_equal(compact_matrix, final_matrix)),
            f"Compact predictions disagree with final confusion: "
            f"{model}, h={horizon}.",
        )

    audit_path = (
        reports_dir
        / "research_audit"
        / f"{args.symbol}_{start}_to_{end}_{args.artifact_tag}"
        / "audit_summary.json"
    )
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    for field in (
        "critical_audit_passed",
        "mid_return_5_all_match_float64",
        "realized_vol_20_all_match_float64",
    ):
        require(bool(audit[field]), f"Research audit failed: {field}.")
    require(
        audit["log_feature_precision_status"] == "pass",
        "Research audit log-feature precision did not pass.",
    )

    print("Integration verification passed.")
    print(f"event_rows: {event_rows:,}")
    print(f"feature_rows: {feature_rows:,}")
    print(f"eligible_test_rows: {test_n:,}")
    print(f"dataset_sha256: {dataset_hash}")
    print(f"final_metric_max_abs_error: {final_metric_max_error:.3e}")
    print(
        "regime_recombined_accuracy_max_abs_error: "
        f"{regime_accuracy_max_error:.3e}"
    )
    print("compact_prediction_confusions_match: True")
    print("research_audit_critical_passed: True")


if __name__ == "__main__":
    main()
