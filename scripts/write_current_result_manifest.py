from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_naming import tagged_artifact_stem  # noqa: E402
from src.data_loader import parse_date  # noqa: E402
from src.model_dataset_io import resolve_model_dataset  # noqa: E402
from src.protocol import validate_protocol_symbol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a compact manifest for the current canonical result artifacts."
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
        "--output-prefix",
        default="current_result_manifest",
        help="Output basename under outputs/reports.",
    )
    parser.add_argument(
        "--include-regime-artifacts",
        action="store_true",
        help=(
            "Include regime tables/plot only after they have been regenerated "
            "from complete frozen model artifacts."
        ),
    )
    return parser.parse_args()


def artifact_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    if path.is_dir():
        size_bytes = sum(
            child.stat().st_size
            for child in path.rglob("*")
            if child.is_file()
        )
    else:
        size_bytes = path.stat().st_size
    return size_bytes / (1024 * 1024)


def add_file(
    rows: list[dict],
    *,
    path: Path,
    category: str,
    description: str,
    included: bool,
    note: str,
) -> None:
    rows.append(
        {
            "category": category,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": path.exists(),
            "size_mb": artifact_size_mb(path),
            "included": included,
            "description": description,
            "note": note,
        }
    )


def load_audit_summary(audit_path: Path) -> dict:
    if not audit_path.exists():
        return {}
    with audit_path.open() as f:
        return json.load(f)


def validate_regime_artifacts(regime_path: Path) -> None:
    if not regime_path.exists():
        raise FileNotFoundError(
            f"Missing regime aggregate table: {regime_path}"
        )
    columns = set(pd.read_csv(regime_path, nrows=1).columns)
    required = {
        "dataset_sha256",
        "experiment_fingerprint",
        "model_created_utc",
        "model_parameter_fingerprint",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(
            "Regime artifacts cannot be included without required "
            f"frozen-artifact provenance columns: {missing}"
        )


def markdown_manifest(
    *,
    symbol: str,
    start: str,
    end: str,
    artifact_tag: str,
    audit: dict,
    rows: list[dict],
) -> str:
    audit_lines = [
        f"- `critical_audit_passed`: `{audit.get('critical_audit_passed', 'missing')}`",
        f"- `log_feature_precision_status`: `{audit.get('log_feature_precision_status', 'missing')}`",
        f"- `mid_return_5_all_match_float64`: `{audit.get('mid_return_5_all_match_float64', 'missing')}`",
        f"- `realized_vol_20_all_match_float64`: `{audit.get('realized_vol_20_all_match_float64', 'missing')}`",
        f"- `main_result_status`: `{audit.get('main_result_status', 'missing')}`",
    ]

    table_lines = [
        "| Category | Path | Exists | Size MB | Included? | Note |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        size = "" if row["size_mb"] is None else f"{row['size_mb']:.2f}"
        included = "yes" if row["included"] else "no"
        table_lines.append(
            "| {category} | `{path}` | {exists} | {size} | {included} | {note} |".format(
                category=row["category"],
                path=row["path"],
                exists=row["exists"],
                size=size,
                included=included,
                note=row["note"],
            )
        )

    return "\n".join(
        [
            "# Current Result Manifest",
            "",
            f"- symbol: `{symbol}`",
            f"- sample: `{start}` to `{end}` UTC",
            f"- artifact tag: `{artifact_tag}`",
            "- status: frozen 7-day checkpoint",
            "",
            "## Audit Status",
            "",
            *audit_lines,
            "",
            "## File Manifest",
            "",
            "`Exists` records the generation machine at manifest creation; only rows",
            "marked `yes` under `Included?` belong to the public snapshot.",
            "",
            *table_lines,
            "",
            "## Artifact Policy",
            "",
            "- Raw, interim, processed, and row-level parquet files are excluded.",
            "- Included artifacts are code, documentation, small CSV/JSON summaries, and selected plots.",
            "- API keys, environment-specific files, caches, and logs are excluded.",
            "- A 14-day replication should use a new artifact tag.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()

    validate_protocol_symbol(args.symbol)

    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")
    artifact_tag = args.artifact_tag or ""

    data_dir = PROJECT_ROOT / "data"
    processed_dir = data_dir / "processed"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    results_dir = PROJECT_ROOT / "outputs" / "results"

    feature_stem = tagged_artifact_stem(
        "feature_table",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    dataset_stem = tagged_artifact_stem(
        "model_dataset",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    model_dataset_path = resolve_model_dataset(
        processed_dir,
        dataset_stem,
    ).path
    final_prefix = tagged_artifact_stem(
        "final_test",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )
    compact_prefix = tagged_artifact_stem(
        "compact_test_predictions",
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
    audit_dir = reports_dir / "research_audit" / (
        f"{args.symbol}_{start_str}_to_{end_str}_{artifact_tag}"
        if artifact_tag
        else f"{args.symbol}_{start_str}_to_{end_str}"
    )
    audit_path = audit_dir / "audit_summary.json"
    if args.include_regime_artifacts:
        validate_regime_artifacts(
            results_dir / f"{final_prefix}_regime_aggregate.csv"
        )

    rows: list[dict] = []
    add_file(
        rows,
        path=processed_dir / f"{feature_stem}.parquet",
        category="data",
        description="Feature table with labels before split metadata.",
        included=False,
        note="excluded large artifact",
    )
    add_file(
        rows,
        path=model_dataset_path,
        category="data",
        description="Model dataset with chronological split metadata.",
        included=False,
        note="excluded large artifact",
    )

    for suffix, description in [
        ("aggregate.csv", "Final-test aggregate metrics."),
        ("class_proportions.csv", "Final-test true and predicted class proportions."),
        ("confusion_counts.csv", "Final-test confusion counts."),
        (
            "confusion_true_normalized.csv",
            "Final-test row-normalized confusion matrices.",
        ),
        ("daily_blocks.csv", "Daily final-test stability metrics."),
        ("logistic_coefficients.csv", "Frozen full-model coefficients."),
        ("majority_fit_summary.csv", "Train-fitted majority classes."),
        ("nonzero_subset.csv", "Final-test nonzero subset metrics."),
        ("per_class.csv", "Final-test per-class metrics."),
    ]:
        add_file(
            rows,
            path=results_dir / f"{final_prefix}_{suffix}",
            category="small_result",
            description=description,
            included=True,
            note="small CSV",
        )
    for suffix, description in [
        ("regime_aggregate.csv", "Train-median regime aggregate metrics."),
        ("regime_nonzero_subset.csv", "Train-median regime nonzero metrics."),
    ]:
        add_file(
            rows,
            path=results_dir / f"{final_prefix}_{suffix}",
            category="small_result",
            description=description,
            included=args.include_regime_artifacts,
            note=(
                "small CSV"
                if args.include_regime_artifacts
                else "excluded unless frozen-model provenance is available"
            ),
        )

    for suffix, description in [
        ("frozen_thresholds.csv", "Frozen validation-selected thresholds."),
        ("full_vs_baselines_deltas.csv", "Final-test deltas versus baselines."),
        ("ranking.csv", "Final-test model ranking."),
        ("signal_decay.csv", "Final-test performance by prediction horizon."),
        ("thresholded_deltas.csv", "Thresholded model deltas."),
    ]:
        add_file(
            rows,
            path=reports_dir / f"{final_prefix}_{suffix}",
            category="small_report",
            description=description,
            included=True,
            note="small CSV",
        )
    add_file(
        rows,
        path=reports_dir / f"{final_prefix}_regime_thresholds.csv",
        category="small_report",
        description="Train-median regime cutoffs.",
        included=args.include_regime_artifacts,
        note=(
            "small CSV"
            if args.include_regime_artifacts
            else "excluded unless frozen-model provenance is available"
        ),
    )

    add_file(
        rows,
        path=results_dir / f"{compact_prefix}.parquet",
        category="diagnostic_data",
        description="Compact test-only probabilities and hard predictions.",
        included=False,
        note="excluded row-level parquet",
    )
    add_file(
        rows,
        path=reports_dir / f"{compact_prefix}_summary.csv",
        category="small_report",
        description="Summary of compact test prediction file.",
        included=True,
        note="small CSV",
    )
    for prefix, description in [
        ("feature_summary", "Feature completeness and distribution summary."),
        ("label_distribution", "Full-sample label distribution."),
        ("split_summary", "Chronological split boundaries and row counts."),
        ("target_diagnostics", "Split-specific target distributions."),
    ]:
        report_stem = tagged_artifact_stem(
            prefix,
            args.symbol,
            start_str,
            end_str,
            args.artifact_tag,
        )
        add_file(
            rows,
            path=reports_dir / f"{report_stem}.csv",
            category="small_report",
            description=description,
            included=True,
            note="small CSV",
        )
    add_file(
        rows,
        path=audit_path,
        category="audit",
        description="Machine-readable audit summary.",
        included=True,
        note="small JSON",
    )
    add_file(
        rows,
        path=audit_dir / "final_audit_conclusion.csv",
        category="audit",
        description="One-row audit conclusion.",
        included=True,
        note="small CSV",
    )

    for plot_name in [
        "target_drift",
        "horizon_performance",
        "thresholded_vs_argmax_recall",
        "nonzero_performance",
    ]:
        add_file(
            rows,
            path=reports_dir / "plots" / f"{plot_prefix}_{plot_name}.png",
            category="plot",
            description=f"{plot_name.replace('_', ' ').title()} plot.",
            included=True,
            note="small PNG",
        )
    add_file(
        rows,
        path=reports_dir / "plots" / f"{plot_prefix}_regime_performance.png",
        category="plot",
        description="Regime Performance plot.",
        included=args.include_regime_artifacts,
        note=(
            "small PNG"
            if args.include_regime_artifacts
            else "excluded unless required regime provenance is available"
        ),
    )
    add_file(
        rows,
        path=reports_dir / "plots" / f"{plot_prefix}_manifest.csv",
        category="plot",
        description="Plot source and output manifest.",
        included=True,
        note="small CSV",
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"{args.output_prefix}.csv"
    md_path = reports_dir / f"{args.output_prefix}.md"

    table = pd.DataFrame(rows)
    table.to_csv(csv_path, index=False)

    audit = load_audit_summary(audit_path)
    md_path.write_text(
        markdown_manifest(
            symbol=args.symbol,
            start=start_str,
            end=end_str,
            artifact_tag=artifact_tag,
            audit=audit,
            rows=rows,
        )
    )

    print(f"Saved manifest CSV: {csv_path}")
    print(f"Saved manifest markdown: {md_path}")


if __name__ == "__main__":
    main()
