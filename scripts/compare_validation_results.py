from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.model_comparison import (  # noqa: E402
    load_all_aggregate_results,
    load_all_nonzero_results,
    save_comparison_table,
    signal_decay_table,
    train_validation_comparison_table,
    validation_comparison_table,
    validation_delta_vs_baselines,
    validation_nonzero_comparison_table,
    validation_ranking_table,
)
from src.data_loader import parse_date  # noqa: E402
from src.artifact_naming import tagged_artifact_stem  # noqa: E402
from src.protocol import validate_protocol_symbol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare validation results across baseline and full-feature models."
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
            "'v3_fixed_window_features'."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    validate_protocol_symbol(args.symbol)
    
    start_str = parse_date(args.start).strftime("%Y-%m-%d")
    end_str = parse_date(args.end).strftime("%Y-%m-%d")

    results_dir = PROJECT_ROOT / "outputs" / "results"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"

    print("=" * 80)
    print("STEP 13: VALIDATION COMPARISON + SIGNAL DECAY")
    print("=" * 80)

    print()
    print("Loading aggregate results...")
    aggregate = load_all_aggregate_results(
        results_dir=results_dir,
        symbol=args.symbol,
        start=start_str,
        end=end_str,
        artifact_tag=args.artifact_tag,
    )

    print("Loading non-zero subset results...")
    nonzero = load_all_nonzero_results(
        results_dir=results_dir,
        symbol=args.symbol,
        start=start_str,
        end=end_str,
        artifact_tag=args.artifact_tag,
    )

    validation = validation_comparison_table(aggregate)
    train_validation = train_validation_comparison_table(aggregate)
    nonzero_validation = validation_nonzero_comparison_table(nonzero)
    deltas = validation_delta_vs_baselines(validation)
    decay = signal_decay_table(validation)
    ranking = validation_ranking_table(validation)

    prefix = tagged_artifact_stem(
        "validation_comparison",
        args.symbol,
        start_str,
        end_str,
        args.artifact_tag,
    )

    print()
    print("Validation comparison:")
    print(validation.to_string(index=False))

    print()
    print("Validation ranking:")
    print(ranking.to_string(index=False))

    print()
    print("Full logistic deltas versus baselines:")
    print(deltas.to_string(index=False))

    print()
    print("Signal decay table:")
    print(decay.to_string(index=False))

    print()
    print("Validation non-zero subset comparison:")
    print(nonzero_validation.to_string(index=False))

    print()
    save_comparison_table(
        train_validation,
        reports_dir / f"{prefix}_train_validation.csv",
    )

    save_comparison_table(
        validation,
        reports_dir / f"{prefix}_validation_only.csv",
    )

    save_comparison_table(
        ranking,
        reports_dir / f"{prefix}_ranking.csv",
    )

    save_comparison_table(
        deltas,
        reports_dir / f"{prefix}_full_vs_baselines_deltas.csv",
    )

    save_comparison_table(
        decay,
        reports_dir / f"{prefix}_signal_decay.csv",
    )

    save_comparison_table(
        nonzero_validation,
        reports_dir / f"{prefix}_nonzero_validation.csv",
    )

    print()
    print("Done. Validation comparison report created.")


if __name__ == "__main__":
    main()
