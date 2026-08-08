from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    evaluate_nonzero_subset,
    evaluate_ternary_predictions,
)


def main() -> None:
    y_true = pd.Series([-1, -1, 0, 0, 1, 1, 1])
    y_pred = pd.Series([-1, 0, 0, 1, 1, -1, 1])

    result = evaluate_ternary_predictions(
        y_true=y_true,
        y_pred=y_pred,
        split="validation",
        horizon=10,
        model_name="smoke_test",
    )

    nonzero = evaluate_nonzero_subset(
        y_true=y_true,
        y_pred=y_pred,
        split="validation",
        horizon=10,
        model_name="smoke_test",
    )
    aggregate = result["aggregate"].iloc[0]
    assert np.isclose(aggregate["accuracy"], 4 / 7)
    assert np.isclose(aggregate["macro_f1"], 5 / 9)
    assert np.isclose(aggregate["balanced_accuracy"], 5 / 9)
    assert int(result["confusion_counts"]["value"].sum()) == len(y_true)
    assert int(nonzero.loc[0, "n_nonzero_obs"]) == 5
    assert np.isclose(nonzero.loc[0, "nonzero_accuracy"], 0.6)

    print("=" * 80)
    print("AGGREGATE")
    print("=" * 80)
    print(result["aggregate"].to_string(index=False))

    print()
    print("=" * 80)
    print("PER CLASS")
    print("=" * 80)
    print(result["per_class"].to_string(index=False))

    print()
    print("=" * 80)
    print("CONFUSION COUNTS")
    print("=" * 80)
    print(result["confusion_counts"].to_string(index=False))

    print()
    print("=" * 80)
    print("NON-ZERO SUBSET")
    print("=" * 80)
    print(nonzero.to_string(index=False))


if __name__ == "__main__":
    main()
