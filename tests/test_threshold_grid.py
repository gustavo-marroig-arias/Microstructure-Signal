import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    aggregate_metric_row,
    evaluate_nonzero_subset,
)
from src.evaluation.probability_diagnostics import (
    frozen_threshold_mapping,
    threshold_grid_results,
    threshold_predict,
)


def test_optimized_threshold_grid_matches_brute_force() -> None:
    rng = np.random.default_rng(91)
    y_true = pd.Series(rng.choice([-1, 0, 1], size=2_003))
    probabilities = pd.DataFrame(
        rng.dirichlet([1.3, 2.4, 1.8], size=len(y_true)),
        columns=["proba_-1", "proba_0", "proba_1"],
    )
    grid = (0.10, 0.25, 0.40, 0.50)

    fast_aggregate, fast_nonzero = threshold_grid_results(
        y_true,
        probabilities,
        50,
        "validation",
        "model",
        grid,
    )
    slow_aggregate = []
    slow_nonzero = []
    for threshold_down in grid:
        for threshold_up in grid:
            prediction = threshold_predict(
                probabilities,
                threshold_down,
                threshold_up,
            )
            aggregate = aggregate_metric_row(
                y_true,
                prediction,
                "validation",
                50,
                "model",
            )
            aggregate.update(
                threshold_down=threshold_down,
                threshold_up=threshold_up,
            )
            slow_aggregate.append(aggregate)

            nonzero = evaluate_nonzero_subset(
                y_true,
                prediction,
                "validation",
                50,
                "model",
            ).iloc[0].to_dict()
            nonzero.update(
                threshold_down=threshold_down,
                threshold_up=threshold_up,
            )
            slow_nonzero.append(nonzero)

    pd.testing.assert_frame_equal(
        fast_aggregate,
        pd.DataFrame(slow_aggregate),
        check_exact=False,
        rtol=1e-14,
        atol=1e-14,
    )
    pd.testing.assert_frame_equal(
        fast_nonzero,
        pd.DataFrame(slow_nonzero),
        check_exact=False,
        rtol=1e-14,
        atol=1e-14,
    )


def test_frozen_thresholds_are_bound_to_exact_model_and_dataset() -> None:
    table = pd.DataFrame(
        {
            "horizon": [10, 20, 50],
            "selection_metric": ["macro_f1"] * 3,
            "threshold_down": [0.2, 0.3, 0.4],
            "threshold_up": [0.25, 0.35, 0.45],
            "dataset_sha256": ["dataset"] * 3,
            "experiment_fingerprint": ["protocol"] * 3,
            "model_created_utc": ["m10", "m20", "m50"],
            "model_parameter_fingerprint": ["p10", "p20", "p50"],
        }
    )

    mapping = frozen_threshold_mapping(
        table,
        selection_metric="macro_f1",
        horizons=(10, 20, 50),
        dataset_sha256="dataset",
        experiment_fingerprint="protocol",
        model_created_utc={10: "m10", 20: "m20", 50: "m50"},
        model_parameter_fingerprint={10: "p10", 20: "p20", 50: "p50"},
    )
    assert mapping == {10: (0.2, 0.25), 20: (0.3, 0.35), 50: (0.4, 0.45)}

    table.loc[table["horizon"] == 20, "model_created_utc"] = "stale"
    with np.testing.assert_raises_regex(
        ValueError,
        "model identity mismatch",
    ):
        frozen_threshold_mapping(
            table,
            selection_metric="macro_f1",
            horizons=(10, 20, 50),
            dataset_sha256="dataset",
            experiment_fingerprint="protocol",
            model_created_utc={10: "m10", 20: "m20", 50: "m50"},
            model_parameter_fingerprint={10: "p10", 20: "p20", 50: "p50"},
        )

    table.loc[table["horizon"] == 20, "model_created_utc"] = "m20"
    table.loc[
        table["horizon"] == 20,
        "model_parameter_fingerprint",
    ] = "stale"
    with np.testing.assert_raises_regex(
        ValueError,
        "parameter fingerprint mismatch",
    ):
        frozen_threshold_mapping(
            table,
            selection_metric="macro_f1",
            horizons=(10, 20, 50),
            dataset_sha256="dataset",
            experiment_fingerprint="protocol",
            model_created_utc={10: "m10", 20: "m20", 50: "m50"},
            model_parameter_fingerprint={10: "p10", 20: "p20", 50: "p50"},
        )
