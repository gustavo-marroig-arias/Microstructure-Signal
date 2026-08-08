import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    aggregate_metric_row,
    evaluate_nonzero_subset,
)
from src.evaluation.regime_analysis import (
    REGIME_SPECS,
    evaluate_regime_predictions,
    training_regime_thresholds,
)


def test_regime_metrics_match_canonical_metrics_with_missing_class() -> None:
    data = pd.DataFrame(
        {
            "relative_spread": [1.0, 2.0, 10.0, 11.0],
            "realized_vol_20": [1.0, 2.0, 10.0, 11.0],
            "trade_intensity_1s": [1.0, 2.0, 10.0, 11.0],
            "y_50": [-1, -1, 0, 1],
        }
    )
    predictions = np.array([-1, 0, 0, -1], dtype=np.int8)
    thresholds = training_regime_thresholds(data)

    aggregate, nonzero = evaluate_regime_predictions(
        data=data,
        y_pred=predictions,
        split="test",
        horizon=50,
        model_name="model",
        thresholds=thresholds,
    )

    for spec in REGIME_SPECS:
        threshold = float(
            thresholds.loc[
                thresholds["feature"] == spec.feature,
                "threshold_value",
            ].iloc[0]
        )
        for label, mask in (
            (spec.lower_label, data[spec.feature] <= threshold),
            (spec.upper_label, data[spec.feature] > threshold),
        ):
            expected_aggregate = aggregate_metric_row(
                data.loc[mask, "y_50"],
                predictions[mask],
                "test",
                50,
                "model",
            )
            actual_aggregate = aggregate.loc[
                (aggregate["feature"] == spec.feature)
                & (aggregate["regime"] == label)
            ].iloc[0]
            for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                assert np.isclose(
                    actual_aggregate[metric],
                    expected_aggregate[metric],
                )

            expected_nonzero = evaluate_nonzero_subset(
                data.loc[mask, "y_50"],
                predictions[mask],
                "test",
                50,
                "model",
            ).iloc[0]
            actual_nonzero = nonzero.loc[
                (nonzero["feature"] == spec.feature)
                & (nonzero["regime"] == label)
            ].iloc[0]
            for metric in (
                "nonzero_accuracy",
                "nonzero_macro_f1",
                "nonzero_balanced_accuracy",
            ):
                assert np.isclose(
                    actual_nonzero[metric],
                    expected_nonzero[metric],
                    equal_nan=True,
                )


def test_validation_regimes_use_train_medians_unchanged() -> None:
    train = pd.DataFrame(
        {
            "relative_spread": [1.0, 2.0, 3.0],
            "realized_vol_20": [4.0, 5.0, 6.0],
            "trade_intensity_1s": [7.0, 8.0, 9.0],
        }
    )
    validation = pd.DataFrame(
        {
            "relative_spread": [100.0, 200.0],
            "realized_vol_20": [300.0, 400.0],
            "trade_intensity_1s": [500.0, 600.0],
            "y_50": [-1, 1],
        }
    )
    thresholds = training_regime_thresholds(train)

    aggregate, _ = evaluate_regime_predictions(
        data=validation,
        y_pred=np.array([-1, 1]),
        split="validation",
        horizon=50,
        model_name="model",
        thresholds=thresholds,
    )

    assert thresholds["threshold_value"].tolist() == [2.0, 5.0, 8.0]
    assert set(aggregate["threshold_value"]) == {2.0, 5.0, 8.0}
    assert set(aggregate["threshold_source"]) == {"train_median"}
