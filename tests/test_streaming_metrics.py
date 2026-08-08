import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    evaluate_daily_blocks,
    evaluate_nonzero_subset,
    evaluate_ternary_predictions,
)
from src.evaluation.streaming_metrics import (
    DailyTernaryMetricAccumulator,
    TernaryMetricAccumulator,
)


def test_streaming_metrics_match_batch_metrics_exactly() -> None:
    rng = np.random.default_rng(2026)
    y_true = rng.choice([-1, 0, 1], size=10_003)
    y_pred = rng.choice([-1, 0, 1], size=10_003)
    accumulator = TernaryMetricAccumulator()
    for start in range(0, len(y_true), 317):
        accumulator.update(
            y_true[start : start + 317],
            y_pred[start : start + 317],
        )

    expected = evaluate_ternary_predictions(
        y_true,
        y_pred,
        split="validation",
        horizon=50,
        model_name="model",
    )
    actual = accumulator.evaluation_tables(
        split="validation",
        horizon=50,
        model_name="model",
    )
    for name in expected:
        pd.testing.assert_frame_equal(
            actual[name],
            expected[name],
            check_dtype=False,
            atol=1e-15,
            rtol=1e-15,
        )

    pd.testing.assert_frame_equal(
        accumulator.nonzero_table(
            split="validation",
            horizon=50,
            model_name="model",
        ),
        evaluate_nonzero_subset(
            y_true,
            y_pred,
            split="validation",
            horizon=50,
            model_name="model",
        ),
        check_dtype=False,
        atol=1e-15,
        rtol=1e-15,
    )


def test_streaming_daily_metrics_match_batch_daily_metrics() -> None:
    rng = np.random.default_rng(17)
    n_rows = 5_000
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-03-01",
                periods=n_rows,
                freq="30s",
                tz="UTC",
            ),
            "y_20": rng.choice([-1, 0, 1], n_rows),
        }
    )
    y_pred = rng.choice([-1, 0, 1], n_rows)
    accumulator = DailyTernaryMetricAccumulator()
    for start in range(0, n_rows, 211):
        stop = start + 211
        accumulator.update(
            frame["timestamp"].iloc[start:stop],
            frame["y_20"].iloc[start:stop],
            y_pred[start:stop],
        )

    expected = evaluate_daily_blocks(
        frame,
        y_pred,
        split="test",
        horizon=20,
        model_name="model",
    )
    actual = accumulator.table(
        split="test",
        horizon=20,
        model_name="model",
    )
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        atol=1e-15,
        rtol=1e-15,
    )


def test_streaming_balanced_accuracy_ignores_absent_true_class() -> None:
    accumulator = TernaryMetricAccumulator()
    accumulator.update(
        np.array([-1, -1, 0, 0]),
        np.array([-1, 1, 0, 1]),
    )
    actual = accumulator.evaluation_tables(
        split="test",
        horizon=10,
        model_name="model",
    )["aggregate"].iloc[0]
    expected = evaluate_ternary_predictions(
        np.array([-1, -1, 0, 0]),
        np.array([-1, 1, 0, 1]),
        split="test",
        horizon=10,
        model_name="model",
    )["aggregate"].iloc[0]
    assert actual["balanced_accuracy"] == expected["balanced_accuracy"]
