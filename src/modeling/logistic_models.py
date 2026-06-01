from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TERNARY_LABELS = [-1, 0, 1]


@dataclass(frozen=True)
class LogisticModelConfig:
    """
    Configuration for multinomial logistic regression.

    Protocol choices:
    - L2 regularization
    - no class weights
    - train-only scaling

    Solver is an implementation choice. For very large datasets, "saga" is often
    more practical than "lbfgs".
    """
    C: float = 1.0
    max_iter: int = 200
    solver: str = "lbfgs"
    random_state: int = 42
    n_jobs: int | None = None


def validate_feature_columns(data: pd.DataFrame, feature_columns: list[str]) -> None:
    """
    Checks that required feature columns exist and contain no missing/non-finite values.
    Uses column-wise checks to avoid creating one huge dense array.
    """
    missing = sorted(set(feature_columns) - set(data.columns))

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    bad_missing = []
    bad_nonfinite = []

    for col in feature_columns:
        s = data[col]

        if s.isna().any():
            bad_missing.append(col)
            continue

        values = s.to_numpy()

        if not np.isfinite(values).all():
            bad_nonfinite.append(col)

    if bad_missing:
        raise ValueError(f"Feature columns contain missing values: {bad_missing}")

    if bad_nonfinite:
        raise ValueError(f"Feature columns contain non-finite values: {bad_nonfinite}")
    

def validate_label(y: Iterable[int]) -> None:
    """
    Checks that labels are in {-1, 0, +1}.
    """
    y_series = pd.Series(y)

    if y_series.empty:
        raise ValueError("Labels are empty.")

    if y_series.isna().any():
        raise ValueError("Labels contain missing values.")

    observed = set(y_series.unique())

    if not observed.issubset(set(TERNARY_LABELS)):
        raise ValueError(f"Unexpected labels: {observed}")

    if len(observed) < 2:
        raise ValueError(
            f"Need at least two observed classes to fit logistic regression, got {observed}."
        )
    

def make_logistic_pipeline(config: LogisticModelConfig = LogisticModelConfig()) -> Pipeline:
    """
    Creates a train-only scaling + logistic regression pipeline.

    StandardScaler is inside the pipeline, so when we call fit() on train,
    scaling parameters are learned only from train.
    """
    allowed_solvers = {"lbfgs", "saga", "newton-cg", "sag"}

    if config.solver not in allowed_solvers:
        raise ValueError(f"Unsupported solver: {config.solver}")

    model = LogisticRegression(
        penalty="l2",
        C=config.C,
        solver=config.solver,
        max_iter=config.max_iter,
        class_weight=None,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    return pipeline


def fit_logistic_model(
        train: pd.DataFrame,
        label_col: str,
        feature_columns: list[str],
        config: LogisticModelConfig = LogisticModelConfig(),
) -> Pipeline:
    """
    Fits multinomial logistic regression on train only.
    """
    if label_col not in train.columns:
        raise ValueError(f"Missing label column: {label_col}")

    validate_feature_columns(train, feature_columns)
    validate_label(train[label_col])

    X_train = train[feature_columns]
    y_train = train[label_col].astype(int)

    pipeline = make_logistic_pipeline(config=config)
    pipeline.fit(X_train, y_train)

    return pipeline


def predict_logistic_model(
        model: Pipeline,
        data: pd.DataFrame,
        feature_columns: list[str],
) -> np.ndarray:
    """
    Predicts class labels
    """
    validate_feature_columns(data,feature_columns)

    X = data[feature_columns]
    return model.predict(X).astype(int)


def predict_logistic_probabilities(
    model: Pipeline,
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Predicts class probabilities and returns a tidy DataFrame.

    Useful later for calibration/score diagnostics.
    """
    validate_feature_columns(data, feature_columns)

    X = data[feature_columns]

    clf = model.named_steps["model"]
    classes = clf.classes_

    proba = model.predict_proba(X)

    out = pd.DataFrame(
        proba,
        columns = [f"proba_{int(c)}" for c in classes],
        index=data.index,
    )

    return out


def coefficient_table(
    model: Pipeline,
    feature_columns: list[str],
    horizon: int,
    model_name: str,        
) -> pd.DataFrame:
    """
    Extracts standardized logistic-regression coefficients.

    Because StandardScaler is inside the pipeline, these coefficients are for
    standardized features.

    For multiclass logistic regression:
    - one coefficient row per predicted class
    - positive coefficient means higher feature value pushes probability toward that class
      relative to the learned multiclass decision structure
    """
    clf = model.named_steps["model"]

    rows = []

    for class_label, coef_row, intercept in zip(
        clf.classes_,
        clf.coef_,
        clf.intercept_,
        strict=True,
    ):
        for feature, coef in zip(feature_columns, coef_row, strict=True):
            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "class_label": int(class_label),
                    "feature": feature,
                    "standardized_coefficient": float(coef),
                    "intercept": float(intercept),
                }
            )

    return pd.DataFrame(rows)