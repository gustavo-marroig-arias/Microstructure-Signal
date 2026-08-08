from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.protocol import TERNARY_LABELS


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
    tol: float = 1e-4
    fail_on_nonconvergence: bool = True

    def __post_init__(self) -> None:
        allowed_solvers = {"lbfgs", "saga", "newton-cg", "sag"}
        if self.C <= 0:
            raise ValueError("C must be strictly positive.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.solver not in allowed_solvers:
            raise ValueError(f"Unsupported solver: {self.solver}")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero.")
        if self.tol <= 0:
            raise ValueError("tol must be strictly positive.")

    def as_dict(self) -> dict:
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "solver": self.solver,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "tol": self.tol,
            "fail_on_nonconvergence": self.fail_on_nonconvergence,
        }


DEFAULT_LOGISTIC_MODEL_CONFIG = LogisticModelConfig()


def validate_feature_columns(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
) -> None:
    """
    Checks that required feature columns exist and contain no missing/non-finite values.
    Uses column-wise checks to avoid creating one huge dense array.
    """
    if data.empty:
        raise ValueError("Feature data is empty.")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns must be unique.")

    missing = sorted(set(feature_columns) - set(data.columns))

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    bad_missing = []
    bad_nonfinite = []

    for col in feature_columns:
        s = data[col]

        if not pd.api.types.is_numeric_dtype(s):
            raise TypeError(f"Feature column must be numeric: {col}")
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


def validate_label(
    y: Iterable[int],
    *,
    require_all_classes: bool = True,
) -> None:
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
    if require_all_classes and observed != set(TERNARY_LABELS):
        raise ValueError(
            "The protocol model requires all ternary classes in training; "
            f"observed {sorted(observed)}."
        )


def make_logistic_pipeline(
    config: LogisticModelConfig = DEFAULT_LOGISTIC_MODEL_CONFIG,
) -> Pipeline:
    """
    Creates a train-only scaling + logistic regression pipeline.

    StandardScaler is inside the pipeline, so when we call fit() on train,
    scaling parameters are learned only from train.
    """
    model = LogisticRegression(
        penalty="l2",
        C=config.C,
        solver=config.solver,
        max_iter=config.max_iter,
        class_weight=None,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        tol=config.tol,
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
    feature_columns: Sequence[str],
    config: LogisticModelConfig = DEFAULT_LOGISTIC_MODEL_CONFIG,
) -> Pipeline:
    """
    Fits multinomial logistic regression on train only.
    """
    if label_col not in train.columns:
        raise ValueError(f"Missing label column: {label_col}")

    validate_feature_columns(train, feature_columns)
    validate_label(train[label_col])

    X_train = train[list(feature_columns)]
    y_train = train[label_col].astype(int)

    pipeline = make_logistic_pipeline(config=config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)

    convergence_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]
    if convergence_warnings and config.fail_on_nonconvergence:
        messages = "; ".join(str(warning.message) for warning in convergence_warnings)
        raise RuntimeError(
            "Logistic regression did not converge. Increase max_iter, adjust tol, "
            f"or investigate feature scaling. Details: {messages}"
        )

    return pipeline


def fit_prestandardized_logistic_model(
    features: np.ndarray,
    labels: Iterable[int],
    config: LogisticModelConfig = DEFAULT_LOGISTIC_MODEL_CONFIG,
) -> LogisticRegression:
    """Fit the protocol estimator on an already train-standardized matrix."""
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix.")
    if not np.issubdtype(features.dtype, np.floating):
        raise TypeError("Prestandardized features must be floating point.")
    y_train = np.asarray(labels, dtype=np.int8)
    if len(y_train) != len(features):
        raise ValueError("Feature and label row counts differ.")
    observed_labels: set[int] = set()
    for start in range(0, len(features), 1_000_000):
        stop = min(start + 1_000_000, len(features))
        if not np.isfinite(features[start:stop]).all():
            raise ValueError("Prestandardized features contain non-finite values.")
        label_chunk = y_train[start:stop]
        if not np.isin(label_chunk, np.asarray(TERNARY_LABELS)).all():
            raise ValueError("Labels contain values outside {-1, 0, 1}.")
        observed_labels.update(int(value) for value in np.unique(label_chunk))
    if observed_labels != set(TERNARY_LABELS):
        raise ValueError(
            "The protocol model requires all ternary classes in training; "
            f"observed {sorted(observed_labels)}."
        )

    model = LogisticRegression(
        penalty="l2",
        C=config.C,
        solver=config.solver,
        max_iter=config.max_iter,
        class_weight=None,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        tol=config.tol,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features, y_train)
    convergence_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]
    if convergence_warnings and config.fail_on_nonconvergence:
        messages = "; ".join(
            str(warning.message) for warning in convergence_warnings
        )
        raise RuntimeError(
            "Logistic regression did not converge. Increase max_iter, adjust "
            f"tol, or investigate scaling. Details: {messages}"
        )
    return model


def predict_logistic_model(
    model: Pipeline,
    data: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    """
    Predicts class labels
    """
    validate_feature_columns(data, feature_columns)

    X = data[list(feature_columns)]
    return model.predict(X).astype(int)


def predict_logistic_probabilities(
    model: Pipeline,
    data: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Predicts class probabilities and returns a tidy DataFrame.

    Useful later for calibration/score diagnostics.
    """
    validate_feature_columns(data, feature_columns)

    X = data[list(feature_columns)]

    clf = model.named_steps["model"]
    classes = clf.classes_

    proba = model.predict_proba(X)

    out = pd.DataFrame(
        proba,
        columns=[f"proba_{int(c)}" for c in classes],
        index=data.index,
    )

    return out


def coefficient_table(
    model: Pipeline,
    feature_columns: Sequence[str],
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


def coefficient_table_from_estimator(
    model: LogisticRegression,
    feature_columns: Sequence[str],
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    """Extract standardized coefficients from a fitted logistic estimator."""
    rows = []
    for class_label, coef_row, intercept in zip(
        model.classes_,
        model.coef_,
        model.intercept_,
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
