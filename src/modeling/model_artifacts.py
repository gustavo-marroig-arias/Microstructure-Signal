from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.protocol import TERNARY_LABELS


MODEL_ARTIFACT_FORMAT_VERSION = 2
FLOAT64_DTYPE = np.dtype("float64")
SUPPORTED_TRANSFORM_DTYPES = {"float32", "float64"}


@dataclass(frozen=True)
class FrozenLogisticArtifact:
    """Portable, pickle-free representation of a fitted scaled logistic model."""

    model_name: str
    horizon: int
    feature_names: tuple[str, ...]
    classes: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    transform_dtype: str
    model_config: dict[str, Any]
    n_iter: tuple[int, ...]
    converged: bool
    experiment_fingerprint: str | None = None
    dataset_sha256: str | None = None
    source_git_commit: str | None = None
    source_tree_sha256: str | None = None
    sklearn_version: str = sklearn.__version__
    created_utc: str = ""
    format_version: int = MODEL_ARTIFACT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not self.created_utc:
            object.__setattr__(
                self,
                "created_utc",
                datetime.now(timezone.utc).isoformat(),
            )
        self.validate()

    def validate(self) -> None:
        n_features = len(self.feature_names)
        n_classes = len(self.classes)

        if self.format_version != MODEL_ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model artifact format: {self.format_version}"
            )
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if n_features == 0:
            raise ValueError("feature_names must not be empty.")
        if len(set(self.feature_names)) != n_features:
            raise ValueError("feature_names must be unique and ordered.")
        if n_classes < 2:
            raise ValueError("At least two model classes are required.")
        if self.coefficients.shape != (n_classes, n_features):
            raise ValueError(
                "coefficients shape does not match classes and feature_names."
            )
        if self.intercepts.shape != (n_classes,):
            raise ValueError("intercepts shape does not match classes.")
        if self.scaler_mean.shape != (n_features,):
            raise ValueError("scaler_mean shape does not match feature_names.")
        if self.scaler_scale.shape != (n_features,):
            raise ValueError("scaler_scale shape does not match feature_names.")
        if not np.isfinite(self.coefficients).all():
            raise ValueError("coefficients contain non-finite values.")
        if not np.isfinite(self.intercepts).all():
            raise ValueError("intercepts contain non-finite values.")
        if not np.isfinite(self.scaler_mean).all():
            raise ValueError("scaler_mean contains non-finite values.")
        if not np.isfinite(self.scaler_scale).all():
            raise ValueError("scaler_scale contains non-finite values.")
        if (self.scaler_scale <= 0.0).any():
            raise ValueError("scaler_scale must be strictly positive.")
        if self.transform_dtype not in SUPPORTED_TRANSFORM_DTYPES:
            raise ValueError(
                "transform_dtype must be one of "
                f"{sorted(SUPPORTED_TRANSFORM_DTYPES)}, "
                f"got {self.transform_dtype!r}."
            )

    def metadata(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "model_name": self.model_name,
            "horizon": self.horizon,
            "feature_names": list(self.feature_names),
            "transform_dtype": self.transform_dtype,
            "model_config": self.model_config,
            "n_iter": list(self.n_iter),
            "converged": self.converged,
            "experiment_fingerprint": self.experiment_fingerprint,
            "dataset_sha256": self.dataset_sha256,
            "source_git_commit": self.source_git_commit,
            "source_tree_sha256": self.source_tree_sha256,
            "sklearn_version": self.sklearn_version,
            "created_utc": self.created_utc,
        }

    def parameter_fingerprint(self) -> str:
        """Hash every value needed to reproduce this artifact's probabilities."""
        self.validate()
        identity = {
            "format_version": self.format_version,
            "model_name": self.model_name,
            "horizon": self.horizon,
            "feature_names": list(self.feature_names),
            "transform_dtype": self.transform_dtype,
            "model_config": self.model_config,
            "experiment_fingerprint": self.experiment_fingerprint,
            "dataset_sha256": self.dataset_sha256,
        }
        digest = hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for name, array in (
            ("classes", self.classes),
            ("coefficients", self.coefficients),
            ("intercepts", self.intercepts),
            ("scaler_mean", self.scaler_mean),
            ("scaler_scale", self.scaler_scale),
        ):
            contiguous = np.ascontiguousarray(array)
            descriptor = {
                "name": name,
                "dtype": contiguous.dtype.str,
                "shape": contiguous.shape,
            }
            digest.update(
                json.dumps(
                    descriptor,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(contiguous.tobytes(order="C"))
        return digest.hexdigest()


def artifact_from_pipeline(
    *,
    pipeline: Pipeline,
    model_name: str,
    horizon: int,
    feature_names: tuple[str, ...] | list[str],
    transform_dtype: str,
    model_config: dict[str, Any],
    experiment_fingerprint: str | None = None,
    dataset_sha256: str | None = None,
    source_git_commit: str | None = None,
    source_tree_sha256: str | None = None,
) -> FrozenLogisticArtifact:
    scaler = pipeline.named_steps.get("scaler")
    model = pipeline.named_steps.get("model")

    if scaler is None or model is None:
        raise ValueError("Pipeline must contain 'scaler' and 'model' steps.")

    required_scaler_attributes = ("mean_", "scale_")
    required_model_attributes = ("classes_", "coef_", "intercept_", "n_iter_")
    if any(not hasattr(scaler, name) for name in required_scaler_attributes):
        raise ValueError("Scaler is not fitted.")
    if any(not hasattr(model, name) for name in required_model_attributes):
        raise ValueError("Logistic model is not fitted.")

    max_iter = int(model_config["max_iter"])
    n_iter = tuple(int(value) for value in np.ravel(model.n_iter_))
    converged = bool(n_iter) and all(value < max_iter for value in n_iter)

    return FrozenLogisticArtifact(
        model_name=model_name,
        horizon=horizon,
        feature_names=tuple(feature_names),
        classes=np.asarray(model.classes_, dtype=np.int64),
        coefficients=np.asarray(model.coef_, dtype=np.float64),
        intercepts=np.asarray(model.intercept_, dtype=np.float64),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        transform_dtype=np.dtype(transform_dtype).name,
        model_config=dict(model_config),
        n_iter=n_iter,
        converged=converged,
        experiment_fingerprint=experiment_fingerprint,
        dataset_sha256=dataset_sha256,
        source_git_commit=source_git_commit,
        source_tree_sha256=source_tree_sha256,
    )


def artifact_from_prestandardized_estimator(
    *,
    model: LogisticRegression,
    model_name: str,
    horizon: int,
    feature_names: tuple[str, ...] | list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    model_config: dict[str, Any],
    experiment_fingerprint: str | None = None,
    dataset_sha256: str | None = None,
    source_git_commit: str | None = None,
    source_tree_sha256: str | None = None,
) -> FrozenLogisticArtifact:
    required = ("classes_", "coef_", "intercept_", "n_iter_")
    if any(not hasattr(model, name) for name in required):
        raise ValueError("Logistic estimator is not fitted.")
    max_iter = int(model_config["max_iter"])
    n_iter = tuple(int(value) for value in np.ravel(model.n_iter_))
    converged = bool(n_iter) and all(value < max_iter for value in n_iter)
    return FrozenLogisticArtifact(
        model_name=model_name,
        horizon=horizon,
        feature_names=tuple(feature_names),
        classes=np.asarray(model.classes_, dtype=np.int64),
        coefficients=np.asarray(model.coef_, dtype=np.float64),
        intercepts=np.asarray(model.intercept_, dtype=np.float64),
        scaler_mean=np.asarray(scaler_mean, dtype=np.float64),
        scaler_scale=np.asarray(scaler_scale, dtype=np.float64),
        transform_dtype="float64",
        model_config=dict(model_config),
        n_iter=n_iter,
        converged=converged,
        experiment_fingerprint=experiment_fingerprint,
        dataset_sha256=dataset_sha256,
        source_git_commit=source_git_commit,
        source_tree_sha256=source_tree_sha256,
    )


def assert_prestandardized_artifact_parity(
    *,
    model: LogisticRegression,
    artifact: FrozenLogisticArtifact,
    sample: pd.DataFrame,
    atol: float = 1e-12,
) -> float:
    """Verify direct-estimator and frozen-artifact probabilities and classes."""
    if sample.empty:
        raise ValueError("Parity sample must not be empty.")
    values = sample.loc[:, list(artifact.feature_names)].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    values -= artifact.scaler_mean
    values /= artifact.scaler_scale
    estimator_probabilities = model.predict_proba(values)
    artifact_probabilities = predict_probabilities_from_artifact(
        artifact,
        sample,
        chunk_size=max(1, len(sample)),
    )
    if not np.array_equal(model.classes_, artifact.classes):
        raise AssertionError("Estimator and artifact class order differ.")
    max_abs_error = float(
        np.max(np.abs(estimator_probabilities - artifact_probabilities))
    )
    if max_abs_error > atol:
        raise AssertionError(
            "Frozen artifact probabilities do not match the estimator: "
            f"max_abs_error={max_abs_error:.3e}, atol={atol:.3e}"
        )
    estimator_predictions = model.predict(values)
    artifact_predictions = artifact.classes[
        np.argmax(artifact_probabilities, axis=1)
    ]
    if not np.array_equal(estimator_predictions, artifact_predictions):
        raise AssertionError("Estimator and artifact hard predictions differ.")
    return max_abs_error


def predict_probabilities_from_artifact(
    artifact: FrozenLogisticArtifact,
    data: pd.DataFrame,
    *,
    chunk_size: int = 1_000_000,
    output_dtype: np.dtype = FLOAT64_DTYPE,
) -> np.ndarray:
    artifact.validate()

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    missing = sorted(set(artifact.feature_names) - set(data.columns))
    if missing:
        raise ValueError(f"Data is missing artifact features: {missing}")

    n_rows = len(data)
    probabilities = np.empty(
        (n_rows, len(artifact.classes)),
        dtype=output_dtype,
    )

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        values = data.iloc[start:stop].loc[
            :,
            list(artifact.feature_names),
        ].to_numpy(
            dtype=np.dtype(artifact.transform_dtype),
            copy=True,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite feature values found in rows {start}:{stop}."
            )

        values -= artifact.scaler_mean
        values /= artifact.scaler_scale

        logits = values @ artifact.coefficients.T
        logits += artifact.intercepts
        logits -= logits.max(axis=1, keepdims=True)

        exp_logits = np.exp(logits)
        chunk_probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probabilities[start:stop] = chunk_probabilities.astype(
            output_dtype,
            copy=False,
        )

    return probabilities


def predict_from_artifact(
    artifact: FrozenLogisticArtifact,
    data: pd.DataFrame,
    *,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    probabilities = predict_probabilities_from_artifact(
        artifact,
        data,
        chunk_size=chunk_size,
    )
    return artifact.classes[np.argmax(probabilities, axis=1)]


def probability_frame_from_artifact(
    artifact: FrozenLogisticArtifact,
    data: pd.DataFrame,
    *,
    chunk_size: int = 1_000_000,
) -> pd.DataFrame:
    probabilities = predict_probabilities_from_artifact(
        artifact,
        data,
        chunk_size=chunk_size,
    )
    return pd.DataFrame(
        probabilities,
        columns=[
            f"proba_{int(class_label)}"
            for class_label in artifact.classes
        ],
        index=data.index,
    )


def coefficient_table_from_artifact(
    artifact: FrozenLogisticArtifact,
) -> pd.DataFrame:
    rows = []
    for class_label, coefficient_row, intercept in zip(
        artifact.classes,
        artifact.coefficients,
        artifact.intercepts,
        strict=True,
    ):
        for feature, coefficient in zip(
            artifact.feature_names,
            coefficient_row,
            strict=True,
        ):
            rows.append(
                {
                    "model": artifact.model_name,
                    "horizon": artifact.horizon,
                    "class_label": int(class_label),
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                    "intercept": float(intercept),
                }
            )
    return pd.DataFrame(rows)


def validate_artifact_compatibility(
    artifact: FrozenLogisticArtifact,
    *,
    model_name: str,
    horizon: int,
    feature_names: tuple[str, ...] | list[str],
    experiment_fingerprint: str | None = None,
    dataset_sha256: str | None = None,
    source_tree_sha256: str | None = None,
    require_converged: bool = True,
) -> None:
    """Fail closed when a frozen model does not belong to this experiment."""
    artifact.validate()
    errors = []

    if artifact.model_name != model_name:
        errors.append(
            f"model_name={artifact.model_name!r}, expected {model_name!r}"
        )
    if artifact.horizon != horizon:
        errors.append(f"horizon={artifact.horizon}, expected {horizon}")
    if artifact.feature_names != tuple(feature_names):
        errors.append("feature order differs from the protocol")
    if not np.array_equal(
        artifact.classes,
        np.asarray(TERNARY_LABELS, dtype=np.int64),
    ):
        errors.append(
            f"class order={artifact.classes.tolist()}, "
            f"expected {list(TERNARY_LABELS)}"
        )
    if require_converged and not artifact.converged:
        errors.append("optimizer convergence was not established")
    if (
        experiment_fingerprint is not None
        and artifact.experiment_fingerprint != experiment_fingerprint
    ):
        errors.append("experiment fingerprint differs")
    if (
        dataset_sha256 is not None
        and artifact.dataset_sha256 != dataset_sha256
    ):
        errors.append("model dataset SHA-256 differs")
    if (
        source_tree_sha256 is not None
        and artifact.source_tree_sha256 != source_tree_sha256
    ):
        errors.append("source-tree SHA-256 differs")

    if errors:
        raise ValueError(
            "Frozen model artifact is incompatible: " + "; ".join(errors)
        )


def assert_pipeline_artifact_parity(
    *,
    pipeline: Pipeline,
    artifact: FrozenLogisticArtifact,
    sample: pd.DataFrame,
    atol: float = 1e-12,
) -> float:
    if sample.empty:
        raise ValueError("Parity sample must not be empty.")
    if atol <= 0:
        raise ValueError("atol must be positive.")

    pipeline_probabilities = pipeline.predict_proba(
        sample.loc[:, list(artifact.feature_names)]
    )
    artifact_probabilities = predict_probabilities_from_artifact(
        artifact,
        sample,
        chunk_size=max(1, len(sample)),
    )

    if not np.array_equal(
        np.asarray(pipeline.named_steps["model"].classes_),
        artifact.classes,
    ):
        raise AssertionError("Pipeline and artifact class order differ.")

    max_abs_error = float(
        np.max(np.abs(pipeline_probabilities - artifact_probabilities))
    )
    if max_abs_error > atol:
        raise AssertionError(
            "Frozen artifact probabilities do not match the fitted pipeline: "
            f"max_abs_error={max_abs_error:.3e}, atol={atol:.3e}"
        )

    pipeline_predictions = pipeline.predict(
        sample.loc[:, list(artifact.feature_names)]
    )
    artifact_predictions = artifact.classes[
        np.argmax(artifact_probabilities, axis=1)
    ]
    if not np.array_equal(pipeline_predictions, artifact_predictions):
        raise AssertionError("Pipeline and artifact hard predictions differ.")

    return max_abs_error


def save_frozen_logistic_artifact(
    artifact: FrozenLogisticArtifact,
    artifact_base: Path,
) -> tuple[Path, Path]:
    artifact.validate()
    artifact_base = Path(artifact_base)
    artifact_base.parent.mkdir(parents=True, exist_ok=True)

    arrays_path = artifact_base.with_suffix(".npz")
    metadata_path = artifact_base.with_suffix(".json")

    _atomic_save_npz(
        arrays_path,
        classes=artifact.classes,
        coefficients=artifact.coefficients,
        intercepts=artifact.intercepts,
        scaler_mean=artifact.scaler_mean,
        scaler_scale=artifact.scaler_scale,
    )
    metadata = artifact.metadata()
    metadata["arrays_sha256"] = file_sha256(arrays_path)
    _atomic_save_json(metadata_path, metadata)

    return arrays_path, metadata_path


def load_frozen_logistic_artifact(
    artifact_base: Path,
) -> FrozenLogisticArtifact:
    artifact_base = Path(artifact_base)
    arrays_path = artifact_base.with_suffix(".npz")
    metadata_path = artifact_base.with_suffix(".json")

    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing model arrays: {arrays_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing model metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    expected_arrays_sha256 = metadata.get("arrays_sha256")
    if not isinstance(expected_arrays_sha256, str):
        raise ValueError(
            f"Model metadata lacks arrays_sha256: {metadata_path}"
        )
    observed_arrays_sha256 = file_sha256(arrays_path)
    if observed_arrays_sha256 != expected_arrays_sha256:
        raise ValueError(
            "Model artifact checksum mismatch. Arrays and metadata may be "
            f"inconsistent: {arrays_path}"
        )

    with np.load(arrays_path, allow_pickle=False) as arrays:
        artifact = FrozenLogisticArtifact(
            model_name=str(metadata["model_name"]),
            horizon=int(metadata["horizon"]),
            feature_names=tuple(metadata["feature_names"]),
            classes=np.asarray(arrays["classes"], dtype=np.int64),
            coefficients=np.asarray(arrays["coefficients"], dtype=np.float64),
            intercepts=np.asarray(arrays["intercepts"], dtype=np.float64),
            scaler_mean=np.asarray(arrays["scaler_mean"], dtype=np.float64),
            scaler_scale=np.asarray(arrays["scaler_scale"], dtype=np.float64),
            transform_dtype=str(metadata["transform_dtype"]),
            model_config=dict(metadata["model_config"]),
            n_iter=tuple(int(value) for value in metadata["n_iter"]),
            converged=bool(metadata["converged"]),
            experiment_fingerprint=metadata.get("experiment_fingerprint"),
            dataset_sha256=metadata.get("dataset_sha256"),
            source_git_commit=metadata.get("source_git_commit"),
            source_tree_sha256=metadata.get("source_tree_sha256"),
            sklearn_version=str(metadata["sklearn_version"]),
            created_utc=str(metadata["created_utc"]),
            format_version=int(metadata["format_version"]),
        )

    return artifact


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        np.savez_compressed(handle, **arrays)

    try:
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    try:
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
