from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import pyarrow.dataset as ds


VALID_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class ModelDatasetLocation:
    """Resolved on-disk layout for one model-ready dataset."""

    dataset_stem: str
    layout: str
    path: Path

    def __post_init__(self) -> None:
        if self.layout not in {"partitioned", "monolithic"}:
            raise ValueError(f"Unsupported model dataset layout: {self.layout}")


def resolve_model_dataset(
    processed_dir: Path,
    dataset_stem: str,
) -> ModelDatasetLocation:
    processed_dir = Path(processed_dir)
    partitioned_path = processed_dir / dataset_stem
    monolithic_path = processed_dir / f"{dataset_stem}.parquet"

    partition_files = [
        partitioned_path / f"{split}.parquet"
        for split in VALID_SPLITS
    ]
    existing_partition_files = [path for path in partition_files if path.exists()]

    if existing_partition_files:
        missing = [
            path.name
            for path in partition_files
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Partitioned model dataset is incomplete. "
                f"Missing files under {partitioned_path}: {missing}"
            )
        return ModelDatasetLocation(
            dataset_stem=dataset_stem,
            layout="partitioned",
            path=partitioned_path,
        )

    if monolithic_path.exists():
        return ModelDatasetLocation(
            dataset_stem=dataset_stem,
            layout="monolithic",
            path=monolithic_path,
        )

    raise FileNotFoundError(
        "Missing model dataset. Expected either "
        f"{partitioned_path}/{{train,validation,test}}.parquet or "
        f"{monolithic_path}."
    )


def load_model_split(
    location: ModelDatasetLocation,
    split: str,
    columns: Sequence[str],
    *,
    eligible_only: bool = True,
) -> pd.DataFrame:
    """Load one chronological split and only the requested columns."""
    _validate_split(split)
    requested = _validated_columns(columns)

    read_columns = list(requested)
    for helper in ("split", "model_eligible"):
        if helper not in read_columns:
            read_columns.append(helper)

    if location.layout == "partitioned":
        path = location.path / f"{split}.parquet"
        filters = (
            [("model_eligible", "==", True)]
            if eligible_only
            else None
        )
        frame = pd.read_parquet(
            path,
            columns=read_columns,
            filters=filters,
        )
        if "split" in frame.columns:
            observed = set(frame["split"].dropna().astype(str).unique())
            if observed and observed != {split}:
                raise ValueError(
                    f"Partition {path} contains unexpected split values: {observed}"
                )
        else:
            frame["split"] = split
    else:
        filters: list[tuple[str, str, object]] = [("split", "==", split)]
        if eligible_only:
            filters.append(("model_eligible", "==", True))
        frame = pd.read_parquet(
            location.path,
            columns=read_columns,
            filters=filters,
        )

    _validate_loaded_split(frame, split, eligible_only=eligible_only)

    if eligible_only:
        frame = frame.loc[frame["model_eligible"]].copy()

    frame["split"] = pd.Categorical(
        frame["split"],
        categories=list(VALID_SPLITS),
        ordered=True,
    )

    output_columns = list(requested)
    if "split" in requested and "split" not in frame.columns:
        frame["split"] = split
    return frame.loc[:, output_columns].reset_index(drop=True)


def load_model_splits(
    location: ModelDatasetLocation,
    splits: Iterable[str],
    columns: Sequence[str],
    *,
    eligible_only: bool = True,
) -> dict[str, pd.DataFrame]:
    split_names = tuple(splits)
    if not split_names:
        raise ValueError("At least one split must be requested.")
    if len(set(split_names)) != len(split_names):
        raise ValueError(f"Duplicate split requested: {split_names}")

    return {
        split: load_model_split(
            location,
            split,
            columns,
            eligible_only=eligible_only,
        )
        for split in split_names
    }


def iter_model_split_batches(
    location: ModelDatasetLocation,
    split: str,
    columns: Sequence[str],
    *,
    eligible_only: bool = True,
    batch_size: int = 1_000_000,
) -> Iterator[pd.DataFrame]:
    """Yield ordered, column-pruned batches for one chronological split."""
    _validate_split(split)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    requested = _validated_columns(columns)
    read_columns = list(requested)
    for helper in ("split", "model_eligible"):
        if helper not in read_columns:
            read_columns.append(helper)

    path = (
        location.path / f"{split}.parquet"
        if location.layout == "partitioned"
        else location.path
    )
    dataset = ds.dataset(path, format="parquet")
    predicate = ds.field("split") == split
    if eligible_only:
        predicate &= ds.field("model_eligible") == True  # noqa: E712

    rows_seen = 0
    for record_batch in dataset.scanner(
        columns=read_columns,
        filter=predicate,
        batch_size=batch_size,
        use_threads=False,
    ).to_batches():
        frame = record_batch.to_pandas()
        if frame.empty:
            continue
        _validate_loaded_split(frame, split, eligible_only=eligible_only)
        rows_seen += len(frame)
        yield frame.loc[:, list(requested)].reset_index(drop=True)

    if rows_seen == 0:
        qualifier = "model-eligible " if eligible_only else ""
        raise ValueError(f"No {qualifier}{split} rows were loaded.")


def model_split_row_count(
    location: ModelDatasetLocation,
    split: str,
    *,
    eligible_only: bool = True,
) -> int:
    """Count one split through Arrow predicates without materializing rows."""
    _validate_split(split)
    path = (
        location.path / f"{split}.parquet"
        if location.layout == "partitioned"
        else location.path
    )
    dataset = ds.dataset(path, format="parquet")
    predicate = ds.field("split") == split
    if eligible_only:
        predicate &= ds.field("model_eligible") == True  # noqa: E712
    count = int(dataset.count_rows(filter=predicate))
    if count == 0:
        qualifier = "model-eligible " if eligible_only else ""
        raise ValueError(f"No {qualifier}{split} rows were found.")
    return count


def model_dataset_sha256(
    location: ModelDatasetLocation,
    *,
    chunk_size: int = 1024 * 1024,
    use_cache: bool = True,
) -> str:
    """Hash dataset content once and reuse it while file identity is unchanged."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    paths = _model_dataset_paths(location)
    identities_before = _file_identities(paths)
    cache_path = _hash_cache_path(location)
    if use_cache:
        cached = _read_hash_cache(cache_path, identities_before)
        if cached is not None:
            return cached

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)

    identities_after = _file_identities(paths)
    if identities_after != identities_before:
        raise RuntimeError(
            "Model dataset changed while its provenance hash was being computed."
        )

    dataset_hash = digest.hexdigest()
    if use_cache:
        _write_hash_cache(
            cache_path,
            identities=identities_after,
            dataset_sha256=dataset_hash,
        )
    return dataset_hash


def _model_dataset_paths(
    location: ModelDatasetLocation,
) -> tuple[Path, ...]:
    if location.layout == "monolithic":
        return (location.path,)
    return tuple(
        location.path / f"{split}.parquet"
        for split in VALID_SPLITS
    )


def _file_identities(paths: tuple[Path, ...]) -> list[dict[str, int | str]]:
    identities = []
    for path in paths:
        stat = path.stat()
        identities.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        )
    return identities


def _hash_cache_path(location: ModelDatasetLocation) -> Path:
    if location.layout == "monolithic":
        return location.path.with_suffix(
            f"{location.path.suffix}.sha256.json"
        )
    return location.path / ".dataset_sha256.json"


def _read_hash_cache(
    path: Path,
    identities: list[dict[str, int | str]],
) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    dataset_hash = payload.get("dataset_sha256")
    if (
        payload.get("format_version") != 1
        or payload.get("files") != identities
        or not isinstance(dataset_hash, str)
        or len(dataset_hash) != 64
    ):
        return None
    return dataset_hash


def _write_hash_cache(
    path: Path,
    *,
    identities: list[dict[str, int | str]],
    dataset_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "dataset_sha256": dataset_sha256,
        "files": identities,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_split(split: str) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(
            f"split must be one of {VALID_SPLITS}, got {split!r}."
        )


def _validated_columns(columns: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(columns)
    if not requested:
        raise ValueError("columns must not be empty.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"columns must be unique: {requested}")
    return requested


def _validate_loaded_split(
    frame: pd.DataFrame,
    split: str,
    *,
    eligible_only: bool,
) -> None:
    if frame.empty:
        qualifier = "model-eligible " if eligible_only else ""
        raise ValueError(f"No {qualifier}{split} rows were loaded.")

    required = {"split", "model_eligible"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Model dataset is missing columns: {missing}")

    observed = set(frame["split"].dropna().astype(str).unique())
    if observed != {split}:
        raise ValueError(
            f"Expected only split {split!r}, observed {sorted(observed)}."
        )

    if not pd.api.types.is_bool_dtype(frame["model_eligible"]):
        unique_eligibility = set(frame["model_eligible"].dropna().unique())
        if not unique_eligibility.issubset({True, False}):
            raise TypeError(
                "model_eligible must contain boolean-compatible values."
            )

    if eligible_only and not bool(frame["model_eligible"].all()):
        raise AssertionError(
            "Eligible-only read returned ineligible observations."
        )
