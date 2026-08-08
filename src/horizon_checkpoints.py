from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd

from src.atomic_io import atomic_write_json


CHECKPOINT_FORMAT_VERSION = 1


def checkpoint_identity_fingerprint(identity: dict[str, Any]) -> str:
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HorizonCheckpointStore:
    """Transactional, compatibility-checked per-horizon result checkpoints."""

    def __init__(
        self,
        root: Path,
        *,
        identity: dict[str, Any],
    ) -> None:
        self.root = Path(root)
        self.identity = identity
        self.identity_fingerprint = checkpoint_identity_fingerprint(identity)

    def path_for(self, horizon: int) -> Path:
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        return self.root / f"horizon_{horizon}"

    def exists(self, horizon: int) -> bool:
        return (self.path_for(horizon) / "manifest.json").exists()

    def save(
        self,
        horizon: int,
        *,
        tables: dict[str, pd.DataFrame],
        artifact_parameter_fingerprint: str,
        overwrite: bool = False,
    ) -> Path:
        target = self.path_for(horizon)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                dir=self.root,
                prefix=f".horizon_{horizon}.",
            )
        )
        try:
            file_hashes: dict[str, str] = {}
            for name, table in sorted(tables.items()):
                path = temporary / f"{name}.csv"
                table.to_csv(path, index=False)
                file_hashes[path.name] = _file_sha256(path)

            manifest = {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "horizon": horizon,
                "identity": self.identity,
                "identity_fingerprint": self.identity_fingerprint,
                "artifact_parameter_fingerprint": artifact_parameter_fingerprint,
                "files": file_hashes,
            }
            atomic_write_json(temporary / "manifest.json", manifest)

            if target.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"Checkpoint already exists: {target}. "
                        "Use --resume or --overwrite."
                    )
                shutil.rmtree(target)
            os.replace(temporary, target)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return target

    def load(
        self,
        horizon: int,
        *,
        artifact_parameter_fingerprint: str,
    ) -> dict[str, pd.DataFrame]:
        checkpoint = self.path_for(horizon)
        manifest_path = checkpoint / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        errors = []
        if manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            errors.append("format version differs")
        if manifest.get("horizon") != horizon:
            errors.append("horizon differs")
        if manifest.get("identity_fingerprint") != self.identity_fingerprint:
            errors.append("run identity differs")
        if manifest.get("identity") != self.identity:
            errors.append("run identity payload differs")
        if (
            manifest.get("artifact_parameter_fingerprint")
            != artifact_parameter_fingerprint
        ):
            errors.append("model parameter fingerprint differs")
        if errors:
            raise ValueError(
                f"Incompatible horizon-{horizon} checkpoint: "
                + "; ".join(errors)
            )

        tables = {}
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError(f"Checkpoint has no result files: {manifest_path}")
        for filename, expected_hash in sorted(files.items()):
            path = checkpoint / filename
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint file is missing: {path}")
            if _file_sha256(path) != expected_hash:
                raise ValueError(f"Checkpoint checksum mismatch: {path}")
            tables[path.stem] = pd.read_csv(path)
        return tables


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
