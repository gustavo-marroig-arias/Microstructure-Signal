from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from types import TracebackType
from typing import Any
from uuid import uuid4

from src.atomic_io import atomic_write_json


SOURCE_GLOBS = (
    "src/**/*.py",
    "scripts/**/*.py",
    "tests/**/*.py",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "environment.yml",
)


def capture_source_state(project_root: Path) -> dict[str, Any]:
    """Capture both Git identity and a content hash that also covers dirty files."""
    root = Path(project_root).resolve()
    git_sha = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    return {
        "git_commit": git_sha,
        "git_dirty": bool(status),
        "source_tree_sha256": source_tree_sha256(root),
    }


def require_clean_source(source_state: dict[str, Any]) -> None:
    if source_state["git_dirty"]:
        raise RuntimeError(
            "Canonical execution requires a clean Git worktree. Commit the "
            "reviewed source snapshot or omit --require-clean-git for a "
            "development-only run."
        )


def source_tree_sha256(project_root: Path) -> str:
    root = Path(project_root).resolve()
    paths: set[Path] = set()
    for pattern in SOURCE_GLOBS:
        paths.update(path for path in root.glob(pattern) if path.is_file())

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


class RunRecorder(AbstractContextManager["RunRecorder"]):
    """Persist success/failure, source identity, timing, and peak RSS atomically."""

    def __init__(
        self,
        path: Path,
        *,
        project_root: Path,
        stage: str,
        arguments: dict[str, Any],
        dataset_sha256: str | None = None,
        experiment_fingerprint: str | None = None,
        require_clean_git: bool = False,
    ) -> None:
        self.path = Path(path)
        self.project_root = Path(project_root)
        self.stage = stage
        self.arguments = arguments
        self.dataset_sha256 = dataset_sha256
        self.experiment_fingerprint = experiment_fingerprint
        self.source_state = capture_source_state(self.project_root)
        if require_clean_git:
            require_clean_source(self.source_state)
        self._started_monotonic = 0.0
        self._started_utc = ""
        self._run_id = ""

    def __enter__(self) -> "RunRecorder":
        self._started_monotonic = time.perf_counter()
        self._started_utc = datetime.now(timezone.utc).isoformat()
        self._run_id = uuid4().hex
        return self

    def set_dataset_sha256(self, value: str) -> None:
        if len(value) != 64:
            raise ValueError("dataset SHA-256 must contain 64 hexadecimal characters.")
        int(value, 16)
        self.dataset_sha256 = value

    def set_experiment_fingerprint(self, value: str) -> None:
        if not value:
            raise ValueError("experiment fingerprint must not be empty.")
        self.experiment_fingerprint = value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        elapsed = time.perf_counter() - self._started_monotonic
        payload: dict[str, Any] = {
            "format_version": 1,
            "run_id": self._run_id,
            "stage": self.stage,
            "status": "success" if exc_type is None else "failed",
            "started_utc": self._started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak_rss_bytes(),
            "command": [sys.executable, *sys.argv],
            "arguments": self.arguments,
            "dataset_sha256": self.dataset_sha256,
            "experiment_fingerprint": self.experiment_fingerprint,
            "source": self.source_state,
            "runtime": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "pid": os.getpid(),
            },
        }
        if exc_type is not None:
            payload["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
            }
        history_path = (
            self.path.parent
            / "history"
            / self.path.stem
            / f"{self._run_id}.json"
        )
        payload["history_path"] = str(history_path)
        atomic_write_json(history_path, payload)
        atomic_write_json(self.path, payload)
        return False


def peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _git_output(project_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()
