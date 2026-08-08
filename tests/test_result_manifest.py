from __future__ import annotations

import pytest

from scripts.write_current_result_manifest import artifact_size_mb


def test_artifact_size_mb_supports_files_and_directories(tmp_path) -> None:
    file_path = tmp_path / "artifact.bin"
    file_path.write_bytes(b"x" * 1024)

    directory = tmp_path / "partitioned"
    directory.mkdir()
    (directory / "train.parquet").write_bytes(b"x" * 1024)
    (directory / "validation.parquet").write_bytes(b"x" * 2048)

    assert artifact_size_mb(file_path) == pytest.approx(1024 / 1024**2)
    assert artifact_size_mb(directory) == pytest.approx(3072 / 1024**2)
    assert artifact_size_mb(tmp_path / "missing") is None
