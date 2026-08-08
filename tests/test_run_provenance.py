import json
from pathlib import Path

import pytest

from src.run_provenance import RunRecorder, capture_source_state


def test_source_tree_hash_covers_untracked_source_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    module = source / "module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    first = capture_source_state(tmp_path)
    module.write_text("VALUE = 2\n", encoding="utf-8")
    second = capture_source_state(tmp_path)

    assert first["git_commit"] is None
    assert first["source_tree_sha256"] != second["source_tree_sha256"]


def test_run_recorder_persists_failure_metadata(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    metadata_path = tmp_path / "run.json"

    with pytest.raises(RuntimeError, match="deliberate"):
        with RunRecorder(
            metadata_path,
            project_root=tmp_path,
            stage="test",
            arguments={"value": 1},
        ) as recorder:
            recorder.set_dataset_sha256("a" * 64)
            recorder.set_experiment_fingerprint("protocol")
            raise RuntimeError("deliberate failure")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["dataset_sha256"] == "a" * 64
    assert payload["experiment_fingerprint"] == "protocol"
    assert payload["elapsed_seconds"] >= 0
    assert payload["peak_rss_bytes"] > 0
    assert payload["error"]["type"] == "RuntimeError"
    history_path = Path(payload["history_path"])
    assert history_path.exists()
    assert json.loads(history_path.read_text(encoding="utf-8")) == payload


def test_run_recorder_preserves_each_invocation(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    metadata_path = tmp_path / "run.json"

    for value in (1, 2):
        with RunRecorder(
            metadata_path,
            project_root=tmp_path,
            stage="test",
            arguments={"value": value},
        ):
            pass

    history = list((tmp_path / "history" / "run").glob("*.json"))
    assert len(history) == 2
    latest = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert latest["arguments"] == {"value": 2}
