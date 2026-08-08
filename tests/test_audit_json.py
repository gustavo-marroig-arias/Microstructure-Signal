from __future__ import annotations

import numpy as np

from scripts.run_research_audit import json_safe_scalar


def test_json_safe_scalar_normalizes_numpy_and_nonfinite_values() -> None:
    assert json_safe_scalar(np.int64(3)) == 3
    assert json_safe_scalar(np.float64(1.5)) == 1.5
    assert json_safe_scalar(np.nan) is None
    assert json_safe_scalar(float("inf")) is None
    assert json_safe_scalar("pass") == "pass"
