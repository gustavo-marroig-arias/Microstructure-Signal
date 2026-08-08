from __future__ import annotations

import sys

from scripts import run_full_logistic


def test_full_logistic_cli_uses_benchmarked_solver_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_logistic.py",
            "--start",
            "2024-03-01",
            "--end",
            "2024-03-01",
        ],
    )

    args = run_full_logistic.parse_args()

    assert args.solver == "lbfgs"
    assert args.max_iter == 200
    assert not args.allow_nonconvergence
