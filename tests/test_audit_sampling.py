import numpy as np
import pandas as pd

from scripts.run_research_audit import (
    audit_split_boundaries,
    build_spread_relative_summary,
    choose_event_id_sample,
    parquet_columns,
    parquet_num_rows,
    write_final_conclusion,
)


def test_audit_sampling_handles_large_population_without_materialization() -> None:
    population_size = 250_000_000
    sample = choose_event_id_sample(
        num_rows=population_size,
        rng=np.random.default_rng(42),
        sample_size=10,
        min_event_id=100,
        max_future_horizon=50,
        lookback=25,
    )
    assert sample.dtype == np.int64
    assert len(sample) == 10
    assert len(np.unique(sample)) == 10
    assert sample.min() >= 100
    assert sample.max() < population_size - 50


def test_split_boundary_audit_streams_row_groups(tmp_path) -> None:
    n_rows = 300
    split = np.repeat(["train", "validation", "test"], 100)
    boundary = np.zeros(n_rows, dtype=bool)
    boundary[50:100] = True
    boundary[150:200] = True
    feature_complete = np.ones(n_rows, dtype=bool)
    frame = pd.DataFrame(
        {
            "event_id": np.arange(n_rows),
            "timestamp": pd.date_range(
                "2024-03-01",
                periods=n_rows,
                freq="s",
                tz="UTC",
            ),
            "split": split,
            "feature_complete": feature_complete,
            "is_boundary_drop": boundary,
            "model_eligible": feature_complete & ~boundary,
            "y_10": np.tile([-1, 0, 1], 100),
            "y_20": np.tile([0, 1, -1], 100),
            "y_50": np.tile([1, -1, 0], 100),
        }
    )
    path = tmp_path / "model.parquet"
    output = tmp_path / "audit"
    output.mkdir()
    frame.to_parquet(path, index=False, row_group_size=37)

    checks = audit_split_boundaries(
        path,
        output,
        (10, 20, 50),
        50,
    )
    assert all(checks.values())
    boundary_rows = pd.read_csv(output / "split_boundary_rows.csv")
    assert len(boundary_rows) == 100


def test_split_boundary_audit_supports_partitioned_dataset(tmp_path) -> None:
    n_rows = 12
    frame = pd.DataFrame(
        {
            "event_id": np.arange(n_rows),
            "timestamp": pd.date_range(
                "2024-03-01",
                periods=n_rows,
                freq="s",
                tz="UTC",
            ),
            "split": np.repeat(["train", "validation", "test"], 4),
            "feature_complete": True,
            "is_boundary_drop": [
                False,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
            ],
            "model_eligible": [
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
            ],
            "y_2": np.tile([-1, 0, 1], 4),
        }
    )
    path = tmp_path / "model_dataset"
    path.mkdir()
    output = tmp_path / "audit"
    output.mkdir()
    for split in ("train", "validation", "test"):
        frame.loc[frame["split"] == split].to_parquet(
            path / f"{split}.parquet",
            index=False,
        )
    (path / "manifest.json").write_text(
        '{"sidecar": "must not be parsed as parquet"}',
        encoding="utf-8",
    )

    checks = audit_split_boundaries(path, output, (2,), 2)

    assert parquet_num_rows(path) == len(frame)
    assert "event_id" in parquet_columns(path)
    assert all(checks.values())


def test_spread_summary_ignores_partition_manifest(tmp_path) -> None:
    event_ids = np.arange(12, dtype=np.int64)
    timestamps = pd.date_range(
        "2024-03-01",
        periods=len(event_ids),
        freq="s",
        tz="UTC",
    )
    midprice = np.array(
        [100.0, 100.0, 100.0, 100.0, 100.0, 101.0,
         101.0, 100.0, 102.0, 101.0, 101.0, 101.0],
        dtype=np.float64,
    )
    events = pd.DataFrame(
        {
            "event_id": event_ids,
            "timestamp": timestamps,
            "bid_price": midprice - 0.5,
            "ask_price": midprice + 0.5,
            "midprice": midprice,
        }
    )
    events_path = tmp_path / "events.parquet"
    events.to_parquet(events_path, index=False)

    model_path = tmp_path / "model_dataset"
    model_path.mkdir()
    split_ranges = {
        "train": np.arange(0, 2),
        "validation": np.arange(2, 4),
        "test": np.arange(4, 8),
    }
    for split, indices in split_ranges.items():
        frame = pd.DataFrame(
            {
                "event_id": indices,
                "timestamp": timestamps[indices],
                "split": split,
                "model_eligible": True,
                "y_2": np.sign(
                    midprice[indices + 2] - midprice[indices]
                ).astype(np.int8),
            }
        )
        frame.to_parquet(model_path / f"{split}.parquet", index=False)
    (model_path / "manifest.json").write_text(
        '{"sidecar": "must not be parsed as parquet"}',
        encoding="utf-8",
    )

    output = tmp_path / "audit"
    output.mkdir()
    result = build_spread_relative_summary(
        model_path,
        events_path,
        output,
        (2,),
    )

    assert result["test_event_ids_contiguous"]
    assert result["events_end_covers_future_horizon"]
    assert (
        pd.read_csv(output / "spread_relative_summary.csv")["n_obs"].max()
        == 4
    )


def test_successful_audit_conclusion_is_version_neutral(tmp_path) -> None:
    precision_status = {
        "log_feature_precision_issue_found": False,
        "precision_issue_interpretation": (
            "strict float64 manual recomputation matched"
        ),
    }
    conclusion = write_final_conclusion(
        output_dir=tmp_path,
        experiment_id="integration",
        critical_audit_passed=True,
        failed_critical_checks=[],
        precision_status=precision_status,
        spread_info={},
        horizons=(10, 20, 50),
    )

    assert "v2" not in conclusion["next_step"]
    assert conclusion["main_caveat"] != "pass"
    assert "sampled audit scope" in conclusion["main_caveat"]
