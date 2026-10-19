"""Tests for broadcast_steps (Etappe 2b, Schritt 8, PR 8)."""

from datetime import datetime, timezone
import numpy as np
import polars as pl
import pytest

from datualizer_core.campaign.derived import broadcast_steps


def test_broadcast_steps_with_target_df():
    results = pl.DataFrame({
        "step_start": [100.0, 200.0],
        "step_end": [150.0, 250.0],
        "B": [1.2, 3.4],
        "c_in": [50.0, 60.0],
    })
    target = pl.DataFrame({
        "time_master": [90.0, 100.0, 120.0, 150.0, 160.0, 200.0, 220.0, 260.0]
    })
    out = broadcast_steps(results, ["B", "c_in"], target=target, time_col="time_master")
    assert "B" in out.columns
    assert "c_in" in out.columns
    assert len(out) == 8

    # 90.0 is before first step -> null
    assert out["B"][0] is None
    # 100.0 is at step_start -> 1.2
    assert out["B"][1] == pytest.approx(1.2)
    # 120.0 is inside step -> 1.2
    assert out["B"][2] == pytest.approx(1.2)
    # 150.0 is at step_end -> 1.2
    assert out["B"][3] == pytest.approx(1.2)
    # 160.0 is in gap -> null
    assert out["B"][4] is None
    # 200.0 is at step_start 2 -> 3.4
    assert out["B"][5] == pytest.approx(3.4)
    # 220.0 is inside step 2 -> 3.4
    assert out["B"][6] == pytest.approx(3.4)
    # 260.0 is after step 2 -> null
    assert out["B"][7] is None


def test_broadcast_steps_minimal_traces_without_target():
    results = pl.DataFrame({
        "step_start": [100.0, 200.0],
        "step_end": [150.0, 250.0],
        "B": [1.2, 3.4],
    })
    out = broadcast_steps(results, ["B"], target=None, time_col="time_master")
    assert "time_master" in out.columns
    assert "B" in out.columns

    # Must contain start and end points of each step
    t = out["time_master"].to_list()
    b = out["B"].to_list()
    assert 100.0 in t
    assert 150.0 in t
    assert 200.0 in t
    assert 250.0 in t
    # There should be null gap points between 150 and 200
    gap_points = [val for time, val in zip(t, b) if 150.0 < time < 200.0]
    assert len(gap_points) > 0
    assert all(v is None for v in gap_points)


def test_broadcast_steps_with_datetime_columns():
    t_start1 = datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc)
    t_end1 = datetime(2026, 10, 12, 10, 30, tzinfo=timezone.utc)
    t_start2 = datetime(2026, 10, 12, 11, 0, tzinfo=timezone.utc)
    t_end2 = datetime(2026, 10, 12, 11, 30, tzinfo=timezone.utc)

    results = pl.DataFrame({
        "step_start": [t_start1, t_start2],
        "step_end": [t_end1, t_end2],
        "val": [10.0, 20.0],
    })
    target = pl.DataFrame({
        "time": [
            datetime(2026, 10, 12, 9, 59, tzinfo=timezone.utc),
            datetime(2026, 10, 12, 10, 15, tzinfo=timezone.utc),
            datetime(2026, 10, 12, 10, 45, tzinfo=timezone.utc),
            datetime(2026, 10, 12, 11, 15, tzinfo=timezone.utc),
        ]
    })

    out = broadcast_steps(results, ["val"], target=target, time_col="time")
    assert out["val"][0] is None
    assert out["val"][1] == pytest.approx(10.0)
    assert out["val"][2] is None
    assert out["val"][3] == pytest.approx(20.0)


def test_broadcast_steps_validation_errors():
    results = pl.DataFrame({
        "step_start": [10.0],
        "step_end": [20.0],
        "B": [1.0],
    })
    with pytest.raises(ValueError, match="not found in results table"):
        broadcast_steps(results, ["unknown_col"])

    broken_results = pl.DataFrame({"step_start": [10.0], "B": [1.0]})
    with pytest.raises(ValueError, match="Step end column"):
        broadcast_steps(broken_results, ["B"])
