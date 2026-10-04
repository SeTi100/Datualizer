"""Etappe 2: Run-Segmentierung (Datenkatalog P10, P11, P18)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from datualizer_core import DualModeDataset, IngestionConfig, RunConfig, load_csv

FIXTURES = Path(__file__).parent / "fixtures" / "runs_export"
ALL_MESSY = FIXTURES / "runs_export_20261004_123307_ALL_MESSY.csv"


@pytest.fixture(scope="module")
def messy() -> DualModeDataset:
    return load_csv(ALL_MESSY)


def _run(ds: DualModeDataset, run_id: int) -> dict:
    return ds.runs.filter(pl.col("run_id") == run_id).row(0, named=True)


class TestRunSummary:
    def test_runs_in_order_of_appearance(self, messy: DualModeDataset) -> None:
        assert messy.run_columns == ["run_id"]
        assert messy.runs["run_id"].to_list() == list(range(1, 21))

    def test_sample_counts_and_duration(self, messy: DualModeDataset) -> None:
        run2 = _run(messy, 2)
        assert run2["n_samples"] == 11
        assert run2["duration_s"] == pytest.approx(run2["t_end"] - run2["t_start"])
        assert run2["duration_s"] == pytest.approx(10.0, abs=0.5)

    def test_mixed_sampling_rates(self, messy: DualModeDataset) -> None:
        """P11: Runs 1–6 mit 1 Hz, Runs 7–20 mit 10 Hz."""
        assert _run(messy, 1)["median_dt_s"] == pytest.approx(1.0, abs=0.05)
        assert _run(messy, 20)["median_dt_s"] == pytest.approx(0.1, abs=0.01)

    def test_aborted_runs(self, messy: DualModeDataset) -> None:
        """P18: Run 2 (11 Samples) und Run 5 (12) sind Mini-Runs, Run 6 (37) nicht."""
        aborted = messy.runs.filter(pl.col("aborted"))["run_id"].to_list()
        assert aborted == [2, 5]

    def test_parameters_per_run(self, messy: DualModeDataset) -> None:
        assert _run(messy, 11)["param_temperatur"] is not None
        assert _run(messy, 1)["param_temperatur"] is None
        assert "masse" not in messy.runs.columns  # Messkanäle gehören nicht in die Run-Übersicht

    def test_aborted_badge_can_be_disabled(self) -> None:
        ds = load_csv(ALL_MESSY, config=IngestionConfig(runs=RunConfig(aborted_fraction=0)))
        assert not ds.runs["aborted"].any()
        assert ds.ingestion_spec.runs.aborted_fraction == 0

    def test_long_export_has_same_runs_as_wide(self) -> None:
        wide = load_csv(FIXTURES / "runs_export_20261004_115339.csv")
        long = load_csv(FIXTURES / "runs_export_20261004_115351_long.csv")
        assert wide.runs["run_id"].to_list() == long.runs["run_id"].to_list() == [2, 3, 4]

    def test_no_run_column(self) -> None:
        ds = load_csv("time,x\n00:00:00,1.0\n00:00:01,2.0\n")
        assert ds.run_columns == []
        assert ds.runs.is_empty()
        assert ds.channel_availability.is_empty()


class TestChannelAvailability:
    def test_sparse_matrix(self, messy: DualModeDataset) -> None:
        """P10: `V_VOC` existiert nur in Run 1, die Rechenkanäle erst ab Run 4 (ohne Run 17)."""
        avail = messy.channel_availability
        assert avail.filter(pl.col("v_voc") > 0)["run_id"].to_list() == [1]
        filled = avail.filter(pl.col("abgaskonzentration_betriebszustand") > 0)["run_id"].to_list()
        assert filled == [r for r in range(4, 21) if r != 17]

    def test_parameter_presence_per_run(self, messy: DualModeDataset) -> None:
        """P10 für Parameter: `Konzentration` ist nur in Run 9–14 gesetzt."""
        present = messy.runs.filter(pl.col("konzentration").is_not_null())["run_id"].to_list()
        assert present == list(range(9, 15))

    def test_run_without_calculated_values(self, messy: DualModeDataset) -> None:
        """P18: Run 17 hat keinen einzigen Rechenwert."""
        run17 = messy.channel_availability.filter(pl.col("run_id") == 17).row(0, named=True)
        for ch in ("massenstrom_waage", "normkonzentration", "abgaskonzentration_betriebszustand"):
            assert run17[ch] == 0.0
        assert run17["masse"] == 1.0

    def test_only_channels(self, messy: DualModeDataset) -> None:
        assert messy.channel_availability.columns == ["run_id", *messy.channels]


class TestRunTime:
    def test_run_time_restarts_per_run(self, messy: DualModeDataset) -> None:
        rt = messy.run_time()
        assert len(rt) == len(messy)
        starts = messy.df.with_columns(rt.alias("rt")).group_by("run_id").agg(pl.col("rt").min())
        assert starts["rt"].to_list() == [0.0] * 20

    def test_select_run(self, messy: DualModeDataset) -> None:
        before = messy.df.clone()
        sub = messy.select_run((2,))
        assert len(sub) == 11
        assert sub[sub.time_col][0] == 0.0
        assert sub["run_id"].unique().to_list() == [2]
        assert sub.channels == messy.channels
        assert sub.run_columns == ["run_id"]
        assert messy.df.equals(before)  # nicht destruktiv

    def test_select_unknown_run_fails_loudly(self, messy: DualModeDataset) -> None:
        with pytest.raises(KeyError):
            messy.select_run((999,))

    def test_select_run_without_runs_fails(self) -> None:
        ds = load_csv("time,x\n00:00:00,1.0\n00:00:01,2.0\n")
        with pytest.raises(KeyError):
            ds.select_run((1,))
