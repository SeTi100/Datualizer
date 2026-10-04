"""Etappe 2: Qualitäts-Flags (Datenkatalog P12–P17). Markieren, nie verändern."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from datualizer_core import (
    DualModeDataset,
    IngestionConfig,
    QualityConfig,
    QualityConfigError,
    load_csv,
    load_csvs,
)

FIXTURES = Path(__file__).parent / "fixtures" / "runs_export"
ALL_MESSY = FIXTURES / "runs_export_20261004_123307_ALL_MESSY.csv"
LONG = FIXTURES / "runs_export_20261004_115351_long.csv"


@pytest.fixture(scope="module")
def messy() -> DualModeDataset:
    return load_csv(ALL_MESSY)


def _rows(ds: DualModeDataset, flag: str, channel: str | None = None) -> list[int]:
    ev = ds.quality_flags.filter(pl.col("flag") == flag)
    ev = ev.filter(pl.col("channel").is_null() if channel is None else pl.col("channel") == channel)
    return ev["row"].to_list()


class TestFixtureFindings:
    def test_session_gaps(self, messy: DualModeDataset) -> None:
        """P12: genau die zwei Sprünge > 5 s (30.09. → 04.10., 11:55 → 12:23)."""
        assert _rows(messy, "gap") == [536, 900]
        assert messy["run_id"][536] == 2 and messy["run_id"][900] == 7

    def test_dropout_as_zero(self, messy: DualModeDataset) -> None:
        """P13: `Masse = 0.0` für 12 Samples, davor ein Leerfeld. Markiert, nicht verändert."""
        assert _rows(messy, "dropout", "masse") == list(range(5, 17))
        assert messy["masse"][5:17].to_list() == [0.0] * 12
        assert _rows(messy, "missing", "masse") == [4]

    def test_refill_jump(self, messy: DualModeDataset) -> None:
        """P14: Nachfüll-Sprung 162,7 → 172,9 g ist ein Sprung, kein Dropout."""
        jumps = _rows(messy, "jump", "masse")
        assert 269 in jumps
        assert messy["masse"][268] == pytest.approx(162.704)
        assert messy["masse"][269] == pytest.approx(172.945)

    def test_stuck_sensor(self, messy: DualModeDataset) -> None:
        """P15: `Masse` 162.704 über ~65 s konstant (Run 1)."""
        stuck = set(_rows(messy, "stuck", "masse"))
        assert set(range(200, 269)) <= stuck
        assert 199 not in stuck and 269 not in stuck

    def test_frozen_sensor_across_runs_then_catch_up(self, messy: DualModeDataset) -> None:
        """Neuer Befund: `Masse` hängt in Run 15–18 ~109 s fest und springt dann um −15 g."""
        stuck = set(_rows(messy, "stuck", "masse"))
        assert {3446, 4531} <= stuck
        assert 4532 in _rows(messy, "jump", "masse")

    def test_out_of_range_needs_user_range(self, messy: DualModeDataset) -> None:
        """P17: Ohne Vorgabe kein Plausibilitätsurteil; mit Band 454 negative Werte markiert."""
        assert not _rows(messy, "out_of_range", "massenstrom_waage")
        cfg = IngestionConfig(quality=QualityConfig(ranges={"Massenstrom Waage (g/s)": (0.0, None)}))
        ds = load_csv(ALL_MESSY, config=cfg)
        rows = _rows(ds, "out_of_range", "massenstrom_waage")
        assert len(rows) == 454
        assert ds["massenstrom_waage"].len() == messy["massenstrom_waage"].len()  # nichts gelöscht
        assert ds.ingestion_spec.quality.ranges == {"massenstrom_waage": (0.0, None)}

    def test_calculated_nulls_kept_apart(self) -> None:
        """P16: Leere Rechenwerte sind `missing_calculated`, nicht `missing` (Long-Attribut)."""
        ds = load_csv(LONG)
        assert ds.ingestion_spec.quality.calculated_channels == [
            "massenstrom_waage", "normkonzentration", "abgaskonzentration_betriebszustand"
        ]
        flags = set(ds.quality_flags.filter(pl.col("channel") == "normkonzentration")["flag"])
        assert flags == {"missing_calculated"}

    def test_calculated_channels_for_wide_data(self) -> None:
        cfg = IngestionConfig(quality=QualityConfig(calculated_channels=["Normkonzentration (g/Nm³)"]))
        ds = load_csv(ALL_MESSY, config=cfg)
        assert _rows(ds, "missing_calculated", "normkonzentration")
        assert not _rows(ds, "missing", "normkonzentration")

    def test_sparse_channels_are_not_missing(self, messy: DualModeDataset) -> None:
        """P10: Ein Kanal, den es in einem Run gar nicht gibt, ist dort nicht „fehlend“."""
        v_voc_missing = set(_rows(messy, "missing", "v_voc"))
        assert all(messy["run_id"][r] == 1 for r in v_voc_missing)
        assert not _rows(messy, "missing", "snad")  # komplett leer (P9), keine Flut


class TestApi:
    def test_data_unchanged(self, messy: DualModeDataset) -> None:
        before = messy.df.clone()
        _ = messy.quality_flags, messy.with_flag_columns()
        assert messy.df.equals(before)

    def test_flag_mask(self, messy: DualModeDataset) -> None:
        mask = messy.flag_mask(["dropout"], channel="masse")
        assert len(mask) == len(messy)
        assert mask.arg_true().to_list() == list(range(5, 17))
        assert messy.flag_mask(["gap"]).sum() == 2

    def test_flag_columns(self, messy: DualModeDataset) -> None:
        wide = messy.with_flag_columns()
        assert wide.columns[: messy.df.width] == messy.columns
        assert wide["masse_flags"][5] == "dropout"
        assert wide["time_seconds_flags"][536] == "gap"
        assert wide["masse_flags"][0] is None

    def test_summary(self, messy: DualModeDataset) -> None:
        row = messy.quality_summary.filter(
            (pl.col("channel") == "masse") & (pl.col("flag") == "dropout")
        ).row(0, named=True)
        assert row["n"] == 12

    def test_select_run_has_own_flags(self, messy: DualModeDataset) -> None:
        run1 = messy.select_run((1,))
        assert _rows(run1, "dropout", "masse") == list(range(5, 17))
        assert not _rows(run1, "gap")

    def test_detection_off(self) -> None:
        ds = load_csv(ALL_MESSY, config=IngestionConfig(quality=QualityConfig(detect=False)))
        assert ds.quality_flags.is_empty()

    def test_in_memory_dataset_without_settings(self) -> None:
        ds = DualModeDataset(pl.DataFrame({"time_seconds": [0.0, 1.0], "x": [1.0, None]}))
        assert ds.quality_flags.is_empty()


class TestConfig:
    def test_spec_holds_resolved_thresholds_and_replays(self, messy: DualModeDataset) -> None:
        spec = messy.ingestion_spec
        assert set(spec.quality.jump_thresholds) == set(messy.channels)
        assert spec.quality.jump_thresholds["masse"] > 1.0
        again = load_csv(ALL_MESSY, config=spec)
        assert again.ingestion_spec == spec
        assert again.quality_flags.equals(messy.quality_flags)

    def test_explicit_jump_threshold_wins(self) -> None:
        cfg = IngestionConfig(quality=QualityConfig(jump_thresholds={"Masse": 0.0}))
        ds = load_csv(ALL_MESSY, config=cfg)
        assert not _rows(ds, "jump", "masse") and not _rows(ds, "dropout", "masse")
        assert ds.ingestion_spec.quality.jump_thresholds["masse"] == 0.0

    def test_thresholds_are_configurable(self) -> None:
        cfg = IngestionConfig(quality=QualityConfig(gap_factor=0, stuck_min_duration_s=0))
        ds = load_csv(ALL_MESSY, config=cfg)
        assert not _rows(ds, "gap")
        assert ds.quality_flags.filter(pl.col("flag") == "stuck").is_empty()

    def test_unknown_channel_fails_loudly(self) -> None:
        cfg = IngestionConfig(quality=QualityConfig(ranges={"Gibtsnicht": (0.0, 1.0)}))
        with pytest.raises(QualityConfigError, match="Gibtsnicht"):
            load_csv(ALL_MESSY, config=cfg)

    def test_parameter_is_not_a_channel(self) -> None:
        cfg = IngestionConfig(quality=QualityConfig(ranges={"Rotameter": (0.0, 1.0)}))
        with pytest.raises(QualityConfigError):
            load_csv(ALL_MESSY, config=cfg)

    def test_merge_keeps_quality_settings(self) -> None:
        """Ein Kanal, den es nur in einer der Dateien gibt, darf im Merge konfiguriert werden."""
        files = [FIXTURES / "runs_export_20261004_115422.csv", FIXTURES / "runs_export_20261004_115339.csv"]
        cfg = IngestionConfig(quality=QualityConfig(ranges={"Massenstrom Waage (g/s)": (0.0, None)}))
        ds = load_csvs(files, config=cfg)
        assert ds.ingestion_spec.quality.ranges == {"massenstrom_waage": (0.0, None)}


class TestSynthetic:
    def test_stuck_ignores_short_repeats(self) -> None:
        text = "time,x\n" + "".join(f"00:00:{s:02d},{1.0 if s < 20 else s}\n" for s in range(40))
        ds = load_csv(text)
        assert not _rows(ds, "stuck", "x")  # 20 s gleich < 30 s
        ds = load_csv(text, config=IngestionConfig(quality=QualityConfig(stuck_min_duration_s=10)))
        assert _rows(ds, "stuck", "x") == list(range(20))

    def test_gap_inside_run(self) -> None:
        text = "time,x\n00:00:00,1\n00:00:01,2\n00:00:02,3\n00:01:00,4\n00:01:01,5\n"
        assert _rows(load_csv(text), "gap") == [3]
