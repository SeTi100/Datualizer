"""Absolute time axis (Etappe 2b, step 1): time origin, time zones, DST, replay.

Several devices of one measurement campaign record in parallel; to put them on one time axis,
a dataset must keep where its relative `time_seconds` start in absolute time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from datualizer_core import IngestionConfig, TimeConfig, TimeConfigError, TimeFormat, load_csv, load_csvs

FIXTURES = Path(__file__).parent / "fixtures" / "runs_export"
WIDE_ISO = FIXTURES / "runs_export_20261004_115339.csv"


def _abs(ds) -> list[datetime]:
    return ds.time_abs.to_list()


class TestDatetimeSources:
    def test_iso_fixture_keeps_first_timestamp_as_origin(self) -> None:
        ds = load_csv(WIDE_ISO)
        assert ds.time_origin == datetime(2026, 10, 4, 11, 49, 39, 12072)
        assert ds.time_zone is None
        assert ds.ingestion_spec.time.format is TimeFormat.DATETIME

    def test_time_abs_reproduces_the_raw_timestamps(self) -> None:
        raw = pl.read_csv(WIDE_ISO, infer_schema_length=0)["timestamp"].head(50).to_list()
        ds = load_csv(WIDE_ISO)
        expected = [datetime.fromisoformat(s) for s in raw]
        assert _abs(ds)[:50] == expected

    def test_german_device_format_tab_separated(self) -> None:
        # Layout of the Testa iFID .dat export
        dat = (
            "Time\tConc [mgC]\tSignal\n"
            "12.10.2026 10:32:52\t12.89\t15.997\n"
            "12.10.2026 10:32:53\t12.92\t16.016\n"
            "12.10.2026 10:33:04\t12.84\t15.925\n"
        )
        ds = load_csv(dat)
        assert ds["time_seconds"].to_list() == [0.0, 1.0, 12.0]
        assert _abs(ds) == [
            datetime(2026, 10, 12, 10, 32, 52),
            datetime(2026, 10, 12, 10, 32, 53),
            datetime(2026, 10, 12, 10, 33, 4),
        ]

    def test_with_time_abs_adds_column_after_time(self) -> None:
        ds = load_csv("time,v\n2026-10-12 10:00:00,1\n2026-10-12 10:00:05,2\n")
        df = ds.with_time_abs()
        assert df.columns[:2] == ["time_seconds", "time_abs"]
        assert df["time_abs"].dtype == pl.Datetime("us")
        assert "time_abs" not in ds.columns  # the dataset itself is unchanged

    def test_extra_datetime_formats_are_configurable(self) -> None:
        csv = "time;v\n12.10.26 10:00;1\n12.10.26 10:02;2\n"
        with pytest.raises(TimeConfigError):
            load_csv(csv, config=IngestionConfig(time=TimeConfig(format=TimeFormat.DATETIME)))
        cfg = IngestionConfig(time=TimeConfig(datetime_formats=["%d.%m.%y %H:%M"]))
        ds = load_csv(csv, config=cfg)
        assert ds["time_seconds"].to_list() == [0.0, 120.0]
        assert ds.time_origin == datetime(2026, 10, 12, 10, 0)


class TestRelativeSources:
    # Layout of a PicoLog CSV export: only elapsed time, decimal comma
    PICO = (
        '"","Kanal 1 Kat_oben Letzte (C)"\n'
        '"00:00:00","70,172"\n'
        '"00:00:01","70,605"\n'
        '"00:00:02","71,025"\n'
    )

    def test_elapsed_time_has_no_absolute_axis_without_origin(self) -> None:
        ds = load_csv(self.PICO)
        assert ds.time_origin is None
        assert ds.time_abs is None
        assert ds.ingestion_spec.time.format is TimeFormat.CLOCK

    def test_origin_places_elapsed_time_on_absolute_axis(self) -> None:
        cfg = IngestionConfig(time=TimeConfig(origin=datetime(2026, 10, 12, 8, 42, 58)))
        ds = load_csv(self.PICO, config=cfg)
        assert _abs(ds) == [
            datetime(2026, 10, 12, 8, 42, 58),
            datetime(2026, 10, 12, 8, 42, 59),
            datetime(2026, 10, 12, 8, 43, 0),
        ]

    def test_numeric_seconds_with_origin(self) -> None:
        cfg = IngestionConfig(time=TimeConfig(origin=datetime(2026, 1, 1, 12, 0)))
        ds = load_csv("t,v\n10,1\n10.5,2\n", config=cfg)
        assert ds.ingestion_spec.time.format is TimeFormat.NUMERIC
        assert _abs(ds) == [datetime(2026, 1, 1, 12, 0), datetime(2026, 1, 1, 12, 0, 0, 500000)]

    def test_origin_conflicts_with_datetime_data(self) -> None:
        cfg = IngestionConfig(time=TimeConfig(origin=datetime(2026, 1, 1)))
        with pytest.raises(TimeConfigError):
            load_csv("time,v\n2026-10-12 10:00:00,1\n", config=cfg)


class TestTimeZones:
    # Europe/Zurich: summer time starts 2026-03-29 02:00 -> 03:00, ends 2026-10-25 03:00 -> 02:00
    SPRING = "time,v\n2026-03-29 01:59:00,1\n2026-03-29 03:01:00,2\n"

    def test_naive_wall_clock_does_not_depend_on_the_machine(self) -> None:
        ds = load_csv(self.SPRING)
        assert ds["time_seconds"].to_list() == [0.0, 3720.0]
        assert ds.time_zone is None

    def test_configured_zone_gives_true_elapsed_seconds(self) -> None:
        ds = load_csv(self.SPRING, config=IngestionConfig(time=TimeConfig(timezone="Europe/Zurich")))
        assert ds["time_seconds"].to_list() == [0.0, 120.0]
        assert ds.time_zone == "Europe/Zurich"
        assert ds.time_abs.dtype == pl.Datetime("us", "Europe/Zurich")
        assert [t.hour for t in _abs(ds)] == [1, 3]

    def test_nonexistent_local_time_is_audited(self) -> None:
        csv = "time,v\n2026-03-29 01:59:00,1\n2026-03-29 02:30:00,2\n2026-03-29 03:01:00,3\n"
        ds = load_csv(csv, config=IngestionConfig(time=TimeConfig(timezone="Europe/Zurich")))
        assert ds["time_seconds"].to_list() == [0.0, None, 120.0]
        assert [e.reason for e in ds.audit_log] == ["nonexistent_local_time"]

    def test_ambiguous_local_time_is_audited_and_kept(self) -> None:
        csv = "time,v\n2026-10-25 01:30:00,1\n2026-10-25 02:30:00,2\n"
        ds = load_csv(csv, config=IngestionConfig(time=TimeConfig(timezone="Europe/Zurich")))
        assert ds["time_seconds"].to_list() == [0.0, 3600.0]  # earliest reading of 02:30
        assert [(e.row_index, e.reason) for e in ds.audit_log] == [(1, "ambiguous_local_time")]

    def test_offsets_in_the_data_are_respected(self) -> None:
        csv = "time,v\n2026-10-12T10:00:00+02:00,1\n2026-10-12T08:00:30Z,2\n"
        ds = load_csv(csv)
        assert ds["time_seconds"].to_list() == [0.0, 30.0]
        assert ds.time_origin == datetime(2026, 10, 12, 8, 0, tzinfo=timezone.utc)
        assert ds.time_zone == "UTC"

    def test_mixed_offset_and_naive_values_are_audited(self) -> None:
        csv = "time,v\n2026-10-12T10:00:00+02:00,1\n2026-10-12 10:00:05,2\n2026-10-12T10:00:10+02:00,3\n"
        ds = load_csv(csv)
        assert ds["time_seconds"].to_list() == [0.0, None, 10.0]
        assert [e.reason for e in ds.audit_log] == ["mixed_timezone"]

    def test_unknown_zone_fails_loudly(self) -> None:
        with pytest.raises(TimeConfigError):
            load_csv(self.SPRING, config=IngestionConfig(time=TimeConfig(timezone="Mars/Olympus")))


class TestForcedFormatAndReplay:
    def test_forced_format_that_does_not_fit_fails_loudly(self) -> None:
        with pytest.raises(TimeConfigError):
            load_csv("time,v\n00:00:01,1\n00:00:02,2\n",
                     config=IngestionConfig(time=TimeConfig(format=TimeFormat.DATETIME)))

    def test_spec_replay_keeps_time_axis(self) -> None:
        cfg = IngestionConfig(time=TimeConfig(timezone="Europe/Zurich"))
        first = load_csv(WIDE_ISO, config=cfg)
        again = load_csv(WIDE_ISO, config=first.ingestion_spec)
        assert again.ingestion_spec == first.ingestion_spec
        assert again.time_origin == first.time_origin
        assert again.time_abs.equals(first.time_abs)

    def test_merge_keeps_origin_of_the_combined_table(self) -> None:
        a = "time,v\n2026-10-12 10:00:00,1\n2026-10-12 10:00:01,2\n"
        b = "time,v\n2026-10-12 10:00:01,2\n2026-10-12 10:00:02,3\n"
        ds = load_csvs([a, b])
        assert ds.time_origin == datetime(2026, 10, 12, 10, 0)
        assert _abs(ds)[-1] == datetime(2026, 10, 12, 10, 0, 2)


class TestSelectRun:
    def test_run_selection_moves_the_origin_to_the_run_start(self) -> None:
        ds = load_csv(FIXTURES / "runs_export_20261004_123307_ALL_MESSY.csv")
        key = tuple(ds.runs.row(1, named=True)[c] for c in ds.run_columns)
        run = ds.select_run(key)
        assert run["time_seconds"][0] == 0.0
        assert run.time_abs[0] == run.time_origin
        full = ds.with_time_abs()
        mask = pl.all_horizontal(pl.col(c) == v for c, v in zip(ds.run_columns, key))
        assert run.time_abs.to_list() == full.filter(mask)["time_abs"].to_list()
