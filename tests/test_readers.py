"""Readers and source profiles (Etappe 2b, step 2) with extracts of a real measurement campaign.

Fixtures in tests/fixtures/sa_12_10/ (see README there): Testa iFID (.dat, .xls), PicoLog CSV,
instrument_reader SQLite database, OCR CSV of a video, lab journal (.xlsx). Never modified.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import os
from pathlib import Path
import shutil

import pytest

from datualizer_core import (
    IngestionConfig,
    ReaderError,
    SourceProfile,
    TimeConfigError,
    TimeFormat,
    builtin_profiles,
    detect_profile,
    get_profile,
    load_source,
)
from datualizer_core.ingestion.readers import ExcelOptions, SqliteOptions, cell_to_text, read_excel, read_sqlite

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
FID_DAT = SA / "fid_ifid_12_10_extract.dat"
FID_XLS = SA / "fid_ifid_05_10.xls"
FID_CSV = SA / "fid_ifid_17_10_extract.csv"
PICO = SA / "picolog_12_10_extract.csv"
DB = SA / "instrument_reader_12_10_extract.db"
OCR = SA / "ocr_recording_105012_extract.csv"
JOURNAL = SA / "labjournal_12_10_extract.xlsx"
RUNS_EXPORT = Path(__file__).parent / "fixtures" / "runs_export" / "runs_export_20261004_115351_long.csv"


def _journal_profile() -> SourceProfile:
    prof = get_profile("labjournal_xlsx")
    prof.ingestion.time.clock_date = date(2026, 10, 12)
    return prof


class TestProfiles:
    def test_shipped_profiles_are_valid_json(self) -> None:
        names = [p.name for p in builtin_profiles()]
        assert {"testa_ifid_dat", "testa_ifid_xls", "picolog_csv", "instrument_reader_db",
                "instrument_reader_ocr_csv", "labjournal_xlsx"} <= set(names)

    @pytest.mark.parametrize("path, name", [
        (FID_DAT, "testa_ifid_dat"),
        (FID_XLS, "testa_ifid_xls"),
        (FID_CSV, "testa_ifid_csv"),
        (PICO, "picolog_csv"),
        (DB, "instrument_reader_db"),
        (OCR, "instrument_reader_ocr_csv"),
        (JOURNAL, "labjournal_xlsx"),
    ])
    def test_detect_profile(self, path: Path, name: str) -> None:
        assert detect_profile(path).name == name

    def test_unknown_layout_gets_no_profile_and_loads_like_load_csv(self) -> None:
        from datualizer_core import load_csv

        assert detect_profile(RUNS_EXPORT) is None
        ds = load_source(RUNS_EXPORT)
        ref = load_csv(RUNS_EXPORT)
        assert ds.df.equals(ref.df)
        assert ds.source_meta["profile"] is None
        assert ds.source_meta["reader"] == "delimited"

    def test_profile_by_name_and_unknown_name(self) -> None:
        assert load_source(FID_DAT, "testa_ifid_dat").source_meta["profile"] == "testa_ifid_dat"
        with pytest.raises(KeyError):
            load_source(FID_DAT, "no_such_profile")


class TestAnalyzerExports:
    def test_ifid_dat_text_export(self) -> None:
        ds = load_source(FID_DAT)
        assert ds.shape == (300, 11)
        assert ds.time_zone == "Europe/Zurich"
        assert ds.time_abs.dt.replace_time_zone(None)[0] == datetime(2026, 10, 12, 14, 27, 2)
        assert ds.units["conc"] == "mgC"
        # 11 s without samples in the device export (FID time 14:29:05-14:29:16)
        assert ds.quality_summary.filter(ds.quality_summary["flag"] == "gap")["n"].to_list() == [1]

    def test_ifid_csv_export_has_the_columns_of_the_dat_export(self) -> None:
        # from 15.10. the iFID wrote .csv: ';', every value quoted (PR 10a)
        ds = load_source(FID_CSV)
        assert ds.source_meta["profile"] == "testa_ifid_csv"
        assert ds.shape == (30, 11)
        assert ds.channels == load_source(FID_DAT).channels
        assert ds.time_abs.dt.replace_time_zone(None)[0] == datetime(2026, 10, 17, 11, 24, 53)
        assert ds.units["conc"] == "mgC"
        assert ds.df["signal"][0] == pytest.approx(9.127)

    def test_ifid_xls_with_metadata_sheets(self) -> None:
        ds = load_source(FID_XLS)
        assert ds.shape == (6512, 13)
        assert ds.ingestion_spec.time.format is TimeFormat.DATETIME
        assert ds.time_abs.dt.replace_time_zone(None)[0] == datetime(2026, 10, 5, 13, 0, 39)
        meta = ds.source_meta["meta_sheets"]
        assert meta["Kalibrierung"]["Sensitivität"] == "4.21 pA/ppm"
        assert meta["Kalibrierung"]["Prüfgas Signal"] == "215.74 pA"
        assert meta["Einstellungen"]["Response-Faktor"] == "1.000"
        assert meta["Einstellungen"]["Kohlenwasserstoff"] == "Propan"
        assert "Mittelwerte" not in meta  # not listed in the profile
        assert {"konzentration", "signal"} <= set(ds.channels)
        assert ds.units["signal"] == "pA"

    def test_ocr_csv_of_a_video(self) -> None:
        ds = load_source(OCR)
        assert ds.channels == ["seconds_from_start", "mass_g", "is_valid"]
        assert set(ds.metadata_columns) == {"frame", "raw_text"}
        assert ds.time_abs.dt.replace_time_zone(None)[0] == datetime(2026, 10, 12, 10, 58, 1, 911000)


class TestPicoLog:
    def test_origin_from_file_save_time(self, tmp_path: Path) -> None:
        f = tmp_path / PICO.name
        shutil.copyfile(PICO, f)
        saved = datetime(2026, 10, 12, 16, 0, 21, tzinfo=timezone.utc)  # 18:00:21 CEST
        os.utime(f, (saved.timestamp(), saved.timestamp()))
        ds = load_source(f)
        last = ds.df[ds.time_col].max()
        assert ds.time_abs[-1] == saved
        assert (ds.time_abs[-1] - ds.time_abs[0]).total_seconds() == last
        rule = ds.source_meta["time_origin_rule"]
        assert rule["rule"] == "file_mtime_end"
        assert "warning" not in rule

    def test_spec_replay_keeps_the_computed_origin(self, tmp_path: Path) -> None:
        f = tmp_path / PICO.name
        shutil.copyfile(PICO, f)
        ds = load_source(f)
        assert ds.ingestion_spec.time.origin == ds.time_origin
        os.utime(f, (0, 0))  # a later copy changes the file time ...
        again = load_source(f, config=ds.ingestion_spec)  # ... the spec still reproduces the axis
        assert again.time_origin == ds.time_origin
        assert again.df.equals(ds.df)

    def test_logger_dropout_stays_as_empty_rows(self) -> None:
        ds = load_source(PICO)
        assert len(ds) == 160
        empty = ds.df.filter(ds.df["kanal_1_kat_oben"].is_null())
        assert len(empty) == 76  # PicoLog kept counting time while the logger delivered nothing


class TestInstrumentReaderDb:
    def test_long_readings_with_runs_and_experiments(self) -> None:
        ds = load_source(DB)
        assert ds.source_format == "long"
        assert ds.channels == ["masse", "massenstrom_waage"]
        assert ds.run_columns == ["run_id"]
        assert set(ds.runs["run_id"].to_list()) == {9, 10}
        assert ds.channel_attrs["massenstrom_waage"]["unit"] == "g/h"
        assert '"V_MFC": "2"' in ds.df.filter(ds.df["run_id"] == 10)["run_parameters"][0]

    def test_local_timestamps_are_read_in_the_zone(self) -> None:
        ds = load_source(DB)
        assert ds.time_abs.dt.replace_time_zone(None)[0] == datetime(2026, 10, 12, 15, 45, 59, 198114)
        assert ds.time_zone == "Europe/Zurich"

    def test_deleted_ids_are_reported(self) -> None:
        meta = load_source(DB).source_meta
        assert meta["missing_ids"]["experiments"] == [[2, 2]]
        assert meta["missing_ids"]["runs"] == [[4, 7]]
        assert meta["tables"]["runs"] == 25

    def test_invalid_ocr_readings_become_empty_values(self) -> None:
        raw = read_sqlite(DB, SqliteOptions(query="SELECT COUNT(*) AS n FROM readings WHERE is_valid = 0"))
        assert int(raw.df["n"][0]) > 0
        ds = load_source(DB)
        assert ds.quality_summary.filter(
            (ds.quality_summary["channel"] == "masse") & (ds.quality_summary["flag"] == "missing")
        )["n"].to_list() == [36]

    def test_table_and_errors(self) -> None:
        runs = read_sqlite(DB, SqliteOptions(table="runs"))
        assert len(runs.df) == 25
        assert runs.df.dtypes == [runs.df.dtypes[0]] * runs.df.width  # all String
        with pytest.raises(ReaderError, match="tables"):
            read_sqlite(DB)
        with pytest.raises(ReaderError):
            read_sqlite(DB, SqliteOptions(table="nope"))
        with pytest.raises(ReaderError):
            read_sqlite(DB, SqliteOptions(query="SELECT * FROM nope"))
        with pytest.raises(ValueError):
            SqliteOptions(table="runs", query="SELECT 1")

    def test_query_cannot_write(self) -> None:
        with pytest.raises(ReaderError):
            read_sqlite(DB, SqliteOptions(query="DELETE FROM runs"))


class TestLabJournal:
    def test_header_row_is_detected_below_the_notes(self) -> None:
        raw = read_excel(JOURNAL)
        assert raw.meta["header_row"] == 10
        assert raw.df.columns[:7] == ["col_A", "Temperatur", "V_MFC [Nl/min]", "Masse_0", "Masse_1", "Zeit_0", "Zeit_1"]
        assert raw.df["Zeit_0"][1] == "10:53:00"
        assert raw.df["Masse_0"][1] == "9674.45"

    def test_journal_on_absolute_time(self) -> None:
        ds = load_source(JOURNAL, _journal_profile())
        assert ds.ingestion_spec.time.format is TimeFormat.CLOCK
        assert ds.time_abs.dt.replace_time_zone(None)[0] == datetime(2026, 10, 12, 10, 0)
        step = ds.with_time_abs().row(1, named=True)
        assert step["time_abs"].replace(tzinfo=None) == datetime(2026, 10, 12, 10, 53)
        assert step["masse_0"] == 9674.45

    def test_spreadsheet_errors_are_sentinels(self) -> None:
        ds = load_source(JOURNAL, _journal_profile())
        reasons = {(e.raw_value, e.reason) for e in ds.audit_log}
        assert ("#DIV/0!", "sentinel_value") in reasons

    def test_clock_date_rules(self) -> None:
        prof = _journal_profile()
        prof.ingestion.time.origin = datetime(2026, 10, 12, 10, 0)
        with pytest.raises(TimeConfigError):
            load_source(JOURNAL, prof)
        with pytest.raises(TimeConfigError):
            load_source(FID_DAT, config=IngestionConfig(time_column="Time", time={"clock_date": date(2026, 10, 12)}))

    def test_sheet_selection(self) -> None:
        with pytest.raises(ReaderError, match="no sheet"):
            read_excel(JOURNAL, ExcelOptions(sheet="5.10.2026"))
        with pytest.raises(ReaderError, match="several sheets"):
            read_excel(FID_XLS)
        assert read_excel(FID_XLS, ExcelOptions(sheet=4, header_row=1)).meta["sheet"] == "Data01"


class TestCellText:
    def test_typed_cells_become_parseable_text(self) -> None:
        from datetime import time, timedelta

        assert cell_to_text(None) is None
        assert cell_to_text(9674.45) == "9674.45"
        assert cell_to_text(200) == "200"
        assert cell_to_text(True) == "1"
        assert cell_to_text(time(10, 53)) == "10:53:00"
        assert cell_to_text(datetime(2026, 10, 12, 10, 53, 0, 5)) == "2026-10-12 10:53:00.000005"
        assert cell_to_text(timedelta(minutes=5)) == "300.0"
