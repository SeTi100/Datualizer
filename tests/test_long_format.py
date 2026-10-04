"""Etappe 2, Schritt 2: Long-Format erkennen und nach wide pivotieren (Datenkatalog P1, P19)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from datualizer_core import ColumnKind, load_csv

FIXTURES = Path(__file__).parent / "fixtures" / "runs_export"

LONG_CSV = (
    "timestamp,run_id,phase_status,roi_name,parsed_value,unit,is_calculated\n"
    "2026-10-04T11:00:00,1,IDLE,Masse,10.0,g,0\n"
    "2026-10-04T11:00:00,1,IDLE,Massenstrom (g/s),0.5,g/s,1\n"
    "2026-10-04T11:00:01,1,IDLE,Masse,9.5,g,0\n"
    "2026-10-04T11:00:01,1,IDLE,Massenstrom (g/s),,g/s,1\n"
    "2026-10-04T11:00:02,2,STAGE_1,Masse,9.0,g,0\n"
)


class TestDetection:
    def test_long_csv_is_pivoted(self) -> None:
        ds = load_csv(LONG_CSV)
        assert ds.source_format == "long"
        assert ds.shape == (3, 5)
        assert ds.columns == ["time_seconds", "run_id", "phase_status", "masse", "massenstrom"]

    def test_wide_csv_with_label_column_is_not_pivoted(self) -> None:
        csv = "time,channel,value\n00:00:00,A,1.0\n00:00:01,A,2.0\n"
        ds = load_csv(csv)
        assert ds.source_format == "wide"

    def test_pivot_can_be_disabled(self) -> None:
        ds = load_csv(LONG_CSV, pivot_long=False)
        assert ds.source_format == "long"
        assert "roi_name" in ds.columns
        assert len(ds) == 5

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("runs_export_20261004_115346.csv", "long"),            # long ohne Suffix
            ("runs_export_20261004_115351_long.csv", "long"),
            ("runs_export_20261004_115428_long.csv", "long"),
            ("runs_export_20261004_115519_long.csv", "long"),
            ("runs_export_20261004_123115_konst_T_V_long.csv", "wide"),  # "_long", aber wide
            ("runs_export_20261004_123138_rand_params_long.csv", "wide"),
            ("runs_export_20261004_123316_ALL_MESSY_lang.csv", "wide"),
        ],
    )
    def test_file_name_is_not_trusted(self, name: str, expected: str) -> None:
        assert load_csv(FIXTURES / name).source_format == expected


class TestPivot:
    def test_values_and_nulls(self) -> None:
        ds = load_csv(LONG_CSV)
        assert ds["masse"].to_list() == [10.0, 9.5, 9.0]
        assert ds["massenstrom"].to_list() == [0.5, None, None]

    def test_row_metadata_and_kinds(self) -> None:
        ds = load_csv(LONG_CSV)
        assert ds["run_id"].to_list() == [1, 1, 2]
        assert ds["phase_status"].to_list() == ["IDLE", "IDLE", "STAGE_1"]
        assert ds.column_kinds["run_id"] is ColumnKind.IDENTIFIER
        assert ds.channels == ["masse", "massenstrom"]

    def test_channel_attributes(self) -> None:
        ds = load_csv(LONG_CSV)
        assert ds.channel_attrs == {
            "masse": {"unit": "g", "is_calculated": 0.0},
            "massenstrom": {"unit": "g/s", "is_calculated": 1.0},
        }

    def test_duplicate_keys_keep_first_and_are_audited(self) -> None:
        csv = LONG_CSV + "2026-10-04T11:00:00,1,IDLE,Masse,99.0,g,0\n"
        ds = load_csv(csv)
        assert ds["masse"][0] == 10.0
        assert [(e.row_index, e.column, e.reason) for e in ds.audit_log] == [
            (5, "masse", "duplicate_long_key")
        ]

    def test_constant_channel_attribute_is_not_row_metadata(self) -> None:
        """115428_long: is_calculated ist überall 0 und muss trotzdem Kanal-Attribut bleiben."""
        ds = load_csv(FIXTURES / "runs_export_20261004_115428_long.csv")
        assert "is_calculated" not in ds.columns
        assert ds.channel_attrs["temperatur"]["unit"] == "°C"
        assert ds.channel_attrs["masse"]["is_calculated"] == 0.0


class TestWideLongEquivalence:
    """Gleiche Messung, zwei Exportformate → identische Wide-Tabelle (auf der zeitlichen Überlappung)."""

    @pytest.mark.parametrize(
        ("wide_name", "long_name"),
        [
            ("runs_export_20261004_115512.csv", "runs_export_20261004_115519_long.csv"),
            ("runs_export_20261004_115339.csv", "runs_export_20261004_115346.csv"),
            ("runs_export_20261004_115422.csv", "runs_export_20261004_115428_long.csv"),
        ],
    )
    def test_same_channels_and_values(self, wide_name: str, long_name: str) -> None:
        wide = load_csv(FIXTURES / wide_name)
        long = load_csv(FIXTURES / long_name)
        assert set(wide.channels) <= set(long.channels)

        cols = ["time_seconds", "run_id", "phase_status", *sorted(wide.channels)]
        n = min(len(wide), len(long))
        assert wide.df.select(cols).head(n).equals(long.df.select(cols).head(n))

    def test_long_dataset_plots_physical_channels(self) -> None:
        ds = load_csv(FIXTURES / "runs_export_20261004_115519_long.csv")
        assert "parsed_value" not in ds.channels
        assert {"masse", "massenstrom_waage", "normkonzentration"} <= set(ds.channels)
        assert ds["experiment_name"].unique().to_list() == ["250"]
        assert ds.df["time_seconds"].is_unique().all()
        assert ds.df.schema["masse"] == pl.Float64
