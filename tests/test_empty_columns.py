"""Etappe 2: Leere Spalten markieren statt löschen (Datenkatalog P9, P19)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from datualizer_core import ColumnKind, DualModeDataset, load_csv

FIXTURES = Path(__file__).parent / "fixtures" / "runs_export"
ALL_MESSY = FIXTURES / "runs_export_20261004_123307_ALL_MESSY.csv"

CSV = (
    "time,run_id,a,leer,halb\n"
    "00:00:00,1,1.0,,5.0\n"
    "00:00:01,1,2.0,,\n"
    "00:00:02,1,3.0,,\n"
    "00:00:03,1,4.0,,\n"
)


class TestFillRatio:
    def test_fill_ratio_per_column(self) -> None:
        ds = load_csv(CSV)
        assert ds.fill_ratio == {"run_id": 1.0, "a": 1.0, "leer": 0.0, "halb": 0.25}

    def test_time_column_is_not_reported(self) -> None:
        assert "time_seconds" not in load_csv(CSV).fill_ratio

    def test_empty_table(self) -> None:
        ds = DualModeDataset(pl.DataFrame(schema={"time_seconds": pl.Float64, "x": pl.Float64}))
        assert ds.fill_ratio == {"x": 0.0}
        assert ds.empty_columns == ["x"]


class TestEmptyColumns:
    def test_empty_column_is_marked_not_dropped(self) -> None:
        ds = load_csv(CSV)
        assert ds.empty_columns == ["leer"]
        assert "leer" in ds.columns
        assert "leer" in ds.channels  # Rolle bleibt Messkanal, nur ohne Werte

    def test_empty_columns_do_not_count_as_audit_errors(self) -> None:
        assert len(load_csv(CSV).audit_log) == 0

    def test_all_messy_snad_is_empty(self) -> None:
        """P9: `snad` hat 0 % Füllgrad, `volumenstrom_voc` ebenfalls."""
        ds = load_csv(ALL_MESSY)
        assert {"snad", "volumenstrom_voc"} <= set(ds.empty_columns)
        assert "masse" not in ds.empty_columns

    def test_long_channel_without_values_is_marked(self) -> None:
        """P19: `Volumenstrom_VOC` hat im Long-Export nur leere `parsed_value`-Zeilen."""
        ds = load_csv(FIXTURES / "runs_export_20261004_115428_long.csv")
        assert ds.source_format == "long"
        assert ds.empty_columns == ["voc_type", "volumenstrom_voc"]  # voc_type: leeres Metadatum (P8)
        assert ds.fill_ratio["volumenstrom_voc"] == 0.0

    def test_wide_and_long_agree(self) -> None:
        wide = load_csv(FIXTURES / "runs_export_20261004_115422.csv")
        long = load_csv(FIXTURES / "runs_export_20261004_115428_long.csv")
        empty_channels = lambda ds: [c for c in ds.channels if c in ds.empty_columns]  # noqa: E731
        assert empty_channels(wide) == empty_channels(long) == ["volumenstrom_voc"]

    def test_empty_column_is_never_a_parameter(self) -> None:
        ds = load_csv(ALL_MESSY)
        assert ds.column_kinds["snad"] is ColumnKind.NUMERIC
