"""Etappe 2: Alles, was die Automatik entscheidet, ist per IngestionConfig einstell- und reproduzierbar."""

from __future__ import annotations

from pathlib import Path

import pytest

from datualizer_core import (
    ColumnKind,
    IngestionConfig,
    LongFormatConfig,
    LongFormatMode,
    Vocabulary,
    load_csv,
)
from datualizer_core.ingestion.long_format import LongFormatError

FIXTURES = Path(__file__).parent / "fixtures" / "runs_export"

# Long-Tabelle mit Bezeichnungen, die kein Default-Vokabular kennt
UNKNOWN_LONG = (
    "Zeitpunkt;Versuch;Groesse;Betrag;Masseinheit\n"
    "00:00:00;7;Druck;1,5;bar\n"
    "00:00:00;7;Temp;20,0;°C\n"
    "00:00:01;7;Druck;1,6;bar\n"
    "00:00:01;7;Temp;20,5;°C\n"
)


class TestUnknownNamingConventions:
    def test_not_detected_without_configuration(self) -> None:
        ds = load_csv(UNKNOWN_LONG)
        assert ds.source_format == "wide"

    def test_explicit_roles(self) -> None:
        cfg = IngestionConfig(
            column_kinds={"Versuch": ColumnKind.IDENTIFIER},
            long_format=LongFormatConfig(
                variable_col="Groesse", value_col="Betrag", channel_attr_cols=["Masseinheit"]
            ),
        )
        ds = load_csv(UNKNOWN_LONG, config=cfg)
        assert ds.source_format == "long"
        assert ds.columns == ["time_seconds", "versuch", "druck", "temp"]
        assert ds["druck"].to_list() == [1.5, 1.6]
        assert ds.channels == ["druck", "temp"]
        assert ds.channel_attrs == {"druck": {"masseinheit": "bar"}, "temp": {"masseinheit": "°C"}}

    def test_extended_vocabulary(self) -> None:
        vocab = Vocabulary()
        vocab.time_names.append("zeitpunkt")
        vocab.long_variable_tokens.append("groesse")
        vocab.long_value_tokens.append("betrag")
        vocab.channel_attr_tokens.append("masseinheit")
        vocab.metadata_tokens.append("versuch")
        vocab.identifier_tokens.append("versuch")
        ds = load_csv(UNKNOWN_LONG, config=IngestionConfig(vocabulary=vocab))
        assert ds.source_format == "long"
        assert ds.column_kinds["versuch"] is ColumnKind.IDENTIFIER
        assert ds.channel_attrs["temp"] == {"masseinheit": "°C"}

    def test_force_without_resolvable_columns_fails_loudly(self) -> None:
        cfg = IngestionConfig(long_format=LongFormatConfig(mode=LongFormatMode.FORCE))
        with pytest.raises(LongFormatError):
            load_csv(UNKNOWN_LONG, config=cfg)

    def test_misconfigured_column_fails_loudly(self) -> None:
        cfg = IngestionConfig(long_format=LongFormatConfig(variable_col="Gibts_nicht", value_col="Betrag"))
        with pytest.raises(LongFormatError):
            load_csv(UNKNOWN_LONG, config=cfg)


class TestOverrides:
    def test_column_kind_override_beats_heuristic(self) -> None:
        cfg = IngestionConfig(column_kinds={"Konzentration": ColumnKind.CATEGORICAL})
        ds = load_csv(FIXTURES / "runs_export_20261004_123307_ALL_MESSY.csv", config=cfg)
        assert len(ds.audit_log) == 0
        assert "20 °C" in ds["konzentration"].to_list()

    def test_override_by_cleaned_name(self) -> None:
        cfg = IngestionConfig(column_kinds={"konzentration": ColumnKind.CATEGORICAL})
        ds = load_csv(FIXTURES / "runs_export_20261004_123307_ALL_MESSY.csv", config=cfg)
        assert ds.column_kinds["konzentration"] is ColumnKind.CATEGORICAL

    def test_forced_identifier_audits_non_integers(self) -> None:
        csv = "time,probe\n00:00:00,1\n00:00:01,A7\n"
        ds = load_csv(csv, config=IngestionConfig(column_kinds={"probe": ColumnKind.IDENTIFIER}))
        assert ds["probe"].to_list() == [1, None]
        assert [(e.row_index, e.reason) for e in ds.audit_log] == [(1, "non_convertible_int")]

    def test_numeric_threshold(self) -> None:
        csv = "time,x\n00:00:00,1\n00:00:01,a\n00:00:02,b\n"
        assert load_csv(csv).column_kinds["x"] is ColumnKind.CATEGORICAL
        cfg = IngestionConfig(numeric_ratio_threshold=0.3)
        assert load_csv(csv, config=cfg).column_kinds["x"] is ColumnKind.NUMERIC

    def test_time_column_via_column_kinds(self) -> None:
        csv = "nr,stamp,x\n1,00:00:00,1.0\n2,00:00:05,2.0\n"
        ds = load_csv(csv, config=IngestionConfig(column_kinds={"stamp": ColumnKind.TIME}))
        assert ds["time_seconds"].to_list() == [0.0, 5.0]
        assert ds.ingestion_spec.time_column == "stamp"

    def test_long_format_off(self) -> None:
        cfg = IngestionConfig(long_format=LongFormatConfig(mode=LongFormatMode.OFF))
        ds = load_csv(FIXTURES / "runs_export_20261004_115519_long.csv", config=cfg)
        assert "roi_name" in ds.columns

    def test_column_varying_per_time_and_channel_is_dropped_with_audit(self) -> None:
        csv = (
            "time,roi_name,parsed_value,quality\n"
            "00:00:00,a,1.0,ok\n00:00:00,b,2.0,bad\n"
            "00:00:01,a,1.5,bad\n00:00:01,b,2.5,ok\n"
        )
        ds = load_csv(csv)
        assert ds.columns == ["time_seconds", "a", "b"]
        assert [(e.column, e.reason) for e in ds.audit_log] == [("quality", "dropped_long_column")]


class TestReplayableSpec:
    @pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.csv")), ids=lambda p: p.name)
    def test_spec_reproduces_dataset(self, path: Path) -> None:
        first = load_csv(path)
        spec = IngestionConfig.model_validate_json(first.ingestion_spec.model_dump_json())
        second = load_csv(path, config=spec)

        assert second.df.equals(first.df)
        assert second.column_kinds == first.column_kinds
        assert second.channel_attrs == first.channel_attrs
        assert len(second.audit_log) == len(first.audit_log)
        assert second.ingestion_spec == first.ingestion_spec  # Spec ist ein Fixpunkt

    def test_spec_writes_out_every_decision(self) -> None:
        spec = load_csv(FIXTURES / "runs_export_20261004_115519_long.csv").ingestion_spec
        assert spec.time_column == "timestamp"
        assert spec.column_kinds["roi_name"] is ColumnKind.CATEGORICAL
        assert spec.column_kinds["run_id"] is ColumnKind.IDENTIFIER
        lf = spec.long_format
        assert lf.mode is LongFormatMode.FORCE
        assert (lf.variable_col, lf.value_col) == ("roi_name", "parsed_value")
        assert set(lf.channel_attr_cols) == {"unit", "is_calculated"}
        assert "run_id" in lf.row_meta_cols

    def test_wide_file_freezes_long_format_off(self) -> None:
        spec = load_csv(FIXTURES / "runs_export_20261004_115512.csv").ingestion_spec
        assert spec.long_format.mode is LongFormatMode.OFF

    def test_keyword_shortcuts_still_work(self) -> None:
        ds = load_csv(FIXTURES / "runs_export_20261004_115519_long.csv", pivot_long=False)
        assert ds.ingestion_spec.long_format.pivot is False
        assert "roi_name" in ds.columns
