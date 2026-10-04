"""Etappe 2, Schritt 1: Typ-Inferenz pro Spalte (Datenkatalog P4, P5, P6, P9)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from datualizer_core import ColumnKind, DualModeDataset, load_csv
from datualizer_core.ingestion.type_inference import infer_column_kind, is_metadata_name

FIXTURES = Path(__file__).parent / "fixtures" / "runs_export"
SENTINELS = {"", "NA", "ERR", "NAN", "OVERFLOW"}


class TestInferColumnKind:
    def test_numeric_with_some_error_codes_stays_numeric(self) -> None:
        kind = infer_column_kind("Druck", ["1.0", "2.0", "E_05", "3.0"], ".", SENTINELS)
        assert kind is ColumnKind.NUMERIC

    def test_mostly_text_is_categorical(self) -> None:
        kind = infer_column_kind("Zustand", ["IDLE", "STAGE_1", "STAGE_1", "1"], ".", SENTINELS)
        assert kind is ColumnKind.CATEGORICAL

    def test_metadata_name_beats_numeric_values(self) -> None:
        assert infer_column_kind("experiment_name", ["250", "250"], ".", SENTINELS) is ColumnKind.CATEGORICAL

    def test_integer_id_column_is_identifier(self) -> None:
        assert infer_column_kind("run_id", ["5", "5", "6"], ".", SENTINELS) is ColumnKind.IDENTIFIER

    def test_non_integer_id_column_is_categorical(self) -> None:
        assert infer_column_kind("probe_id", ["A1", "A2"], ".", SENTINELS) is ColumnKind.CATEGORICAL

    def test_empty_column_stays_numeric(self) -> None:
        assert infer_column_kind("snad", ["", None, "ERR"], ".", SENTINELS) is ColumnKind.NUMERIC

    def test_german_decimals_count_as_numeric(self) -> None:
        assert infer_column_kind("ch1", ["24,1", "1.013,25"], ",", SENTINELS) is ColumnKind.NUMERIC

    @pytest.mark.parametrize("name", ["phase_status", "roi_name", "unit", "voc_type", "Experiment ID"])
    def test_metadata_names(self, name: str) -> None:
        assert is_metadata_name(name)

    @pytest.mark.parametrize("name", ["Masse", "Kanal 1 Letzte (C)", "target_temperature", "parsed_value"])
    def test_channel_names(self, name: str) -> None:
        assert not is_metadata_name(name)


class TestLoaderWithMixedColumns:
    CSV = (
        "timestamp,run_id,experiment_name,phase_status,Masse\n"
        "2026-10-04T11:54:54,5,250,STAGE_1,166.205\n"
        "2026-10-04T11:54:55,5,250,STAGE_1,ERR\n"
        "2026-10-04T11:54:56,6,250,,166.165\n"
    )

    def test_dtypes_per_kind(self) -> None:
        ds = load_csv(self.CSV)
        assert ds["run_id"].dtype == pl.Int64
        assert ds["experiment_name"].to_list() == ["250", "250", "250"]
        assert ds["phase_status"].to_list() == ["STAGE_1", "STAGE_1", None]
        assert ds["masse"].dtype == pl.Float64

    def test_audit_only_for_numeric_channels(self) -> None:
        ds = load_csv(self.CSV)
        assert [(e.column, e.raw_value) for e in ds.audit_log] == [("masse", "ERR")]

    def test_channels_exclude_metadata(self) -> None:
        ds = load_csv(self.CSV)
        assert ds.channels == ["masse"]
        assert ds.metadata_columns == ["run_id", "experiment_name", "phase_status"]

    def test_to_long_keeps_metadata_as_ids(self) -> None:
        long_df = load_csv(self.CSV).to_long()
        assert long_df.columns == [
            "time_seconds", "run_id", "experiment_name", "phase_status", "channel", "value"
        ]
        assert long_df["channel"].unique().to_list() == ["masse"]

    def test_manual_dataset_derives_kinds_from_dtypes(self) -> None:
        ds = DualModeDataset(pl.DataFrame({"time_seconds": [0.0], "a": [1.0], "tag": ["x"]}))
        assert ds.channels == ["a"]
        assert ds.metadata_columns == ["tag"]


class TestRunsExportCorpus:
    """Regression gegen echte Pipeline-Exporte (siehe docs/DATENKATALOG_RUNS_EXPORT.md)."""

    @pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.csv")), ids=lambda p: p.name)
    def test_no_false_audit_entries_on_text_columns(self, path: Path) -> None:
        ds = load_csv(path)
        text_cols = {"phase_status", "experiment_name", "roi_name", "unit", "voc_type", "run_id"}
        assert not [e for e in ds.audit_log if e.column in text_cols]

    def test_wide_export_column_kinds(self) -> None:
        ds = load_csv(FIXTURES / "runs_export_20261004_115512.csv")
        kinds = ds.column_kinds
        assert kinds["run_id"] is ColumnKind.IDENTIFIER
        assert kinds["experiment_name"] is ColumnKind.CATEGORICAL
        assert kinds["phase_status"] is ColumnKind.CATEGORICAL
        assert ds["experiment_name"].unique().to_list() == ["250"]  # P5: Name bleibt String
        assert len(ds.audit_log) == 0

    def test_long_export_column_kinds(self) -> None:
        ds = load_csv(FIXTURES / "runs_export_20261004_115519_long.csv")
        assert ds.column_kinds["roi_name"] is ColumnKind.CATEGORICAL
        assert ds.column_kinds["unit"] is ColumnKind.CATEGORICAL
        assert ds["roi_name"].n_unique() == 4
        assert "parsed_value" in ds.channels
        assert len(ds.audit_log) == 0

    def test_all_messy_only_audits_value_with_unit(self) -> None:
        """P6: 'Konzentration = 20 °C' ist der einzige echte Konvertierungsfehler im Gesamtexport."""
        ds = load_csv(FIXTURES / "runs_export_20261004_123307_ALL_MESSY.csv")
        audit = ds.audit_log.to_dataframe()
        assert len(audit) == 306
        assert audit["column"].unique().to_list() == ["konzentration"]
        assert audit["raw_value"].unique().to_list() == ["20 °C"]
        assert ds["run_id"].dtype == pl.Int64
        assert ds["run_id"].n_unique() == 20
