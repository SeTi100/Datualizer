"""Etappe 2: Rollen Messwert vs. Parameter (Datenkatalog P7).

Heuristik: Eine numerische Spalte, die innerhalb jedes Runs konstant ist, ist ein Parameter
(Sollwert, Einstellung) und kein Messkanal. Alles ist über `IngestionConfig` überschreibbar.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datualizer_core import ColumnKind, IngestionConfig, RoleConfig, Vocabulary, load_csv
from datualizer_core.ingestion.roles import RoleConfigError

FIXTURES = Path(__file__).parent / "fixtures" / "runs_export"
ALL_MESSY = FIXTURES / "runs_export_20261004_123307_ALL_MESSY.csv"

WIDE_CSV = (
    "time,run_id,sollwert,messwert\n"
    "00:00:00,1,100,20.1\n"
    "00:00:01,1,100,20.4\n"
    "00:00:02,2,180,30.2\n"
    "00:00:03,2,180,30.9\n"
)


class TestHeuristic:
    def test_constant_per_run_becomes_parameter(self) -> None:
        ds = load_csv(WIDE_CSV)
        assert ds.column_kinds["sollwert"] is ColumnKind.PARAMETER
        assert ds.parameters == ["sollwert"]
        assert ds.channels == ["messwert"]

    def test_parameter_values_are_kept_as_float(self) -> None:
        ds = load_csv(WIDE_CSV)
        assert ds["sollwert"].to_list() == [100.0, 100.0, 180.0, 180.0]

    def test_parameters_are_not_melted(self) -> None:
        long_df = load_csv(WIDE_CSV).to_long()
        assert long_df["channel"].unique().to_list() == ["messwert"]
        assert "sollwert" in long_df.columns

    def test_no_run_column_means_no_parameters(self) -> None:
        csv = "time,a,b\n00:00:00,5,1.0\n00:00:01,5,2.0\n"
        ds = load_csv(csv)
        assert ds.parameters == []
        assert ds.ingestion_spec.roles.run_columns == []

    def test_single_sample_per_run_is_no_evidence(self) -> None:
        csv = "time,run_id,x\n00:00:00,1,5.0\n00:00:01,2,6.0\n"
        assert load_csv(csv).channels == ["x"]

    def test_empty_column_is_not_a_parameter(self) -> None:
        csv = "time,run_id,x,leer\n00:00:00,1,1.0,\n00:00:01,1,2.0,\n"
        ds = load_csv(csv)
        assert ds.column_kinds["leer"] is ColumnKind.NUMERIC

    def test_frozen_sensor_in_one_run_stays_measurement(self) -> None:
        """konst_T_V, Run 17: `Masse` steht 311 Samples still, schwankt aber in allen anderen Runs."""
        ds = load_csv(FIXTURES / "runs_export_20261004_123102_konst_T_V.csv")
        assert "masse" in ds.channels


class TestRealData:
    def test_all_messy_setpoints_are_parameters(self) -> None:
        ds = load_csv(ALL_MESSY)
        assert {"param_temperatur", "rotameter", "volumenstrom", "konzentration", "snad2"} <= set(
            ds.parameters
        )
        assert {"masse", "massenstrom_waage", "normkonzentration"} <= set(ds.channels)

    def test_long_row_column_target_temperature_is_parameter(self) -> None:
        """Vorher wurde `target_temperature` aus Long-Exporten als Messkanal geplottet."""
        for name in ("runs_export_20261004_115346.csv", "runs_export_20261004_115519_long.csv"):
            ds = load_csv(FIXTURES / name)
            assert ds.column_kinds["target_temperature"] is ColumnKind.PARAMETER
            assert "target_temperature" not in ds.channels

    def test_wide_and_long_export_agree_on_roles(self) -> None:
        wide = load_csv(FIXTURES / "runs_export_20261004_115422.csv")
        long = load_csv(FIXTURES / "runs_export_20261004_115428_long.csv")
        assert sorted(wide.channels) == sorted(long.channels)
        assert set(wide.parameters) <= set(long.parameters)

    def test_constant_sensor_is_ambiguous_and_needs_override(self) -> None:
        """P7: `Temperatur` in Run 1 ist ein Sensor, misst aber konstant 25.0.

        Aus den Werten allein ist das nicht von einem Sollwert zu unterscheiden. Die Heuristik
        schlägt Parameter vor, der Nutzer korrigiert per `column_kinds`.
        """
        path = FIXTURES / "runs_export_20261004_115422.csv"
        assert load_csv(path).column_kinds["temperatur"] is ColumnKind.PARAMETER

        cfg = IngestionConfig(column_kinds={"Temperatur": ColumnKind.NUMERIC})
        assert "temperatur" in load_csv(path, config=cfg).channels


class TestConfiguration:
    def test_detection_can_be_disabled(self) -> None:
        ds = load_csv(WIDE_CSV, config=IngestionConfig(roles=RoleConfig(detect_parameters=False)))
        assert ds.parameters == []
        assert ds.channels == ["sollwert", "messwert"]

    def test_explicit_numeric_beats_heuristic(self) -> None:
        cfg = IngestionConfig(column_kinds={"sollwert": ColumnKind.NUMERIC})
        assert load_csv(WIDE_CSV, config=cfg).channels == ["sollwert", "messwert"]

    def test_explicit_parameter_without_runs(self) -> None:
        csv = "time,a,b\n00:00:00,5,1.0\n00:00:01,5,2.0\n"
        ds = load_csv(csv, config=IngestionConfig(column_kinds={"a": ColumnKind.PARAMETER}))
        assert ds.parameters == ["a"]
        assert ds["a"].to_list() == [5.0, 5.0]

    def test_explicit_parameter_that_varies_is_audited(self) -> None:
        cfg = IngestionConfig(column_kinds={"messwert": ColumnKind.PARAMETER})
        ds = load_csv(WIDE_CSV, config=cfg)
        assert ds.parameters == ["sollwert", "messwert"]
        assert [(e.column, e.reason) for e in ds.audit_log] == [("messwert", "non_constant_parameter")]

    def test_explicit_run_columns(self) -> None:
        csv = "time,charge,x,y\n00:00:00,A,1.0,5\n00:00:01,A,2.0,5\n00:00:02,B,3.0,7\n00:00:03,B,4.0,7\n"
        assert load_csv(csv).parameters == []
        ds = load_csv(csv, config=IngestionConfig(roles=RoleConfig(run_columns=["charge"])))
        assert ds.parameters == ["y"]
        assert ds.ingestion_spec.roles.run_columns == ["charge"]

    def test_run_tokens_from_vocabulary(self) -> None:
        csv = "time,versuchsnr,y\n00:00:00,1,5\n00:00:01,1,5\n"
        vocab = Vocabulary()
        vocab.metadata_tokens.append("versuchsnr")
        vocab.identifier_tokens.append("versuchsnr")
        assert load_csv(csv, config=IngestionConfig(vocabulary=vocab)).parameters == []
        vocab.run_tokens.append("versuchsnr")
        assert load_csv(csv, config=IngestionConfig(vocabulary=vocab)).parameters == ["y"]

    def test_unknown_run_column_fails_loudly(self) -> None:
        cfg = IngestionConfig(roles=RoleConfig(run_columns=["gibts_nicht"]))
        with pytest.raises(RoleConfigError):
            load_csv(WIDE_CSV, config=cfg)

    def test_long_channel_override(self) -> None:
        path = FIXTURES / "runs_export_20261004_115428_long.csv"
        cfg = IngestionConfig(column_kinds={"Temperatur": ColumnKind.NUMERIC})
        ds = load_csv(path, config=cfg)
        assert "temperatur" in ds.channels
        assert ds.column_kinds["target_temperature"] is ColumnKind.PARAMETER


class TestSpec:
    def test_spec_writes_out_roles(self) -> None:
        spec = load_csv(ALL_MESSY).ingestion_spec
        assert spec.roles.run_columns == ["run_id"]
        assert spec.column_kinds["Rotameter"] is ColumnKind.PARAMETER
        assert spec.column_kinds["Masse"] is ColumnKind.NUMERIC

    def test_spec_contains_pivoted_channel_roles(self) -> None:
        spec = load_csv(FIXTURES / "runs_export_20261004_115428_long.csv").ingestion_spec
        assert spec.column_kinds["target_temperature"] is ColumnKind.PARAMETER
        assert spec.column_kinds["masse"] is ColumnKind.NUMERIC
