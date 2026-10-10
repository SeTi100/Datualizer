"""Etappe 2: Einheiten aus Headern (P20) und Spalten ohne Einheit markieren (Rest von P9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from datualizer_core import IngestionConfig, UnitConfig, UnitConfigError, load_csv, load_csvs
from datualizer_core.ingestion.config import DEFAULT_UNIT_PATTERNS
from datualizer_core.ingestion.units import unit_from_header

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "runs_export"
ALL_MESSY = FIXTURES / "runs_export_20261004_123307_ALL_MESSY.csv"
WIDE = FIXTURES / "runs_export_20261004_115339.csv"
LONG = FIXTURES / "runs_export_20261004_115351_long.csv"


class TestHeaderUnits:
    @pytest.mark.parametrize(
        ("header", "unit"),
        [
            ("Massenstrom Waage (g/s)", "g/s"),
            ("Normkonzentration (g/Nm³)", "g/Nm³"),
            ("Kanal 1 Letzte (C)", "C"),
            ("Temp [°C]", "°C"),
            ("Druck {bar} ", "bar"),
            ("Kanal (1)", None),  # Kanalnummer, keine Einheit
            ("Masse", None),
        ],
    )
    def test_default_patterns(self, header: str, unit: str | None) -> None:
        assert unit_from_header(header, DEFAULT_UNIT_PATTERNS) == unit

    def test_fixture_units_are_kept_as_metadata(self) -> None:
        """P20: Der Name verliert die Einheit, das Dataset behält sie."""
        ds = load_csv(ALL_MESSY)
        assert "massenstrom_waage" in ds.channels
        assert ds.units == {
            "abgaskonzentration_betriebszustand": "g/m³",
            "massenstrom_waage": "g/s",
            "normkonzentration": "g/Nm³",
        }

    def test_device_export(self) -> None:
        ds = load_csv(ROOT / "SA_testmessung_1.csv")
        assert ds.units["kanal_1"] == "C"
        assert set(ds.units) == set(ds.channels)

    def test_wide_and_long_agree(self) -> None:
        """Gleiche Messung: Header-Einheiten (wide) = Attribut-Einheiten (long)."""
        wide, long = load_csv(WIDE), load_csv(LONG)
        for ch, unit in wide.units.items():
            assert long.units[ch] == unit
        assert long.units["masse"] == "g"  # nur der Long-Export kennt sie
        assert "masse" in wide.unitless_columns

    def test_metadata_columns_have_no_unit(self) -> None:
        ds = load_csv("time,run_id,status (text),x (V)\n00:00:00,1,ok,1.0\n00:00:01,1,ok,2.0\n")
        assert ds.units == {"x": "V"}


class TestUnitless:
    def test_parameters_without_unit(self) -> None:
        """Rest von P9: `snad2 = 180` ohne Einheit wird markiert."""
        ds = load_csv(ALL_MESSY)
        assert "snad2" in ds.parameters
        assert "snad2" in ds.unitless_columns
        assert "massenstrom_waage" not in ds.unitless_columns
        assert set(ds.unitless_columns) <= set(ds.channels) | set(ds.parameters)

    def test_select_run_keeps_units(self) -> None:
        ds = load_csv(ALL_MESSY)
        assert ds.select_run((4,)).units == ds.units


class TestConfig:
    def test_explicit_unit_wins(self) -> None:
        cfg = IngestionConfig(units=UnitConfig(units={"snad2": "°C", "Massenstrom Waage (g/s)": "kg/h"}))
        ds = load_csv(ALL_MESSY, config=cfg)
        assert ds.units["snad2"] == "°C"
        assert ds.units["massenstrom_waage"] == "kg/h"
        assert "snad2" not in ds.unitless_columns

    def test_explicit_empty_removes_unit(self) -> None:
        cfg = IngestionConfig(units=UnitConfig(units={"massenstrom_waage": ""}))
        ds = load_csv(ALL_MESSY, config=cfg)
        assert "massenstrom_waage" not in ds.units
        assert ds.ingestion_spec.units.units["massenstrom_waage"] == ""

    def test_header_extraction_off(self) -> None:
        ds = load_csv(ALL_MESSY, config=IngestionConfig(units=UnitConfig(from_headers=False)))
        assert ds.units == {}

    def test_custom_pattern(self) -> None:
        """Andere Anlagen schreiben `Druck / bar`: Muster ist konfigurierbar, nicht verdrahtet."""
        text = "time,Druck / bar\n00:00:00,1.0\n00:00:01,1.1\n"
        assert load_csv(text).units == {}
        cfg = IngestionConfig(units=UnitConfig(header_patterns=[r"/\s*(?P<unit>\S+)\s*$"]))
        ds = load_csv(text, config=cfg)
        assert ds.channels == ["druck"]  # Einheit aus dem Namen gelöst, nicht verworfen
        assert ds.units == {"druck": "bar"}

    def test_spec_is_fixpoint(self) -> None:
        for path in (ALL_MESSY, LONG):
            first = load_csv(path)
            again = load_csv(path, config=first.ingestion_spec)
            assert again.units == first.units
            assert again.ingestion_spec == first.ingestion_spec

    def test_unknown_column_fails_loudly(self) -> None:
        cfg = IngestionConfig(units=UnitConfig(units={"phase_status": "s"}))
        with pytest.raises(UnitConfigError, match="phase_status"):
            load_csv(ALL_MESSY, config=cfg)

    def test_bad_pattern_fails_loudly(self) -> None:
        with pytest.raises(UnitConfigError, match="named group"):
            load_csv(ALL_MESSY, config=IngestionConfig(units=UnitConfig(header_patterns=[r"\((.*)\)"])))

    def test_merge_keeps_units(self) -> None:
        files = [FIXTURES / "runs_export_20261004_115422.csv", WIDE]
        cfg = IngestionConfig(units=UnitConfig(units={"Masse": "g"}))
        ds = load_csvs(files, config=cfg)
        assert ds.units["masse"] == "g"
        assert ds.units["massenstrom_waage"] == "g/s"
