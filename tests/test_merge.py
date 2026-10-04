"""Etappe 2: Multi-File-Merge mit Dedupe (Datenkatalog P2, P3)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from datualizer_core import (
    IngestionConfig,
    MergeConfig,
    MergeConflictMode,
    MergeError,
    load_csv,
    load_csvs,
)

FIXTURES = Path(__file__).parent / "fixtures" / "runs_export"
P = "runs_export_20261004_"
KONST = FIXTURES / f"{P}123102_konst_T_V.csv"
KONST_NEWER = FIXTURES / f"{P}123115_konst_T_V_long.csv"  # wide, +104 Zeilen
LONG_PART = FIXTURES / f"{P}115346.csv"
LONG_FULL = FIXTURES / f"{P}115351_long.csv"
RAND = FIXTURES / f"{P}123129_rand_params.csv"
RAND_COPY = FIXTURES / f"{P}123138_rand_params_long.csv"  # byte-identisch
RUN1_WIDE = FIXTURES / f"{P}115422.csv"
RUN1_LONG = FIXTURES / f"{P}115428_long.csv"
RUNS2_4_WIDE = FIXTURES / f"{P}115339.csv"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


class TestDuplicateFiles:
    """P2: Byte-identische Duplikate werden erkannt und nur einmal geladen."""

    def test_identical_file_loaded_once(self) -> None:
        merged = load_csvs([RAND, RAND_COPY])
        single = load_csv(RAND)
        assert merged.df.equals(single.df)
        dup = [e for e in merged.audit_log if e.reason == "duplicate_file"]
        assert [e.raw_value for e in dup] == [str(RAND_COPY)]
        assert dup[0].raw_column == str(RAND)

    def test_sources_report_duplicate(self) -> None:
        src = load_csvs([RAND, RAND_COPY]).sources
        assert src["duplicate_of"].to_list() == [None, str(RAND)]
        assert src["n_kept"].to_list() == [1861, 0]
        assert src["sha256"][0] == src["sha256"][1] == hashlib.sha256(RAND.read_bytes()).hexdigest()

    def test_single_load_records_hash(self) -> None:
        src = load_csv(RAND).sources
        assert src.height == 1
        assert src["sha256"][0] == hashlib.sha256(RAND.read_bytes()).hexdigest()
        assert src["n_kept"][0] == 1861


class TestOverlappingSnapshots:
    """P3: Ein späterer Snapshot wiederholt alle Zeilen des früheren und hängt neue an."""

    def test_wide_snapshot_equals_newest(self) -> None:
        merged = load_csvs([KONST, KONST_NEWER])
        assert merged.df.equals(load_csv(KONST_NEWER).df)
        assert len(merged) == 2592
        assert not [e for e in merged.audit_log if e.reason == "merge_conflict"]
        assert merged.sources["n_replaced"].to_list() == [2488, 0]

    def test_long_snapshot_equals_newest(self) -> None:
        merged = load_csvs([LONG_PART, LONG_FULL])
        assert merged.source_format == "long"
        assert merged.df.equals(load_csv(LONG_FULL).df)
        assert merged.ingestion_spec.merge.key_columns == ["timestamp", "roi_name"]
        assert merged.channel_attrs == load_csv(LONG_FULL).channel_attrs

    def test_order_only_decides_conflicts(self) -> None:
        """Identische überlappende Zeilen: Die Reihenfolge der Dateien ändert das Ergebnis nicht."""
        assert load_csvs([LONG_FULL, LONG_PART]).df.equals(load_csvs([LONG_PART, LONG_FULL]).df)

    def test_disjoint_files_are_appended(self) -> None:
        """Zwei Exporte verschiedener Runs mit unterschiedlichen Spalten ergeben die Vereinigung."""
        merged = load_csvs([RUN1_WIDE, RUNS2_4_WIDE])
        assert len(merged) == 536 + 244
        assert merged.runs["run_id"].to_list() == [1, 2, 3, 4]
        assert "v_voc" in merged.channels  # nur in Run 1 vorhanden
        assert merged.sources["n_replaced"].to_list() == [0, 0]

    def test_inputs_not_modified(self) -> None:
        before = KONST.read_bytes()
        load_csvs([KONST, KONST_NEWER])
        assert KONST.read_bytes() == before


class TestConflicts:
    OLD = "time,run_id,x,y\n00:00:00,1,1.0,5\n00:00:01,1,2.0,6\n"
    NEW = "time,run_id,x,y\n00:00:01,1,2.5,6\n00:00:02,1,3.0,7\n"

    def test_newest_wins_and_conflict_is_audited(self, tmp_path: Path) -> None:
        ds = load_csvs([_write(tmp_path, "a.csv", self.OLD), _write(tmp_path, "b.csv", self.NEW)])
        assert ds["x"].to_list() == [1.0, 2.5, 3.0]
        conflicts = [e for e in ds.audit_log if e.reason == "merge_conflict"]
        assert [(e.row_index, e.column, e.raw_value) for e in conflicts] == [(1, "x", "2.0")]
        assert ds.sources["n_conflicts"].to_list() == [1, 0]

    def test_conflict_mode_error_fails_loudly(self, tmp_path: Path) -> None:
        cfg = IngestionConfig(merge=MergeConfig(conflict=MergeConflictMode.ERROR))
        with pytest.raises(MergeError, match="differ"):
            load_csvs([_write(tmp_path, "a.csv", self.OLD), _write(tmp_path, "b.csv", self.NEW)], config=cfg)

    def test_value_added_later_is_no_conflict(self, tmp_path: Path) -> None:
        """Ein Rechenkanal, der erst im späteren Snapshot gefüllt ist, ist ein Update."""
        old = _write(tmp_path, "a.csv", "time,x,calc\n00:00:00,1.0,\n00:00:01,2.0,\n")
        new = _write(tmp_path, "b.csv", "time,x,calc\n00:00:00,1.0,0.5\n00:00:01,2.0,0.6\n")
        ds = load_csvs([old, new])
        assert ds["calc"].to_list() == [0.5, 0.6]
        assert not ds.audit_log.has_errors()

    def test_value_lost_later_is_conflict(self, tmp_path: Path) -> None:
        old = _write(tmp_path, "a.csv", "time,x\n00:00:00,1.0\n00:00:01,2.0\n")
        new = _write(tmp_path, "b.csv", "time,x\n00:00:00,\n00:00:01,2.0\n")
        ds = load_csvs([old, new])
        assert ds["x"].to_list() == [None, 2.0]
        assert [e.reason for e in ds.audit_log] == ["merge_conflict"]

    def test_explicit_key_columns(self, tmp_path: Path) -> None:
        """Gleiche Zeit, aber verschiedene Sensoren: Mit Schlüssel (time, sensor) kein Konflikt."""
        a = _write(tmp_path, "a.csv", "time,sensor,x\n00:00:00,A,1.0\n")
        b = _write(tmp_path, "b.csv", "time,sensor,x\n00:00:00,B,2.0\n")
        cfg = IngestionConfig(merge=MergeConfig(key_columns=["time", "sensor"]))
        ds = load_csvs([a, b], config=cfg)
        assert len(ds) == 2
        assert not ds.audit_log.has_errors()
        assert len(load_csvs([a, b])) == 1  # Standard-Schlüssel: nur die Zeit


class TestIncompatibleFiles:
    def test_wide_and_long_fail_loudly(self) -> None:
        with pytest.raises(MergeError, match="layout"):
            load_csvs([RUN1_WIDE, RUN1_LONG])

    def test_different_decimal_separators(self, tmp_path: Path) -> None:
        a = _write(tmp_path, "a.csv", "time;x\n00:00:00;1,5\n00:00:01;2,5\n")
        b = _write(tmp_path, "b.csv", "time,x\n00:00:02,3.5\n00:00:03,4.5\n")
        with pytest.raises(MergeError, match="decimal"):
            load_csvs([a, b])

    def test_missing_key_column(self, tmp_path: Path) -> None:
        a = _write(tmp_path, "a.csv", "time,x\n00:00:00,1.0\n")
        cfg = IngestionConfig(merge=MergeConfig(key_columns=["time", "sensor"]))
        with pytest.raises(MergeError, match="key"):
            load_csvs([a], config=cfg)

    def test_no_sources(self) -> None:
        with pytest.raises(MergeError):
            load_csvs([])


class TestReplay:
    @pytest.mark.parametrize(
        "files", [[KONST, KONST_NEWER], [LONG_PART, LONG_FULL], [RUN1_WIDE, RUNS2_4_WIDE]]
    )
    def test_spec_is_fixpoint(self, files: list[Path]) -> None:
        first = load_csvs(files)
        again = load_csvs(files, config=first.ingestion_spec)
        assert again.df.equals(first.df)
        assert again.column_kinds == first.column_kinds
        assert again.ingestion_spec == first.ingestion_spec

    def test_single_file_merge_equals_load_csv(self) -> None:
        assert load_csvs([LONG_FULL]).df.equals(load_csv(LONG_FULL).df)

    def test_spec_is_json_serializable(self) -> None:
        spec = load_csvs([KONST, KONST_NEWER]).ingestion_spec
        assert IngestionConfig.model_validate_json(spec.model_dump_json()) == spec
