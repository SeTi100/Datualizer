"""Robust step mapping (PR 9b): the lab journal of 17.10.2026 as it changes while it is written.

Real data: `labjournal_17_10_extract.xlsx` is the sheet as of 19.10.2026 (new column C "Index",
steps of the 15.10. pasted below row 30). Variants are made from it in tmp_path with openpyxl:
columns inserted, rows inserted above the header, a column renamed, end times missing. Each
must give the same 20 steps (9 dosings) with the step 200 °C / 10 Nl/min from 13:04:16 to
13:56:10, or fail loudly with a suggestion.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import shutil

import openpyxl
import pytest

from datualizer_core.campaign import (
    CampaignSession,
    CellPair,
    ClockCells,
    IntervalConfigError,
    IntervalMapping,
    to_epoch_s,
)
from datualizer_core.campaign.intervals import CellValue
from datualizer_core.campaign.sync import ClockCellsError
from datualizer_core.ingestion.grid import Grid, GridError, match_header
from datualizer_core.ingestion.profiles import get_profile
from datualizer_core.ingestion.source_loader import with_choices
from datualizer_core.ingestion.steps import EndKind, EndRule, RowSelection, StopRule

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
JOURNAL = SA / "labjournal_17_10_extract.xlsx"
OLD_PAIRS = SA / "labjournal_17_10_clock_pairs.xlsx"
ZONE = "Europe/Zurich"
SHEET = "17.10.2026"


def zurich(*args) -> float:
    return to_epoch_s(datetime(*args), ZONE)


def session_for(path: Path) -> CampaignSession:
    s = CampaignSession()
    profile = with_choices(path, get_profile("labjournal_xlsx"), sheet=SHEET, clock_date=date(2026, 10, 17))
    s.add_source(path, profile, source_id="journal")
    return s


def mapping(**changes) -> IntervalMapping:
    """The mapping of the day configuration: headers, stop at the time jump, missing end = next start."""
    base = dict(
        source_id="journal", start_column="Zeit_0", label_columns=["col_A"], fill_down=["col_A"],
        end=EndRule(column="Zeit_1", fallback=EndRule(kind=EndKind.NEXT_START)),
        parameter_columns=["Temperatur", "V_MFC", "Masse_0", "Masse_1"],
        rows=RowSelection(stop=StopRule.TIME_JUMP),
    )
    return IntervalMapping(**{**base, **changes})


def variant(tmp_path: Path, edit) -> Path:
    path = tmp_path / "journal.xlsx"
    shutil.copy(JOURNAL, path)
    wb = openpyxl.load_workbook(path)
    edit(wb[SHEET])
    wb.save(path)
    return path


def assert_the_20_steps(steps) -> None:
    assert len(steps) == 20
    assert sum(iv.params["v_mfc"] > 0 for iv in steps) == 9
    stage = next(iv for iv in steps if iv.params["temperatur"] == 200 and iv.params["v_mfc"] == 10)
    assert (stage.start, stage.end) == (zurich(2026, 10, 17, 13, 4, 16), zurich(2026, 10, 17, 13, 56, 10))
    assert stage.params["masse_0"] == 9608.75 and stage.params["masse_1"] == 9597.05
    assert steps[0].start == zurich(2026, 10, 17, 11, 51, 16)
    assert steps[-1].start == zurich(2026, 10, 17, 17, 4, 58)


# --------------------------------------------------------------- real sheet
class TestJournal17:
    def test_headers_stop_and_fallback(self) -> None:
        s = session_for(JOURNAL)
        d = s.derive_mapping(mapping())
        assert d.header.row == 10
        assert d.header.headers[4] == "V_MFC [Nl/min]"  # 'V_MFC' found despite the unit
        assert_the_20_steps(d.intervals)
        assert d.intervals[5].origin == "journal:row 16"
        assert any(
            p.startswith("journal: stop: start in row 31 (09:49:00) is 436 min before row 30 (17:04:58)")
            and p.endswith("rows 31–62 not read.") for p in d.problems
        )
        assert "journal: rows 11–12, 21: no end from 'Zeit_1': next start used." in d.problems
        first = d.intervals[0]
        assert first.end == zurich(2026, 10, 17, 12, 5, 23) and "next start used" in first.note
        assert d.rows[16].status == "step" and d.rows[31].status == "excluded"
        assert "another day or block" in d.rows[40].reason
        assert d.rows[10].status == "header" and d.rows[3].status == "above"
        assert d.roles[7] == "start" and d.roles[8] == "end" and d.roles[4] == "parameter"

    def test_profile_suggests_time_jump(self) -> None:
        s = session_for(JOURNAL)
        m = s.suggest_mapping("journal")
        assert m.rows.stop is StopRule.TIME_JUMP
        steps, problems = s.preview_steps(m)
        # without the end fallback the three steps without end time are reported, not guessed
        assert len(steps) == 17 and "journal: rows 11–12, 21: no readable end (skipped)." in problems

    def test_without_stop_the_other_day_is_read(self) -> None:
        s = session_for(JOURNAL)
        steps, _ = s.preview_steps(mapping(rows=RowSelection()))
        assert len(steps) > 40

    @pytest.mark.parametrize("rows", [
        RowSelection(where="index <= 20"),
        RowSelection(first_row=11, last_row=30),
        RowSelection(stop=StopRule.TIME_JUMP, time_jump_min=60),
    ])
    def test_other_ways_to_pick_the_rows(self, rows: RowSelection) -> None:
        d = session_for(JOURNAL).derive_mapping(mapping(rows=rows))
        assert_the_20_steps(d.intervals)
        if rows.where:
            assert d.rows[31].reason == "where 'index <= 20' is not true"

    def test_where_with_text_and_unknown_names(self) -> None:
        s = session_for(JOURNAL)
        d = s.derive_mapping(mapping(rows=RowSelection(stop=StopRule.TIME_JUMP, where="v_mfc > 0")))
        assert len(d.intervals) == 9
        d = s.derive_mapping(mapping(rows=RowSelection(stop=StopRule.TIME_JUMP, where="col_a == '222'")))
        assert [iv.origin for iv in d.intervals] == ["journal:row 16", "journal:row 25"]  # raw cells, not filled down
        with pytest.raises(IntervalConfigError, match="temperatur"):
            s.derive_mapping(mapping(rows=RowSelection(where="temp <= 200")))

    def test_skip_rows_and_first_empty_start(self) -> None:
        s = session_for(JOURNAL)
        d = s.derive_mapping(mapping(rows=RowSelection(stop=StopRule.TIME_JUMP, skip_rows=[13])))
        assert len(d.intervals) == 19 and "journal: row 13: left out by hand (not read)." in d.problems
        d = s.derive_mapping(mapping(rows=RowSelection(stop=StopRule.FIRST_EMPTY_START)))
        assert len(d.intervals) == 31  # rows 11-41; row 42 has no start time
        assert any("stop: row 42 is the first without start time" in p for p in d.problems)

    def test_parameter_defaults(self) -> None:
        s = session_for(JOURNAL)
        m = mapping(parameter_columns=["Temperatur", "V_MFC", "Luft"], parameter_defaults={"luft": 100})
        d = s.derive_mapping(m)
        assert all(iv.params["luft"] == 100 for iv in d.intervals)
        assert "journal: no column 'Luft': parameter 'luft' = 100 for every step." in d.problems
        with pytest.raises(IntervalConfigError, match="no column 'Luft'"):
            s.derive_mapping(mapping(parameter_columns=["Temperatur", "Luft"]))  # no default: loud

    def test_id_column(self) -> None:
        d = session_for(JOURNAL).derive_mapping(mapping(id_column="Index"))
        assert [iv.id for iv in d.intervals[:2]] == ["journal:1", "journal:2"]
        assert d.intervals[5].id == "journal:6" and d.roles[2] == "id"

    def test_cell_value_by_label(self) -> None:
        s = session_for(JOURNAL)
        cv = CellValue(name="raumluft", label="Baseline mgC")
        hit = s.cell_value(cv, mapping())
        assert (hit.address, hit.value, hit.label_cell) == ("E5", 2.09, "C5")  # D5 is empty
        d = s.derive_mapping(mapping(cell_values=[cv]))
        assert all(iv.params["raumluft"] == 2.09 for iv in d.intervals)
        assert "journal: 'raumluft' = 2.09 from cell E5 (right of 'Baseline mgC')." in d.problems
        with pytest.raises(IntervalConfigError, match="did you mean 'Baseline mgC'"):
            s.derive_mapping(mapping(cell_values=[CellValue(name="b", label="Baseline mg")]))

    def test_clock_pairs_by_label(self) -> None:
        s = session_for(JOURNAL)
        cells = ClockCells(source_id="journal", sheet=SHEET, day=date(2026, 10, 17), pairs=[
            CellPair(source_label="Start FID", master_label="Start Real", nth=1),
            CellPair(source_label="Start FID", master_label="Start Real", nth=2),
        ])
        model, anchors = s.clock_from_cells(cells)
        assert (model.offset_s, model.uncertainty_s) == (974.5, 0.5)
        assert [a.note for a in anchors] == ["D3 → D4", "E3 → E4"]  # moved from C3/D3 with the new column
        with pytest.raises(ClockCellsError, match="Start FID"):
            s.clock_from_cells(cells.model_copy(update={"pairs": [CellPair(source_label="Start FID", master_cell="D4", nth=3)]}))

    def test_clock_pairs_by_label_in_the_old_layout(self) -> None:
        """The same configuration reads the journal as it was before the new column."""
        s = CampaignSession()
        s.add_source(OLD_PAIRS, with_choices(OLD_PAIRS, get_profile("labjournal_xlsx"), sheet=SHEET), source_id="old")
        cells = ClockCells(source_id="old", sheet=SHEET, day=date(2026, 10, 17), pairs=[
            CellPair(source_label="start fid", master_label="Start  Real", nth=n) for n in (1, 2)
        ])
        model, anchors = s.clock_from_cells(cells)
        assert model.offset_s == 974.5 and [a.note for a in anchors] == ["C3 → C4", "D3 → D4"]


# ----------------------------------------------------------- changed sheets
class TestVariants:
    def test_columns_inserted(self, tmp_path: Path) -> None:
        def edit(ws):
            ws.insert_cols(2)
            ws["B10"] = "Bemerkung"
            ws.insert_cols(8)  # between Masse_1 and Zeit_0
            ws["H10"] = "Neu"
        d = session_for(variant(tmp_path, edit)).derive_mapping(mapping(cell_values=[CellValue(name="b", label="Baseline mgC")]))
        assert_the_20_steps(d.intervals)
        assert d.intervals[0].params["b"] == 2.09

    def test_rows_inserted_above_the_header(self, tmp_path: Path) -> None:
        def edit(ws):
            ws.insert_rows(1, 4)
            ws["A2"] = "Notiz"
            ws["H3"] = "Zeit_0"  # a lonely header-like word above: not the header row
        d = session_for(variant(tmp_path, edit)).derive_mapping(mapping())
        assert d.header.row == 14
        assert_the_20_steps(d.intervals)
        assert d.intervals[5].origin == "journal:row 20"

    def test_column_renamed_fails_with_a_suggestion(self, tmp_path: Path) -> None:
        def edit(ws):
            ws["D10"] = "Temperatur_Soll"
        s = session_for(variant(tmp_path, edit))
        with pytest.raises(IntervalConfigError, match=r"no column 'Temperatur' \(did you mean 'Temperatur_Soll'\?\)"):
            s.derive_mapping(mapping())
        assert s.intervals().intervals == []  # never a silent fallback to the column position
        fixed = s.derive_mapping(mapping(parameter_columns=["Temperatur_Soll", "V_MFC", "Masse_0", "Masse_1"]))
        assert len(fixed.intervals) == 20 and fixed.intervals[5].params["temperatur_soll"] == 200

    def test_block_of_another_day_with_dates(self, tmp_path: Path) -> None:
        def edit(ws):
            ws["K10"] = "Datum"
            for r in range(11, 63):
                ws.cell(r, 11, "17.10.2026" if r <= 30 else "15.10.2026")
        s = session_for(variant(tmp_path, edit))
        d = s.derive_mapping(mapping(rows=RowSelection(where="datum == '17.10.2026'")))
        assert_the_20_steps(d.intervals)

    def test_missing_end_times(self, tmp_path: Path) -> None:
        def edit(ws):
            ws["I16"] = None  # Zeit_1 of the stage 200 °C / 10 Nl/min
        s = session_for(variant(tmp_path, edit))
        d = s.derive_mapping(mapping())
        assert len(d.intervals) == 20
        stage = d.intervals[5]
        assert stage.end == zurich(2026, 10, 17, 13, 56, 39) and "next start used" in stage.note
        assert "journal: rows 11–12, 16, 21: no end from 'Zeit_1': next start used." in d.problems
        _, problems = s.preview_steps(mapping(end=EndRule(column="Zeit_1")))
        assert "journal: rows 11–12, 16, 21: no readable end (skipped)." in problems


# ------------------------------------------------------- reopening a campaign
class TestChangedFile:
    def test_diff_and_overrides_follow_their_step(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.xlsx"
        shutil.copy(JOURNAL, path)
        s = session_for(path)
        s.set_mapping(mapping())
        stage = s.intervals().intervals[5]
        assert stage.id == "journal:5"
        s.set_window_override(stage.id, stage.end - 300, stage.end - 60)
        s.save(tmp_path / "campaign.json")
        assert s.campaign.interval_mappings[0].resolved.steps[5].row == 16

        wb = openpyxl.load_workbook(path)
        ws = wb[SHEET]
        ws.insert_rows(11)  # a new first step: every row below moves down
        ws["C11"], ws["D11"], ws["E11"], ws["H11"], ws["I11"] = 0, 150, 0, datetime(2026, 10, 17, 11, 45).time(), datetime(2026, 10, 17, 11, 51).time()
        ws.insert_cols(2)
        ws["B10"] = "Bemerkung"
        wb.save(path)

        again = CampaignSession.open(tmp_path / "campaign.json")
        w = again.warnings
        assert any("file changed since it was added" in x for x in w)
        assert any("the saved reading spec does not fit the changed file" in x for x in w)  # read anew, not a crash
        assert "journal: new columns: Bemerkung." in w
        assert "journal: table has 1 more rows (last row 63, was 62)." in w
        assert "journal: new steps: journal:0 'journal row 11' 11:45:00." in w
        assert "journal: 20 unchanged steps have a new id (rows moved; an id column keeps ids stable)." in w
        assert not any("steps changed" in x or "steps gone" in x for x in w)
        assert "journal:5: window set by hand moved to journal:6 (same step '222')." in w
        moved = again.intervals().get("journal:6")
        assert moved.window_override and moved.start == stage.start
        assert not any("window set by hand" in p for p in again.intervals().problems)

    def test_with_id_column_overrides_keep_their_id(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.xlsx"
        shutil.copy(JOURNAL, path)
        s = session_for(path)
        s.set_mapping(mapping(id_column="Index"))
        stage = s.intervals().get("journal:6")
        s.set_window_override("journal:6", stage.end - 300, stage.end - 60)
        s.save(tmp_path / "campaign.json")
        wb = openpyxl.load_workbook(path)
        wb[SHEET].insert_rows(12, 2)  # empty rows in the middle
        wb.save(path)
        again = CampaignSession.open(tmp_path / "campaign.json")
        assert not any("moved" in x for x in again.warnings)
        assert again.intervals().get("journal:6").window_override

    def test_steps_that_no_longer_fit_are_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.xlsx"
        shutil.copy(JOURNAL, path)
        s = session_for(path)
        s.set_mapping(mapping())
        stage = s.intervals().intervals[5]
        s.set_window_override(stage.id, stage.end - 300, stage.end - 60)
        s.save(tmp_path / "campaign.json")
        wb = openpyxl.load_workbook(path)
        wb[SHEET]["A16"] = "anders"  # the step got another name
        wb.save(path)
        again = CampaignSession.open(tmp_path / "campaign.json")
        assert any("steps changed: journal:5 ('222' → 'anders')" in x for x in again.warnings)
        assert "journal:5: window set by hand on step '222', the step is now 'anders'." in again.intervals().problems


# ------------------------------------------------------------------- grid
def test_header_matching() -> None:
    headers = ["col_A", "Index", "Temperatur", "V_MFC [Nl/min]", "Masse_0", "Masse 1", "mg_C FID [RF1]"]
    assert match_header("v_mfc", headers) == 3
    assert match_header("V_MFC", headers) == 3
    assert match_header("masse_1", headers) == 5
    assert match_header(" index ", headers) == 1
    assert match_header("col_a", headers) == 0
    assert match_header("Zeit_0", headers) is None
    with pytest.raises(GridError, match="ambiguous"):
        match_header("masse", ["Masse [g]", "Masse (kg)"])


def test_table_grid_from_csv() -> None:
    """A CSV is a table with its header in row 1; the time column keeps its original name."""
    s = CampaignSession()
    path = SA / "picolog_12_10_extract.csv"
    s.add_source(path, source_id="pico")
    grid = s.step_grid("pico")
    assert isinstance(grid, Grid) and grid.kind == "table" and len(grid.rows) == len(s.datasets["pico"].df) + 1
