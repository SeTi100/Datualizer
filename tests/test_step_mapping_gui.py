"""Dialog "Steps from Source" on the sheet itself (PR 9b), with the journal of 17.10.2026."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from datualizer_core.campaign import CampaignSession
from datualizer_core.ingestion.profiles import get_profile
from datualizer_core.ingestion.source_loader import with_choices
from datualizer_core.ingestion.steps import EndKind
from datualizer_gui.components.step_mapping_dialog import (
    ROW_COLORS,
    StepMappingDialog,
    label_left,
    parse_defaults,
)

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
JOURNAL = SA / "labjournal_17_10_extract.xlsx"
DB = SA / "instrument_reader_12_10_extract.db"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture()
def session() -> CampaignSession:
    s = CampaignSession()
    profile = with_choices(JOURNAL, get_profile("labjournal_xlsx"), sheet="17.10.2026", clock_date=date(2026, 10, 17))
    s.add_source(JOURNAL, profile, source_id="journal")
    s.add_source(DB, source_id="db")
    return s


@pytest.fixture()
def dialog(qapp: QApplication, session: CampaignSession) -> StepMappingDialog:
    d = StepMappingDialog(session, source_id="journal")
    yield d
    d.close()


def actions(menu) -> dict[str, object]:
    return {a.text(): a for a in menu.actions() if a.text()}


def bg(d: StepMappingDialog, row: int, col: int = 3):
    return d.model.data(d.model.index(row - 1, col), Qt.ItemDataRole.BackgroundRole)


def test_the_sheet_as_grid_with_roles_and_row_states(dialog: StepMappingDialog) -> None:
    m = dialog.model
    assert m.rowCount() == 62 and m.columnCount() == 10  # all rows, also those above the header
    assert m.headerData(0, Qt.Orientation.Horizontal) == "A\nname"
    assert m.headerData(7, Qt.Orientation.Horizontal) == "H\nstart"
    assert m.headerData(8, Qt.Orientation.Horizontal) == "I\nend"
    assert m.data(m.index(2, 2)) == "Start FID" and m.data(m.index(2, 3)) == "11:26:00"
    assert dialog.header_row() == 10 and "first row with all headers" in dialog.header_info.text()
    assert "17 steps" in dialog.preview.text()  # the profile: stop at the time jump, ends not guessed
    assert bg(dialog, 16) == ROW_COLORS["step"]
    assert bg(dialog, 11) == ROW_COLORS["problem"]  # no end time
    assert bg(dialog, 40) == ROW_COLORS["excluded"]
    tip = m.data(m.index(39, 3), Qt.ItemDataRole.ToolTipRole)
    assert "left out" in tip and "another day or block" in tip


def test_end_fallback_roles_and_rows(dialog: StepMappingDialog) -> None:
    dialog.fallback_kind.setCurrentIndex(dialog.fallback_kind.findData(EndKind.NEXT_START.value))
    dialog.refresh_now()
    assert "20 steps" in dialog.preview.text() and dialog.steps_table.rowCount() == 20
    assert dialog.timeline.steps and bg(dialog, 11) == ROW_COLORS["step"]

    actions(dialog.role_menu(2))["ID"].trigger()  # column C 'Index'
    assert dialog.mapping().id_column == "Index"
    assert dialog.steps_table.item(5, 0).data(Qt.ItemDataRole.UserRole) == "journal:6"
    actions(dialog.role_menu(5))["Parameter"].trigger()  # Masse_0
    assert "Masse_0" in dialog.mapping().parameter_columns
    actions(dialog.role_menu(5))["Ignore"].trigger()
    assert "Masse_0" not in dialog.mapping().parameter_columns

    actions(dialog.row_menu_for(13))["Leave out row 13"].trigger()
    assert "19 steps" in dialog.preview.text() and dialog.mapping().rows.skip_rows == [13]
    actions(dialog.row_menu_for(13))["Read row 13 again"].trigger()
    actions(dialog.row_menu_for(15))["Table starts here (row 15)"].trigger()
    assert dialog.mapping().rows.first_row == 15 and "16 steps" in dialog.preview.text()
    dialog.btn_range_clear.click()

    dialog.stop.setCurrentIndex(dialog.stop.findData("none"))
    dialog.where.setText("index <= 20")
    dialog.refresh_now()
    assert "20 steps" in dialog.preview.text() and "fits" in dialog.where_status.text()
    dialog.where.setText("indx <= 20")
    dialog.refresh_now()
    assert "indx" in dialog.where_status.text() and "0 steps" in dialog.preview.text()
    dialog.where.setText("")
    dialog.stop.setCurrentIndex(dialog.stop.findData("time_jump"))

    dialog.defaults.setText("luft = 100")
    dialog.refresh_now()
    assert dialog.mapping().parameter_defaults == {"luft": 100.0}

    actions(dialog.row_menu_for(10))["Header row here (row 10)"].trigger()
    assert dialog.mapping().rows.header_row == 10 and "set by hand" in dialog.header_info.text()

    dialog.accept()
    m = dialog._session.campaign.interval_mappings[0]
    assert m.end.fallback.kind is EndKind.NEXT_START and m.id_column == "Index"
    assert len(dialog._session.intervals().intervals) == 20


def test_cells_by_clicking(dialog: StepMappingDialog) -> None:
    dialog.add_cell_value(5, 4, name="raumluft")  # E5, right of 'Baseline mgC' (D5 empty)
    cv = dialog.mapping().cell_values[0]
    assert (cv.label, cv.nth, cv.cell) == ("Baseline mgC", 1, None)
    assert dialog.cells_table.item(0, 2).text() == "2.09"
    assert bg(dialog, 5, 4) is not None
    for row, side in ((3, "source"), (4, "master")):
        dialog.add_pair_side(row, 3, side)
    for row, side in ((3, "source"), (4, "master")):
        dialog.add_pair_side(row, 4, side)
    assert [(p.source_label, p.master_label, p.nth) for p in dialog.pairs] == [
        ("Start FID", "Start Real", 1), ("Start FID", "Start Real", 2)]
    assert "Offset 974.5 s ± 0.5 s" in dialog.pair_result.text()
    dialog.pair_target.setCurrentIndex(dialog.pair_target.findData("db"))
    dialog.accept()
    assert dialog._session.campaign.source("db").clock.offset_s == 974.5


def test_menus_on_cells(dialog: StepMappingDialog) -> None:
    assert "Start" in actions(dialog.cell_menu_for(10, 7))  # header row: the role menu
    menu = actions(dialog.cell_menu_for(5, 4))
    assert "Keep as cell value…" in menu and "Leave out row 5" in menu


def test_table_sources_show_their_table(qapp: QApplication, session: CampaignSession) -> None:
    d = StepMappingDialog(session, source_id="db")
    assert not d.sheet.isVisible() and d.header_row() == 1
    assert d.model.rowCount() == len(session.datasets["db"].df) + 1
    d.close()


def test_helpers() -> None:
    assert parse_defaults("luft = 100; gas = N2") == {"luft": 100.0, "gas": "N2"}
    with pytest.raises(ValueError):
        parse_defaults("luft 100")
    from datualizer_core.ingestion.grid import Grid

    grid = Grid(rows=[["Start FID", None, "11:26", "11:31"]])
    assert label_left(grid, 1, 3) == ("Start FID", 2) and label_left(grid, 1, 0) is None
