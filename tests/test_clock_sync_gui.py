"""Dialog „Align Clock“ im Quellen-Dock (Etappe 2b, Schritt 7), offscreen.

Zwei Kopien des FID-Exports (die zweite mit falscher Uhr, −37 s) und die Uhrenpaare aus dem
Labjournal vom 17.10.: Kreuzkorrelation findet den Versatz, der Schieber verschiebt von Hand,
die Journal-Paare ergeben 974,5 s ± 0,5 s; „Apply“ schreibt das ClockModel mit method und note.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog

from datualizer_core.campaign import CROSS_CORRELATION, JOURNAL_PAIRS, Campaign, CampaignSession, ClockModel
from datualizer_gui.campaign_window import CampaignWindow
from datualizer_gui.components.clock_sync_dialog import ClockSyncDialog

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
FID = SA / "fid_ifid_12_10_extract.dat"
PAIRS = SA / "labjournal_17_10_clock_pairs.xlsx"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture()
def window(qapp: QApplication) -> CampaignWindow:
    session = CampaignSession(Campaign(timezone="Europe/Zurich"))
    session.add_source(FID, source_id="fid")
    session.add_source(FID, source_id="fid_copy")
    session.add_source(PAIRS, source_id="journal")
    session.set_clock("fid_copy", ClockModel(offset_s=-37.0, method="manual"))
    win = CampaignWindow(session)
    win.show()
    qapp.processEvents()
    yield win
    win.close()


def select(combo, data) -> None:
    idx = combo.findData(data)
    assert idx >= 0, data
    combo.setCurrentIndex(idx)


def xcorr_dialog(win: CampaignWindow) -> ClockSyncDialog:
    start, end = win.session.span("fid")
    dlg = ClockSyncDialog(win.session, "fid_copy", (start, end), win)
    dlg.set_reference("fid", "signal")
    dlg.set_moving("signal")
    dlg.set_span(start + 60, end - 60)
    dlg.lag_min_spin.setValue(-120)
    dlg.lag_max_spin.setValue(120)
    dlg._update_slider_range()
    return dlg


def test_correlation_finds_the_shift_and_fills_the_model(qapp: QApplication, window: CampaignWindow) -> None:
    dlg = xcorr_dialog(window)
    assert dlg.mov_combo.count() == 10  # only channels of this source
    assert all(dlg.ref_combo.itemData(i)[0] != "fid_copy" for i in range(dlg.ref_combo.count()))
    with pytest.raises(KeyError):
        dlg.set_reference("fid_copy", "signal")  # never its own source
    res = dlg.correlate()
    qapp.processEvents()
    assert res is not None and res.lag_s == pytest.approx(37.0, abs=res.step_s)
    assert dlg.shift() == pytest.approx(res.lag_s, abs=1e-3)
    assert dlg.shift_slider.value() == round(res.lag_s * 10)
    assert dlg.best_line.isVisible() and dlg.shift_line.value() == pytest.approx(dlg.shift())
    assert "shift" in dlg.xcorr_label.text()
    model = dlg.clock_model()
    assert model.method == CROSS_CORRELATION and model.uncertainty_s == pytest.approx(res.step_s / 2)
    assert model.offset_s == pytest.approx(-37.0 + res.lag_s)
    assert "Cross-correlation" in model.note and dlg.note_edit.text() == model.note

    # moved by hand: the model says so
    dlg.shift_slider.setValue(400)
    assert dlg.shift() == pytest.approx(40.0)
    manual = dlg.clock_model()
    assert manual.method == "manual" and manual.offset_s == pytest.approx(3.0) and manual.uncertainty_s is None
    assert "by hand" in manual.note
    x_mov = dlg._mov_curve.getOriginalDataset()[0]
    np.testing.assert_allclose(x_mov[0], window.session.track("fid_copy", "signal").x[0] + 40.0 - dlg._view[0])
    dlg.close()


def test_own_note_is_kept(qapp: QApplication, window: CampaignWindow) -> None:
    dlg = xcorr_dialog(window)
    dlg.note_edit.setText("FID clock, checked against the copy")
    dlg.correlate()
    assert dlg.clock_model().note == "FID clock, checked against the copy"
    dlg.close()


def test_journal_pairs_tab(qapp: QApplication, window: CampaignWindow) -> None:
    dlg = ClockSyncDialog(window.session, "fid", window.session.span("fid"), window)
    dlg.tabs.setCurrentIndex(1)
    assert [dlg.cells_source_combo.itemData(i) for i in range(dlg.cells_source_combo.count())] == ["journal"]
    select(dlg.cells_source_combo, "journal")
    assert dlg.sheet_combo.currentText() == "17.10.2026"
    assert dlg.date_edit.date().toPython() == date(2026, 10, 17)  # suggested from the sheet name
    apply = dlg.buttons.button(dlg.buttons.StandardButton.Apply)
    assert not apply.isEnabled()  # nothing read yet
    dlg.pairs_table.setRowCount(0)
    dlg.add_pair("C3", "C4", "Punkt 1")
    dlg.add_pair("d3", "d4")
    model = dlg.read_pairs()
    assert model is not None and model.offset_s == pytest.approx(974.5) and model.uncertainty_s == 0.5
    assert apply.isEnabled() and dlg.anchor_table.rowCount() == 2
    assert dlg.anchor_table.item(0, 3).text() == "+975.000" and dlg.anchor_table.item(1, 0).text() == "D3 → D4"
    assert dlg.clock_model().method == JOURNAL_PAIRS
    assert [p.source_cell for p in dlg.clock_model().cells.pairs] == ["C3", "D3"]

    dlg.add_pair("E3", "E4")  # empty cells: loud, no model
    assert dlg.read_pairs() is None and "E3 is empty" in dlg.pairs_label.text()
    assert not apply.isEnabled()
    dlg.close()


def test_apply_through_sources_dock(qapp: QApplication, window: CampaignWindow, monkeypatch) -> None:
    panel = window.sources_panel
    panel.select_source("fid_copy")
    made = []

    def dialog(source_id: str) -> ClockSyncDialog:
        dlg = xcorr_dialog(window)
        dlg.correlate()
        dlg.exec = lambda: QDialog.DialogCode.Accepted
        made.append(dlg)
        return dlg

    monkeypatch.setattr(panel, "align_dialog", dialog)
    assert panel.btn_align.isEnabled()
    panel.btn_align.click()
    qapp.processEvents()
    clock = window.session.campaign.source("fid_copy").clock
    assert clock.method == CROSS_CORRELATION and clock.offset_s == pytest.approx(0.0, abs=1.0)
    assert "cross_correlation" in panel.method_label.text()
    np.testing.assert_allclose(window.session.master_time("fid_copy"), window.session.master_time("fid"), atol=1.0)


def test_dialog_reopens_with_the_last_pairs(qapp: QApplication, window: CampaignWindow) -> None:
    session = window.session
    dlg = ClockSyncDialog(session, "fid", None, window)
    dlg.tabs.setCurrentIndex(1)
    dlg.pairs_table.setRowCount(0)
    dlg.add_pair("C3", "C4")
    dlg.add_pair("D3", "D4")
    session.set_clock("fid", dlg.read_pairs())
    dlg.close()
    again = ClockSyncDialog(session, "fid", None, window)
    assert again.pairs_table.rowCount() == 2 and again.pairs_table.item(1, 1).text() == "D4"
    assert again.sheet_combo.currentText() == "17.10.2026"
    again.close()
