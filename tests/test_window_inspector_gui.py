"""Fenster-Inspektor im Kampagnenfenster (Etappe 2b, Schritt 6, Paket D), offscreen.

Waage (SQLite, OCR) und Labjournal vom 12.10.: Schritt „T Einstellung 155“ öffnen, Kennwerte
lesen, mit der Tastatur weiterschalten, Fensterränder ziehen (Override in campaign.json).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pyqtgraph as pg
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from datualizer_core.campaign import master_datetime
from datualizer_core.ingestion.profiles import detect_profile, get_profile
from datualizer_core.ingestion.source_loader import load_source, with_choices
from datualizer_gui import campaign_window
from datualizer_gui.campaign_window import CampaignWindow
from datualizer_gui.components.window_inspector import OVERRIDE_PEN, value_and_u

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
JOURNAL = SA / "labjournal_12_10_extract.xlsx"
DB = SA / "instrument_reader_12_10_extract.db"
T155_V2 = 19  # row of "T Einstellung 155", V_MFC = 2 in the Steps table (inside the scale recording)
T155_V1 = 18  # V_MFC = 1


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture()
def window(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> CampaignWindow:
    def load(_parent, path, *, ask_clock_date=False):
        path = Path(path)
        profile = detect_profile(path)
        if path == JOURNAL:
            profile = with_choices(path, get_profile("labjournal_xlsx"), clock_date=date(2026, 10, 12))
        return load_source(path, profile), profile

    monkeypatch.setattr(campaign_window, "load_source_interactive", load)
    win = CampaignWindow()
    win.add_sources([DB, JOURNAL])
    tree = win.sources_panel.tree
    for i in range(tree.topLevelItemCount()):  # both scale channels, nothing from the journal
        top = tree.topLevelItem(i)
        for j in range(top.childCount()):
            top.child(j).setCheckState(0, Qt.CheckState.Checked if i == 0 else Qt.CheckState.Unchecked)
    win.resize(1400, 1000)
    win.show()
    qapp.processEvents()
    yield win
    win.close()


def step_id(win: CampaignWindow, row: int) -> str:
    return win.session.intervals().intervals[row].id


def test_double_click_zooms_and_opens_the_inspector(qapp: QApplication, window: CampaignWindow) -> None:
    assert not window.dock_window.isVisible()
    window.intervals_panel._on_double_click(T155_V2, 0)
    qapp.processEvents()
    iv = window.session.intervals().intervals[T155_V2]
    lo, hi = next(iter(window.canvas.plot_items.values())).getViewBox().viewRange()[0]
    assert lo < iv.start < iv.end < hi  # still zooms
    insp = window.window_inspector
    assert window.dock_window.isVisible() and insp.step_id == iv.id
    assert "T Einstellung 155" in insp.title.text() and "v_mfc = 2.0" in insp.title.text()
    assert set(insp.results) == {(DB.stem, "masse"), (DB.stem, "massenstrom_waage")}
    r = insp.results[(DB.stem, "masse")]
    assert r.n == 120 and -2.2 < r.slope * 3600 < -1.3
    assert len(insp.views) == 2
    view = insp.views[0]
    assert view.ols is not None and view.theil_sen is not None and view.support is not None
    assert view.theil_sen.opts["pen"].style() == Qt.PenStyle.DashLine
    assert view.region.getRegion() == pytest.approx((0.0, iv.duration_s))
    rows = insp.table_rows()
    assert ["masse", "Samples (left out / empty)", "120 (0 / 0)", "", ""] in rows
    mean_row = next(r for r in rows if r[1] == "Mean ± SEM_eff")
    assert mean_row[2].startswith("9683.6")
    levels = {insp.checks.item(i).data(Qt.ItemDataRole.UserRole) for i in range(insp.checks.count())}
    assert "warn" in levels  # the mass falls: a trend, and n_eff is small
    assert "AR(1)" in insp.assumptions.toPlainText()


def test_context_menus_offer_inspect(qapp: QApplication, window: CampaignWindow) -> None:
    window.set_bands_visible(True)
    canvas = window.canvas
    iv = window.session.intervals().intervals[T155_V2]
    vb = next(iter(canvas.plot_items.values())).getViewBox()
    assert canvas.action_inspect_step in vb.menu.actions()
    canvas._prepare_step_menu((iv.start + iv.end) / 2)
    assert canvas.action_inspect_step.isVisible() and "T Einstellung 155" in canvas.action_inspect_step.text()
    canvas.action_inspect_step.trigger()
    assert window.window_inspector.step_id == iv.id and window.dock_window.isVisible()
    canvas._prepare_step_menu(iv.start - 1e6)  # no step there
    assert not canvas.action_inspect_step.isVisible()

    panel = window.intervals_panel
    assert panel.action_inspect in panel.table.actions() and panel.action_zoom in panel.table.actions()
    panel.table.selectRow(T155_V1)
    panel.action_inspect.trigger()
    assert window.window_inspector.step_id == step_id(window, T155_V1)


def test_keyboard_steps_through(qapp: QApplication, window: CampaignWindow) -> None:
    window.inspect_step(step_id(window, T155_V2))
    insp = window.window_inspector
    insp.setFocus()
    qapp.processEvents()
    QTest.keyClick(insp, Qt.Key.Key_PageDown)
    assert insp.step_id == step_id(window, T155_V2 + 1)
    assert window.intervals_panel.selected_id() == insp.step_id  # the table follows
    assert "Step 21/38" in insp.title.text()
    QTest.keyClick(insp, Qt.Key.Key_PageUp)
    QTest.keyClick(insp, Qt.Key.Key_PageUp)
    assert insp.step_id == step_id(window, T155_V1)
    insp.btn_next.click()
    assert insp.step_id == step_id(window, T155_V2)
    window.inspect_step(step_id(window, 0))
    assert not insp.btn_prev.isEnabled()
    insp.previous_step()  # stays at the first step
    assert insp.step_id == step_id(window, 0)


def test_excluded_samples_are_crosses(window: CampaignWindow) -> None:
    """A step set by hand around the spikes of the scale's mass flow (rows 194–196)."""
    t = window.session.master_time(DB.stem)
    zone = window.session.campaign.timezone
    iv = window.session.add_interval(master_datetime(t[150], zone), master_datetime(t[230], zone), label="Misread")
    window.refresh_steps()
    window.inspect_step(iv.id)
    insp = window.window_inspector
    assert insp.results[(DB.stem, "masse")].n_excluded == 0  # flags count per channel
    r = insp.results[(DB.stem, "massenstrom_waage")]
    assert list(r.excluded_rows) == [194, 195, 196]
    assert insp.views[0].excluded is None
    crosses = insp.views[1].excluded
    assert crosses is not None and len(crosses.data) == r.n_excluded
    assert crosses.opts["symbol"] == "x"
    assert any("3 points excluded by flags" in insp.checks.item(i).text() for i in range(insp.checks.count()))


def test_dragging_the_window_sets_an_override(qapp: QApplication, window: CampaignWindow, tmp_path: Path) -> None:
    sid = step_id(window, T155_V2)
    window.inspect_step(sid)
    insp = window.window_inspector
    insp.set_window(10, 110)  # as if both edges were dragged
    qapp.processEvents()
    iv = window.session.intervals().get(sid)
    assert iv.window_override and iv.window == pytest.approx((iv.start + 10, iv.start + 110))
    assert sid in window.session.campaign.window_overrides
    assert insp.results[(DB.stem, "masse")].n == 100
    assert "(set by hand)" in insp.title.text() and insp.btn_reset.isEnabled()
    assert insp.views[0].region.lines[0].pen.color().rgb() == OVERRIDE_PEN.rgb()
    assert all(v.region.getRegion() == pytest.approx((10, 110)) for v in insp.views)
    assert "window ✎" in window.intervals_panel.table.item(T155_V2, 4).text()
    # other steps keep the window rule
    assert not window.session.intervals().intervals[T155_V1].window_override

    path = tmp_path / "campaign.json"
    window.save_campaign(path)
    other = CampaignWindow()
    other.open_campaign(path)
    other.inspect_step(sid)
    assert other.session.intervals().get(sid).window == pytest.approx(iv.window)
    assert "(set by hand)" in other.window_inspector.title.text()
    other.close()

    insp.btn_reset.click()
    qapp.processEvents()
    assert not window.session.campaign.window_overrides
    assert insp.results[(DB.stem, "masse")].n == 120 and not insp.btn_reset.isEnabled()


def test_region_cannot_leave_the_step(qapp: QApplication, window: CampaignWindow) -> None:
    sid = step_id(window, T155_V2)
    window.inspect_step(sid)
    window.window_inspector.set_window(-50, 500)
    qapp.processEvents()
    iv = window.session.intervals().get(sid)
    assert iv.window == pytest.approx((iv.start, iv.end))  # clamped to the step: no override needed
    assert not window.session.campaign.window_overrides


def test_channel_selection_follows_the_sources_dock(qapp: QApplication, window: CampaignWindow) -> None:
    window.inspect_step(step_id(window, T155_V2))
    top = window.sources_panel.tree.topLevelItem(0)
    top.child(1).setCheckState(0, Qt.CheckState.Unchecked)
    qapp.processEvents()
    assert list(window.window_inspector.results) == [(DB.stem, "masse")]
    top.child(0).setCheckState(0, Qt.CheckState.Unchecked)
    qapp.processEvents()
    insp = window.window_inspector
    assert insp.results == {} and "Select channels" in insp.checks.item(0).text()


def test_window_rule_change_updates_the_inspector(window: CampaignWindow) -> None:
    window.inspect_step(step_id(window, T155_V2))
    panel = window.intervals_panel
    panel.length.setValue(60)
    panel._apply_rule()
    r = window.window_inspector.results[(DB.stem, "masse")]
    assert 55 <= r.n <= 62
    assert "window 60–120 s of 120 s (window rule)" in window.window_inspector.title.text()


def test_values_rounded_to_their_uncertainty() -> None:
    assert value_and_u(9683.63441, 0.00681) == ("9683.6344", "0.0068")
    assert value_and_u(-1.7372, 0.172) == ("-1.74", "0.17")
    assert value_and_u(12345.6, 230.0) == ("12346", "2.3e+02")
    assert value_and_u(float("nan"), 1.0) == ("—", "1")
