"""Measurement steps in the campaign window (Etappe 2b, step 5): bands, Marker ○ ○, Steps dock."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pyqtgraph as pg
import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from datualizer_core.campaign import to_epoch_s
from datualizer_core.ingestion.profiles import detect_profile, get_profile
from datualizer_core.ingestion.source_loader import load_source, with_choices
from datualizer_gui import campaign_window
from datualizer_gui.campaign_window import CampaignWindow
from datualizer_gui.components.interval_bands import MARKER_END, MARKER_START, marker_icon
from datualizer_gui.components.intervals_panel import MappingDialog
from datualizer_gui.settings import INTERVAL_BANDS, get_bool

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
JOURNAL = SA / "labjournal_12_10_extract.xlsx"
DB = SA / "instrument_reader_12_10_extract.db"
ZONE = "Europe/Zurich"


def zurich(*args) -> float:
    return to_epoch_s(datetime(*args), ZONE)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture()
def window(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> CampaignWindow:
    """Scale readings (SQLite) plus the lab journal of the same day, as if picked in the dialogs."""

    def load(_parent, path, *, ask_clock_date=False):
        path = Path(path)
        profile = detect_profile(path)
        if path == JOURNAL:  # the date the dialog would ask for
            profile = with_choices(path, get_profile("labjournal_xlsx"), clock_date=date(2026, 10, 12))
        return load_source(path, profile), profile

    monkeypatch.setattr(campaign_window, "load_source_interactive", load)
    win = CampaignWindow()
    win.add_sources([DB, JOURNAL])
    win.set_bands_visible(True)  # off by default
    qapp.processEvents()
    return win


def steps_in(win: CampaignWindow) -> int:
    return len(win.session.intervals().intervals)


def test_journal_steps_are_read_with_the_profile_suggestion(window: CampaignWindow) -> None:
    assert [m.source_id for m in window.session.campaign.interval_mappings] == [JOURNAL.stem]
    assert steps_in(window) == 38
    assert "38 steps" in window.status_bar.currentMessage()
    assert window.intervals_panel.table.rowCount() == 38
    assert window.intervals_panel.table.item(0, 0).text() == "Konditionierung"
    assert "no start time" in window.intervals_panel.problem_label.text()


def test_bands_over_all_subplots(window: CampaignWindow) -> None:
    canvas = window.canvas
    plots = list(canvas.plot_items.values())
    assert len(plots) == 2  # first channel of the scale and of the journal
    # one painting item per subplot (not one region per step and subplot: that lagged)
    assert len(canvas.steps_items) == len(plots) + (canvas.coverage_plot is not None)
    assert all(len(item.intervals) == 38 for item in canvas.steps_items)
    assert canvas.bands == []  # draggable regions only for steps set by hand
    text = canvas.update_crosshair(zurich(2026, 10, 12, 15, 47))
    assert text.endswith("step: T Einstellung 155")


def test_marker_icon_shows_the_state() -> None:
    def colors(state: int) -> tuple[QColor, QColor]:
        img = marker_icon(state).pixmap(36, 16).toImage()
        return img.pixelColor(8, 8), img.pixelColor(26, 8)

    empty, _ = colors(0)
    assert empty.alpha() == 0  # ○ ○
    first, second = colors(1)
    assert first.rgb() == MARKER_START.rgb() and second.alpha() == 0  # ● ○ green
    first, second = colors(2)
    assert (first.rgb(), second.rgb()) == (MARKER_START.rgb(), MARKER_END.rgb())


def test_marker_sets_a_step_in_two_clicks(qapp: QApplication, window: CampaignWindow) -> None:
    canvas = window.canvas
    assert canvas.marker_state == 0 and canvas.action_marker.text() == "Marker ○ ○"
    canvas.place_marker(zurich(2026, 10, 12, 15, 52))
    assert canvas.marker_state == 1 and canvas.action_marker.text() == "Marker ● ○"
    assert len(canvas._pending_lines) == len(canvas.plot_items)
    canvas.place_marker(zurich(2026, 10, 12, 15, 48))  # before the start: swapped, no error
    qapp.processEvents()
    assert canvas.marker_state == 0 and canvas._pending_lines == []
    (iv,) = window.session.campaign.intervals
    assert (iv.start, iv.end, iv.label) == (datetime(2026, 10, 12, 15, 48), datetime(2026, 10, 12, 15, 52), "Step 1")
    manual = [b for b in canvas.bands if b.interval.manual]
    assert len(manual) == len(canvas.plot_items) and manual[0].region.movable
    assert steps_in(window) == 39


def test_discard_marker(window: CampaignWindow) -> None:
    canvas = window.canvas
    canvas.place_marker(zurich(2026, 10, 12, 15, 50))
    canvas._prepare_step_menu(zurich(2026, 10, 12, 15, 51))
    assert canvas.action_marker_cancel.isVisible()
    canvas.cancel_marker()
    assert canvas.marker_state == 0 and canvas._pending_lines == []
    assert window.session.campaign.intervals == []


def test_dragging_a_manual_band_moves_the_step(qapp: QApplication, window: CampaignWindow) -> None:
    canvas = window.canvas
    canvas.place_marker(zurich(2026, 10, 12, 15, 48))
    canvas.place_marker(zurich(2026, 10, 12, 15, 50))
    band = next(b for b in canvas.bands if b.interval.manual)
    band.region.setRegion((zurich(2026, 10, 12, 15, 49), zurich(2026, 10, 12, 15, 53)))
    band.region.sigRegionChangeFinished.emit(band.region)
    qapp.processEvents()
    (iv,) = window.session.campaign.intervals
    assert (iv.start, iv.end) == (datetime(2026, 10, 12, 15, 49), datetime(2026, 10, 12, 15, 53))


def test_context_menu_entries(window: CampaignWindow) -> None:
    canvas = window.canvas
    vb = next(iter(canvas.plot_items.values())).getViewBox()
    assert canvas.action_marker in vb.menu.actions()
    canvas._prepare_step_menu(zurich(2026, 10, 12, 15, 47))  # inside a journal step
    assert canvas.action_zoom_step.isVisible() and not canvas.action_delete_step.isVisible()
    canvas.place_marker(zurich(2026, 10, 12, 15, 48))
    canvas.place_marker(zurich(2026, 10, 12, 15, 49))
    canvas._prepare_step_menu(zurich(2026, 10, 12, 15, 48, 30))  # the manual step wins
    assert canvas.action_delete_step.isVisible() and canvas.action_rename_step.isVisible()
    window.rename_step("m1", "Blindwert")
    assert window.session.intervals().get("m1").label == "Blindwert"
    canvas.action_delete_step.trigger()
    assert window.session.campaign.intervals == []


def test_double_click_in_the_table_zooms(qapp: QApplication, window: CampaignWindow) -> None:
    window.show()
    window.intervals_panel._on_double_click(19, 0)  # T Einstellung 155
    qapp.processEvents()
    iv = window.session.intervals().intervals[19]
    lo, hi = next(iter(window.canvas.plot_items.values())).getViewBox().viewRange()[0]
    assert lo < iv.start < iv.end < hi and hi - lo < 2 * iv.duration_s
    window.close()


def test_window_rule_from_the_panel(window: CampaignWindow) -> None:
    panel = window.intervals_panel
    panel.length.setValue(120)
    panel._apply_rule()
    assert window.session.campaign.window_rule.length_s == 120
    iv = window.session.intervals().intervals[1]
    assert iv.window == (iv.end - 120, iv.end)
    assert window.canvas.steps_items[0].intervals[1].window == iv.window  # drawn darker inside the band


def test_mapping_dialog_prefilled_and_changeable(window: CampaignWindow) -> None:
    dialog = MappingDialog(window.session, window, source_id=JOURNAL.stem)
    m = dialog.mapping()
    assert (m.end.column, m.label_columns, m.fill_down) == ("Zeit_1", ["col_A"], ["col_A"])  # header texts
    assert "38 steps" in dialog.preview.text()
    dialog.end_kind.setCurrentIndex(dialog.end_kind.findText("Fixed length"))
    dialog.end_seconds.setValue(60)
    dialog.refresh_now()  # the preview follows after a short pause while typing
    assert "38 steps" in dialog.preview.text()
    dialog.accept()
    assert window.session.intervals().intervals[0].duration_s == 60


def test_bands_are_off_by_default(qapp: QApplication) -> None:
    win = CampaignWindow()
    assert not win.bands_visible and not win.action_bands.isChecked()
    win.close()


def test_steps_item_paints_bands_and_window(qapp: QApplication, window: CampaignWindow) -> None:
    """Zoomed onto one step: band colour inside the step, lighter window, background outside."""
    window.session.set_window_rule(window.session.campaign.window_rule.model_copy(update={"length_s": 120}))
    window.refresh_steps()
    window.resize(1200, 700)
    window.show()
    steps = window.session.intervals().intervals
    # a step longer than twice the window, with a pause before it
    i = next(k for k in range(1, len(steps)) if steps[k].duration_s >= 240 and steps[k].start - steps[k - 1].end >= 20)
    prev, iv = steps[i - 1], steps[i]
    window.canvas.set_view_range(prev.end - 5, iv.end + 5)
    qapp.processEvents()
    plot = next(iter(window.canvas.plot_items.values()))
    gl = window.canvas.graphics_layout
    image = gl.grab().toImage()
    vb = plot.getViewBox()
    y = vb.viewRange()[1][0] + 0.25 * (vb.viewRange()[1][1] - vb.viewRange()[1][0])

    def color(t: float) -> int:
        pt = gl.mapFromScene(vb.mapViewToScene(pg.QtCore.QPointF(t, y)))
        return image.pixel(pt.x(), pt.y())

    outside, band, win = color((prev.end + iv.start) / 2), color(iv.start + 10), color(iv.end - 10)
    assert len({outside, band, win}) == 3
    window.close()


def test_bands_toggle_in_settings(window: CampaignWindow) -> None:
    assert window.bands_visible and window.canvas.steps_items
    window.action_bands.trigger()
    assert window.canvas.steps_items == [] and window.intervals_panel.table.rowCount() == 38
    assert not get_bool(INTERVAL_BANDS)  # remembered (no throwaway window: see test_campaign_gui)
    window.action_bands.trigger()
    assert window.canvas.steps_items
