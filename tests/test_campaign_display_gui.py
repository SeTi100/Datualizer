"""Campaign window, display per channel: plot style, y scale with edge markers, alias (offscreen)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtWidgets import QApplication

from datualizer_core.campaign import ChannelDisplay, YScale, YScaleMode
from datualizer_core.ingestion.profiles import get_profile
from datualizer_core.schema import PlotStyle
from datualizer_gui.campaign_window import CampaignWindow
from datualizer_gui.components.curves import FastCurve, peak_lod, plot_arrays, step_arrays

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
FID_DAT = SA / "fid_ifid_12_10_extract.dat"
DB = SA / "instrument_reader_12_10_extract.db"
JOURNAL = SA / "labjournal_12_10_extract.xlsx"
NO_GAPS = np.zeros(0, dtype=np.int64)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture()
def window(qapp: QApplication) -> CampaignWindow:
    win = CampaignWindow()
    profile = get_profile("labjournal_xlsx")
    profile.excel.sheet = "12.10.2026"
    profile.ingestion.time.clock_date = date(2026, 10, 12)
    win.session.add_source(JOURNAL, profile)
    win.session.add_source(DB)
    win.session.add_source(FID_DAT)
    win._refresh_all()
    panel = win.sources_panel
    for sid, ch in panel.checked_channels():
        panel.set_checked(sid, ch, False)
    panel.set_checked(JOURNAL.stem, "v_mfc")
    panel.set_checked(DB.stem, "massenstrom_waage")
    qapp.processEvents()
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()
    yield win
    win.close()


# ---------------------------------------------------------------- plot styles
def test_step_arrays_hold_the_value_until_the_next_sample() -> None:
    x = np.array([0.0, 10.0, 20.0, 30.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    connect = np.array([True, False, True, False])  # a break between 10 and 20
    xs, ys, cs = step_arrays(x, y, connect)
    assert xs.tolist() == [0.0, 10.0, 10.0, 20.0, 30.0, 30.0]
    assert ys.tolist() == [1.0, 1.0, 2.0, 3.0, 3.0, 4.0]
    assert cs.tolist() == [True, True, False, True, True, False]  # 2 does not hold across the break


def test_steps_survive_peak_downsampling_without_bridging_gaps() -> None:
    """Many steps on few columns: reduced, but no line over the gap between the two halves."""
    x = np.concatenate([np.arange(5000.0), np.arange(5000.0) + 20000])
    y = np.sin(x / 50)
    sx, sy, sc, _ = plot_arrays(x, y, np.array([5000]))
    xs, ys, cs = peak_lod(*step_arrays(sx, sy, sc), None, 100)
    assert len(xs) < 1000
    gap = np.flatnonzero((xs[:-1] < 5000) & (xs[1:] >= 20000))
    assert len(gap) == 1 and not cs[gap[0]]


def test_journal_is_drawn_as_steps_by_profile(window: CampaignWindow) -> None:
    """Labjournal channels are measurement steps: no zigzag between the steps."""
    curve = window.canvas.curves[f"{JOURNAL.stem}/v_mfc"]
    assert curve.style is PlotStyle.STEP
    xs, ys = curve.getData()
    drawn = curve.opts["connect"][:-1]  # segment i -> i+1 is drawn
    samples_x, _ = curve.getOriginalDataset()
    assert len(xs) > len(samples_x)  # risers added
    horizontal = drawn & (xs[1:] > xs[:-1])
    assert horizontal.any()
    assert np.all(ys[1:][horizontal] == ys[:-1][horizontal])  # no slanted segment: flat or vertical


@pytest.mark.parametrize("style", list(PlotStyle))
def test_every_style_keeps_all_samples(qapp: QApplication, window: CampaignWindow, style: PlotStyle) -> None:
    key = f"{DB.stem}/massenstrom_waage"
    window.sources_panel.apply_display(DB.stem, "massenstrom_waage", ChannelDisplay(style=style))
    qapp.processEvents()
    curve = window.canvas.curves[key]
    assert isinstance(curve, FastCurve) and curve.style is style
    tr = window.session.track(DB.stem, "massenstrom_waage")
    assert len(curve.getOriginalDataset()[0]) == np.isfinite(tr.y).sum()
    has_markers = style in (PlotStyle.POINTS, PlotStyle.LINE_POINTS)
    assert (curve.markers is not None) == has_markers
    if has_markers:
        assert curve.markers in window.canvas.plot_items[key].listDataItems() or curve.markers.scene() is not None
        assert len(curve.markers.data) > 0
    if style is PlotStyle.POINTS:
        assert not curve.getData()[0].size or not np.any(curve.opts["connect"])  # no line at all


def test_style_from_the_context_menu(qapp: QApplication, window: CampaignWindow) -> None:
    menu = window.sources_panel.channel_menu(JOURNAL.stem, "v_mfc")
    styles = next(a.menu() for a in menu.actions() if a.text() == "Plot Style")
    checked = [a.data() for a in styles.actions() if a.isChecked()]
    assert checked == [PlotStyle.STEP.value]
    next(a for a in styles.actions() if a.data() == PlotStyle.POINTS.value).trigger()
    qapp.processEvents()
    assert window.canvas.curves[f"{JOURNAL.stem}/v_mfc"].style is PlotStyle.POINTS
    assert window.session.campaign.source(JOURNAL.stem).display["v_mfc"].style is PlotStyle.POINTS


# --------------------------------------------------------------------- y scale
def test_robust_scale_marks_outliers_at_the_edge(qapp: QApplication, window: CampaignWindow) -> None:
    """massenstrom_waage: the robust range shows the bulk; the extreme values are marked, not dropped."""
    key = f"{DB.stem}/massenstrom_waage"
    tr = window.session.track(DB.stem, "massenstrom_waage")
    window.sources_panel.apply_display(DB.stem, "massenstrom_waage", ChannelDisplay(
        y_scale=YScale(mode=YScaleMode.ROBUST, low_percentile=5, high_percentile=95)
    ))
    qapp.processEvents()
    lo, hi = window.canvas.plot_items[key].getViewBox().viewRange()[1]
    finite = tr.y[np.isfinite(tr.y)]
    assert finite.min() < lo and hi < finite.max()  # narrower than all values
    markers = window.canvas.edge_markers[key]
    above, below = markers.counts
    assert above > 0 and below > 0
    assert len(window.canvas.curves[key].getOriginalDataset()[0]) == len(finite)  # nothing dropped
    window.canvas.reset_zoom()  # keeps the robust range
    qapp.processEvents()
    assert window.canvas.plot_items[key].getViewBox().viewRange()[1] == pytest.approx([lo, hi])


def test_fixed_scale(qapp: QApplication, window: CampaignWindow) -> None:
    key = f"{DB.stem}/massenstrom_waage"
    window.sources_panel.apply_display(DB.stem, "massenstrom_waage", ChannelDisplay(
        y_scale=YScale(mode=YScaleMode.FIXED, min=0.0, max=10.0)
    ))
    qapp.processEvents()
    lo, hi = window.canvas.plot_items[key].getViewBox().viewRange()[1]
    assert lo == pytest.approx(-0.2) and hi == pytest.approx(10.2)  # 2 % padding
    assert window.canvas.edge_markers[key].counts[1] > 0  # negative flows below the range
    assert f"{JOURNAL.stem}/v_mfc" not in window.canvas.edge_markers  # auto: nothing to mark


# ----------------------------------------------------------------------- alias
def test_alias_in_axis_tree_and_readout(qapp: QApplication, window: CampaignWindow) -> None:
    window.sources_panel.set_checked(JOURNAL.stem, "col_9")
    window.sources_panel.apply_display(JOURNAL.stem, "col_9", ChannelDisplay(alias="Massenstrom Waage"))
    qapp.processEvents()
    key = f"{JOURNAL.stem}/col_9"
    assert key in window.canvas.plot_items  # still checked after the tree was rebuilt
    assert "Massenstrom Waage" in window.canvas.plot_items[key].getAxis("left").labelText
    tree = window.sources_panel.tree
    journal = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).text(1) == "labjournal_xlsx")
    names = [journal.child(j).text(0) for j in range(journal.childCount())]
    assert "Massenstrom Waage (col_9)" in names
    x = window.canvas.curves[key].getOriginalDataset()[0][3]
    assert "Massenstrom Waage =" in window.canvas.update_crosshair(x)
    assert "col_9" in window.session.datasets[JOURNAL.stem].df.columns


# ---------------------------------------------------------------------- dialog
def test_dialog_roundtrip(qapp: QApplication, window: CampaignWindow) -> None:
    panel = window.sources_panel
    dialog = panel.display_dialog(DB.stem, "massenstrom_waage")
    assert dialog.style() is None  # profile default
    assert not dialog.min_spin.isEnabled()
    dialog.alias_edit.setText("Waage")
    dialog.style_combo.setCurrentIndex(dialog.style_combo.findData(PlotStyle.LINE_POINTS.value))
    dialog.scale_combo.setCurrentIndex(dialog.scale_combo.findData(YScaleMode.FIXED.value))
    assert dialog.min_spin.isEnabled()
    dialog.min_spin.setValue(5.0)
    dialog.max_spin.setValue(1.0)
    dialog.accept()
    assert dialog.result() == 0 and "min < max" in dialog.error_label.text()  # invalid range refused
    dialog.max_spin.setValue(20.0)
    dialog.accept()
    assert dialog.result() == 1
    panel.apply_display(DB.stem, "massenstrom_waage", dialog.display())
    stored = window.session.campaign.source(DB.stem).display["massenstrom_waage"]
    assert stored == ChannelDisplay(
        alias="Waage", style=PlotStyle.LINE_POINTS, y_scale=YScale(mode=YScaleMode.FIXED, min=5.0, max=20.0)
    )
    again = panel.display_dialog(DB.stem, "massenstrom_waage")  # opens with what is stored
    assert (again.alias_edit.text(), again.style(), again.max_spin.value()) == (
        "Waage", PlotStyle.LINE_POINTS, 20.0
    )


def test_settings_menu_has_channel_display(window: CampaignWindow) -> None:
    menus = {a.text(): a.menu() for a in window.menuBar().actions()}
    assert window.action_channel_display in menus["S&ettings"].actions()


def test_display_survives_save_and_open(qapp: QApplication, window: CampaignWindow, tmp_path: Path) -> None:
    window.sources_panel.apply_display(JOURNAL.stem, "col_9", ChannelDisplay(alias="Massenstrom Waage"))
    target = tmp_path / "campaign.json"
    window.save_campaign(target)
    other = CampaignWindow()
    other.open_campaign(target)
    assert other.session.campaign.source(JOURNAL.stem).display["col_9"].alias == "Massenstrom Waage"
    assert other.session.track(JOURNAL.stem, "v_mfc").style is PlotStyle.STEP
    other.close()


def test_points_curve_draws_markers_on_screen(qapp: QApplication, window: CampaignWindow) -> None:
    """Pixel check: points mode paints marker dots in the curve colour."""
    window.sources_panel.apply_display(JOURNAL.stem, "v_mfc", ChannelDisplay(style=PlotStyle.POINTS))
    qapp.processEvents()
    curve = window.canvas.curves[f"{JOURNAL.stem}/v_mfc"]
    image = window.canvas.graphics_layout.grab().toImage()
    color = curve.markers.opts["brush"].color().rgb()
    vb = window.canvas.plot_items[f"{JOURNAL.stem}/v_mfc"].getViewBox()
    xs, ys = curve.getOriginalDataset()
    hits = 0
    for x, y in zip(xs, ys):
        pt = window.canvas.graphics_layout.mapFromScene(vb.mapViewToScene(pg.QtCore.QPointF(x, y)))
        around = {
            image.pixel(int(pt.x()) + dx, int(pt.y()) + dy)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if 0 <= int(pt.x()) + dx < image.width() and 0 <= int(pt.y()) + dy < image.height()
        }
        hits += color in around
    assert hits > len(xs) // 2
