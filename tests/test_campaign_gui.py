"""Campaign window (Etappe 2b, step 3): sources dock, clock editor, plot on master time."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtWidgets import QApplication

from datualizer_core.campaign import to_epoch_s
from datualizer_gui.campaign_window import CampaignWindow
from datualizer_gui.components.campaign_canvas import connect_mask, utc_offset_s
from datualizer_gui.main_window import DatualizerMainWindow

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
FID_DAT = SA / "fid_ifid_12_10_extract.dat"
PICO = SA / "picolog_12_10_extract.csv"
ZONE = "Europe/Zurich"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture()
def pico_copy(tmp_path: Path) -> Path:
    f = tmp_path / PICO.name
    shutil.copyfile(PICO, f)
    saved = datetime(2026, 10, 12, 16, 0, 21, tzinfo=timezone.utc).timestamp()
    os.utime(f, (saved, saved))
    return f


@pytest.fixture()
def window(qapp: QApplication, pico_copy: Path) -> CampaignWindow:
    win = CampaignWindow()
    win.add_sources([FID_DAT, pico_copy])
    qapp.processEvents()
    return win


def test_connect_mask_breaks_at_nan_and_gaps() -> None:
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    assert connect_mask(y, np.array([4])).tolist() == [True, False, False, False, True, False]


def test_utc_offset_follows_daylight_saving() -> None:
    assert utc_offset_s(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp(), ZONE) == 7200
    assert utc_offset_s(datetime(2026, 12, 1, tzinfo=timezone.utc).timestamp(), ZONE) == 3600


def test_sources_and_default_channels(window: CampaignWindow) -> None:
    tree = window.sources_panel.tree
    assert tree.topLevelItemCount() == 2
    assert [tree.topLevelItem(i).text(1) for i in range(2)] == ["testa_ifid_dat", "picolog_csv"]
    assert window.sources_panel.checked_channels() == [
        (FID_DAT.stem, "conc"), (PICO.stem, "kanal_1_kat_oben")
    ]
    assert len(window.canvas.plot_items) == 2
    axis = next(iter(window.canvas.plot_items.values())).getAxis("bottom")
    assert isinstance(axis, pg.DateAxisItem)


def test_curves_use_master_time(window: CampaignWindow) -> None:
    x = window.canvas.curves[f"{FID_DAT.stem}/conc"].getOriginalDataset()[0]
    assert x[0] == to_epoch_s(datetime(2026, 10, 12, 14, 27, 2), ZONE)
    text = window.canvas.update_crosshair(x[0])
    assert text.startswith("12.10.2026 14:27:02")


def test_clock_editor_shifts_only_that_source(qapp: QApplication, window: CampaignWindow) -> None:
    pico_key = f"{PICO.stem}/kanal_1_kat_oben"
    pico_before = window.canvas.curves[pico_key].getOriginalDataset()[0]
    panel = window.sources_panel
    panel.select_source(FID_DAT.stem)
    panel.offset_spin.setValue(970.0)
    panel.offset_spin.editingFinished.emit()
    qapp.processEvents()
    clock = window.session.campaign.source(FID_DAT.stem).clock
    assert (clock.offset_s, clock.method) == (970.0, "manual")
    x = window.canvas.curves[f"{FID_DAT.stem}/conc"].getOriginalDataset()[0]
    assert x[0] == to_epoch_s(datetime(2026, 10, 12, 14, 43, 12), ZONE)
    np.testing.assert_array_equal(window.canvas.curves[pico_key].getOriginalDataset()[0], pico_before)
    assert panel.tree.topLevelItem(0).text(3) == "+970.0 s"


def test_checking_channels_adds_subplots(qapp: QApplication, window: CampaignWindow) -> None:
    window.sources_panel.set_checked(PICO.stem, "kanal_8_heizung")
    qapp.processEvents()
    assert f"{PICO.stem}/kanal_8_heizung" in window.canvas.plot_items
    assert len(window.canvas.plot_items) == 3


def test_save_and_reopen(qapp: QApplication, window: CampaignWindow, tmp_path: Path) -> None:
    window.session.campaign.source(FID_DAT.stem).clock.offset_s = 970
    target = tmp_path / "campaign.json"
    window.save_campaign(target)
    other = CampaignWindow()
    other.open_campaign(target)
    qapp.processEvents()
    assert other.session.campaign.source(FID_DAT.stem).clock.offset_s == 970
    assert other.sources_panel.tree.topLevelItemCount() == 2
    assert other.session.warnings == []


def test_remove_source(qapp: QApplication, window: CampaignWindow) -> None:
    window.remove_source(PICO.stem)
    qapp.processEvents()
    assert window.sources_panel.tree.topLevelItemCount() == 1
    assert list(window.canvas.plot_items) == [f"{FID_DAT.stem}/conc"]


def test_main_window_opens_one_campaign_window(qapp: QApplication) -> None:
    main = DatualizerMainWindow(auto_load=False)
    first = main.open_campaign_window()
    assert main.open_campaign_window() is first
    first.close()


def test_rebuilt_plot_shows_all_data(qapp: QApplication, window: CampaignWindow) -> None:
    """Regression: a rebuild must not inherit the default [0, 1] range of a not yet auto-ranged view."""
    window.show()  # pyqtgraph computes auto-range for visible views
    window.refresh_plot()
    qapp.processEvents()
    for key, curve in window.canvas.curves.items():
        x = curve.getOriginalDataset()[0]
        lo, hi = window.canvas.plot_items[key].getViewBox().viewRange()[0]
        assert lo <= np.nanmin(x) and np.nanmax(x) <= hi, key
    window.close()


def test_user_zoom_survives_a_rebuild(qapp: QApplication, window: CampaignWindow) -> None:
    window.show()
    qapp.processEvents()
    t0 = to_epoch_s(datetime(2026, 10, 12, 14, 28), ZONE)
    window.canvas.set_view_range(t0, t0 + 60)
    window.sources_panel.set_checked(PICO.stem, "kanal_8_heizung")
    qapp.processEvents()
    lo, hi = next(iter(window.canvas.plot_items.values())).getViewBox().viewRange()[0]
    assert (round(lo), round(hi)) == (round(t0), round(t0 + 60))
    window.canvas.reset_zoom()
    qapp.processEvents()
    lo, hi = next(iter(window.canvas.plot_items.values())).getViewBox().viewRange()[0]
    for curve in window.canvas.curves.values():  # back to all tracks of all sources
        x = curve.getOriginalDataset()[0]
        assert lo <= np.nanmin(x) and np.nanmax(x) <= hi
    window.close()


def test_readout_never_resizes_the_window(qapp: QApplication, window: CampaignWindow) -> None:
    """Hovering from left to right must not widen the window when values get longer (99 -> 100)."""
    window.resize(900, 600)
    window.show()
    qapp.processEvents()
    width = window.width()
    label = window.canvas.crosshair_label
    label.setText("x" * 2000)  # far longer than the window
    qapp.processEvents()
    assert window.width() == width
    assert label.minimumSizeHint().width() == 0
    window.close()
    main = DatualizerMainWindow(auto_load=False)  # the main window's readout behaves the same
    assert main.plot_canvas.crosshair_label.minimumSizeHint().width() == 0


def test_readout_values_have_a_fixed_width(window: CampaignWindow) -> None:
    from datualizer_gui.components.campaign_canvas import format_value

    assert len(format_value(99.5)) == len(format_value(100.25)) == len(format_value(-1234.5))
    assert len(format_value(float("nan"))) == len(format_value(1.0))


# ------------------------------------------------------------ coverage strip (step 4)
def test_coverage_is_off_by_default_and_toggled_in_settings(qapp: QApplication, window: CampaignWindow) -> None:
    assert not window.coverage_visible and window.canvas.coverage_plot is None
    menus = {a.text(): a.menu() for a in window.menuBar().actions()}
    tools = {a.text(): a.menu() for a in menus["S&ettings"].actions()}["&Tools"]
    assert window.action_coverage in tools.actions()
    window.action_coverage.trigger()
    qapp.processEvents()
    assert window.canvas.coverage_plot is not None
    assert CampaignWindow().coverage_visible  # remembered for the next window
    window.action_coverage.trigger()
    assert window.canvas.coverage_plot is None and not CampaignWindow().coverage_visible


@pytest.fixture()
def coverage_window(qapp: QApplication, window: CampaignWindow) -> CampaignWindow:
    window.set_coverage_visible(True)
    qapp.processEvents()
    return window


def test_coverage_strip_has_one_lane_per_source(coverage_window: CampaignWindow) -> None:
    window = coverage_window
    canvas = window.canvas
    assert canvas.coverage_plot is not None
    assert [c.source_id for c in canvas.coverage] == [FID_DAT.stem, PICO.stem]
    ticks = canvas.coverage_plot.getAxis("left")._tickLevels[0]
    assert [label for _, label in ticks] == [FID_DAT.name, PICO.name]
    # every subplot follows the strip, and all left axes have the same width
    for p in canvas.plot_items.values():
        assert p.getViewBox().linkedView(0) is canvas.coverage_plot.getViewBox()
    widths = {p.getAxis("left").width() for p in [canvas.coverage_plot, *canvas.plot_items.values()]}
    assert len(widths) == 1


def test_coverage_shown_without_checked_channels(qapp: QApplication, coverage_window: CampaignWindow) -> None:
    window = coverage_window
    for sid, ch in window.sources_panel.checked_channels():
        window.sources_panel.set_checked(sid, ch, False)
    qapp.processEvents()
    assert window.canvas.plot_items == {}
    assert window.canvas.coverage_plot is not None
    assert window.canvas.coverage_plot.getAxis("bottom").style["showValues"]
    lo, hi = window.canvas.full_range()
    assert (lo, hi) == (window.session.span(FID_DAT.stem)[0], window.session.span(PICO.stem)[1])


def test_readout_names_the_gap(coverage_window: CampaignWindow) -> None:
    window = coverage_window
    gap = window.canvas.coverage[0].gaps[0]
    text = window.canvas.update_crosshair((gap.start + gap.end) / 2)
    assert text.endswith(f"{FID_DAT.name}: gap 11 s (time gap)")
    assert "gap" not in window.canvas.update_crosshair(gap.start - 5)


def test_hatched_gap_is_painted(qapp: QApplication, coverage_window: CampaignWindow) -> None:
    window = coverage_window
    """Zoomed onto the FID gap, the lane shows filled bars left and right and a hatch in between."""
    window.resize(1200, 700)
    window.show()
    gap = window.canvas.coverage[0].gaps[0]
    window.canvas.set_view_range(gap.start - 20, gap.end + 20)
    qapp.processEvents()
    plot = window.canvas.coverage_plot
    image = window.canvas.graphics_layout.grab().toImage()
    vb = plot.getViewBox()

    def pixels(t0: float, t1: float) -> set[int]:
        out = set()
        for t in np.linspace(t0, t1, 15):
            pt = vb.mapViewToScene(pg.QtCore.QPointF(t, 0.0))
            view_pt = window.canvas.graphics_layout.mapFromScene(pt)
            out.add(image.pixel(view_pt.x(), view_pt.y()))
        return out

    covered = pixels(gap.start - 15, gap.start - 3)
    hatched = pixels(gap.start + 2, gap.end - 2)
    background = pg.mkColor("#18181c").rgb()
    assert len(covered) == 1 and covered != {background}  # solid fill in the source colour
    assert background in hatched and len(hatched) > 1  # stripes alternate with the background
    window.close()
