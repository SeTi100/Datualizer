"""Campaign window: zoom behaviour, time axis (clock, date + time, t - t0), time zone, subplot height."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from datualizer_core import TimeConfigError
from datualizer_core.campaign import Campaign, CampaignSession, to_epoch_s
from datualizer_gui.campaign_window import CampaignWindow
from datualizer_gui.components.time_axis import (
    AbsoluteTimeAxis,
    RelativeTimeAxis,
    TimeAxisMode,
    TimeViewBox,
    format_absolute,
    format_relative,
    nice_spacing,
)
from datualizer_gui.settings import TIME_AXIS, get_str

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
FID_DAT = SA / "fid_ifid_12_10_extract.dat"
PICO = SA / "picolog_12_10_extract.csv"
ZONE = "Europe/Zurich"
FID_KEY = f"{FID_DAT.stem}/conc"


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
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()
    yield win
    win.close()


# ------------------------------------------------------------------ formatting
def test_relative_labels() -> None:
    assert format_relative(12.5, 0.5) == "12.5 s"
    assert format_relative(-5, 1) == "-5 s"
    assert format_relative(3725, 60) == "1:02:05"
    assert format_relative(-90, 30) == "-0:01:30"
    assert format_relative(61.25, 0.25) == "0:01:01.250"


def test_absolute_labels_follow_the_spacing() -> None:
    t = to_epoch_s(datetime(2026, 10, 12, 14, 27, 2, 500000), ZONE)
    offset = 7200
    assert format_absolute(t, offset, TimeAxisMode.CLOCK, 600) == "14:27"
    assert format_absolute(t, offset, TimeAxisMode.CLOCK, 10) == "14:27:02"
    assert format_absolute(t, offset, TimeAxisMode.CLOCK, 0.1) == "14:27:02.500"
    assert format_absolute(t, offset, TimeAxisMode.DATETIME, 10) == "12.10.2026 14:27:02"


def test_nice_spacing() -> None:
    assert nice_spacing(3600, 6) == 600
    assert nice_spacing(50, 10) == 5
    assert nice_spacing(0.5, 5) == 0.1


# ---------------------------------------------------------------------- zoom
class _Wheel:
    """Minimal wheel event for ViewBox.wheelEvent."""

    def __init__(self, pos: QPointF) -> None:
        self._pos = pos

    def delta(self) -> int:
        return 120

    def pos(self) -> QPointF:
        return self._pos

    def accept(self) -> None: ...

    def ignore(self) -> None: ...


def test_wheel_zooms_only_time_by_default(qapp: QApplication, window: CampaignWindow) -> None:
    assert window.action_wheel_time.isChecked()  # default on
    vb = window.canvas.plot_items[FID_KEY].getViewBox()
    assert isinstance(vb, TimeViewBox) and vb.wheel_time_only
    vb.enableAutoRange(axis="y", enable=False)  # make y changes visible
    (x0, x1), y = vb.viewRange()
    vb.wheelEvent(_Wheel(QPointF(vb.width() / 2, vb.height() / 2)))
    (nx0, nx1), ny = vb.viewRange()
    assert nx1 - nx0 < x1 - x0
    assert ny == pytest.approx(y)
    window.action_wheel_time.setChecked(False)  # both axes again
    vb.wheelEvent(_Wheel(QPointF(vb.width() / 2, vb.height() / 2)))
    assert vb.viewRange()[1][1] - vb.viewRange()[1][0] < y[1] - y[0]


def test_y_fits_the_visible_time_span(qapp: QApplication, window: CampaignWindow) -> None:
    assert window.action_y_fit.isChecked()
    curve = window.canvas.curves[FID_KEY]
    x, y = curve.getOriginalDataset()
    window.canvas.reset_zoom()
    qapp.processEvents()
    full = window.canvas.plot_items[FID_KEY].getViewBox().viewRange()[1]
    quiet = slice(0, 20)  # a short stretch at the start
    window.canvas.set_view_range(x[quiet][0], x[quiet][-1])
    qapp.processEvents()
    lo, hi = window.canvas.plot_items[FID_KEY].getViewBox().viewRange()[1]
    assert hi - lo < full[1] - full[0]
    assert lo <= y[quiet].min() and y[quiet].max() <= hi


# ----------------------------------------------------------------- time axis
def test_time_axis_modes(qapp: QApplication, window: CampaignWindow) -> None:
    canvas = window.canvas
    t = canvas.curves[FID_KEY].getOriginalDataset()[0][0]  # 12.10.2026 14:27:02 on master time
    window.set_time_axis_mode(TimeAxisMode.CLOCK)
    axis = canvas.plot_items[FID_KEY].getAxis("bottom")
    assert isinstance(axis, AbsoluteTimeAxis)
    assert axis.tickStrings([t], 1, 10) == ["14:27:02"]
    window.set_time_axis_mode(TimeAxisMode.DATETIME)
    axis = canvas.plot_items[FID_KEY].getAxis("bottom")
    assert axis.tickStrings([t], 1, 10) == ["12.10.2026 14:27:02"]
    assert get_str(TIME_AXIS) == "datetime"
    assert CampaignWindow().canvas.time_axis_mode is TimeAxisMode.DATETIME  # personal preference, remembered


def test_relative_axis_starts_at_the_first_sample(qapp: QApplication, window: CampaignWindow) -> None:
    canvas = window.canvas
    window.set_time_axis_mode(TimeAxisMode.RELATIVE)
    axis = canvas.plot_items[FID_KEY].getAxis("bottom")
    assert isinstance(axis, RelativeTimeAxis)
    first = min(float(np.nanmin(c.getOriginalDataset()[0])) for c in canvas.curves.values())
    assert canvas.t0 == first
    assert axis.tickStrings([first + 90], 1, 30) == ["0:01:30"]
    assert "(t − t0 = 0:01:30)" in canvas.update_crosshair(first + 90)


def test_t0_at_crosshair_is_saved_with_the_campaign(qapp: QApplication, window: CampaignWindow, tmp_path: Path) -> None:
    t0 = to_epoch_s(datetime(2026, 10, 12, 14, 28), ZONE)
    window.canvas.update_crosshair(t0)
    window.action_t0_here.trigger()
    assert window.canvas.time_axis_mode is TimeAxisMode.RELATIVE
    assert window.canvas.t0 == pytest.approx(t0)
    assert window.session.campaign.time_reference == datetime(2026, 10, 12, 12, 28, tzinfo=timezone.utc)
    target = tmp_path / "campaign.json"
    window.save_campaign(target)
    other = CampaignWindow()
    other.open_campaign(target)
    assert other.canvas.t0 == pytest.approx(t0)
    window.set_time_reference(None)  # back to the start of the data
    assert window.session.campaign.time_reference is None
    assert window.canvas.t0 < t0
    other.close()


# ------------------------------------------------------------------ time zone
def test_campaign_time_zone(qapp: QApplication, window: CampaignWindow) -> None:
    """The FID has its own zone (profile): its instants stay, only the labels move to UTC."""
    x_before = window.canvas.curves[FID_KEY].getOriginalDataset()[0].copy()
    assert window.canvas.update_crosshair(x_before[0]).startswith("12.10.2026 14:27:02")
    window.set_timezone("UTC")
    qapp.processEvents()
    np.testing.assert_array_equal(window.canvas.curves[FID_KEY].getOriginalDataset()[0], x_before)
    assert window.canvas.update_crosshair(x_before[0]).startswith("12.10.2026 12:27:02")
    with pytest.raises(TimeConfigError):
        window.set_timezone("Mars/Olympus_Mons")
    assert window.session.campaign.timezone == "UTC"


def test_time_reference_and_zone_roundtrip(tmp_path: Path) -> None:
    s = CampaignSession(Campaign())
    s.add_source(FID_DAT)
    s.set_timezone("America/New_York")
    s.set_time_reference(to_epoch_s(datetime(2026, 10, 12, 14, 28), ZONE))
    s.save(tmp_path / "c.json")
    again = CampaignSession.open(tmp_path / "c.json")
    assert again.campaign.timezone == "America/New_York"
    assert again.time_reference_s == s.time_reference_s
    assert Campaign().time_reference is None  # older campaign files have no t0


# ------------------------------------------------------------ subplot height
def test_minimum_subplot_height_scrolls(qapp: QApplication, window: CampaignWindow) -> None:
    window.sources_panel.set_checked(PICO.stem, "kanal_8_heizung")
    window.sources_panel.set_checked(PICO.stem, "kanal_2_kat_unten")
    window.resize(1000, 500)
    qapp.processEvents()
    window.canvas.set_min_subplot_height(200)
    qapp.processEvents()
    assert window.canvas.graphics_layout.minimumHeight() == 4 * 200
    assert window.canvas.scroll_area.verticalScrollBar().maximum() > 0
    shown = [p.getAxis("bottom").style["showValues"] for p in window.canvas.plot_items.values()]
    assert shown == [True] * 4  # every subplot carries time labels while scrolling
    window.canvas.set_min_subplot_height(0)
    qapp.processEvents()
    assert window.canvas.scroll_area.verticalScrollBar().maximum() == 0
    shown = [p.getAxis("bottom").style["showValues"] for p in window.canvas.plot_items.values()]
    assert shown == [False, False, False, True]


def test_settings_menu_entries(window: CampaignWindow) -> None:
    menus = {a.text(): a.menu() for a in window.menuBar().actions()}
    texts = [a.text() for a in menus["S&ettings"].actions()]
    for entry in ("&Time Axis", "Campaign Time &Zone...", "Mouse &Wheel Zooms Time Only",
                  "&Fit Y to Visible Time Span", "Minimum Subplot &Height...", "&Tools"):
        assert entry in texts
