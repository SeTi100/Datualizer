"""Campaign window: values under the cursor as bar, tooltip or table (dock), with chosen decimals."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolTip

from datualizer_gui.campaign_window import CampaignWindow
from datualizer_gui.components.readout import ReadoutMode, format_value
from datualizer_gui.settings import READOUT_DECIMALS, READOUT_MODE, get_int, get_str

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
FID_DAT = SA / "fid_ifid_12_10_extract.dat"
PICO = SA / "picolog_12_10_extract.csv"
PICO_CHANNELS = ["kanal_1_kat_oben", "kanal_2_kat_unten", "kanal_3_rohrbogen", "kanal_4_wt",
                 "kanal_5_luefter", "kanal_6_ansaug_raum", "kanal_7_raum_voc", "kanal_8_heizung"]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture()
def window(qapp: QApplication, tmp_path: Path) -> CampaignWindow:
    pico = tmp_path / PICO.name
    shutil.copyfile(PICO, pico)
    saved = datetime(2026, 10, 12, 16, 0, 21, tzinfo=timezone.utc).timestamp()
    os.utime(pico, (saved, saved))
    win = CampaignWindow()
    win.add_sources([FID_DAT, pico])
    for ch in PICO_CHANNELS:  # many channels: far more than one line can hold
        win.sources_panel.set_checked(PICO.stem, ch)
    win.resize(900, 700)
    win.show()
    qapp.processEvents()
    yield win
    win.close()


def fid_time(window: CampaignWindow, i: int = 3) -> float:
    return float(window.canvas.curves[f"{FID_DAT.stem}/conc"].getOriginalDataset()[0][i])


def test_decimals() -> None:
    assert format_value(12.3456).strip() == "12.35"  # automatic: 4 significant digits
    assert format_value(12.3456, 1).strip() == "12.3"
    assert format_value(12.0, 3).strip() == "12.000"
    assert format_value(float("nan"), 2).strip() == "—"


def test_bar_is_the_default(window: CampaignWindow) -> None:
    assert window.canvas.readout_mode is ReadoutMode.BAR
    assert window.canvas.crosshair_label.isVisible() and not window.dock_values.isVisible()


def test_table_shows_every_track_in_full(qapp: QApplication, window: CampaignWindow) -> None:
    """With nine channels the bar clips; the table lists each one in its own row."""
    window.set_readout(mode=ReadoutMode.TABLE)
    qapp.processEvents()
    assert window.dock_values.isVisible() and not window.canvas.crosshair_label.isVisible()
    window.canvas.update_crosshair(fid_time(window))
    table = window.canvas.readout_table
    rows = table.rows()
    assert len(rows) == 1 + len(PICO_CHANNELS)
    assert rows[0][0] == f"{FID_DAT.name}: conc" and rows[0][2] == "mgC"
    assert table.time_label.text().startswith("12.10.2026 14:27:")
    assert get_str(READOUT_MODE) == "table"


def test_tooltip_at_the_cursor(qapp: QApplication, window: CampaignWindow) -> None:
    window.set_readout(mode=ReadoutMode.TOOLTIP)
    assert not window.canvas.crosshair_label.isVisible()
    pos = window.canvas.graphics_layout.mapToGlobal(QPoint(300, 200))
    text = window.canvas.update_crosshair(fid_time(window), pos)
    qapp.processEvents()
    assert QToolTip.isVisible()
    assert QToolTip.text().count("\n") == len(PICO_CHANNELS) + 1  # time + one line per track
    assert text.startswith("12.10.2026")  # the one-line text is still returned
    window.set_readout(mode=ReadoutMode.BAR)
    QTest.qWait(500)  # Qt hides a tooltip after a short delay
    assert not QToolTip.isVisible()


def test_decimals_apply_to_every_mode(qapp: QApplication, window: CampaignWindow) -> None:
    window.set_readout(decimals=3)
    x = fid_time(window)
    value = window.canvas.readout_rows(x)[1][0][1]
    assert len(value.split(".")[1]) == 3
    assert f"= {value.rjust(10)} mgC" in window.canvas.update_crosshair(x)
    assert get_int(READOUT_DECIMALS) == 3
    window.set_readout(mode=ReadoutMode.TABLE)
    window.canvas.update_crosshair(x)
    assert window.canvas.readout_table.rows()[0][1] == value
    assert CampaignWindow().canvas.readout_mode is ReadoutMode.TABLE  # remembered


def test_settings_menu_has_readout(window: CampaignWindow) -> None:
    menus = {a.text(): a.menu() for a in window.menuBar().actions()}
    sub = {a.text(): a.menu() for a in menus["S&ettings"].actions()}["&Values under Cursor"]
    assert [a.text() for a in sub.actions() if a.text()] == [
        "&Bar Above the Plots", "&Tooltip at the Cursor", "T&able in a Dock", "&Decimals..."
    ]
