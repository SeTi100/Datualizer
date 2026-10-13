"""Campaign window: channels grouped into subplots (per channel, unit, source) with legend; colour schemes."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import numpy as np
import pyqtgraph as pg
import pytest
from pydantic import ValidationError
from PySide6.QtWidgets import QApplication

from datualizer_core.campaign import ChannelDisplay, Track, YScale, YScaleMode
from datualizer_gui.campaign_window import CampaignWindow
from datualizer_gui.components.campaign_canvas import group_limits
from datualizer_gui.components.plot_canvas import CHANNEL_COLORS
from datualizer_gui.components.plot_groups import ColorBy, GroupBy, group_tracks, track_colors
from datualizer_gui.settings import COLOR_BY, GROUP_BY, get_str

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
FID_DAT = SA / "fid_ifid_12_10_extract.dat"
PICO = SA / "picolog_12_10_extract.csv"
PICO_CHANNELS = ["kanal_1_kat_oben", "kanal_2_kat_unten", "kanal_8_heizung"]


def fake_track(source: str, channel: str, unit: str, y=(1.0, 2.0, 3.0), **kw) -> Track:
    x = np.arange(len(y), dtype=float)
    return Track(source, channel, f"{source}: {channel}", unit, x, np.asarray(y, dtype=float), np.zeros(0, dtype=np.int64), **kw)


# ---------------------------------------------------------------- pure logic
def test_group_by_unit_and_source() -> None:
    tracks = [fake_track("fid", "conc", "mgC"), fake_track("pico", "t1", "C"), fake_track("pico", "t2", "C"),
              fake_track("db", "flow", "")]
    by_channel = group_tracks(tracks, GroupBy.CHANNEL)
    assert [g.key for g in by_channel] == ["fid/conc", "pico/t1", "pico/t2", "db/flow"]
    by_unit = group_tracks(tracks, GroupBy.UNIT)
    assert [[t.channel for t in g.tracks] for g in by_unit] == [["conc"], ["t1", "t2"], ["flow"]]
    assert by_unit[1].legend and not by_unit[0].legend
    assert by_unit[1].title == "Unit: C" and by_unit[2].title == "Unit: (no unit)"
    by_source = group_tracks(tracks, GroupBy.SOURCE, {"pico": "Picolog.csv"})
    assert [g.title for g in by_source] == ["fid", "Picolog.csv", "db"]
    mixed = group_tracks([fake_track("a", "x", "C"), fake_track("a", "y", "g/h")], GroupBy.SOURCE)[0]
    assert (mixed.unit, mixed.axis_label) == ("", "C, g/h")


def test_colours() -> None:
    tracks = [fake_track("fid", "conc", "mgC"), fake_track("pico", "t1", "C"),
              fake_track("pico", "t2", "C", color="#ff0000")]
    assert list(track_colors(tracks, ColorBy.CHANNEL, CHANNEL_COLORS).values()) == CHANNEL_COLORS[:3]
    by_source = track_colors(tracks, ColorBy.SOURCE, CHANNEL_COLORS, ["db", "fid", "pico"])
    assert list(by_source.values()) == [CHANNEL_COLORS[1], CHANNEL_COLORS[2], CHANNEL_COLORS[2]]
    manual = track_colors(tracks, ColorBy.MANUAL, CHANNEL_COLORS)
    assert list(manual.values()) == [CHANNEL_COLORS[0], CHANNEL_COLORS[1], "#ff0000"]


def test_group_range_is_the_union_of_fixed_ranges() -> None:
    a = fake_track("s", "a", "C", y_scale=YScale(mode=YScaleMode.FIXED, min=0, max=10))
    b = fake_track("s", "b", "C", y_scale=YScale(mode=YScaleMode.FIXED, min=-5, max=5))
    assert group_limits([a, b]) == (-5.0, 10.0)
    assert group_limits([a, fake_track("s", "c", "C")]) is None  # one auto track: auto


def test_colour_must_be_hex() -> None:
    assert ChannelDisplay(color="#A0b1C2").color == "#A0b1C2"
    with pytest.raises(ValidationError):
        ChannelDisplay(color="red")


# ------------------------------------------------------------------------ GUI
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
    for ch in PICO_CHANNELS:
        win.sources_panel.set_checked(PICO.stem, ch)
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()
    yield win
    win.close()


def pen_color(curve) -> str:
    return pg.mkPen(curve.opts["pen"]).color().name()


def test_default_is_one_subplot_per_channel(window: CampaignWindow) -> None:
    assert window.group_actions[GroupBy.CHANNEL].isChecked() and window.color_actions[ColorBy.CHANNEL].isChecked()
    assert len(window.canvas.plot_items) == 4
    assert all(p.legend is None for p in window.canvas.plot_items.values())


def test_group_by_unit_shares_a_subplot_with_legend(qapp: QApplication, window: CampaignWindow) -> None:
    window.set_grouping(group_by=GroupBy.UNIT)
    qapp.processEvents()
    canvas = window.canvas
    assert list(canvas.plot_items) == ["unit:mgC", "unit:C"]
    temps = canvas.plot_items["unit:C"]
    assert len(temps.legend.items) == 3
    assert {f"{PICO.stem}/{ch}" for ch in PICO_CHANNELS} <= set(canvas.curves)  # curves stay per track
    assert all(canvas.curves[f"{PICO.stem}/{ch}"] in temps.listDataItems() for ch in PICO_CHANNELS)
    assert len({pen_color(canvas.curves[f"{PICO.stem}/{ch}"]) for ch in PICO_CHANNELS}) == 3
    assert get_str(GROUP_BY) == "unit" and CampaignWindow().canvas.groups == []  # remembered, nothing loaded
    x = canvas.curves[f"{PICO.stem}/kanal_8_heizung"].getOriginalDataset()[0][5]
    assert "kanal_8_heizung =" in canvas.update_crosshair(x)  # readout still lists every track


def test_group_by_source(qapp: QApplication, window: CampaignWindow) -> None:
    window.set_grouping(group_by=GroupBy.SOURCE)
    qapp.processEvents()
    assert list(window.canvas.plot_items) == [f"source:{FID_DAT.stem}", f"source:{PICO.stem}"]
    names = [label.text for _, label in window.canvas.plot_items[f"source:{PICO.stem}"].legend.items]
    assert names == PICO_CHANNELS  # channel names only: the subplot title names the source


def test_colours_by_source_match_the_coverage_lanes(qapp: QApplication, window: CampaignWindow) -> None:
    window.set_coverage_visible(True)
    window.set_grouping(color_by=ColorBy.SOURCE)
    qapp.processEvents()
    canvas = window.canvas
    pico_colors = {pen_color(canvas.curves[f"{PICO.stem}/{ch}"]) for ch in PICO_CHANNELS}
    assert pico_colors == {CHANNEL_COLORS[1]}  # second source of the campaign
    assert pen_color(canvas.curves[f"{FID_DAT.stem}/conc"]) == CHANNEL_COLORS[0]
    assert canvas.coverage_item is not None
    assert get_str(COLOR_BY) == "source"


def test_manual_colour(qapp: QApplication, window: CampaignWindow) -> None:
    window.sources_panel.apply_display(PICO.stem, "kanal_8_heizung", ChannelDisplay(color="#ff00ff"))
    key = f"{PICO.stem}/kanal_8_heizung"
    assert pen_color(window.canvas.curves[key]) != "#ff00ff"  # only used with the manual scheme
    window.set_grouping(color_by=ColorBy.MANUAL)
    qapp.processEvents()
    assert pen_color(window.canvas.curves[key]) == "#ff00ff"
    assert window.canvas.colors[f"{FID_DAT.stem}/conc"] == CHANNEL_COLORS[0]  # others automatic


def test_dialog_sets_the_colour(window: CampaignWindow) -> None:
    dialog = window.sources_panel.display_dialog(PICO.stem, "kanal_1_kat_oben")
    assert dialog.display().color is None
    dialog.set_color("#12ab34")
    window.sources_panel.apply_display(PICO.stem, "kanal_1_kat_oben", dialog.display())
    assert window.session.campaign.source(PICO.stem).display["kanal_1_kat_oben"].color == "#12ab34"
    again = window.sources_panel.display_dialog(PICO.stem, "kanal_1_kat_oben")
    assert again.color_button.text() == "#12ab34"


def test_regrouping_keeps_the_zoom(qapp: QApplication, window: CampaignWindow) -> None:
    x = window.canvas.curves[f"{FID_DAT.stem}/conc"].getOriginalDataset()[0]
    window.canvas.set_view_range(x[10], x[100])
    window.set_grouping(group_by=GroupBy.UNIT)
    qapp.processEvents()
    lo, hi = next(iter(window.canvas.plot_items.values())).getViewBox().viewRange()[0]
    assert (lo, hi) == pytest.approx((x[10], x[100]))
