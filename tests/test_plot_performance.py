"""Plots stay responsive with large data: downsampling, clip to view, cached curves, kept subplots.

No timing asserts (they would be flaky on CI); the tests check that the mechanisms are on and
that nothing visible changes: gaps still break the line, the drawn data stays the full data.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsItem

from datualizer_core.campaign import ClockModel
from datualizer_gui.campaign_window import CampaignWindow
from datualizer_gui.components.curves import (
    FastCurve,
    add_curve,
    connect_mask,
    curve_pen,
    ensure_pixmap_cache,
    peak_lod,
    plot_arrays,
)
from datualizer_gui.components.plot_canvas import MultiChannelPlotCanvas
from tests.test_campaign_gui import FID_DAT, PICO

CACHED = QGraphicsItem.CacheMode.DeviceCoordinateCache


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
    qapp.processEvents()
    return win


class TestPlotArrays:
    def test_connect_mask_per_row(self) -> None:
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        assert connect_mask(y, np.array([4])).tolist() == [True, False, False, False, True, False]

    def test_only_finite_points_and_breaks_where_rows_were_left_out(self) -> None:
        x = np.array([np.nan, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        y = np.array([1.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        xs, ys, connect, x_sorted = plot_arrays(x, y, np.array([6]))
        assert xs.tolist() == [10.0, 11.0, 13.0, 14.0, 15.0]
        assert ys.tolist() == [1.0, 2.0, 4.0, 5.0, 6.0]
        # 10-11 joined, break over the missing value, 13-14 joined, time gap before row 6
        assert connect.tolist() == [True, False, True, False, False]
        assert x_sorted
        assert np.isnan(x[0]) and np.isnan(y[3])  # input untouched

    def test_unsorted_time_is_reported(self) -> None:
        assert not plot_arrays(np.array([0.0, 2.0, 1.0]), np.ones(3), np.array([], dtype=np.int64))[3]


def bridged_breaks(x_out: np.ndarray, c_out: np.ndarray, break_lo: float, break_hi: float) -> int:
    """Number of drawn segments that start before a break and end after it."""
    a, b = x_out[:-1][c_out[:-1]], x_out[1:][c_out[:-1]]
    return int(((a <= break_lo) & (b >= break_hi)).sum())


class TestPeakLod:
    def test_never_merges_across_a_dropout(self) -> None:
        # 1 Hz logger with a 78-min dropout (values empty, time runs on), as in the PicoLog export
        n = 36_000
        x = np.arange(n, dtype=float)
        y = np.sin(x / 500.0)
        y[18_000:18_000 + 78 * 60] = np.nan
        xs, ys, cs, x_sorted = plot_arrays(x, y, np.array([], dtype=np.int64))
        for columns in (300, 1000, 1600, 4000):
            xo, _, co = peak_lod(xs, ys, cs, None, columns, x_sorted)
            assert len(xo) < len(xs)
            assert bridged_breaks(xo, co, 17_999, 18_000 + 78 * 60) == 0, columns
        xo, _, co = peak_lod(xs, ys, cs, (10_000.0, 30_000.0), 800, x_sorted)
        assert bridged_breaks(xo, co, 17_999, 18_000 + 78 * 60) == 0

    def test_never_merges_across_a_time_gap(self) -> None:
        x = np.arange(10_000, dtype=float)
        xs, ys, cs, _ = plot_arrays(x, np.cos(x), np.array([5_000]))
        xo, _, co = peak_lod(xs, ys, cs, None, 200)
        assert bridged_breaks(xo, co, 4_999, 5_000) == 0

    def test_keeps_min_and_max_per_column_and_clips(self) -> None:
        x = np.arange(100_000, dtype=float)
        y = np.zeros_like(x)
        y[54_321], y[54_322] = 7.0, -3.0
        c = np.ones(len(x), dtype=bool)
        xo, yo, _ = peak_lod(x, y, c, None, 500)
        assert yo.max() == 7.0 and yo.min() == -3.0 and len(xo) <= 2 * 500 + 2
        xo, _, _ = peak_lod(x, y, c, (1_000.0, 2_000.0), 500)
        assert xo.min() >= 999 and xo.max() <= 2_001

    def test_few_points_are_drawn_as_they_are(self) -> None:
        x = np.arange(50, dtype=float)
        c = np.ones(50, dtype=bool)
        xo, yo, co = peak_lod(x, x * 2, c, None, 100)
        assert xo.tolist() == x.tolist() and not co[-1]

    def test_unsorted_time_is_not_reduced(self) -> None:
        x = np.array([0.0, 2.0, 1.0])
        xo, _, _ = peak_lod(x, x, np.ones(3, dtype=bool), (0.0, 1.0), 1, x_sorted=False)
        assert xo.tolist() == x.tolist()

    def test_columns_follow_the_view_not_a_far_neighbour(self) -> None:
        # A sample long before the data (as in the instrument reader DB) is kept as the left
        # neighbour while the view starts in the gap. The columns must still be one screen
        # column wide: spanning them over the neighbour made a sawtooth of a few wide columns.
        x = np.concatenate([[0.0], np.arange(100_000.0, 110_000.0)])
        y = np.sin(x / 30.0)
        c = np.ones(len(x), dtype=bool)
        x_range, columns = (99_000.0, 110_000.0), 500
        xo, yo, _ = peak_lod(x, y, c, x_range, columns)
        width = (x_range[1] - x_range[0]) / columns
        inside = np.unique(xo[(xo >= 100_000.0)])
        assert np.diff(inside).max() <= 1.01 * width
        assert yo.max() == y[1:].max() and yo.min() == y[1:].min()

    def test_panning_by_whole_columns_keeps_the_columns(self) -> None:
        # Columns are fixed on the time axis, so the curve does not jitter while panning.
        x = np.concatenate([[0.0], np.arange(100_000.0, 110_000.0)])
        y = np.sin(x / 7.0)
        c = np.ones(len(x), dtype=bool)
        columns = 400
        width = 8_000.0 / columns
        a = peak_lod(x, y, c, (101_000.0, 109_000.0), columns)
        b = peak_lod(x, y, c, (101_000.0 + 3 * width, 109_000.0 + 3 * width), columns)
        overlap = (101_500.0, 108_500.0)
        pick = [np.column_stack(r[:2])[(r[0] >= overlap[0]) & (r[0] <= overlap[1])] for r in (a, b)]
        assert len(pick[0]) > 100 and np.array_equal(pick[0], pick[1])


class TestFastCurve:
    def test_thin_pen_and_large_pixmap_cache(self, qapp: QApplication) -> None:
        pen = curve_pen("#ff0000")
        assert pen.widthF() == 1 and pen.isCosmetic()
        assert ensure_pixmap_cache() >= 64 * 1024

    def test_large_curve_draws_one_min_max_pair_per_pixel(self, qapp: QApplication) -> None:
        w = pg.GraphicsLayoutWidget()
        w.resize(600, 300)
        p = w.addPlot()
        n = 1_000_000
        x = np.arange(n, dtype=float)
        y = np.sin(x / 1000.0)
        y[500_001] = 50.0  # a single spike must survive downsampling
        y[::10] = np.nan  # scattered dropouts must not blank the overview
        curve = add_curve(p, x, y, np.array([], dtype=np.int64), pg.mkPen("w"))
        w.show()
        qapp.processEvents()
        xd, yd = curve.getData()
        assert len(xd) < n / 100  # drawn: about two points per screen column
        assert np.nanmax(yd) == 50.0
        assert np.isfinite(yd).sum() > len(yd) * 0.9
        assert len(curve.getOriginalDataset()[0]) == n - n // 10  # all finite samples stay on the item
        assert curve.cacheMode() == CACHED
        assert p.getAxis("left").cacheMode() == CACHED

        p.setXRange(100_000, 101_000, padding=0)
        qapp.processEvents()
        xd, _ = curve.getData()
        assert xd.min() >= 99_000 and xd.max() <= 102_000  # only the visible part is drawn
        w.close()

    def test_no_plot_options_menu_for_fast_curves(self, qapp: QApplication) -> None:
        """pyqtgraphs "Plot Options" (Alpha, FFT, Downsample, Log) rufen PlotDataItem-Methoden auf.

        FastCurve hat sie nicht (eigenes Downsampling), jeder Klick warf einen AttributeError.
        """
        p = pg.PlotItem()
        with pytest.raises(AttributeError):  # so sah der Fehler aus
            add_curve(p, np.arange(3.0), np.arange(3.0), np.array([], dtype=np.int64), pg.mkPen("w"))
            p.updateAlpha()
        assert p.getContextMenus(None) is None  # kein "Plot Options"
        assert p.getViewBox().menuEnabled()  # View All, Achsen, Export bleiben

    def test_main_canvas_uses_fast_curves(self, qapp: QApplication) -> None:
        from datualizer_core import load_source

        canvas = MultiChannelPlotCanvas()
        canvas.set_dataset(load_source(FID_DAT))
        curves = [item for p in canvas.plot_items.values() for item in p.listDataItems()]
        assert curves and all(isinstance(c, FastCurve) for c in curves)
        assert all(c.cacheMode() == CACHED for c in curves)
        grid = next(iter(canvas.plot_items.values())).getAxis("left")
        assert grid.grid and grid.style["maxTickLevel"] == 0  # grid at labelled ticks only

    def test_main_canvas_does_not_bridge_the_logger_dropout(self, qapp: QApplication, tmp_path: Path) -> None:
        """PicoLog export: time runs on through a dropout, the values are empty (Datenkatalog P24)."""
        from datualizer_core import load_source

        rows = ['"","Kanal 1 Kat_oben Letzte (C)"']
        for i in range(20_000):
            value = "" if 9_000 <= i < 9_000 + 78 * 60 else f"{150 + np.sin(i / 300):.3f}".replace(".", ",")
            rows.append(f'"{i // 3600:02d}:{i % 3600 // 60:02d}:{i % 60:02d}","{value}"')
        f = tmp_path / "picolog_dropout.csv"
        f.write_text("\n".join(rows) + "\n", encoding="utf-8")
        canvas = MultiChannelPlotCanvas()
        canvas.resize(900, 400)
        canvas.set_dataset(load_source(f))
        canvas.show()
        qapp.processEvents()
        curve = next(iter(canvas.plot_items.values())).listDataItems()[0]
        xd, _ = curve.getData()
        assert len(xd) < 20_000 - 78 * 60  # reduced to the screen
        assert bridged_breaks(xd, curve.opts["connect"], 8_999, 9_000 + 78 * 60) == 0
        canvas.close()


class TestCampaignRefresh:
    def test_ticking_a_channel_keeps_the_other_subplots(self, qapp: QApplication, window: CampaignWindow) -> None:
        canvas = window.canvas
        before_plots = dict(canvas.plot_items)
        before_curves = dict(canvas.curves)
        window.sources_panel.set_checked(PICO.stem, "kanal_8_heizung")
        qapp.processEvents()
        assert f"{PICO.stem}/kanal_8_heizung" in canvas.curves
        for key, plot in before_plots.items():
            assert canvas.plot_items[key] is plot
            assert canvas.curves[key] is before_curves[key]
        # only the last subplot shows time labels
        plots = list(canvas.plot_items.values())
        assert [p.getAxis("bottom").style["showValues"] for p in plots] == [False] * (len(plots) - 1) + [True]

    def test_clock_change_redraws_only_that_source(self, qapp: QApplication, window: CampaignWindow) -> None:
        canvas = window.canvas
        fid_key, pico_key = f"{FID_DAT.stem}/conc", f"{PICO.stem}/kanal_1_kat_oben"
        fid_curve, pico_curve = canvas.curves[fid_key], canvas.curves[pico_key]
        window.session.set_clock(FID_DAT.stem, ClockModel(offset_s=60.0))
        window.refresh_plot()
        assert canvas.curves[fid_key] is not fid_curve
        assert canvas.curves[pico_key] is pico_curve
        assert canvas.curves[fid_key].getOriginalDataset()[0][0] == fid_curve.getOriginalDataset()[0][0] + 60.0

    def test_session_caches_and_notices_direct_clock_edits(self, window: CampaignWindow) -> None:
        s = window.session
        t1 = s.master_time(FID_DAT.stem)
        assert s.master_time(FID_DAT.stem) is t1
        assert s.track(FID_DAT.stem, "conc") is s.track(FID_DAT.stem, "conc")
        assert not t1.flags.writeable
        s.campaign.source(FID_DAT.stem).clock = ClockModel(offset_s=5.0)  # not via set_clock
        t2 = s.master_time(FID_DAT.stem)
        assert t2[0] == t1[0] + 5.0
        assert s.track(FID_DAT.stem, "conc").x is t2

    def test_mouse_moves_reach_the_crosshair(self, qapp: QApplication, window: CampaignWindow) -> None:
        window.show()
        qapp.processEvents()
        plot = next(iter(window.canvas.plot_items.values()))
        pos = plot.sceneBoundingRect().center()
        window.canvas.graphics_layout.scene().sigMouseMoved.emit(QPointF(pos))
        QTest.qWait(100)
        assert window.canvas.crosshair_label.text().startswith("12.10.2026")
        window.close()
