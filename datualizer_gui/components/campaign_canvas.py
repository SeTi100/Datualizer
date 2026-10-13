"""Plot canvas for a campaign: one subplot per track, each with its own x array on master time."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import polars as pl
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from datualizer_core.campaign import Track
from datualizer_gui.components import curves
from datualizer_gui.components.curves import (
    FastCurve,
    add_curve,
    curve_pen,
    ensure_pixmap_cache,
    light_grid,
    rate_limited_mouse,
)
from datualizer_gui.components.plot_canvas import CHANNEL_COLORS
from datualizer_gui.components.readout import ReadoutLabel, format_value


def utc_offset_s(epoch_s: float, zone: str) -> int:
    """Offset of `zone` from UTC at an instant, in seconds (east positive)."""
    s = pl.Series([int(epoch_s * 1e6)]).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    local = s.dt.convert_time_zone(zone).dt.replace_time_zone(None)
    return int((local.dt.epoch("us")[0] - int(epoch_s * 1e6)) / 1e6)


def connect_mask(values: np.ndarray, gap_rows: np.ndarray) -> np.ndarray:
    """Connect i to i+1 only if both are finite and no time gap starts at i+1."""
    return curves.connect_mask(values, gap_rows)


class CampaignPlotCanvas(QWidget):
    """Stacked subplots with linked x axes on absolute master time.

    Unlike the dataset canvas, every track keeps its own x array, so sources with different
    sampling and clocks share one time axis without resampling.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tracks: list[Track] = []
        self._plot_items: dict[str, pg.PlotItem] = {}
        self._curves: dict[str, FastCurve] = {}
        self._crosshair_lines: dict[str, pg.InfiniteLine] = {}
        self._track_of: dict[str, Track] = {}  # track each kept subplot was drawn from
        self._zone = "UTC"
        self._utc_offset = 0
        self._user_range: tuple[float, float] | None = None  # set when the user zooms or pans

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.crosshair_label = ReadoutLabel("Add sources and select channels to plot them on one time axis.")
        layout.addWidget(self.crosshair_label)
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground("#18181c")
        ensure_pixmap_cache()
        self._mouse_proxy = rate_limited_mouse(self.graphics_layout.scene(), self._on_mouse_moved)
        layout.addWidget(self.graphics_layout)

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    @property
    def plot_items(self) -> dict[str, pg.PlotItem]:
        return self._plot_items

    @property
    def curves(self) -> dict[str, FastCurve]:
        return self._curves

    def set_tracks(self, tracks: Sequence[Track], zone: str = "UTC") -> None:
        """Show these tracks; a range the user zoomed to is kept, otherwise all tracks are shown.

        Subplots of tracks that are still shown (the same Track object) are kept with their
        drawn curve, so ticking one more channel does not redraw all the others.
        """
        self._tracks = list(tracks)
        starts = [float(np.nanmin(t.x)) for t in self._tracks if np.isfinite(t.x).any()]
        offset = utc_offset_s(min(starts), zone) if starts else 0
        old_plots, old_curves, old_lines, old_tracks = self._plot_items, self._curves, self._crosshair_lines, self._track_of
        if (offset, zone) != (self._utc_offset, self._zone):
            old_tracks = {}  # date axis labels depend on the offset: build all subplots anew
        self._zone, self._utc_offset = zone, offset
        self.graphics_layout.clear()  # detaches the subplots, kept ones are added back below
        self._plot_items, self._curves, self._crosshair_lines, self._track_of = {}, {}, {}, {}

        first: pg.PlotItem | None = None
        last = len(self._tracks) - 1
        for i, tr in enumerate(self._tracks):
            key = f"{tr.source_id}/{tr.channel}"
            pen = curve_pen(CHANNEL_COLORS[i % len(CHANNEL_COLORS)])
            if old_tracks.get(key) is tr:
                p, curve, line = old_plots[key], old_curves[key], old_lines[key]
                self.graphics_layout.addItem(p, row=i, col=0)
                curve.setPen(pen)
            else:
                p, curve, line = self._new_plot(i, tr, pen)
            p.setXLink(first)  # None for the first subplot
            p.getAxis("bottom").setStyle(showValues=i == last)
            if first is None:
                first = p
            self._plot_items[key], self._curves[key], self._crosshair_lines[key] = p, curve, line
            self._track_of[key] = tr
        if first is not None:
            self._apply_range()
        if not self._tracks:
            self.crosshair_label.setText("Add sources and select channels to plot them on one time axis.")

    def _new_plot(self, row: int, tr: Track, pen) -> tuple[pg.PlotItem, FastCurve, pg.InfiniteLine]:
        # pyqtgraph's utcOffset is seconds west of UTC
        axis = pg.DateAxisItem(orientation="bottom", utcOffset=-self._utc_offset)
        p = self.graphics_layout.addPlot(row=row, col=0, axisItems={"bottom": axis})
        light_grid(p, alpha=0.25)
        p.setLabel("left", tr.channel, units=tr.unit or None)
        p.setTitle(tr.label, size="9pt", color="#aaaaaa")
        curve = add_curve(p, tr.x, tr.y, tr.gap_rows, pen)
        line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#888888", style=pg.QtCore.Qt.DashLine))
        p.addItem(line, ignoreBounds=True)
        p.getViewBox().sigRangeChangedManually.connect(self._on_manual_range)
        return p, curve, line

    def full_range(self) -> tuple[float, float] | None:
        """Union of the time spans of all tracks (linked axes otherwise follow the first track only)."""
        spans = [(np.nanmin(t.x), np.nanmax(t.x)) for t in self._tracks if np.isfinite(t.x).any()]
        if not spans:
            return None
        return float(min(a for a, _ in spans)), float(max(b for _, b in spans))

    def set_view_range(self, start: float, end: float) -> None:
        """Show master time `start`..`end`; kept across rebuilds until `reset_zoom`."""
        self._user_range = (start, end)
        self._apply_range()

    def reset_zoom(self) -> None:
        self._user_range = None
        self._apply_range()
        for p in self._plot_items.values():
            p.enableAutoRange(axis="y")

    def _apply_range(self) -> None:
        if not self._plot_items:
            return
        first = next(iter(self._plot_items.values()))
        rng = self._user_range or self.full_range()
        if rng is not None:
            # x auto-range of a newly added subplot would pull all linked subplots to its own span
            for p in self._plot_items.values():
                p.enableAutoRange(axis="x", enable=False)
            first.setXRange(*rng, padding=0 if self._user_range else 0.01)

    def _on_manual_range(self, *_args) -> None:
        if not self._plot_items:
            return
        first = next(iter(self._plot_items.values()))
        lo, hi = first.getViewBox().viewRange()[0]
        self._user_range = (lo, hi)

    def format_time(self, epoch_s: float) -> str:
        t = datetime.fromtimestamp(epoch_s + self._utc_offset, tz=timezone.utc).replace(tzinfo=None)
        return t.strftime("%d.%m.%Y %H:%M:%S")

    def update_crosshair(self, x: float) -> str:
        """Move the crosshair to master time `x` and return the readout text."""
        for line in self._crosshair_lines.values():
            line.setPos(x)
        parts = [self.format_time(x)]
        for tr in self._tracks:
            if not len(tr.x):
                continue
            idx = int(np.clip(np.searchsorted(tr.x, x), 1, len(tr.x) - 1))
            j = idx - 1 if abs(tr.x[idx - 1] - x) <= abs(tr.x[idx] - x) else idx
            val = tr.y[j]
            parts.append(f"{tr.label} = {format_value(val)} {tr.unit}".rstrip())
        text = " | ".join(parts)
        self.crosshair_label.setText(text)
        return text

    def _on_mouse_moved(self, pos) -> None:
        for p in self._plot_items.values():
            if p.sceneBoundingRect().contains(pos):
                self.update_crosshair(p.getViewBox().mapSceneToView(pos).x())
                return
