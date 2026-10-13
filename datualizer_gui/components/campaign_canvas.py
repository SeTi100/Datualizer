"""Plot canvas for a campaign: one subplot per track, each with its own x array on master time."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import polars as pl
import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from datualizer_core.campaign import Track
from datualizer_gui.components.plot_canvas import CHANNEL_COLORS


def utc_offset_s(epoch_s: float, zone: str) -> int:
    """Offset of `zone` from UTC at an instant, in seconds (east positive)."""
    s = pl.Series([int(epoch_s * 1e6)]).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    local = s.dt.convert_time_zone(zone).dt.replace_time_zone(None)
    return int((local.dt.epoch("us")[0] - int(epoch_s * 1e6)) / 1e6)


def connect_mask(values: np.ndarray, gap_rows: np.ndarray) -> np.ndarray:
    """Connect i to i+1 only if both are finite and no time gap starts at i+1."""
    ok = np.isfinite(values)
    mask = ok[:-1] & ok[1:] if len(values) > 1 else np.zeros(0, dtype=bool)
    gaps = gap_rows[(gap_rows > 0) & (gap_rows < len(values))]
    mask[gaps - 1] = False
    return np.append(mask, False).astype(bool) if len(values) else np.zeros(0, dtype=bool)


class CampaignPlotCanvas(QWidget):
    """Stacked subplots with linked x axes on absolute master time.

    Unlike the dataset canvas, every track keeps its own x array, so sources with different
    sampling and clocks share one time axis without resampling.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tracks: list[Track] = []
        self._plot_items: dict[str, pg.PlotItem] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._crosshair_lines: list[pg.InfiniteLine] = []
        self._zone = "UTC"
        self._utc_offset = 0
        self._user_range: tuple[float, float] | None = None  # set when the user zooms or pans

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.crosshair_label = QLabel("Add sources and select channels to plot them on one time axis.")
        self.crosshair_label.setStyleSheet(
            "QLabel { background-color: #1e1e24; color: #dcdcdc; font-family: 'Consolas', monospace;"
            " font-size: 11px; padding: 5px 10px; border-radius: 4px; border: 1px solid #32323e; }"
        )
        layout.addWidget(self.crosshair_label)
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground("#18181c")
        self.graphics_layout.scene().sigMouseMoved.connect(self._on_mouse_moved)
        layout.addWidget(self.graphics_layout)

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    @property
    def plot_items(self) -> dict[str, pg.PlotItem]:
        return self._plot_items

    @property
    def curves(self) -> dict[str, pg.PlotDataItem]:
        return self._curves

    def set_tracks(self, tracks: Sequence[Track], zone: str = "UTC") -> None:
        """Replace all subplots; a range the user zoomed to is kept, otherwise all tracks are shown."""
        self._tracks = list(tracks)
        self._zone = zone
        starts = [float(np.nanmin(t.x)) for t in self._tracks if np.isfinite(t.x).any()]
        self._utc_offset = utc_offset_s(min(starts), zone) if starts else 0
        self.graphics_layout.clear()
        self._plot_items, self._curves, self._crosshair_lines = {}, {}, []

        first: pg.PlotItem | None = None
        for i, tr in enumerate(self._tracks):
            key = f"{tr.source_id}/{tr.channel}"
            # pyqtgraph's utcOffset is seconds west of UTC
            axis = pg.DateAxisItem(orientation="bottom", utcOffset=-self._utc_offset)
            p = self.graphics_layout.addPlot(row=i, col=0, axisItems={"bottom": axis})
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("left", tr.channel, units=tr.unit or None)
            p.setTitle(tr.label, size="9pt", color="#aaaaaa")
            if first is None:
                first = p
            else:
                p.setXLink(first)
            if i < len(self._tracks) - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            curve = p.plot(
                tr.x, tr.y, pen=pg.mkPen(CHANNEL_COLORS[i % len(CHANNEL_COLORS)], width=1.3),
                connect=connect_mask(tr.y, tr.gap_rows),
            )
            line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#888888", style=pg.QtCore.Qt.DashLine))
            p.addItem(line, ignoreBounds=True)
            self._plot_items[key], self._curves[key] = p, curve
            self._crosshair_lines.append(line)
        if first is not None:
            first.getViewBox().sigRangeChangedManually.connect(self._on_manual_range)
            self._apply_range()
        if not self._tracks:
            self.crosshair_label.setText("Add sources and select channels to plot them on one time axis.")

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
            first.setXRange(*rng, padding=0 if self._user_range else 0.01)

    def _on_manual_range(self, *_args) -> None:
        first = next(iter(self._plot_items.values()))
        lo, hi = first.getViewBox().viewRange()[0]
        self._user_range = (lo, hi)

    def format_time(self, epoch_s: float) -> str:
        t = datetime.fromtimestamp(epoch_s + self._utc_offset, tz=timezone.utc).replace(tzinfo=None)
        return t.strftime("%d.%m.%Y %H:%M:%S")

    def update_crosshair(self, x: float) -> str:
        """Move the crosshair to master time `x` and return the readout text."""
        for line in self._crosshair_lines:
            line.setPos(x)
        parts = [self.format_time(x)]
        for tr in self._tracks:
            if not len(tr.x):
                continue
            idx = int(np.clip(np.searchsorted(tr.x, x), 1, len(tr.x) - 1))
            j = idx - 1 if abs(tr.x[idx - 1] - x) <= abs(tr.x[idx] - x) else idx
            val = tr.y[j]
            parts.append(f"{tr.label} = {'—' if not np.isfinite(val) else f'{val:.4g}'} {tr.unit}".strip())
        text = " | ".join(parts)
        self.crosshair_label.setText(text)
        return text

    def _on_mouse_moved(self, pos) -> None:
        for p in self._plot_items.values():
            if p.sceneBoundingRect().contains(pos):
                self.update_crosshair(p.getViewBox().mapSceneToView(pos).x())
                return
