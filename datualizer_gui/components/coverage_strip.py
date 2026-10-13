"""Coverage strip of a campaign: one lane per source, covered time filled, gaps hatched (step 4)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QPen

from datualizer_core.campaign import NO_VALUES, SourceCoverage

LANE_HEIGHT = 0.6  # of 1.0 per lane
LANE_PX = 18  # screen height per lane
LABEL_CHARS = 28  # longer source names are elided in the lane labels
LEFT_AXIS_MIN_PX = 90  # common left axis width; linked x axes only show equal ranges with equal widths


def format_duration(seconds: float) -> str:
    s = abs(seconds)
    if s < 120:
        return f"{seconds:.3g} s"
    if s < 2 * 3600:
        return f"{seconds / 60:.3g} min"
    if s < 2 * 86400:
        return f"{seconds / 3600:.3g} h"
    return f"{seconds / 86400:.3g} d"


def lane_label(cov: SourceCoverage) -> str:
    label = cov.label if len(cov.label) <= LABEL_CHARS else cov.label[: LABEL_CHARS - 1] + "…"
    return label if cov.span is not None else f"{label} (no time)"


def left_axis_width(plot: pg.PlotItem, labels: Sequence[str]) -> int:
    """Pixels the left axis of `plot` needs for these tick labels (at least LEFT_AXIS_MIN_PX)."""
    axis = plot.getAxis("left")
    font = axis.style["tickFont"] or axis.font()
    text = max((QFontMetrics(font).horizontalAdvance(t) for t in labels), default=0)
    return max(LEFT_AXIS_MIN_PX, text + axis.style["tickTextOffset"][0] + max(0, axis.style["tickLength"]) + 4)


def describe_gap(cov: SourceCoverage, t: float) -> str | None:
    """Readout text if master time `t` falls into a gap of this source."""
    g = cov.gap_at(t)
    if g is None:
        return None
    why = "no values" if g.reason == NO_VALUES else "time gap"
    return f"{cov.label}: gap {format_duration(g.duration_s)} ({why})"


class CoverageItem(pg.GraphicsObject):
    """Filled bars for covered intervals and hatched bars for gaps, lane `i` centred at y = -i.

    The hatching is drawn in device pixels, so it keeps its density at any zoom; bars narrower
    than a pixel are widened to one pixel so a short gap never disappears.
    """

    def __init__(self, coverage: Sequence[SourceCoverage], colors: Sequence[str]) -> None:
        super().__init__()
        self._lanes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, QColor]] = []
        lo, hi = np.inf, -np.inf
        for i, cov in enumerate(coverage):
            seg = np.array(cov.segments, dtype=float).reshape(-1, 2)
            gap = np.array([(g.start, g.end) for g in cov.gaps], dtype=float).reshape(-1, 2)
            self._lanes.append((seg[:, 0], seg[:, 1], gap[:, 0], gap[:, 1], QColor(colors[i % len(colors)])))
            if cov.span is not None:
                lo, hi = min(lo, cov.span[0]), max(hi, cov.span[1])
        self._x_bounds = (lo, hi) if np.isfinite(lo) else (0.0, 1.0)
        self._n = len(coverage)

    def boundingRect(self) -> QRectF:
        lo, hi = self._x_bounds
        return QRectF(lo, -(self._n - 1) - 0.5, max(hi - lo, 1e-9), max(self._n, 1))

    def paint(self, painter, option, widget=None) -> None:  # noqa: ARG002 (Qt signature)
        vb = self.getViewBox()
        if vb is None or not self._lanes:
            return
        (x_lo, x_hi), _ = vb.viewRange()
        to_device = painter.transform()
        painter.save()
        painter.resetTransform()
        for i, (s0, s1, g0, g1, color) in enumerate(self._lanes):
            y0 = -i - LANE_HEIGHT / 2
            fill = QColor(color)
            fill.setAlpha(170)
            self._draw(painter, to_device, s0, s1, y0, QBrush(fill), QPen(Qt.PenStyle.NoPen), x_lo, x_hi)
            hatch = QBrush(QColor("#e8e8e8"), Qt.BrushStyle.BDiagPattern)
            outline = QPen(QColor(color))
            outline.setCosmetic(True)
            self._draw(painter, to_device, g0, g1, y0, hatch, outline, x_lo, x_hi)
        painter.restore()

    @staticmethod
    def _draw(painter, to_device, x0, x1, y0, brush, pen, x_lo, x_hi) -> None:
        visible = (x1 >= x_lo) & (x0 <= x_hi)
        painter.setBrush(brush)
        painter.setPen(pen)
        for a, b in zip(x0[visible], x1[visible]):
            r = to_device.mapRect(QRectF(a, y0, b - a, LANE_HEIGHT)).normalized()
            if r.width() < 1.0:
                r.setWidth(1.0)
            painter.drawRect(r)


def build_coverage_plot(
    layout: pg.GraphicsLayout | pg.GraphicsLayoutWidget,
    coverage: Sequence[SourceCoverage],
    colors: Sequence[str],
    utc_offset: int,
    axis: pg.AxisItem | None = None,
) -> tuple[pg.PlotItem, CoverageItem]:
    """Add the coverage strip as a plot (row 0) with a time axis and one labelled lane per source.

    `axis`: bottom axis to use (default: a date axis for `utc_offset`).
    """
    if axis is None:
        axis = pg.DateAxisItem(orientation="bottom", utcOffset=-utc_offset)  # pyqtgraph: seconds west
    p = layout.addPlot(row=0, col=0, axisItems={"bottom": axis})
    p.setTitle("Coverage (hatched = gap)", size="9pt", color="#aaaaaa")
    p.setMouseEnabled(x=True, y=False)
    p.hideButtons()
    p.getViewBox().setMenuEnabled(False)
    n = len(coverage)
    p.setYRange(-(n - 1) - 0.5, 0.5, padding=0)
    ticks = [(-i, lane_label(cov)) for i, cov in enumerate(coverage)]
    p.getAxis("left").setTicks([ticks, []])
    height = LANE_PX * n + 40
    p.setMinimumHeight(height)
    p.setMaximumHeight(height)
    item = CoverageItem(coverage, colors)
    p.addItem(item)
    return p, item
