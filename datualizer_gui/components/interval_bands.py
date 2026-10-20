"""Measurement steps in the campaign plots: bands over all subplots and the "Marker ○ ○" icon (step 5).

All steps of a subplot are painted by one `StepsItem` (only the visible ones, in device pixels).
One `LinearRegionItem` per step and subplot made panning about five times slower (38 journal
steps × 7 subplots: 64 instead of 13 ms per frame). Only steps set by hand get a draggable
`StepBand` on top.
"""

from __future__ import annotations

from typing import Callable, Sequence

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QIcon, QPainter, QPen, QPixmap

from datualizer_core.campaign import ResolvedInterval

MARKER_START = QColor("#2ecc71")  # first circle: start set
MARKER_END = QColor("#e74c3c")  # second circle: end set
PENDING_PEN = pg.mkPen(MARKER_START, width=1, style=Qt.PenStyle.DashLine)

DERIVED_BRUSH = QColor(120, 160, 255, 28)  # steps from a step table (fixed)
MANUAL_BRUSH = QColor(255, 190, 90, 38)  # steps set by hand (draggable)
WINDOW_BRUSH = QColor(255, 255, 255, 30)  # evaluated part of a step (WindowRule)
LABEL_COLOR = QColor(200, 200, 200)
LABEL_MIN_PX = 24  # no label on bands narrower than this


def marker_icon(state: int, size: int = 16) -> QIcon:
    """Two circles side by side: 0 = ○○, 1 = start set (first green), 2 = both set (second red)."""
    pix = QPixmap(2 * size + 4, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    for i, fill in enumerate([MARKER_START if state >= 1 else None, MARKER_END if state >= 2 else None]):
        pen = QPen(QColor("#bbbbbb"))
        pen.setWidthF(1.5)
        p.setPen(pen)
        p.setBrush(QBrush(fill) if fill is not None else Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(1 + i * (size + 2), 1, size - 2, size - 2))
    p.end()
    return QIcon(pix)


def marker_text(state: int) -> str:
    return "Marker " + ("● ○" if state == 1 else "○ ○")


class StepsItem(pg.GraphicsObject):
    """All steps of one subplot: filled bands, the evaluated window inside, optional labels at the top.

    Painted in device pixels and only for steps inside the view, so its cost does not grow with
    the number of subplots times steps. It takes no mouse input (right click reaches the view box).
    """

    def __init__(self, intervals: Sequence[ResolvedInterval], with_labels: bool, fill_manual: bool = False) -> None:
        super().__init__()
        self.intervals = list(intervals)
        self._with_labels = with_labels
        self._fill_manual = fill_manual  # manual steps are filled by their draggable StepBand
        self.setZValue(-20)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._font = QFont()
        self._font.setPointSizeF(8)

    def viewRangeChanged(self) -> None:  # called by pyqtgraph when the view moves
        self.prepareGeometryChange()

    def boundingRect(self) -> QRectF:
        vb = self.getViewBox()
        return vb.viewRect() if vb is not None else QRectF()

    def paint(self, painter, option, widget=None) -> None:  # noqa: ARG002 (Qt signature)
        vb = self.getViewBox()
        if vb is None or not self.intervals:
            return
        (x_lo, x_hi), (y_lo, y_hi) = vb.viewRange()
        to_device = painter.transform()
        painter.save()
        painter.resetTransform()
        painter.setPen(Qt.PenStyle.NoPen)
        metrics = QFontMetrics(self._font)
        labels: list[tuple[QRectF, str]] = []
        for iv in self.intervals:
            if iv.end < x_lo or iv.start > x_hi:
                continue
            band = to_device.mapRect(QRectF(iv.start, y_lo, iv.end - iv.start, y_hi - y_lo)).normalized()
            if not iv.manual or self._fill_manual:
                painter.setBrush(MANUAL_BRUSH if iv.manual else DERIVED_BRUSH)
                painter.drawRect(band)
            if iv.window is not None and iv.window != (iv.start, iv.end):
                w0, w1 = iv.window
                win = to_device.mapRect(QRectF(w0, y_lo, w1 - w0, y_hi - y_lo)).normalized()
                painter.setBrush(WINDOW_BRUSH)
                painter.drawRect(win)
            if self._with_labels and band.width() >= LABEL_MIN_PX:
                labels.append((band, iv.label))
        if labels:
            painter.setFont(self._font)
            painter.setPen(LABEL_COLOR)
            for band, text in labels:
                text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(band.width()) - 6)
                painter.drawText(band.left() + 3, band.top() + metrics.ascent() + 2, text)
        painter.restore()


class ColorBandsItem(pg.GraphicsObject):
    """Coloured bands over the full height of a plot, e.g. the dosing steps of a timeline view.

    `bands`: (start, end, colour, label) on the x axis. One item for all bands, painted only where
    visible (same reason as `StepsItem`); a label sits at the top of its band where it fits.
    Takes no mouse input.
    """

    def __init__(self, bands: Sequence[tuple[float, float, QColor, str]], label_color: QColor | None = None) -> None:
        super().__init__()
        self.bands = list(bands)
        self._label_color = label_color
        self.setZValue(-20)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._font = QFont()
        self._font.setPointSizeF(7.5)

    def viewRangeChanged(self) -> None:
        self.prepareGeometryChange()

    def boundingRect(self) -> QRectF:
        vb = self.getViewBox()
        return vb.viewRect() if vb is not None else QRectF()

    def paint(self, painter, option, widget=None) -> None:  # noqa: ARG002 (Qt signature)
        vb = self.getViewBox()
        if vb is None or not self.bands:
            return
        (x_lo, x_hi), (y_lo, y_hi) = vb.viewRange()
        to_device = painter.transform()
        painter.save()
        painter.resetTransform()
        painter.setPen(Qt.PenStyle.NoPen)
        metrics = QFontMetrics(self._font)
        labels: list[tuple[QRectF, str]] = []
        for start, end, color, label in self.bands:
            if end < x_lo or start > x_hi:
                continue
            band = to_device.mapRect(QRectF(start, y_lo, end - start, y_hi - y_lo)).normalized()
            painter.setBrush(color)
            painter.drawRect(band)
            if label and self._label_color is not None:
                labels.append((band, label))
        if labels:
            painter.setFont(self._font)
            painter.setPen(self._label_color)
            for band, text in labels:
                w = metrics.horizontalAdvance(text)
                if w <= band.width() - 2:
                    painter.drawText(band.center().x() - w / 2, band.top() + metrics.ascent() + 2, text)
        painter.restore()


class StepBand:
    """A step set by hand in one subplot: a draggable region (the StepsItem draws window and label)."""

    def __init__(
        self,
        plot: pg.PlotItem,
        iv: ResolvedInterval,
        on_moved: Callable[[str, float, float], None] | None,
    ) -> None:
        self.plot, self.interval = plot, iv
        self.region = pg.LinearRegionItem(
            values=(iv.start, iv.end), brush=MANUAL_BRUSH, movable=on_moved is not None,
            pen=pg.mkPen(MANUAL_BRUSH.darker(150)),
        )
        self.region.setZValue(-19)
        self.region.setToolTip(f"{iv.label}\n{_param_text(iv)}".strip())
        if on_moved is not None:
            self.region.sigRegionChangeFinished.connect(lambda r: on_moved(iv.id, *sorted(r.getRegion())))
        plot.addItem(self.region, ignoreBounds=True)

    def remove(self) -> None:
        self.plot.removeItem(self.region)


def _param_text(iv: ResolvedInterval) -> str:
    return ", ".join(f"{k} = {v}" for k, v in iv.params.items() if v is not None)
