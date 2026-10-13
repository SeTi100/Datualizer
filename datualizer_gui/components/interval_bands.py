"""Measurement steps in the campaign plots: bands over all subplots and the "Marker ○ ○" icon (step 5)."""

from __future__ import annotations

from typing import Callable
import weakref

import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap

from datualizer_core.campaign import ResolvedInterval

MARKER_START = QColor("#2ecc71")  # first circle: start set
MARKER_END = QColor("#e74c3c")  # second circle: end set
PENDING_PEN = pg.mkPen(MARKER_START, width=1, style=Qt.PenStyle.DashLine)

DERIVED_BRUSH = QColor(120, 160, 255, 28)  # steps from a step table (fixed)
MANUAL_BRUSH = QColor(255, 190, 90, 38)  # steps set by hand (draggable)
WINDOW_BRUSH = QColor(255, 255, 255, 30)  # evaluated part of a step (WindowRule)


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


class StepBand:
    """One step drawn into one subplot: the step band, the evaluated window inside it, an optional label."""

    def __init__(
        self,
        plot: pg.PlotItem,
        iv: ResolvedInterval,
        on_moved: Callable[[str, float, float], None] | None,
        with_label: bool,
    ) -> None:
        self.plot, self.interval = plot, iv
        self.items: list[pg.GraphicsObject] = []
        brush = MANUAL_BRUSH if iv.manual else DERIVED_BRUSH
        self.region = pg.LinearRegionItem(
            values=(iv.start, iv.end), brush=brush, movable=iv.manual and on_moved is not None,
            pen=pg.mkPen(brush.darker(150) if iv.manual else QColor(0, 0, 0, 0)),
        )
        self.region.setZValue(-20)
        self.region.setToolTip(f"{iv.label}\n{_param_text(iv)}".strip())
        if iv.manual and on_moved is not None:
            self.region.sigRegionChangeFinished.connect(
                lambda r: on_moved(iv.id, *sorted(r.getRegion()))
            )
        plot.addItem(self.region, ignoreBounds=True)
        self.items.append(self.region)
        if iv.window is not None and iv.window != (iv.start, iv.end):
            win = pg.LinearRegionItem(values=iv.window, brush=WINDOW_BRUSH, movable=False, pen=pg.mkPen(None))
            win.setZValue(-19)
            win.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            plot.addItem(win, ignoreBounds=True)
            self.items.append(win)
        if with_label:
            text = pg.TextItem(iv.label, color="#c8c8c8", anchor=(0, 0))
            text.setPos(QPointF(iv.start, 0))
            plot.addItem(text, ignoreBounds=True)
            self.items.append(text)
            vb = plot.getViewBox()
            text_ref, vb_ref = weakref.ref(text), weakref.ref(vb)  # no cycle through the signal

            def place(*_args) -> None:  # keep the label at the top edge while y changes
                t, v = text_ref(), vb_ref()
                if t is not None and v is not None:
                    t.setPos(t.pos().x(), v.viewRange()[1][1])

            vb.sigYRangeChanged.connect(place)

            def unpin() -> None:
                v = vb_ref()
                if v is not None:
                    try:
                        v.sigYRangeChanged.disconnect(place)
                    except (RuntimeError, TypeError):  # view already deleted or disconnected
                        pass

            self._unpin = unpin
            place()

    _unpin: Callable[[], None] | None = None

    def remove(self) -> None:
        if self._unpin is not None:
            self._unpin()
            self._unpin = None
        for item in self.items:
            self.plot.removeItem(item)
        self.items = []


def _param_text(iv: ResolvedInterval) -> str:
    return ", ".join(f"{k} = {v}" for k, v in iv.params.items() if v is not None)
