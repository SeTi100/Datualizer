"""Masked time spans in the campaign plots: hatched, the reason in the tooltip (Etappe 2b, PR 2).

All masks of a subplot are painted by one `MasksItem`, in device pixels and only where visible,
like `interval_bands.StepsItem`. Its shape covers the masks only, so the tooltip appears over a
mask and nowhere else; it takes no mouse buttons (a right click reaches the view box).
"""

from __future__ import annotations

from typing import Sequence

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen

from datualizer_core.campaign import MaskSpan, Track

MASK_FILL = QColor(255, 110, 110, 22)
MASK_HATCH = QColor(255, 120, 120, 120)
MIN_PX = 2.0  # a short mask stays visible


def masks_for(masks: Sequence[MaskSpan], tracks: Sequence[Track]) -> list[MaskSpan]:
    """The masks that concern at least one of these tracks (same source, all or this channel)."""
    return [m for m in masks if any(tr.source_id == m.source_id and m.applies_to(tr.channel) for tr in tracks)]


def mask_text(m: MaskSpan, source_label: str = "") -> str:
    what = ", ".join(m.channels) if m.channels else "all channels"
    return f"Masked ({source_label or m.source_id}: {what}): {m.reason or 'no reason given'}"


class MasksItem(pg.GraphicsObject):
    """Hatched spans of the masks of one subplot."""

    def __init__(self, masks: Sequence[MaskSpan], labels: dict[str, str] | None = None) -> None:
        super().__init__()
        self.masks = list(masks)
        self._labels = labels or {}
        self.setZValue(-21)  # under the step bands, whose draggable regions need the hover
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(True)

    def viewRangeChanged(self) -> None:  # called by pyqtgraph when the view moves
        self.prepareGeometryChange()

    def boundingRect(self) -> QRectF:
        vb = self.getViewBox()
        return vb.viewRect() if vb is not None else QRectF()

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        vb = self.getViewBox()
        if vb is None:
            return path
        (x_lo, x_hi), (y_lo, y_hi) = vb.viewRange()
        for m in self.masks:
            if m.end >= x_lo and m.start <= x_hi:
                path.addRect(QRectF(m.start, y_lo, max(m.end - m.start, 1e-9), y_hi - y_lo))
        return path

    def masks_at(self, x: float) -> list[MaskSpan]:
        return [m for m in self.masks if m.start <= x <= m.end]

    def hoverMoveEvent(self, ev) -> None:
        hits = self.masks_at(ev.pos().x())
        self.setToolTip("\n".join(mask_text(m, self._labels.get(m.source_id, "")) for m in hits))

    def hoverLeaveEvent(self, ev) -> None:  # noqa: ARG002 (Qt signature)
        self.setToolTip("")

    def paint(self, painter, option, widget=None) -> None:  # noqa: ARG002 (Qt signature)
        vb = self.getViewBox()
        if vb is None or not self.masks:
            return
        (x_lo, x_hi), (y_lo, y_hi) = vb.viewRange()
        to_device = painter.transform()
        painter.save()
        painter.resetTransform()
        painter.setPen(Qt.PenStyle.NoPen)
        hatch = QBrush(MASK_HATCH, Qt.BrushStyle.BDiagPattern)
        edge = QPen(MASK_HATCH)
        edge.setCosmetic(True)
        for m in self.masks:
            if m.end < x_lo or m.start > x_hi:
                continue
            rect = to_device.mapRect(QRectF(m.start, y_lo, m.end - m.start, y_hi - y_lo)).normalized()
            if rect.width() < MIN_PX:
                rect.adjust(-(MIN_PX - rect.width()) / 2, 0, (MIN_PX - rect.width()) / 2, 0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(MASK_FILL)
            painter.drawRect(rect)
            painter.setBrush(hatch)
            painter.drawRect(rect)
            painter.setPen(edge)
            painter.drawLine(rect.topLeft(), rect.bottomLeft())
            painter.drawLine(rect.topRight(), rect.bottomRight())
        painter.restore()
