"""One-line value readout above a plot that never changes the window size."""

from __future__ import annotations

import math

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

READOUT_STYLE = (
    "QLabel { background-color: #1e1e24; color: #dcdcdc; font-family: 'Consolas', 'Courier New', monospace;"
    " font-size: 11px; padding: 5px 10px; border-radius: 4px; border: 1px solid #32323e; }"
)
VALUE_WIDTH = 10  # characters per value, so the text does not shift while hovering


def format_value(value: float) -> str:
    """Value right-aligned to a fixed width (4 significant digits), '—' for missing ones."""
    if not math.isfinite(value):
        return "—".rjust(VALUE_WIDTH)
    return f"{value:>{VALUE_WIDTH}.4g}"


class ReadoutLabel(QLabel):
    """QLabel whose text length never drives the layout.

    A plain QLabel reports its text width as minimum width, so a longer readout (a value going
    from 99 to 100) widens the whole window while hovering. Text that does not fit is clipped.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setStyleSheet(READOUT_STYLE)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setWordWrap(False)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:
        return QSize(0, super().sizeHint().height())
