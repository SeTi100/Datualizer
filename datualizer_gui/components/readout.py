"""Value readout under the crosshair: one-line bar, tooltip at the cursor, or a table in a dock.

The bar never changes the window size; long contents are clipped there, the tooltip and the
table show everything.
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Sequence

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

READOUT_STYLE = (
    "QLabel { background-color: #1e1e24; color: #dcdcdc; font-family: 'Consolas', 'Courier New', monospace;"
    " font-size: 11px; padding: 5px 10px; border-radius: 4px; border: 1px solid #32323e; }"
)
VALUE_WIDTH = 10  # characters per value, so the text does not shift while hovering
AUTO_DECIMALS = -1  # 4 significant digits


class ReadoutMode(str, Enum):
    BAR = "bar"          # one line above the plots
    TOOLTIP = "tooltip"  # multi-line tooltip at the cursor
    TABLE = "table"      # table in a dock: channel, value, unit


def format_value(value: float, decimals: int | None = None) -> str:
    """Value right-aligned to a fixed width, '—' for missing ones.

    `decimals` None or AUTO_DECIMALS: 4 significant digits; otherwise that many decimals.
    """
    if not math.isfinite(value):
        return "—".rjust(VALUE_WIDTH)
    if decimals is None or decimals < 0:
        return f"{value:>{VALUE_WIDTH}.4g}"
    return f"{value:>{VALUE_WIDTH}.{decimals}f}"


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


class ReadoutTable(QWidget):
    """Time on top, one row per track below (name, value, unit); rows are reused while hovering."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.time_label = QLabel("Move the mouse over the plots.")
        self.time_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.time_label)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Channel", "Value", "Unit"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

    def show_values(self, time_text: str, rows: Sequence[tuple[str, str, str]], notes: Sequence[str] = ()) -> None:
        """Rows of (name, value, unit); `notes` (e.g. gaps) go below the time."""
        self.time_label.setText("\n".join([time_text, *notes]))
        if self.table.rowCount() != len(rows):
            self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, text in enumerate(row):
                item = self.table.item(i, j)
                if item is None:
                    item = QTableWidgetItem()
                    if j == 1:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(i, j, item)
                if item.text() != text:
                    item.setText(text)

    def rows(self) -> list[tuple[str, str, str]]:
        return [
            tuple(self.table.item(i, j).text() for j in range(3))  # type: ignore[misc]
            for i in range(self.table.rowCount())
        ]
