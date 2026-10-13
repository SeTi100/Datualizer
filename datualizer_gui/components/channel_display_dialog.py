"""Dialog for the display settings of one campaign channel: alias, plot style, y scale.

Display only: the settings go to campaign.json, the dataset stays as loaded.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.campaign import ChannelDisplay, YScale, YScaleMode
from datualizer_core.schema import PlotStyle

STYLE_NAMES = {
    PlotStyle.LINE: "Line",
    PlotStyle.POINTS: "Points",
    PlotStyle.STEP: "Steps (value holds until the next one)",
    PlotStyle.LINE_POINTS: "Line + points",
}
SCALE_NAMES = {
    YScaleMode.AUTO: "Auto (all values)",
    YScaleMode.ROBUST: "Robust (percentiles, outliers marked at the edge)",
    YScaleMode.FIXED: "Fixed (min, max)",
}


def _spin(lo: float, hi: float, decimals: int) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    return spin


class ChannelDisplayDialog(QDialog):
    """Edit a `ChannelDisplay`; `display()` returns the result after `accept`.

    `default_style` is the style of the source profile, offered as the first choice.
    `values` (the channel's samples) only pre-fill the fixed range with the data range.
    """

    def __init__(
        self,
        channel: str,
        display: ChannelDisplay,
        default_style: PlotStyle,
        values: np.ndarray | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Display of {channel}")
        layout = QVBoxLayout(self)
        note = QLabel("Display only: the data in the source is never changed.")
        note.setStyleSheet("color: #888888;")
        layout.addWidget(note)
        form = QFormLayout()
        layout.addLayout(form)

        self.alias_edit = QLineEdit(display.alias)
        self.alias_edit.setPlaceholderText(channel)
        form.addRow("Display name", self.alias_edit)

        self.style_combo = QComboBox()
        self.style_combo.addItem(f"Profile default ({STYLE_NAMES[default_style].split(' (')[0].lower()})", None)
        for style, name in STYLE_NAMES.items():
            self.style_combo.addItem(name, style.value)  # Qt returns str enums as plain str
        if display.style is not None:
            self.style_combo.setCurrentIndex(self.style_combo.findData(display.style.value))
        form.addRow("Plot style", self.style_combo)

        scale = display.y_scale
        self.scale_combo = QComboBox()
        for mode, name in SCALE_NAMES.items():
            self.scale_combo.addItem(name, mode.value)
        self.scale_combo.setCurrentIndex(self.scale_combo.findData(scale.mode.value))
        form.addRow("Y scale", self.scale_combo)
        self.low_pct = _spin(0.0, 100.0, 2)
        self.high_pct = _spin(0.0, 100.0, 2)
        self.low_pct.setSuffix(" %")
        self.high_pct.setSuffix(" %")
        self.low_pct.setValue(scale.low_percentile)
        self.high_pct.setValue(scale.high_percentile)
        form.addRow("Lower percentile", self.low_pct)
        form.addRow("Upper percentile", self.high_pct)
        finite = np.asarray(values, dtype=float) if values is not None else np.zeros(0)
        finite = finite[np.isfinite(finite)]
        data_lo, data_hi = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
        self.min_spin = _spin(-1e15, 1e15, 6)
        self.max_spin = _spin(-1e15, 1e15, 6)
        self.min_spin.setValue(scale.min if scale.min is not None else data_lo)
        self.max_spin.setValue(scale.max if scale.max is not None else data_hi)
        form.addRow("Min", self.min_spin)
        form.addRow("Max", self.max_spin)
        self._form = form

        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #ff6b6b;")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.scale_combo.currentIndexChanged.connect(self._update_enabled)
        self._update_enabled()

    def style(self) -> PlotStyle | None:
        """Chosen style, None = profile default."""
        data = self.style_combo.currentData()
        return PlotStyle(data) if data else None

    def scale_mode(self) -> YScaleMode:
        return YScaleMode(self.scale_combo.currentData())

    def _update_enabled(self) -> None:
        mode = self.scale_mode()
        for widget, on in (
            (self.low_pct, mode is YScaleMode.ROBUST),
            (self.high_pct, mode is YScaleMode.ROBUST),
            (self.min_spin, mode is YScaleMode.FIXED),
            (self.max_spin, mode is YScaleMode.FIXED),
        ):
            widget.setEnabled(on)
            self._form.labelForField(widget).setEnabled(on)

    def display(self) -> ChannelDisplay:
        """The edited settings; raises ValueError for an invalid range."""
        mode = self.scale_mode()
        if mode is YScaleMode.ROBUST:
            scale = YScale(mode=mode, low_percentile=self.low_pct.value(), high_percentile=self.high_pct.value())
        elif mode is YScaleMode.FIXED:
            scale = YScale(mode=mode, min=self.min_spin.value(), max=self.max_spin.value())
        else:
            scale = YScale()
        return ChannelDisplay(alias=self.alias_edit.text().strip(), style=self.style(), y_scale=scale)

    def accept(self) -> None:
        try:
            self.display()
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            errors = exc.errors() if hasattr(exc, "errors") else []
            self.error_label.setText(errors[0]["msg"] if errors else str(exc))
            return
        super().accept()
