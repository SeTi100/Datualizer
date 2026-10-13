"""Dialog for the plot export options (size, font, line width, resolution, theme)."""

from __future__ import annotations

from dataclasses import asdict, fields
import json

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from datualizer_gui.components.plot_export import ExportFormat, ExportOptions


def options_from_json(text: str) -> ExportOptions:
    """Options saved as JSON (unknown or broken entries fall back to the defaults)."""
    try:
        data = json.loads(text) if text else {}
    except json.JSONDecodeError:
        data = {}
    known = {f.name for f in fields(ExportOptions)}
    try:
        return ExportOptions(**{k: v for k, v in data.items() if k in known})
    except TypeError:
        return ExportOptions()


def options_to_json(options: ExportOptions) -> str:
    return json.dumps(asdict(options))


def _spin(lo: float, hi: float, decimals: int, suffix: str, value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSuffix(suffix)
    spin.setValue(value)
    return spin


class ExportDialog(QDialog):
    """Edit `ExportOptions` for one file format; `options()` returns the result."""

    def __init__(self, options: ExportOptions, fmt: ExportFormat, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Export Plot as {fmt.value.upper()}")
        layout = QVBoxLayout(self)
        note = QLabel("Exports the time span and y ranges shown. Lines are drawn wider only in the file.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #888888;")
        layout.addWidget(note)
        form = QFormLayout()
        layout.addLayout(form)
        self.width_spin = _spin(20, 1000, 1, " mm", options.width_mm)
        self.height_spin = _spin(0, 2000, 1, " mm", options.height_mm)
        self.height_spin.setSpecialValueText("automatic")
        self.subplot_spin = _spin(10, 300, 1, " mm", options.subplot_height_mm)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setSuffix(" dpi")
        self.dpi_spin.setValue(options.dpi)
        self.dpi_spin.setEnabled(fmt is ExportFormat.PNG)
        if fmt is not ExportFormat.PNG:
            self.dpi_spin.setToolTip("SVG and PDF are vector files")
        self.font_spin = _spin(4, 36, 1, " pt", options.font_pt)
        self.line_spin = _spin(0.1, 5, 2, " pt", options.line_width_pt)
        self.light_check = QCheckBox("Light background (for print)")
        self.light_check.setChecked(options.light)
        form.addRow("Width", self.width_spin)
        form.addRow("Height", self.height_spin)
        form.addRow("Height per subplot (automatic)", self.subplot_spin)
        form.addRow("Resolution", self.dpi_spin)
        form.addRow("Font size", self.font_spin)
        form.addRow("Line width", self.line_spin)
        form.addRow("", self.light_check)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self) -> ExportOptions:
        return ExportOptions(
            width_mm=self.width_spin.value(),
            height_mm=self.height_spin.value(),
            dpi=self.dpi_spin.value(),
            font_pt=self.font_spin.value(),
            line_width_pt=self.line_spin.value(),
            light=self.light_check.isChecked(),
            subplot_height_mm=self.subplot_spin.value(),
        )
