"""Recipe Panel: Visualizes data wrangling pipeline steps and offers transformation actions."""

from __future__ import annotations

from typing import Sequence
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.dataset import DualModeDataset


class RecipePanel(QWidget):
    """Panel displaying the automated wrangling pipeline recipe and execution controls.

    Features:
    - Visual timeline of applied transformations (pre-scan, clean_names, drop_footer, cast).
    - Toggle between Primary Wide-Format and On-Demand Tidy Long-Format.
    - Reset Zoom trigger for the synchronized multi-plot canvas.
    """

    view_mode_changed = Signal(str)  # "wide" or "long"
    reset_zoom_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_mode = "wide"
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 1. Pipeline Recipe Steps
        recipe_group = QGroupBox("Transformation Recipe")
        recipe_layout = QVBoxLayout(recipe_group)
        recipe_layout.setSpacing(4)

        self.step_list = QListWidget()
        self.step_list.setStyleSheet(
            "QListWidget::item {"
            "  padding: 6px;"
            "  border-bottom: 1px solid #2a2a36;"
            "  font-family: 'Consolas', 'Courier New', monospace;"
            "  font-size: 11px;"
            "}"
        )
        recipe_layout.addWidget(self.step_list)
        layout.addWidget(recipe_group)

        # 2. Interactive Transformation Actions
        actions_group = QGroupBox("Recipe Actions & Canvas Controls")
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setSpacing(6)

        self.btn_toggle_mode = QPushButton("Tidy Melt: Switch to Long (to_long)")
        self.btn_toggle_mode.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_toggle_mode.clicked.connect(self._on_toggle_mode_clicked)
        actions_layout.addWidget(self.btn_toggle_mode)

        self.btn_reset_zoom = QPushButton("Reset Zoom (Auto-Range All)")
        self.btn_reset_zoom.setStyleSheet("padding: 6px;")
        self.btn_reset_zoom.clicked.connect(self.reset_zoom_clicked.emit)
        actions_layout.addWidget(self.btn_reset_zoom)

        layout.addWidget(actions_group)
        layout.addStretch()

        self._populate_default_recipe()

    @property
    def current_mode(self) -> str:
        return self._current_mode

    def _populate_default_recipe(self) -> None:
        steps = [
            "1. load_csv(source) -> Dialect & structural pre-scan",
            "2. clean_names() -> Snake_case, deduplicated headers",
            "3. drop_footer() -> Truncated trailing non-data lines",
            "4. harvest_errors() -> Non-strict Float64 casting",
            f"5. representation() -> Active: {self._current_mode.upper()} view",
        ]
        self.set_pipeline_steps(steps)

    def set_pipeline_steps(self, steps: Sequence[str]) -> None:
        """Update displayed recipe steps."""
        self.step_list.clear()
        for s in steps:
            item = QListWidgetItem(f"✓  {s}")
            self.step_list.addItem(item)

    def set_view_mode(self, mode: str) -> None:
        """Update current view mode reflection ('wide' or 'long')."""
        self._current_mode = mode.lower()
        if self._current_mode == "long":
            self.btn_toggle_mode.setText("Wide View: Switch to Wide Format")
        else:
            self.btn_toggle_mode.setText("Tidy Melt: Switch to Long (to_long)")
        self._populate_default_recipe()

    def _on_toggle_mode_clicked(self) -> None:
        new_mode = "long" if self._current_mode == "wide" else "wide"
        self.set_view_mode(new_mode)
        self.view_mode_changed.emit(new_mode)

    def set_dataset(self, dataset: DualModeDataset | None) -> None:
        """Update recipe presentation based on the dataset."""
        if dataset is None:
            self._populate_default_recipe()
            return

        has_audit = len(dataset.audit_log) > 0
        steps = [
            f"1. load_csv() -> {len(dataset):,} valid rows",
            f"2. clean_names() -> {len(dataset.columns)} columns normalized",
            "3. drop_footer() -> Clean footer truncation applied",
            f"4. harvest_errors() -> {len(dataset.audit_log)} audit entries",
            f"5. representation() -> Active: {self._current_mode.upper()} view",
        ]
        self.set_pipeline_steps(steps)
