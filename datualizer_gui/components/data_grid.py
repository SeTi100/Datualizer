"""Virtual high-performance Qt Table Model and Widget for Polars DataFrames."""

from __future__ import annotations

from typing import Any
import numpy as np
import polars as pl
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.dataset import DualModeDataset


class PolarsTableModel(QAbstractTableModel):
    """Virtual high-performance QAbstractTableModel for Polars DataFrames.

    Fetches cell values on-demand without memory duplication.
    Supports switching between Wide-Format and Tidy Long-Format seamlessly.
    """

    mode_changed = Signal(str)

    def __init__(
        self,
        dataset: DualModeDataset | None = None,
        mode: str = "wide",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dataset: DualModeDataset | None = None
        self._df: pl.DataFrame | None = None
        self._mode: str = mode.lower()
        self._column_names: list[str] = []
        self._series_list: list[pl.Series] = []
        self._row_count: int = 0
        self._col_count: int = 0

        if dataset is not None:
            self.set_dataset(dataset, mode=self._mode)

    @property
    def current_mode(self) -> str:
        """Return currently active mode ('wide' or 'long')."""
        return self._mode

    @property
    def dataframe(self) -> pl.DataFrame | None:
        """Return the currently displayed Polars DataFrame."""
        return self._df

    @property
    def dataset(self) -> DualModeDataset | None:
        """Return the underlying DualModeDataset if set."""
        return self._dataset

    def set_dataset(self, dataset: DualModeDataset | None, mode: str | None = None) -> None:
        """Bind a new DualModeDataset and reset the model."""
        self.beginResetModel()
        self._dataset = dataset
        if mode is not None:
            self._mode = mode.lower()

        if self._dataset is None:
            self._df = None
            self._column_names = []
            self._series_list = []
            self._row_count = 0
            self._col_count = 0
        else:
            if self._mode == "long":
                self._df = self._dataset.to_long()
            else:
                self._df = self._dataset.wide
            self._update_internal_buffers()

        self.endResetModel()
        self.mode_changed.emit(self._mode)

    def set_dataframe(self, df: pl.DataFrame | None) -> None:
        """Directly bind a Polars DataFrame (e.g. for audit log or custom views)."""
        self.beginResetModel()
        self._dataset = None
        self._df = df
        if self._df is None:
            self._column_names = []
            self._series_list = []
            self._row_count = 0
            self._col_count = 0
        else:
            self._update_internal_buffers()
        self.endResetModel()

    def set_mode(self, mode: str) -> None:
        """Switch between 'wide' and 'long' view modes."""
        mode_clean = mode.lower()
        if mode_clean == self._mode and self._df is not None:
            return

        self._mode = mode_clean
        if self._dataset is None:
            self.mode_changed.emit(self._mode)
            return

        self.beginResetModel()
        if self._mode == "long":
            self._df = self._dataset.to_long()
        else:
            self._df = self._dataset.wide
        self._update_internal_buffers()
        self.endResetModel()
        self.mode_changed.emit(self._mode)

    def toggle_mode(self) -> str:
        """Toggle between 'wide' and 'long' and return new mode."""
        new_mode = "long" if self._mode == "wide" else "wide"
        self.set_mode(new_mode)
        return new_mode

    def _update_internal_buffers(self) -> None:
        """Cache zero-copy Series references for microsecond cell indexing."""
        if self._df is None:
            self._column_names = []
            self._series_list = []
            self._row_count = 0
            self._col_count = 0
            return

        self._column_names = list(self._df.columns)
        self._series_list = [self._df[c] for c in self._column_names]
        self._row_count = len(self._df)
        self._col_count = len(self._column_names)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return self._row_count

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return self._col_count

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()
        if row < 0 or row >= self._row_count or col < 0 or col >= self._col_count:
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            val = self._series_list[col][row]
            if val is None:
                series = self._series_list[col]
                if series.dtype.is_numeric():
                    return "NaN"
                return ""
            if isinstance(val, (float, np.floating)):
                if np.isnan(val):
                    return "NaN"
                # Format floats cleanly
                if abs(val) < 1e-4 and val != 0.0:
                    return f"{val:.4e}"
                return f"{val:.4f}"
            return str(val)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            val = self._series_list[col][row]
            if isinstance(val, (int, float, np.number)):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._column_names):
                return self._column_names[section]
        elif orientation == Qt.Orientation.Vertical:
            return str(section + 1)

        return None


class DataGridWidget(QWidget):
    """Complete table view widget wrapping PolarsTableModel with toolbar and mode toggle."""

    mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.mode_label = QLabel("No data loaded")
        self.mode_label.setStyleSheet("font-weight: bold; color: #a0a0a0;")
        toolbar.addWidget(self.mode_label)

        toolbar.addStretch()

        self.mode_btn = QPushButton("Switch to Long View (to_long)")
        self.mode_btn.setEnabled(False)
        self.mode_btn.clicked.connect(self._on_mode_button_clicked)
        toolbar.addWidget(self.mode_btn)

        layout.addLayout(toolbar)

        # Table View
        self.table_view = QTableView()
        self.model = PolarsTableModel(parent=self)
        self.model.mode_changed.connect(self._on_model_mode_changed)
        self.table_view.setModel(self.model)

        # View settings for performance and ergonomics
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_view.horizontalHeader().setStretchLastSection(False)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_view.verticalHeader().setDefaultSectionSize(24)

        layout.addWidget(self.table_view)

    def set_dataset(self, dataset: DualModeDataset | None, mode: str = "wide") -> None:
        """Bind a dataset to the view."""
        self.model.set_dataset(dataset, mode=mode)
        self._update_status()

    def set_mode(self, mode: str) -> None:
        """Set the view mode ('wide' or 'long')."""
        self.model.set_mode(mode)
        self._update_status()

    def _on_mode_button_clicked(self) -> None:
        new_mode = self.model.toggle_mode()
        self._update_status()
        self.mode_changed.emit(new_mode)

    def _on_model_mode_changed(self, mode: str) -> None:
        self._update_status()
        self.mode_changed.emit(mode)

    def _update_status(self) -> None:
        df = self.model.dataframe
        if df is None:
            self.mode_label.setText("No data loaded")
            self.mode_btn.setEnabled(False)
            return

        self.mode_btn.setEnabled(True)
        mode = self.model.current_mode
        n_rows = len(df)
        n_cols = len(df.columns)

        if mode == "wide":
            self.mode_label.setText(f"Wide Format: {n_rows:,} rows × {n_cols} columns")
            self.mode_btn.setText("Tidy Melt: Switch to Long View (to_long)")
        else:
            self.mode_label.setText(f"Tidy Long Format: {n_rows:,} rows × {n_cols} columns")
            self.mode_btn.setText("Switch to Wide View")
