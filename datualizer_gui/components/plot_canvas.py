"""Synchronized Multi-Channel PyQtGraph Plot Canvas with linked X-axes and crosshair."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from datualizer_core.dataset import DualModeDataset

# High-contrast color palette for clear channel identification
CHANNEL_COLORS = [
    "#00d2ff",  # Bright Cyan
    "#ff9f43",  # Vibrant Orange
    "#10ac84",  # Emerald Green
    "#ee5253",  # Crimson Red
    "#9b59b6",  # Amethyst Purple
    "#feca57",  # Sun Yellow
    "#48dbfb",  # Sky Blue
    "#ff6b6b",  # Coral
    "#1dd1a1",  # Mint
    "#54a0ff",  # Cobalt
]


class MultiChannelPlotCanvas(QWidget):
    """PyQtGraph canvas displaying synchronized subplots for selected measurement channels.

    Key Features:
    - Subplot per active channel stacked vertically.
    - Synchronized X-Axes: All active subplots are linked via `setXLink`, ensuring pan/zoom
      in one subplot seamlessly moves all other subplots in real time.
    - Synchronized Crosshair: Vertical dash-line across all subplots tracking the mouse cursor
      with high-precision timestamp and value readout across all visible channels.
    - Fast zero-copy data binding: Uses `dataset.to_numpy(col, allow_copy=False)` when possible,
      with graceful fallback to `allow_copy=True` when nulls/sentinels require NaN conversion.
    """

    def __init__(
        self,
        dataset: DualModeDataset | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dataset: DualModeDataset | None = None
        self._active_channels: list[str] = []
        self._plot_items: dict[str, pg.PlotItem] = {}
        self._crosshair_lines: list[pg.InfiniteLine] = []
        self._time_data: np.ndarray | None = None
        self._cached_channel_arrays: dict[str, np.ndarray] = {}

        self._init_ui()
        if dataset is not None:
            self.set_dataset(dataset)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # Crosshair / status overlay header
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(4, 2, 4, 2)

        self.crosshair_label = QLabel("Crosshair: Hover over plots to inspect channel values")
        self.crosshair_label.setStyleSheet(
            "QLabel {"
            "  background-color: #1e1e24;"
            "  color: #dcdcdc;"
            "  font-family: 'Consolas', 'Courier New', monospace;"
            "  font-size: 11px;"
            "  padding: 5px 10px;"
            "  border-radius: 4px;"
            "  border: 1px solid #32323e;"
            "}"
        )
        header_layout.addWidget(self.crosshair_label)
        layout.addLayout(header_layout)

        # PyQtGraph GraphicsLayoutWidget
        pg.setConfigOptions(antialias=True, useOpenGL=False)
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground("#18181c")
        self.graphics_layout.scene().sigMouseMoved.connect(self._on_mouse_moved)
        layout.addWidget(self.graphics_layout)

    @property
    def dataset(self) -> DualModeDataset | None:
        """Return the bound DualModeDataset."""
        return self._dataset

    @property
    def active_channels(self) -> list[str]:
        """Return currently active channel names."""
        return list(self._active_channels)

    @property
    def plot_items(self) -> dict[str, pg.PlotItem]:
        """Return dictionary mapping channel name to its pg.PlotItem."""
        return self._plot_items

    @property
    def crosshair_lines(self) -> list[pg.InfiniteLine]:
        """Return list of active crosshair vertical lines."""
        return self._crosshair_lines

    def set_dataset(
        self,
        dataset: DualModeDataset | None,
        active_channels: Sequence[str] | None = None,
    ) -> None:
        """Set a new dataset and refresh the subplots."""
        self._dataset = dataset
        self._cached_channel_arrays.clear()
        self._time_data = None

        if self._dataset is None:
            self._active_channels = []
            self._rebuild_plots()
            return

        # Extract time data (zero-copy if possible)
        time_col = self._dataset.time_col
        try:
            self._time_data = self._dataset.to_numpy(time_col, allow_copy=False)
        except Exception:
            self._time_data = self._dataset.to_numpy(time_col, allow_copy=True)

        if active_channels is not None:
            self._active_channels = [c for c in active_channels if c in self._dataset.columns]
        else:
            # Default to all numeric measurement channels (metadata is not plotted)
            self._active_channels = list(self._dataset.channels)

        self._rebuild_plots()

    def set_active_channels(self, channels: Sequence[str]) -> None:
        """Dynamically update active channels and reconfigure synchronized subplots."""
        if self._dataset is None:
            self._active_channels = list(channels)
            return

        valid = [c for c in channels if c in self._dataset.columns]
        if valid == self._active_channels and len(self._plot_items) == len(valid):
            return

        self._active_channels = valid
        self._rebuild_plots()

    def _extract_channel_data(self, col: str) -> np.ndarray:
        """Extract column data via zero-copy fast path with graceful fallback for NaNs."""
        if self._dataset is None:
            return np.array([], dtype=np.float64)

        try:
            return self._dataset.to_numpy(col, allow_copy=False)
        except Exception:
            return self._dataset.to_numpy(col, allow_copy=True)

    def _rebuild_plots(self) -> None:
        """Reconstruct the stacked subplots, link X-axes, and attach crosshair lines."""
        self.graphics_layout.clear()
        self._plot_items.clear()
        self._crosshair_lines.clear()

        if self._dataset is None or not self._active_channels or self._time_data is None:
            self.crosshair_label.setText("Crosshair: No channels active or data loaded")
            return

        first_plot: pg.PlotItem | None = None
        total = len(self._active_channels)

        for idx, ch in enumerate(self._active_channels):
            # Extract and cache channel data
            if ch not in self._cached_channel_arrays:
                self._cached_channel_arrays[ch] = self._extract_channel_data(ch)
            ch_data = self._cached_channel_arrays[ch]

            # Add subplot in row `idx`, col 0
            p: pg.PlotItem = self.graphics_layout.addPlot(row=idx, col=0)
            self._plot_items[ch] = p

            # Synchronize X-Axes across all subplots
            if first_plot is None:
                first_plot = p
            else:
                p.setXLink(first_plot)

            color = CHANNEL_COLORS[idx % len(CHANNEL_COLORS)]

            # Configure axes styling
            p.setLabel("left", ch, color=color, **{"font-size": "10pt", "font-weight": "bold"})
            p.showGrid(x=True, y=True, alpha=0.22)
            p.getAxis("left").setWidth(65)

            if idx < total - 1:
                # Hide bottom axis for upper subplots to avoid redundant clutter
                p.showAxis("bottom", False)
            else:
                # Bottom-most subplot shows the master X-axis
                p.showAxis("bottom", True)
                p.setLabel("bottom", "Time (seconds)", color="#a0a0a0")

            # Plot series data
            p.plot(
                self._time_data,
                ch_data,
                pen=pg.mkPen(color=color, width=1.5),
                name=ch,
                connect="finite",
            )

            # Add crosshair vertical line
            v_line = pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pg.mkPen("#ffdd57", width=1.2, style=Qt.PenStyle.DashLine),
            )
            p.addItem(v_line, ignoreBounds=True)
            self._crosshair_lines.append(v_line)

        self.reset_zoom()

    def reset_zoom(self) -> None:
        """Reset view range to auto-fit all active subplots."""
        for p in self._plot_items.values():
            p.enableAutoRange()

    def update_crosshair_by_x(self, x_val: float) -> None:
        """Update crosshair position and overlay values for a specific X (time) coordinate."""
        for line in self._crosshair_lines:
            line.setValue(x_val)

        if self._time_data is None or len(self._time_data) == 0 or not self._active_channels:
            return

        time_arr = self._time_data
        idx = int(np.searchsorted(time_arr, x_val))
        if idx >= len(time_arr):
            idx = len(time_arr) - 1
        elif idx > 0 and abs(time_arr[idx - 1] - x_val) < abs(time_arr[idx] - x_val):
            idx -= 1

        t_val = time_arr[idx]
        parts = [f"t = {t_val:7.2f}s"]

        for ch in self._active_channels:
            arr = self._cached_channel_arrays.get(ch)
            if arr is not None and idx < len(arr):
                val = arr[idx]
                if np.isnan(val):
                    parts.append(f"{ch}: NaN (dropout)")
                else:
                    parts.append(f"{ch}: {val:6.3f}")

        self.crosshair_label.setText("   |   ".join(parts))

    def _on_mouse_moved(self, pos: QPointF) -> None:
        """Handle mouse movement over the graphics scene and broadcast crosshair updates."""
        if not self._plot_items:
            return

        for p in self._plot_items.values():
            if p.sceneBoundingRect().contains(pos):
                mouse_point = p.vb.mapSceneToView(pos)
                self.update_crosshair_by_x(mouse_point.x())
                return
