"""Datualizer Main Window: 3-Panel Docking Layout integrating Inspector, Canvas, Recipe, and Grid."""

from __future__ import annotations

import logging
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)

from datualizer_core import DualModeDataset, load_csv, pre_scan
from datualizer_gui.components.data_grid import DataGridWidget
from datualizer_gui.components.inspector import InspectorPanel
from datualizer_gui.components.plot_canvas import MultiChannelPlotCanvas
from datualizer_gui.components.recipe_panel import RecipePanel

logger = logging.getLogger(__name__)


def apply_modern_theme(widget: QWidget) -> None:
    """Apply a modern dark theme palette to the application or main window."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(28, 29, 36))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 222, 230))
    palette.setColor(QPalette.ColorRole.Base, QColor(20, 21, 26))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(32, 34, 42))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(40, 42, 54))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 222, 230))
    palette.setColor(QPalette.ColorRole.Button, QColor(38, 40, 50))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 222, 230))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 180, 216))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    widget.setPalette(palette)


class DatualizerMainWindow(QMainWindow):
    """Main Application Window for Datualizer.

    Arranges GUI components in a 3-panel docking layout:
    - Central Widget: Synchronized MultiChannelPlotCanvas
    - Left Dock: InspectorPanel (file open, metadata, channel checklist, audit log)
    - Right Dock: RecipePanel (transformation steps, to_long toggle, reset zoom)
    - Bottom Dock: DataGridWidget (virtual PolarsTableModel for wide & long format)
    """

    def __init__(
        self,
        auto_load: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Datualizer - Sensor Data Wrangling & Visualization")
        self.resize(1360, 860)
        self.setDockNestingEnabled(True)

        apply_modern_theme(self)

        self._dataset: DualModeDataset | None = None
        self._current_file: Path | None = None

        self._create_components()
        self._setup_docking_layout()
        self._connect_signals()
        self._create_menu_and_status()

        if auto_load:
            self._auto_load_default()

    def _create_components(self) -> None:
        """Instantiate primary GUI components."""
        self.plot_canvas = MultiChannelPlotCanvas(parent=self)
        self.inspector = InspectorPanel(parent=self)
        self.recipe_panel = RecipePanel(parent=self)
        self.data_grid = DataGridWidget(parent=self)

    def _setup_docking_layout(self) -> None:
        """Assemble the 3-panel docking layout around the central plot canvas."""
        # 1. Central Widget: Plot Canvas
        self.setCentralWidget(self.plot_canvas)

        # 2. Left Dock: Inspector & Channels
        self.dock_inspector = QDockWidget("Inspector & Channels", self)
        self.dock_inspector.setObjectName("dock_inspector")
        self.dock_inspector.setWidget(self.inspector)
        self.dock_inspector.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_inspector)

        # 3. Right Dock: Recipe & Actions
        self.dock_recipe = QDockWidget("Pipeline & Recipe", self)
        self.dock_recipe.setObjectName("dock_recipe")
        self.dock_recipe.setWidget(self.recipe_panel)
        self.dock_recipe.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_recipe)

        # 4. Bottom Dock: Virtual Data Table
        self.dock_data_grid = QDockWidget("Data Table (Wide / Long View)", self)
        self.dock_data_grid.setObjectName("dock_data_grid")
        self.dock_data_grid.setWidget(self.data_grid)
        self.dock_data_grid.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_data_grid)

        # Give appropriate default widths and heights
        self.resizeDocks([self.dock_inspector], [320], Qt.Orientation.Horizontal)
        self.resizeDocks([self.dock_recipe], [280], Qt.Orientation.Horizontal)
        self.resizeDocks([self.dock_data_grid], [220], Qt.Orientation.Vertical)

    def _connect_signals(self) -> None:
        """Wire signals across components."""
        # Inspector -> Load File
        self.inspector.file_selected.connect(self.load_file)

        # Inspector -> Channel Checklist changed -> Update Plots
        self.inspector.channels_toggled.connect(self.plot_canvas.set_active_channels)

        # Recipe -> Mode Changed (wide / long) -> Update Data Grid
        self.recipe_panel.view_mode_changed.connect(self.set_view_mode)

        # Recipe -> Reset Zoom -> Plot Canvas
        self.recipe_panel.reset_zoom_clicked.connect(self.plot_canvas.reset_zoom)

        # Data Grid -> Mode Changed -> Synchronize Recipe Panel
        self.data_grid.mode_changed.connect(self.recipe_panel.set_view_mode)

    def _create_menu_and_status(self) -> None:
        """Create menu bar and status bar."""
        # Menu Bar
        menubar = self.menuBar()

        # File Menu
        menu_file = menubar.addMenu("&File")

        action_open = QAction("&Open CSV...", self)
        action_open.setShortcut(QKeySequence.StandardKey.Open)
        action_open.triggered.connect(self._on_menu_open_file)
        menu_file.addAction(action_open)

        action_quick = QAction("&Quick Load SA_testmessung_1.csv", self)
        action_quick.triggered.connect(self._auto_load_default)
        menu_file.addAction(action_quick)

        menu_file.addSeparator()

        action_exit = QAction("E&xit", self)
        action_exit.setShortcut(QKeySequence("Ctrl+Q"))
        action_exit.triggered.connect(self.close)
        menu_file.addAction(action_exit)

        # View Menu
        menu_view = menubar.addMenu("&View")
        menu_view.addAction(self.dock_inspector.toggleViewAction())
        menu_view.addAction(self.dock_recipe.toggleViewAction())
        menu_view.addAction(self.dock_data_grid.toggleViewAction())
        menu_view.addSeparator()

        action_reset_zoom = QAction("&Reset Zoom", self)
        action_reset_zoom.setShortcut(QKeySequence("Ctrl+R"))
        action_reset_zoom.triggered.connect(self.plot_canvas.reset_zoom)
        menu_view.addAction(action_reset_zoom)

        action_toggle_mode = QAction("Toggle &Wide / Long View", self)
        action_toggle_mode.setShortcut(QKeySequence("Ctrl+T"))
        action_toggle_mode.triggered.connect(self._toggle_view_mode)
        menu_view.addAction(action_toggle_mode)

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — Open a measurement CSV or quick-load to begin.")

    @property
    def dataset(self) -> DualModeDataset | None:
        """Return currently loaded dataset."""
        return self._dataset

    def load_file(self, file_path: str | Path) -> None:
        """Load and parse measurement CSV into DualModeDataset and distribute to all views."""
        p = Path(file_path)
        if not p.is_file():
            # Try resolving relative to workspace if needed
            workspace_candidate = Path.cwd() / file_path
            if workspace_candidate.is_file():
                p = workspace_candidate
            else:
                QMessageBox.warning(
                    self, "File Not Found", f"Could not find measurement file: {file_path}"
                )
                return

        try:
            # 1. Pre-scan for metadata inspection
            res = pre_scan(p)

            # 2. Ingest into DualModeDataset
            dataset = load_csv(p, pre_scan_result=res)
            self._dataset = dataset
            self._current_file = p

            # 3. Distribute to views
            self.inspector.set_dataset(
                dataset,
                file_path=p,
                delimiter=res.delimiter,
                decimal_sep=res.decimal_separator,
            )

            channels = self.inspector.get_selected_channels()
            self.plot_canvas.set_dataset(dataset, active_channels=channels)
            self.data_grid.set_dataset(dataset, mode=self.recipe_panel.current_mode)
            self.recipe_panel.set_dataset(dataset)

            # 4. Status Bar message
            msg = (
                f"Loaded {p.name}: {len(dataset):,} rows, "
                f"{len(channels)} active channels | Audit Errors: {len(dataset.audit_log)}"
            )
            self.status_bar.showMessage(msg)
            logger.info(msg)

        except Exception as exc:
            logger.exception("Failed to load measurement file: %s", exc)
            QMessageBox.critical(
                self,
                "Loading Error",
                f"An error occurred while loading {p.name}:\n\n{exc}",
            )

    def set_view_mode(self, mode: str) -> None:
        """Coordinate view mode change between recipe panel and data grid."""
        self.data_grid.set_mode(mode)
        self.recipe_panel.set_view_mode(mode)

    def _toggle_view_mode(self) -> None:
        new_mode = "long" if self.recipe_panel.current_mode == "wide" else "wide"
        self.set_view_mode(new_mode)

    def _on_menu_open_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Measurement CSV",
            "",
            "CSV / Delimited Files (*.csv *.tsv *.txt);;All Files (*)",
        )
        if file_path:
            self.load_file(file_path)

    def _auto_load_default(self) -> None:
        """Automatically load SA_testmessung_1.csv if present in the environment."""
        target = "SA_testmessung_1.csv"
        candidates = [
            Path.cwd() / target,
            Path(__file__).resolve().parent.parent / target,
        ]
        for c in candidates:
            if c.is_file():
                self.load_file(c)
                return
