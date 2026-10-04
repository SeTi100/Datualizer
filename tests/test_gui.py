"""Tests for Datualizer PySide6 Desktop GUI & PyQtGraph Multi-Plot Canvas."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import polars as pl
import pytest

# Ensure Qt runs offscreen for headless CI/testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from datualizer_core import AuditLog, DualModeDataset, load_csv
from datualizer_gui.components.data_grid import DataGridWidget, PolarsTableModel
from datualizer_gui.components.inspector import InspectorPanel
from datualizer_gui.components.plot_canvas import MultiChannelPlotCanvas
from datualizer_gui.components.recipe_panel import RecipePanel
from datualizer_gui.main_window import DatualizerMainWindow


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Session-scoped QApplication instance running in offscreen mode."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(scope="module")
def csv_path() -> Path:
    """Resolve path to SA_testmessung_1.csv."""
    path = Path(__file__).resolve().parent.parent / "SA_testmessung_1.csv"
    assert path.is_file(), f"Test CSV not found: {path}"
    return path


@pytest.fixture(scope="module")
def dataset(csv_path: Path) -> DualModeDataset:
    """Load real DualModeDataset from SA_testmessung_1.csv."""
    return load_csv(csv_path)


# ==============================================================================
# 1. PolarsTableModel & DataGridWidget Unit Tests
# ==============================================================================


def test_polars_table_model_empty(qapp: QApplication) -> None:
    """Verify empty PolarsTableModel defaults."""
    model = PolarsTableModel()
    assert model.rowCount() == 0
    assert model.columnCount() == 0
    assert model.current_mode == "wide"
    assert model.dataframe is None
    assert model.dataset is None
    assert model.data(model.index(0, 0)) is None
    assert model.headerData(0, Qt.Orientation.Horizontal) is None


def test_polars_table_model_wide_data(qapp: QApplication, dataset: DualModeDataset) -> None:
    """Verify PolarsTableModel in Wide-Format."""
    model = PolarsTableModel(dataset=dataset, mode="wide")

    # Dimensions
    assert model.rowCount() == 4931
    assert model.columnCount() == 9
    assert model.current_mode == "wide"

    # Header Data
    expected_cols = [
        "time_seconds",
        "kanal_1",
        "kanal_2",
        "kanal_3",
        "kanal_4",
        "kanal_5",
        "kanal_6",
        "kanal_7",
        "kanal_8",
    ]
    for col_idx, col_name in enumerate(expected_cols):
        assert model.headerData(col_idx, Qt.Orientation.Horizontal) == col_name

    # Vertical Header (1-based row index)
    assert model.headerData(0, Qt.Orientation.Vertical) == "1"
    assert model.headerData(4930, Qt.Orientation.Vertical) == "4931"

    # Data extraction for first row: time=0.0, kanal_1=24.109
    time_idx = model.index(0, 0)
    assert model.data(time_idx) == "0.0000"

    k1_idx = model.index(0, 1)
    assert model.data(k1_idx) == "24.1090"

    # Dropout row at index 1221 contains NaN
    k1_dropout = model.index(1221, 1)
    assert model.data(k1_dropout) == "NaN"

    # Alignment roles
    align = model.data(time_idx, Qt.ItemDataRole.TextAlignmentRole)
    assert align == int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)


def test_polars_table_model_long_mode_switch(qapp: QApplication, dataset: DualModeDataset) -> None:
    """Verify switching between Wide and Tidy Long modes in PolarsTableModel."""
    model = PolarsTableModel(dataset=dataset, mode="wide")

    # Switch to Long Mode
    model.set_mode("long")
    assert model.current_mode == "long"
    assert model.rowCount() == 4931 * 8  # 39,448 rows
    assert model.columnCount() == 3
    assert [
        model.headerData(i, Qt.Orientation.Horizontal) for i in range(3)
    ] == ["time_seconds", "channel", "value"]

    # Verify first row in long mode
    assert model.data(model.index(0, 0)) == "0.0000"
    assert model.data(model.index(0, 1)) == "kanal_1"
    assert model.data(model.index(0, 2)) == "24.1090"

    # Toggle back to Wide Mode
    new_mode = model.toggle_mode()
    assert new_mode == "wide"
    assert model.current_mode == "wide"
    assert model.rowCount() == 4931
    assert model.columnCount() == 9


def test_polars_table_model_custom_dataframe(qapp: QApplication) -> None:
    """Verify directly setting an arbitrary Polars DataFrame."""
    model = PolarsTableModel()
    df = pl.DataFrame({"status": ["OK", "WARN", "ERR"], "code": [200, 400, 500]})
    model.set_dataframe(df)

    assert model.rowCount() == 3
    assert model.columnCount() == 2
    assert model.data(model.index(0, 0)) == "OK"
    assert model.data(model.index(2, 1)) == "500"


def test_data_grid_widget(qapp: QApplication, dataset: DualModeDataset) -> None:
    """Verify DataGridWidget UI and mode switching."""
    widget = DataGridWidget()
    assert "No data loaded" in widget.mode_label.text()
    assert not widget.mode_btn.isEnabled()

    widget.set_dataset(dataset)
    assert "Wide Format: 4,931 rows" in widget.mode_label.text()
    assert widget.mode_btn.isEnabled()

    # Click toggle button
    received_modes: list[str] = []
    widget.mode_changed.connect(received_modes.append)

    widget.mode_btn.click()
    assert "long" in received_modes
    assert "Tidy Long Format: 39,448 rows" in widget.mode_label.text()

    # Toggle back
    widget.mode_btn.click()
    assert "wide" in received_modes
    assert "Wide Format: 4,931 rows" in widget.mode_label.text()


# ==============================================================================
# 2. MultiChannelPlotCanvas Unit Tests
# ==============================================================================


def test_plot_canvas_empty_initialization(qapp: QApplication) -> None:
    """Verify empty MultiChannelPlotCanvas defaults safely."""
    canvas = MultiChannelPlotCanvas()
    assert canvas.dataset is None
    assert canvas.active_channels == []
    assert len(canvas.plot_items) == 0
    assert len(canvas.crosshair_lines) == 0

    # Calling crosshair update or reset_zoom on empty canvas should not crash
    canvas.update_crosshair_by_x(10.0)
    canvas.reset_zoom()


def test_plot_canvas_with_channels(qapp: QApplication, dataset: DualModeDataset) -> None:
    """Verify MultiChannelPlotCanvas builds subplots, links X-axes, and creates crosshairs."""
    canvas = MultiChannelPlotCanvas(dataset=dataset)

    # All 8 measurement channels should be active by default
    assert len(canvas.plot_items) == 8
    assert len(canvas.crosshair_lines) == 8
    assert canvas.active_channels == [f"kanal_{i}" for i in range(1, 9)]

    # Verify X-Axis Synchronization across all subplots
    plot_items = list(canvas.plot_items.values())
    first_plot = plot_items[0]

    for idx, p in enumerate(plot_items[1:], start=1):
        # Linked view along X-axis (index 0) must be first_plot's ViewBox
        assert p.getViewBox().linkedView(0) == first_plot.getViewBox(), (
            f"Plot item {idx} ({canvas.active_channels[idx]}) is not X-linked to first plot!"
        )

    # Master X-axis: Top subplots hide bottom axis, bottom-most subplot shows it
    for p in plot_items[:-1]:
        assert not p.getAxis("bottom").isVisible()
    assert plot_items[-1].getAxis("bottom").isVisible()


def test_plot_canvas_dynamic_channel_selection(
    qapp: QApplication, dataset: DualModeDataset
) -> None:
    """Verify dynamically filtering active channels updates subplots and preserves X-link."""
    canvas = MultiChannelPlotCanvas(dataset=dataset)
    assert len(canvas.plot_items) == 8

    # Filter to only kanal_1 and kanal_4
    canvas.set_active_channels(["kanal_1", "kanal_4"])
    assert len(canvas.plot_items) == 2
    assert canvas.active_channels == ["kanal_1", "kanal_4"]
    assert len(canvas.crosshair_lines) == 2

    p1 = canvas.plot_items["kanal_1"]
    p4 = canvas.plot_items["kanal_4"]
    assert p4.getViewBox().linkedView(0) == p1.getViewBox()

    # Clear channels
    canvas.set_active_channels([])
    assert len(canvas.plot_items) == 0
    assert len(canvas.crosshair_lines) == 0


def test_plot_canvas_crosshair_tracking(qapp: QApplication, dataset: DualModeDataset) -> None:
    """Verify synchronized crosshair updates line positions and readout text."""
    canvas = MultiChannelPlotCanvas(dataset=dataset)
    canvas.set_active_channels(["kanal_1", "kanal_2"])

    # Update crosshair at timestamp 100.0s
    canvas.update_crosshair_by_x(100.0)

    for line in canvas.crosshair_lines:
        assert line.value() == pytest.approx(100.0, rel=1e-3)

    text = canvas.crosshair_label.text()
    assert "t =  100.00s" in text
    assert "kanal_1:" in text
    assert "kanal_2:" in text

    # Update crosshair during the dropout period (e.g. 1221.0s)
    canvas.update_crosshair_by_x(1221.0)
    dropout_text = canvas.crosshair_label.text()
    assert "t = 1221.00s" in dropout_text
    assert "kanal_1: NaN (dropout)" in dropout_text
    assert "kanal_2: NaN (dropout)" in dropout_text

    # Test mouse move signal handling
    p1 = canvas.plot_items["kanal_1"]
    scene_pos = p1.mapToScene(QPointF(50, 50))
    canvas._on_mouse_moved(scene_pos)
    assert canvas.crosshair_label.text() != ""


def test_plot_canvas_reset_zoom(qapp: QApplication, dataset: DualModeDataset) -> None:
    """Verify reset_zoom executes without error."""
    canvas = MultiChannelPlotCanvas(dataset=dataset)
    canvas.reset_zoom()


# ==============================================================================
# 3. InspectorPanel Unit Tests
# ==============================================================================


def test_inspector_panel(qapp: QApplication, dataset: DualModeDataset, csv_path: Path) -> None:
    """Verify InspectorPanel metadata, checklist selection, and audit display."""
    inspector = InspectorPanel()
    assert "No file loaded" in inspector.lbl_filepath.text()
    assert inspector.channel_list_widget.count() == 0

    # Track emitted channel toggle signals
    emitted_channels: list[list[str]] = []
    inspector.channels_toggled.connect(emitted_channels.append)

    inspector.set_dataset(dataset, file_path=csv_path, delimiter=",", decimal_sep=",")

    # Metadata check
    assert "4,931" in inspector.lbl_rows.text()
    assert "9" in inspector.lbl_cols.text()
    assert "',' (comma)" in repr(inspector.lbl_delimiter.text()) or "',' (comma)" in inspector.lbl_delimiter.text() or "','" in inspector.lbl_delimiter.text()
    assert "0.0s" in inspector.lbl_timespan.text()

    # Channels checklist check
    assert inspector.channel_list_widget.count() == 8
    selected = inspector.get_selected_channels()
    assert len(selected) == 8
    assert selected == [f"kanal_{i}" for i in range(1, 9)]

    # Deselect and Select All
    inspector.deselect_all_channels()
    assert len(inspector.get_selected_channels()) == 0

    inspector.select_all_channels()
    assert len(inspector.get_selected_channels()) == 8

    # Set specific channels
    inspector.set_selected_channels(["kanal_1", "kanal_5"])
    assert inspector.get_selected_channels() == ["kanal_1", "kanal_5"]

    # Audit log: clean dataset has 0 errors
    assert "0 Conversion Errors" in inspector.lbl_audit_summary.text()
    assert inspector.audit_table.rowCount() == 0


def test_inspector_panel_with_audit_errors(qapp: QApplication) -> None:
    """Verify InspectorPanel cleanly reports and lists conversion audit entries."""
    audit_log = AuditLog()
    audit_log.add(42, "kanal_1", "CORRUPT", "non_convertible_float")
    audit_log.add(99, "kanal_2", "ERR#99", "sentinel_value")

    df = pl.DataFrame({
        "time_seconds": [0.0, 1.0, 2.0],
        "kanal_1": [20.0, None, 22.0],
        "kanal_2": [10.0, 11.0, None],
    })
    ds = DualModeDataset(df=df, audit_log=audit_log, time_col="time_seconds")

    inspector = InspectorPanel()
    inspector.set_dataset(ds, file_path="corrupt_test.csv")

    assert "2 Conversion Anomalies" in inspector.lbl_audit_summary.text()
    assert inspector.audit_table.rowCount() == 2
    assert inspector.audit_table.item(0, 0).text() == "42"
    assert inspector.audit_table.item(0, 1).text() == "kanal_1"
    assert inspector.audit_table.item(0, 2).text() == "CORRUPT"
    assert inspector.audit_table.item(0, 3).text() == "non_convertible_float"


def test_inspector_quick_load_signal(qapp: QApplication) -> None:
    """Verify Quick Load button triggers file_selected signal."""
    inspector = InspectorPanel()
    files_selected: list[str] = []
    inspector.file_selected.connect(files_selected.append)

    inspector.btn_quick_load.click()
    assert len(files_selected) == 1
    assert "SA_testmessung_1.csv" in files_selected[0]


# ==============================================================================
# 4. RecipePanel Unit Tests
# ==============================================================================


def test_recipe_panel(qapp: QApplication, dataset: DualModeDataset) -> None:
    """Verify RecipePanel step listing, view mode toggling, and zoom reset signal."""
    recipe = RecipePanel()
    assert recipe.current_mode == "wide"
    assert recipe.step_list.count() == 5

    # Test Mode Toggle Signal
    modes: list[str] = []
    recipe.view_mode_changed.connect(modes.append)

    recipe.btn_toggle_mode.click()
    assert "long" in modes
    assert recipe.current_mode == "long"
    assert "Switch to Wide" in recipe.btn_toggle_mode.text()

    recipe.btn_toggle_mode.click()
    assert "wide" in modes
    assert recipe.current_mode == "wide"

    # Test Reset Zoom Signal
    zoom_clicks: list[bool] = []
    recipe.reset_zoom_clicked.connect(lambda: zoom_clicks.append(True))
    recipe.btn_reset_zoom.click()
    assert len(zoom_clicks) == 1

    # Update with dataset
    recipe.set_dataset(dataset)
    assert any("4,931" in recipe.step_list.item(i).text() for i in range(recipe.step_list.count()))


# ==============================================================================
# 5. DatualizerMainWindow Integration Tests
# ==============================================================================


def test_main_window_auto_load(qapp: QApplication) -> None:
    """Verify DatualizerMainWindow initializes, auto-loads SA_testmessung_1.csv, and wires views."""
    win = DatualizerMainWindow(auto_load=True)

    # 1. Dataset verification
    assert win.dataset is not None
    assert len(win.dataset) == 4931
    assert "SA_testmessung_1.csv" in win.status_bar.currentMessage()

    # 2. Docking Layout & Component Presence
    assert win.dock_inspector.widget() is win.inspector
    assert win.dock_recipe.widget() is win.recipe_panel
    assert win.dock_data_grid.widget() is win.data_grid
    assert win.centralWidget() is win.plot_canvas

    # 3. Canvas & Subplot Verification
    assert len(win.plot_canvas.plot_items) == 8
    first_p = list(win.plot_canvas.plot_items.values())[0]
    second_p = list(win.plot_canvas.plot_items.values())[1]
    assert second_p.getViewBox().linkedView(0) == first_p.getViewBox()

    # 4. Data Grid Verification
    assert win.data_grid.model.rowCount() == 4931
    assert win.data_grid.model.columnCount() == 9

    # 5. Inspector Verification
    assert len(win.inspector.get_selected_channels()) == 8


def test_main_window_channel_toggle_interaction(qapp: QApplication) -> None:
    """Verify deselecting channels in Inspector immediately updates Canvas subplots."""
    win = DatualizerMainWindow(auto_load=True)
    assert len(win.plot_canvas.plot_items) == 8

    # Uncheck all except kanal_1 and kanal_2
    win.inspector.set_selected_channels(["kanal_1", "kanal_2"])
    qapp.processEvents()

    assert len(win.plot_canvas.plot_items) == 2
    assert win.plot_canvas.active_channels == ["kanal_1", "kanal_2"]

    p1 = win.plot_canvas.plot_items["kanal_1"]
    p2 = win.plot_canvas.plot_items["kanal_2"]
    assert p2.getViewBox().linkedView(0) == p1.getViewBox()


def test_main_window_mode_toggle_interaction(qapp: QApplication) -> None:
    """Verify toggling Tidy Melt in RecipePanel updates the DataGrid and vice versa."""
    win = DatualizerMainWindow(auto_load=True)
    assert win.data_grid.model.current_mode == "wide"

    # Click toggle in RecipePanel
    win.recipe_panel.btn_toggle_mode.click()
    qapp.processEvents()

    assert win.data_grid.model.current_mode == "long"
    assert win.data_grid.model.rowCount() == 4931 * 8

    # Click toggle in DataGrid
    win.data_grid.mode_btn.click()
    qapp.processEvents()

    assert win.data_grid.model.current_mode == "wide"
    assert win.recipe_panel.current_mode == "wide"
    assert win.data_grid.model.rowCount() == 4931


def test_main_window_reset_zoom_interaction(qapp: QApplication) -> None:
    """Verify clicking Reset Zoom in RecipePanel calls auto-range without errors."""
    win = DatualizerMainWindow(auto_load=True)
    win.recipe_panel.btn_reset_zoom.click()
    qapp.processEvents()


def test_main_window_fresh_without_autoload(qapp: QApplication) -> None:
    """Verify DatualizerMainWindow starts cleanly when auto_load=False."""
    win = DatualizerMainWindow(auto_load=False)
    assert win.dataset is None
    assert len(win.plot_canvas.plot_items) == 0
    assert win.data_grid.model.rowCount() == 0
    assert "Ready" in win.status_bar.currentMessage()


# ==============================================================================
# 6. Edge Cases & Robustness Tests
# ==============================================================================


def test_crosshair_out_of_bounds(qapp: QApplication, dataset: DualModeDataset) -> None:
    """Verify crosshair clamps correctly when X is outside [min_time, max_time]."""
    canvas = MultiChannelPlotCanvas(dataset=dataset)
    canvas.set_active_channels(["kanal_1"])

    # Far before start time (< 0)
    canvas.update_crosshair_by_x(-999.0)
    text_before = canvas.crosshair_label.text()
    assert "t =    0.00s" in text_before

    # Far after end time (> 4930)
    canvas.update_crosshair_by_x(999999.0)
    text_after = canvas.crosshair_label.text()
    assert "t = 4930.00s" in text_after


def test_plot_canvas_single_channel(qapp: QApplication, dataset: DualModeDataset) -> None:
    """Verify single channel plot displays bottom axis without X-link error."""
    canvas = MultiChannelPlotCanvas(dataset=dataset)
    canvas.set_active_channels(["kanal_3"])
    assert len(canvas.plot_items) == 1
    p3 = canvas.plot_items["kanal_3"]
    assert p3.getAxis("bottom").isVisible()
    assert p3.getViewBox().linkedView(0) is None


def test_plot_canvas_invalid_channel_names(qapp: QApplication, dataset: DualModeDataset) -> None:
    """Verify setting non-existent channel names filters them out safely."""
    canvas = MultiChannelPlotCanvas(dataset=dataset)
    canvas.set_active_channels(["non_existent_1", "non_existent_2"])
    assert len(canvas.plot_items) == 0
    assert canvas.active_channels == []


def test_polars_table_model_scientific_and_negative(qapp: QApplication) -> None:
    """Verify table model correctly handles tiny numbers, negatives, and strings."""
    df = pl.DataFrame({
        "label": ["A", "B", None],
        "tiny": [1.23e-6, -4.56e-8, None],
        "negative": [-12.34567, -0.001, 100.0],
    })
    model = PolarsTableModel()
    model.set_dataframe(df)

    assert model.rowCount() == 3
    assert model.columnCount() == 3

    # Text column
    assert model.data(model.index(0, 0)) == "A"
    assert model.data(model.index(2, 0)) == ""  # String None displays as empty string

    # Tiny float (scientific notation < 1e-4)
    tiny_val = model.data(model.index(0, 1))
    assert "e" in tiny_val.lower()

    # Negative float
    neg_val = model.data(model.index(0, 2))
    assert neg_val == "-12.3457"

    # Numeric null displays as NaN
    assert model.data(model.index(2, 1)) == "NaN"


def test_main_window_load_invalid_file(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify loading non-existent file does not crash the application."""
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    win = DatualizerMainWindow(auto_load=False)
    win.load_file("completely_missing_file_12345.csv")
    assert win.dataset is None

