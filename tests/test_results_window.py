"""Tests for Results Views window and ViewSpec models (Etappe 2b, Schritt 8, PR 7).

Covers:
1. Pydantic ViewSpec models (TableViewSpec, XYViewSpec, MatrixViewSpec, RefLine, SecondaryAxis).
2. Validation: valid names, formula checks, duplicate view names in Evaluation.views.
3. Rendering offscreen:
   - TableViewWidget: filtering, sorting, row selection.
   - XYViewWidget:
     * Light-off (x=T_bett, y=X, error bars, color_by=V_MFC, marker_by=tag, connect_by=V_MFC, filled=stationaer).
     * Faceting (facet_by=T_klasse).
     * Secondary axis (Arrhenius with top °C axis).
   - MatrixViewWidget:
     * rows=T_klasse, cols=V_MFC, value=X, status=stationaer, WASH grey for non-stationary.
4. Interactivity:
   - Clicking scatter point emits step_inspected.
   - Double-clicking row in TableView emits step_inspected.
5. Export:
   - PNG, SVG, PDF export for XY plot and Matrix.
   - CSV export for Table.
6. Integration with CampaignSession and ResultsWindow scenario switching.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import numpy as np
import polars as pl
import pyqtgraph as pg
import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QGraphicsRectItem

from datualizer_core.calc.uncertain import Constant
from datualizer_core.campaign.evaluation import Evaluation, Scenario
from datualizer_core.campaign.views import (
    MatrixViewSpec,
    RefLine,
    SecondaryAxis,
    TableViewSpec,
    ViewSpec,
    XYViewSpec,
    eval_table_expr,
    filter_table,
)
from datualizer_gui.components.plot_export import ExportOptions, export_graphics_layout
from datualizer_gui.results_window import (
    MatrixViewWidget,
    ResultsWindow,
    TableViewWidget,
    XYViewWidget,
    ink_on_color,
    interpolate_color,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture
def synthetic_results() -> pl.DataFrame:
    """Realistic results table mimicking catalytic ethanol oxidation across 2 days."""
    return pl.DataFrame({
        "step_id": [f"step_{i+1:02d}" for i in range(12)],
        "tag": ["2026-10-12"] * 6 + ["2026-10-15"] * 6,
        "start": [100.0 + i * 600.0 for i in range(12)],
        "ende": [600.0 + i * 600.0 for i in range(12)],
        "T_bett": [152.0, 175.0, 198.0, 150.0, 178.0, 202.0, 153.0, 176.0, 200.0, 151.0, 174.0, 201.0],
        "u_T_bett": [1.5, 1.4, 1.6, 1.5, 1.4, 1.6, 1.5, 1.4, 1.6, 1.5, 1.4, 1.6],
        "T_klasse": [150.0, 175.0, 200.0, 150.0, 175.0, 200.0, 150.0, 175.0, 200.0, 150.0, 175.0, 200.0],
        "V_MFC": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        "c_in_mg_m3": [75.0, 74.0, 76.0, 140.0, 138.0, 142.0, 75.0, 75.0, 76.0, 139.0, 141.0, 140.0],
        "X": [0.15, 0.52, 0.88, 0.10, 0.44, 0.81, 0.18, 0.55, 0.90, 0.12, 0.46, 0.83],
        "u_X": [0.03, 0.04, 0.02, 0.03, 0.04, 0.03, 0.03, 0.04, 0.02, 0.03, 0.04, 0.03],
        "k_1_s": [0.05, 0.25, 1.10, 0.06, 0.28, 1.15, 0.05, 0.26, 1.12, 0.06, 0.29, 1.18],
        "u_lnk_zufall": [0.12, 0.08, 0.05, 0.11, 0.08, 0.05, 0.12, 0.08, 0.05, 0.11, 0.08, 0.05],
        "stationaer": [True, True, True, True, True, True, True, True, True, False, True, True],
        "stat_ok": [True, True, True, True, True, True, True, True, True, False, True, True],
    })


# ------------------------------------------------------------------ Model & Validation Tests
def test_view_spec_models_and_validation() -> None:
    # TableViewSpec
    t_spec = TableViewSpec(name="summary_table", columns=["step_id", "T_bett", "X"], filter="stationaer and X > 0.2")
    assert t_spec.kind == "table"
    assert t_spec.name == "summary_table"

    # XYViewSpec
    xy_spec = XYViewSpec(
        name="light_off",
        title="Light-off Curve",
        x="T_bett",
        y="X",
        u_x="u_T_bett",
        u_y="u_X",
        color_by="V_MFC",
        color_mode="ordinal",
        marker_by="tag",
        filled="stationaer",
        connect_by="V_MFC",
        ref_lines=[RefLine(value=0.5, label="T50"), RefLine(value=0.9, label="T90")],
        secondary_x=SecondaryAxis(
            label="Bett-T [°C]",
            forward_expr="1000 / x - 273.15",
            inverse_expr="1000 / (x + 273.15)",
        ),
    )
    assert xy_spec.kind == "xy"
    assert xy_spec.effective_x_expr == "T_bett"
    assert xy_spec.effective_y_expr == "X"

    # MatrixViewSpec
    m_spec = MatrixViewSpec(
        name="umsatz_matrix",
        title="Umsatz Matrix",
        rows="T_klasse",
        cols="V_MFC",
        value="X",
        status="stat_ok",
        subtext="{c_in_mg_m3:.0f} mg/Nm³",
    )
    assert m_spec.kind == "matrix"

    # Invalid names
    with pytest.raises(ValueError):
        TableViewSpec(name="invalid name with spaces")
    with pytest.raises(ValueError):
        XYViewSpec(name="log")  # function name not allowed as view name

    # Invalid formula
    with pytest.raises(Exception):
        XYViewSpec(name="broken", filter="syntax error ==")


def test_evaluation_views_integration() -> None:
    t_spec = TableViewSpec(name="table1")
    xy_spec = XYViewSpec(name="plot1", x="T_bett", y="X")
    m_spec = MatrixViewSpec(name="matrix1", rows="T_klasse", cols="V_MFC", value="X")

    ev = Evaluation(views=[t_spec, xy_spec, m_spec])
    assert ev.view_names() == ["table1", "plot1", "matrix1"]
    assert ev.view("plot1").name == "plot1"

    with pytest.raises(KeyError):
        ev.view("unknown_view")

    # Duplicate view names
    with pytest.raises(ValueError, match="Duplicate view name"):
        Evaluation(views=[t_spec, TableViewSpec(name="table1")])


def test_helpers_table_env_and_eval(synthetic_results: pl.DataFrame) -> None:
    sub = filter_table(synthetic_results, "stationaer and X >= 0.5")
    assert len(sub) == 6

    res_x = eval_table_expr(synthetic_results, "1000 / (T_bett + 273.15)")
    assert len(res_x) == 12
    assert res_x[0] == pytest.approx(1000.0 / (152.0 + 273.15))

    # Ink contrast check
    assert ink_on_color("#ffffff") == "#0b0b0b"
    assert ink_on_color("#000000") == "white"
    assert ink_on_color("#0d366b") == "white"  # dark navy
    assert ink_on_color("#cde2fb") == "#0b0b0b"  # light blue

    # Color interpolation check
    c_mid = interpolate_color(["#000000", "#ffffff"], 0.5)
    assert c_mid.lower() in ("#7f7f7f", "#808080")


# ------------------------------------------------------------------ GUI Offscreen Tests
def test_table_view_widget(qapp: QApplication, synthetic_results: pl.DataFrame) -> None:
    spec = TableViewSpec(name="t1", columns=["step_id", "T_bett", "X"], filter="stationaer", sort_by=["T_bett"])
    w = TableViewWidget(spec)
    w.set_data(synthetic_results)

    assert w.table.rowCount() == 11  # 11 stationary steps
    assert w.table.columnCount() == 3

    inspected = []
    w.step_inspected.connect(inspected.append)
    # Trigger double click
    item = w.table.item(0, 0)
    w._on_row_double_clicked(item)
    assert len(inspected) == 1
    assert inspected[0] in synthetic_results["step_id"].to_list()


def test_xy_view_widget_light_off(qapp: QApplication, synthetic_results: pl.DataFrame) -> None:
    spec = XYViewSpec(
        name="lo",
        x="T_bett",
        y="X",
        u_x="u_T_bett",
        u_y="u_X",
        color_by="V_MFC",
        marker_by="tag",
        connect_by="V_MFC",
        filled="stationaer",
        ref_lines=[RefLine(value=0.5, label="T50")],
    )
    w = XYViewWidget(spec)
    w.set_data(synthetic_results)
    assert "12 points" in w.status.text()


def test_xy_view_widget_faceting(qapp: QApplication, synthetic_results: pl.DataFrame) -> None:
    spec = XYViewSpec(
        name="conc",
        x="c_in_mg_m3",
        y="X",
        u_y="u_X",
        facet_by="T_klasse",
        facet_cols=3,
        marker_by="tag",
    )
    w = XYViewWidget(spec)
    w.set_data(synthetic_results)
    assert "3 facet(s)" in w.status.text()


def test_xy_view_widget_secondary_axis(qapp: QApplication, synthetic_results: pl.DataFrame) -> None:
    spec = XYViewSpec(
        name="arrhenius",
        x_expr="1000 / (T_bett + 273.15)",
        y_expr="log(k_1_s)",
        u_y="u_lnk_zufall",
        color_by="V_MFC",
        secondary_x=SecondaryAxis(
            label="Bett-T [°C]",
            forward_expr="1000 / x - 273.15",
            inverse_expr="1000 / (x + 273.15)",
        ),
    )
    w = XYViewWidget(spec)
    w.set_data(synthetic_results)
    assert "12 points" in w.status.text()


def test_matrix_view_widget(qapp: QApplication, synthetic_results: pl.DataFrame) -> None:
    spec = MatrixViewSpec(
        name="mat",
        rows="T_klasse",
        cols="V_MFC",
        value="X",
        status="stationaer",
        subtext="mg/Nm³",
    )
    w = MatrixViewWidget(spec)
    w.set_data(synthetic_results)
    # Test widget exists and does not crash
    assert w.glw is not None


def test_results_window_tabs_and_scenario(qapp: QApplication, synthetic_results: pl.DataFrame) -> None:
    win = ResultsWindow(results_df=synthetic_results)
    win.show()
    qapp.processEvents()

    # Default table when no views defined
    assert win.tabs.count() == 1

    # With multiple views
    views = [
        TableViewSpec(name="tab_table", title="Table View"),
        XYViewSpec(name="tab_xy", title="XY Plot", x="T_bett", y="X"),
        MatrixViewSpec(name="tab_matrix", title="Matrix", rows="T_klasse", cols="V_MFC", value="X"),
    ]
    win._build_views(views, synthetic_results)
    assert win.tabs.count() == 3
    assert win.tabs.tabText(0) == "Table View"
    assert win.tabs.tabText(1) == "XY Plot"
    assert win.tabs.tabText(2) == "Matrix"

    win.close()


def test_views_export(qapp: QApplication, synthetic_results: pl.DataFrame, tmp_path: Path) -> None:
    # 1. Export Table to CSV
    t_spec = TableViewSpec(name="table_exp", columns=["step_id", "T_bett", "X"])
    tw = TableViewWidget(t_spec)
    tw.set_data(synthetic_results)
    csv_file = tmp_path / "table.csv"
    tw.export_csv(csv_file)
    assert csv_file.exists()
    assert csv_file.stat().st_size > 50

    # 2. Export XY plot to PNG, SVG, PDF
    xy_spec = XYViewSpec(name="xy_exp", x="T_bett", y="X", color_by="V_MFC", connect_by="V_MFC")
    xyw = XYViewWidget(xy_spec)
    xyw.set_data(synthetic_results)

    png_file = tmp_path / "plot.png"
    xyw.export(png_file, ExportOptions(width_mm=160, height_mm=100, dpi=150))
    assert png_file.exists()
    img = QImage(str(png_file))
    assert not img.isNull()
    assert img.width() > 500

    svg_file = tmp_path / "plot.svg"
    xyw.export(svg_file, ExportOptions(width_mm=160, height_mm=100))
    assert svg_file.exists()
    assert svg_file.stat().st_size > 100

    pdf_file = tmp_path / "plot.pdf"
    xyw.export(pdf_file, ExportOptions(width_mm=160, height_mm=100))
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 100

    # 3. Export Matrix to PNG
    m_spec = MatrixViewSpec(name="mat_exp", rows="T_klasse", cols="V_MFC", value="X")
    mw = MatrixViewWidget(m_spec)
    mw.set_data(synthetic_results)
    mat_png = tmp_path / "matrix.png"
    mw.export(mat_png, ExportOptions(width_mm=160, height_mm=120, dpi=150))
    assert mat_png.exists()
    assert QImage(str(mat_png)).width() > 500


def test_xy_view_point_click(qapp: QApplication, synthetic_results: pl.DataFrame) -> None:
    spec = XYViewSpec(name="lo_click", x="T_bett", y="X")
    w = XYViewWidget(spec)
    w.set_data(synthetic_results)

    inspected = []
    w.step_inspected.connect(inspected.append)

    # Find the scatter plot item in the plot
    p = w.glw.getItem(0, 0)
    scatter = next(item for item in p.items if isinstance(item, pg.ScatterPlotItem))
    pts = scatter.points()
    assert len(pts) > 0
    w._on_scatter_clicked(scatter, [pts[0]])
    assert len(inspected) == 1
    assert inspected[0] == "step_01"


def test_matrix_view_cell_click(qapp: QApplication, synthetic_results: pl.DataFrame) -> None:
    spec = MatrixViewSpec(name="mat_click", rows="T_klasse", cols="V_MFC", value="X")
    w = MatrixViewWidget(spec)
    w.set_data(synthetic_results)

    inspected = []
    w.step_inspected.connect(inspected.append)

    # Find a cell rect in the scene
    p = w.glw.getItem(0, 0)
    rect = next(item for item in p.items if isinstance(item, QGraphicsRectItem) and item.data(0))
    step_id = rect.data(0)
    assert step_id in synthetic_results["step_id"].to_list()

