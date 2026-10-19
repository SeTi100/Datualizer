"""Results views window: renders Table, XY and Matrix views with pyqtgraph (Etappe 2b, Schritt 8, PR 7).

Displays campaign evaluation results as configured in `Evaluation.views` (or default table view).
Supports:
1. `table`: Columns, filter expression, sort.
2. `xy`: Scatter & line plots, error bars (u_x, u_y), color_by (ordinal blue ramp or categorical palette),
   marker_by, filled/open points, connect_by lines, facet_by (small multiples with shared axes),
   ref_lines, secondary top x-axis (e.g. °C to 1000/T), side legend, tooltip & click-to-inspect.
3. `matrix`: Heatmap with rows, cols, value, status condition (grey wash + '*' if false),
   cell text (value + subtext), sequential blue color scale with colorbar, luminance-based text color.
4. Export: PNG, SVG, PDF (mm, pt, dpi) via existing plot export, plus CSV for tables.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.calc.expr import FUNCTIONS, compile_expr
from datualizer_core.campaign.campaign import CampaignSession
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
    table_env,
)
from datualizer_gui.components.export_dialog import ExportDialog
from datualizer_gui.components.plot_export import (
    ExportFormat,
    ExportOptions,
    export_graphics_layout,
)
from datualizer_gui.components.readout import ReadoutLabel

# Visual conventions & palettes (from docs/PLAN_AUSWERTUNG_KAMPAGNE.md & katalyse_plots.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WASH = "#f0efec"

# Ordinal ramp (light blue to dark navy)
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
RAMP_SUBSET = {
    1: [2],
    2: [1, 3],
    3: [1, 2, 4],
    4: [1, 2, 3, 4],
    5: [0, 1, 2, 3, 4],
}

# Sequential colormap for matrix (13 shades)
SEQ = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

# Categorical palette
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# pyqtgraph marker symbols
MARKERS = ["o", "s", "t", "d", "t1", "+", "x"]


def ink_on_color(hex_color: str) -> str:
    """Return 'white' or '#0b0b0b' based on background luminance (sRGB WCAG)."""
    c = hex_color.lstrip("#")
    if len(c) == 6:
        r = int(c[0:2], 16) / 255.0
        g = int(c[2:4], 16) / 255.0
        b = int(c[4:6], 16) / 255.0
        r_lin = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
        g_lin = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
        b_lin = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
        lum = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
        return "white" if lum < 0.30 else INK
    return INK


def interpolate_color(palette: Sequence[str], frac: float) -> str:
    """Interpolate a color hex from a palette list given frac in [0, 1]."""
    frac = max(0.0, min(1.0, float(frac)))
    if len(palette) == 1:
        return palette[0]
    idx_float = frac * (len(palette) - 1)
    i = int(idx_float)
    if i >= len(palette) - 1:
        return palette[-1]
    t = idx_float - i
    c1, c2 = palette[i].lstrip("#"), palette[i + 1].lstrip("#")
    r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
    r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
    r = round(r1 + t * (r2 - r1))
    g = round(g1 + t * (g2 - g1))
    b = round(b1 + t * (b2 - b1))
    return f"#{r:02x}{g:02x}{b:02x}"


def _fmt(val: Any) -> str:
    if val is None:
        return "–"
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return "NaN"
        if np.isinf(val):
            return "inf" if val > 0 else "-inf"
        if abs(val) >= 1e4 or (0 < abs(val) < 1e-3):
            return f"{val:.3e}"
        if abs(val - round(val)) < 1e-9:
            return f"{round(val)}"
        return f"{val:.3g}"
    return str(val)


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem with proper numeric sorting."""

    def __init__(self, text: str, sort_key: Any = None) -> None:
        super().__init__(text)
        self.sort_key = sort_key if sort_key is not None else text

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, NumericTableWidgetItem):
            k1, k2 = self.sort_key, other.sort_key
            try:
                if k1 is None:
                    return True
                if k2 is None:
                    return False
                return float(k1) < float(k2)
            except (ValueError, TypeError):
                return str(k1) < str(k2)
        return super().__lt__(other)


class TransformedAxisItem(pg.AxisItem):
    """Secondary axis item transforming data coordinates via compile_expr."""

    def __init__(
        self,
        orientation: str = "top",
        forward_expr: str | None = None,
        inverse_expr: str | None = None,
        label: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(orientation=orientation, **kwargs)
        self.forward_fn = None
        self.inverse_fn = None
        if forward_expr:
            try:
                f_expr = compile_expr(forward_expr, names=["x"])
                self.forward_fn = lambda x: float(f_expr.evaluate({"x": float(x)}).value)
            except Exception:
                pass
        if inverse_expr:
            try:
                inv_expr = compile_expr(inverse_expr, names=["x"])
                self.inverse_fn = lambda x: float(inv_expr.evaluate({"x": float(x)}).value)
            except Exception:
                pass
        if label:
            self.setLabel(label)

    def tickStrings(self, values: Sequence[float], scale: float, spacing: float) -> list[str]:
        if self.forward_fn is None:
            return super().tickStrings(values, scale, spacing)
        strings = []
        for v in values:
            try:
                trans = self.forward_fn(v)
                if np.isfinite(trans):
                    strings.append(f"{trans:.0f}")
                else:
                    strings.append("")
            except Exception:
                strings.append("")
        return strings


# ------------------------------------------------------------------ Table View Widget
class TableViewWidget(QWidget):
    """Render a TableViewSpec into a QTableWidget."""

    step_inspected = Signal(str)

    def __init__(self, spec: TableViewSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._df: pl.DataFrame | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header_layout = QHBoxLayout()
        if spec.title:
            title_lbl = QLabel(f"<b>{spec.title}</b>")
            if spec.subtitle:
                title_lbl.setToolTip(spec.subtitle)
            header_layout.addWidget(title_lbl)
        header_layout.addStretch()

        self.btn_export_csv = QPushButton("Export CSV...")
        self.btn_export_csv.clicked.connect(self.export_csv)
        header_layout.addWidget(self.btn_export_csv)
        layout.addLayout(header_layout)

        self.table = QTableWidget(0, 0)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionsMovable(True)
        self.table.itemDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table, stretch=1)

        self.status = ReadoutLabel("No data.")
        layout.addWidget(self.status)

    def set_data(self, df: pl.DataFrame) -> None:
        sub = filter_table(df, self.spec.filter)
        if self.spec.sort_by:
            valid_sort = [c for c in self.spec.sort_by if c in sub.columns]
            if valid_sort:
                sub = sub.sort(valid_sort, descending=self.spec.descending)

        cols = self.spec.columns if self.spec.columns else sub.columns
        cols = [c for c in cols if c in sub.columns]
        self._df = sub

        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(sub))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)

        id_col = next((c for c in ("step_id", "interval_id", "schritt", "id") if c in sub.columns), None)

        for r_idx in range(len(sub)):
            for c_idx, col in enumerate(cols):
                raw = sub[col][r_idx]
                item = NumericTableWidgetItem(_fmt(raw), raw)
                if id_col is not None:
                    item.setData(Qt.ItemDataRole.UserRole, str(sub[id_col][r_idx]))
                self.table.setItem(r_idx, c_idx, item)

        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.status.setText(f"{len(sub)} rows, {len(cols)} columns.")

    def _on_row_double_clicked(self, item: QTableWidgetItem) -> None:
        step_id = item.data(Qt.ItemDataRole.UserRole)
        if step_id:
            self.step_inspected.emit(str(step_id))

    def export_csv(self, path: str | Path | None = None) -> Path | None:
        if self._df is None or len(self._df) == 0:
            QMessageBox.information(self, "Export CSV", "Table is empty.")
            return None
        if not path:
            chosen, _ = QFileDialog.getSaveFileName(
                self, f"Export CSV ({self.spec.name})", f"{self.spec.name}.csv", "CSV Files (*.csv);;All Files (*)"
            )
            if not chosen:
                return None
            path = chosen
        p = Path(path)
        self._df.write_csv(p, separator=";")
        return p


# ------------------------------------------------------------------ XY View Widget
class XYViewWidget(QWidget):
    """Render an XYViewSpec into a pyqtgraph GraphicsLayoutWidget."""

    step_inspected = Signal(str)

    def __init__(self, spec: XYViewSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._df: pl.DataFrame | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header_layout = QHBoxLayout()
        title_text = spec.title or spec.name
        lbl = QLabel(f"<b>{title_text}</b>" + (f" – <small>{spec.subtitle}</small>" if spec.subtitle else ""))
        header_layout.addWidget(lbl)
        header_layout.addStretch()

        self.btn_export = QPushButton("Export Plot...")
        self.btn_export.clicked.connect(self._on_export_clicked)
        header_layout.addWidget(self.btn_export)
        layout.addLayout(header_layout)

        main_split = QHBoxLayout()
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(SURFACE)
        self.glw.scene().setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        main_split.addWidget(self.glw, stretch=4)

        # Side legend widget
        self.legend_panel = QWidget()
        self.legend_layout = QVBoxLayout(self.legend_panel)
        self.legend_layout.setContentsMargins(8, 8, 8, 8)
        self.legend_layout.addStretch()
        self.legend_panel.setMaximumWidth(250)
        self.legend_panel.setStyleSheet(f"background: {SURFACE}; color: {INK2}; font-size: 9pt;")
        main_split.addWidget(self.legend_panel, stretch=1)

        layout.addLayout(main_split, stretch=1)
        self.status = ReadoutLabel("")
        layout.addWidget(self.status)

    def set_data(self, df: pl.DataFrame) -> None:
        self._df = df
        sub = filter_table(df, self.spec.filter)
        self.glw.clear()
        self._clear_legend()

        if len(sub) == 0:
            self.status.setText("No data matching filter.")
            return

        x_raw = eval_table_expr(sub, self.spec.effective_x_expr)
        y_raw = eval_table_expr(sub, self.spec.effective_y_expr)
        if x_raw is None or y_raw is None:
            self.status.setText("Missing x or y data.")
            return

        x = np.asarray(x_raw, dtype=float)
        y = np.asarray(y_raw, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        sub_valid = sub.filter(valid)
        x, y = x[valid], y[valid]
        n_pts = len(x)
        if n_pts == 0:
            self.status.setText("No finite (x, y) coordinates.")
            return

        u_x = None
        if self.spec.u_x:
            ux_raw = eval_table_expr(sub_valid, self.spec.u_x)
            if ux_raw is not None:
                u_x = np.asarray(ux_raw, dtype=float)

        u_y = None
        if self.spec.u_y:
            uy_raw = eval_table_expr(sub_valid, self.spec.u_y)
            if uy_raw is not None:
                u_y = np.asarray(uy_raw, dtype=float)

        filled_mask = None
        if self.spec.filled:
            f_raw = eval_table_expr(sub_valid, self.spec.filled)
            if f_raw is not None:
                filled_mask = np.asarray(f_raw, dtype=bool)

        id_col = next((c for c in ("step_id", "interval_id", "schritt", "id") if c in sub_valid.columns), None)
        step_ids = sub_valid[id_col].to_list() if id_col else [f"step_{i}" for i in range(n_pts)]

        # Color mapping
        color_vals = None
        color_map: dict[Any, str] = {}
        if self.spec.color_by and self.spec.color_by in sub_valid.columns:
            color_vals = sub_valid[self.spec.color_by].to_list()
            distinct_colors = sorted(list(set(color_vals)), key=lambda v: (v is None, v))
            is_numeric = all(isinstance(v, (int, float, np.number)) for v in distinct_colors if v is not None)
            if (self.spec.color_mode == "ordinal" or (self.spec.color_mode == "auto" and is_numeric)) and len(
                distinct_colors
            ) <= 5:
                indices = RAMP_SUBSET.get(len(distinct_colors), list(range(len(distinct_colors))))
                for idx, v in enumerate(distinct_colors):
                    color_map[v] = RAMP[indices[idx % len(indices)]]
            elif self.spec.color_mode == "ordinal" or (self.spec.color_mode == "auto" and is_numeric):
                for idx, v in enumerate(distinct_colors):
                    frac = idx / max(1, len(distinct_colors) - 1)
                    color_map[v] = interpolate_color(RAMP, frac)
            else:
                for idx, v in enumerate(distinct_colors):
                    color_map[v] = CAT[idx % len(CAT)]

        # Marker mapping
        marker_vals = None
        marker_map: dict[Any, str] = {}
        if self.spec.marker_by and self.spec.marker_by in sub_valid.columns:
            marker_vals = sub_valid[self.spec.marker_by].to_list()
            distinct_markers = sorted(list(set(marker_vals)), key=lambda v: (v is None, v))
            for idx, v in enumerate(distinct_markers):
                marker_map[v] = MARKERS[idx % len(MARKERS)]

        # Faceting
        facets: list[tuple[Any, np.ndarray]] = []
        if self.spec.facet_by and self.spec.facet_by in sub_valid.columns:
            facet_vals = sub_valid[self.spec.facet_by].to_list()
            for f_val in sorted(list(set(facet_vals)), key=lambda v: (v is None, v)):
                mask = np.array([v == f_val for v in facet_vals], dtype=bool)
                facets.append((f_val, mask))
        else:
            facets.append((None, np.ones(n_pts, dtype=bool)))

        n_facets = len(facets)
        ncols = min(self.spec.facet_cols, n_facets)
        first_plot: pg.PlotItem | None = None

        for k, (f_name, f_mask) in enumerate(facets):
            row = k // ncols
            col = k % ncols

            # Top secondary axis if configured
            axis_items = {}
            if self.spec.secondary_x is not None:
                sec = self.spec.secondary_x
                axis_items["top"] = TransformedAxisItem(
                    orientation="top",
                    forward_expr=sec.forward_expr,
                    inverse_expr=sec.inverse_expr,
                    label=sec.label,
                )

            p = self.glw.addPlot(row=row, col=col, axisItems=axis_items)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("bottom").setPen(AXIS)
            p.getAxis("left").setPen(AXIS)
            p.getAxis("bottom").setTextPen(INK2)
            p.getAxis("left").setTextPen(INK2)

            if self.spec.secondary_x is not None:
                p.showAxis("top")
                p.getAxis("top").setPen(AXIS)
                p.getAxis("top").setTextPen(INK2)

            if f_name is not None:
                title_f = f"≈ {f_name:.0f} °C" if isinstance(f_name, (int, float)) else str(f_name)
                p.setTitle(title_f, color=INK)
            elif self.spec.title:
                p.setTitle(self.spec.title, color=INK)

            if first_plot is not None:
                p.setXLink(first_plot)
                p.setYLink(first_plot)
            else:
                first_plot = p

            if self.spec.x_label:
                p.setLabel("bottom", self.spec.x_label)
            if self.spec.y_label and col == 0:
                p.setLabel("left", self.spec.y_label)

            # Sub-arrays for this facet
            fx, fy = x[f_mask], y[f_mask]
            f_ids = [step_ids[i] for i, m in enumerate(f_mask) if m]
            f_filled = filled_mask[f_mask] if filled_mask is not None else np.ones(len(fx), dtype=bool)
            f_cvals = [color_vals[i] for i, m in enumerate(f_mask) if m] if color_vals is not None else None
            f_mvals = [marker_vals[i] for i, m in enumerate(f_mask) if m] if marker_vals is not None else None
            f_ux = u_x[f_mask] if u_x is not None else None
            f_uy = u_y[f_mask] if u_y is not None else None

            # Reference lines
            for r in self.spec.ref_lines:
                angle = 0 if r.axis == "y" else 90
                line = pg.InfiniteLine(
                    pos=r.value,
                    angle=angle,
                    pen=pg.mkPen(r.color, width=r.width),
                    label=r.label,
                    labelOpts={"position": 0.05, "color": MUTED},
                )
                p.addItem(line)

            # Connect by lines
            if self.spec.connect_by and self.spec.connect_by in sub_valid.columns:
                f_conn_vals = [sub_valid[self.spec.connect_by][i] for i, m in enumerate(f_mask) if m]
                for c_group in sorted(list(set(f_conn_vals)), key=lambda v: (v is None, v)):
                    c_mask = np.array([v == c_group for v in f_conn_vals], dtype=bool)
                    cx, cy = fx[c_mask], fy[c_mask]
                    if len(cx) > 1:
                        order = np.argsort(cx)
                        line_c = color_map.get(c_group, CAT[0]) if color_map else CAT[0]
                        curve = pg.PlotCurveItem(
                            cx[order], cy[order], pen=pg.mkPen(line_c, width=1.6)
                        )
                        p.addItem(curve)

            # Error bars
            if f_ux is not None or f_uy is not None:
                err = pg.ErrorBarItem(
                    x=fx,
                    y=fy,
                    top=f_uy if f_uy is not None else np.zeros_like(fy),
                    bottom=f_uy if f_uy is not None else np.zeros_like(fy),
                    left=f_ux if f_ux is not None else np.zeros_like(fx),
                    right=f_ux if f_ux is not None else np.zeros_like(fx),
                    beam=0.0,
                    pen=pg.mkPen(INK2, width=0.9, alpha=160),
                )
                p.addItem(err)

            # Scatter points
            spots = []
            for i in range(len(fx)):
                pt_color = color_map[f_cvals[i]] if f_cvals is not None else CAT[0]
                pt_symbol = marker_map[f_mvals[i]] if f_mvals is not None else "o"
                is_filled = bool(f_filled[i])

                if is_filled:
                    brush = pg.mkBrush(pt_color)
                    pen = pg.mkPen(SURFACE, width=1.2)
                else:
                    brush = pg.mkBrush(SURFACE)
                    pen = pg.mkPen(pt_color, width=1.5)

                spots.append(
                    {
                        "pos": (fx[i], fy[i]),
                        "size": 8,
                        "symbol": pt_symbol,
                        "brush": brush,
                        "pen": pen,
                        "data": f_ids[i],
                    }
                )

            scatter = pg.ScatterPlotItem(
                spots=spots,
                hoverable=True,
                tip=lambda x, y, data: f"Step: {data}\nx: {x:.4g}\ny: {y:.4g}",
            )
            scatter.sigClicked.connect(self._on_scatter_clicked)
            p.addItem(scatter)

            if self.spec.x_lim:
                p.setXRange(self.spec.x_lim[0], self.spec.x_lim[1], padding=0)
            if self.spec.y_lim:
                p.setYRange(self.spec.y_lim[0], self.spec.y_lim[1], padding=0)

        # Build Side Legend
        self._build_side_legend(color_map, marker_map, filled_mask is not None)
        self.status.setText(f"{n_pts} points rendered across {n_facets} facet(s).")

    def _on_scatter_clicked(self, item: Any, points: Sequence[Any]) -> None:
        if points:
            step_id = points[0].data()
            if step_id:
                self.step_inspected.emit(str(step_id))

    def _clear_legend(self) -> None:
        while self.legend_layout.count() > 1:
            item = self.legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _build_side_legend(
        self, color_map: dict[Any, str], marker_map: dict[Any, str], has_filled: bool
    ) -> None:
        self._clear_legend()

        if color_map:
            lbl_col = QLabel(f"<b>{self.spec.color_by or 'Color'}</b>")
            self.legend_layout.insertWidget(self.legend_layout.count() - 1, lbl_col)
            for val, c in color_map.items():
                row = QHBoxLayout()
                swatch = QLabel("■")
                swatch.setStyleSheet(f"color: {c}; font-size: 14pt;")
                row.addWidget(swatch)
                row.addWidget(QLabel(f"{val:g}" if isinstance(val, (int, float)) else str(val)))
                row.addStretch()
                self.legend_layout.insertLayout(self.legend_layout.count() - 1, row)

        if marker_map:
            lbl_m = QLabel(f"<b>{self.spec.marker_by or 'Marker'}</b>")
            self.legend_layout.insertWidget(self.legend_layout.count() - 1, lbl_m)
            for val, m in marker_map.items():
                row = QHBoxLayout()
                m_lbl = QLabel(f"[{m}]")
                m_lbl.setStyleSheet(f"color: {INK2}; font-weight: bold;")
                row.addWidget(m_lbl)
                row.addWidget(QLabel(str(val)))
                row.addStretch()
                self.legend_layout.insertLayout(self.legend_layout.count() - 1, row)

        if has_filled:
            lbl_s = QLabel(f"<b>Status</b>")
            self.legend_layout.insertWidget(self.legend_layout.count() - 1, lbl_s)
            r1 = QHBoxLayout()
            r1.addWidget(QLabel("● filled:"))
            r1.addWidget(QLabel("True"))
            r1.addStretch()
            self.legend_layout.insertLayout(self.legend_layout.count() - 1, r1)

            r2 = QHBoxLayout()
            r2.addWidget(QLabel("○ open:"))
            r2.addWidget(QLabel("False"))
            r2.addStretch()
            self.legend_layout.insertLayout(self.legend_layout.count() - 1, r2)

    def _on_export_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export Plot ({self.spec.name})", f"{self.spec.name}.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if path:
            try:
                export_graphics_layout(self.glw, path, ExportOptions(width_mm=170, dpi=300))
                QMessageBox.information(self, "Export Plot", f"Saved to {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", f"Cannot export plot:\n{exc}")

    def export(self, path: str | Path, options: ExportOptions = ExportOptions()) -> Path:
        return export_graphics_layout(self.glw, path, options)


# ------------------------------------------------------------------ Matrix View Widget
class MatrixViewWidget(QWidget):
    """Render a MatrixViewSpec into a pyqtgraph GraphicsLayoutWidget."""

    step_inspected = Signal(str)

    def __init__(self, spec: MatrixViewSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._df: pl.DataFrame | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header_layout = QHBoxLayout()
        title_text = spec.title or spec.name
        lbl = QLabel(f"<b>{title_text}</b>" + (f" – <small>{spec.subtitle}</small>" if spec.subtitle else ""))
        header_layout.addWidget(lbl)
        header_layout.addStretch()

        self.btn_export = QPushButton("Export Matrix...")
        self.btn_export.clicked.connect(self._on_export_clicked)
        header_layout.addWidget(self.btn_export)
        layout.addLayout(header_layout)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(SURFACE)
        self.glw.scene().setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.glw.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        layout.addWidget(self.glw, stretch=1)

        self.footer_lbl = QLabel(spec.footer or "* nur nicht stationäre Stufen: Wert am Stufenende (Orientierung)")
        self.footer_lbl.setStyleSheet(f"color: {MUTED}; font-size: 8pt;")
        layout.addWidget(self.footer_lbl)

    def _on_mouse_clicked(self, event: Any) -> None:
        pos = event.scenePos()
        for item in self.glw.scene().items(pos):
            step_id = item.data(0)
            if step_id:
                self.step_inspected.emit(str(step_id))
                break

    def set_data(self, df: pl.DataFrame) -> None:
        self._df = df
        sub = filter_table(df, self.spec.filter)
        self.glw.clear()

        if len(sub) == 0:
            return

        # Check required columns
        if self.spec.rows not in sub.columns or self.spec.cols not in sub.columns:
            return

        row_vals = sub[self.spec.rows].to_list()
        col_vals = sub[self.spec.cols].to_list()
        val_data = eval_table_expr(sub, self.spec.value)
        if val_data is None:
            return

        stat_mask = None
        if self.spec.status:
            st_raw = eval_table_expr(sub, self.spec.status)
            if st_raw is not None:
                stat_mask = np.asarray(st_raw, dtype=bool)

        id_col = next((c for c in ("step_id", "interval_id", "schritt", "id") if c in sub.columns), None)
        step_ids = sub[id_col].to_list() if id_col else [f"step_{i}" for i in range(len(sub))]

        # Unique sorted rows (descending e.g. for temperature) and cols (ascending e.g. for MFC)
        rows_unique = sorted(list(set(row_vals)), reverse=True, key=lambda v: (v is None, v))
        cols_unique = sorted(list(set(col_vals)), key=lambda v: (v is None, v))

        n_rows = len(rows_unique)
        n_cols = len(cols_unique)

        p = self.glw.addPlot(row=0, col=0)
        p.getViewBox().setMouseEnabled(x=False, y=False)
        p.showGrid(x=False, y=False)
        p.getAxis("bottom").setPen(AXIS)
        p.getAxis("left").setPen(AXIS)
        p.getAxis("bottom").setTextPen(INK2)
        p.getAxis("left").setTextPen(INK2)

        # X-Ticks
        x_ticks = [(j, f"{v:g}" if isinstance(v, (int, float)) else str(v)) for j, v in enumerate(cols_unique)]
        p.getAxis("bottom").setTicks([x_ticks])
        p.setLabel("bottom", self.spec.x_label or self.spec.cols)

        # Y-Ticks
        y_ticks = []
        for i, r_val in enumerate(rows_unique):
            # Check if there is T_bett in sub to show range (min–max)
            t_col = next((c for c in ("T_bett", "t_bett") if c in sub.columns), None)
            if t_col and isinstance(r_val, (int, float)):
                pts_r = sub.filter(pl.col(self.spec.rows) == r_val)
                if len(pts_r) > 0 and pts_r[t_col].is_not_null().any():
                    t_min = pts_r[t_col].min()
                    t_max = pts_r[t_col].max()
                    y_ticks.append((i, f"{r_val:.0f} °C\n({t_min:.0f}–{t_max:.0f})"))
                else:
                    y_ticks.append((i, f"{r_val:.0f} °C"))
            else:
                y_ticks.append((i, f"{r_val}"))
        p.getAxis("left").setTicks([y_ticks])
        p.setLabel("left", self.spec.y_label or self.spec.rows)

        p.setXRange(-0.5, n_cols - 0.5, padding=0)
        p.setYRange(-0.5, n_rows - 0.5, padding=0)

        vmin = self.spec.vmin if self.spec.vmin is not None else 0.0
        vmax = self.spec.vmax if self.spec.vmax is not None else 1.0

        for i, r_val in enumerate(rows_unique):
            for j, c_val in enumerate(cols_unique):
                cell_mask = np.array([r == r_val and c == c_val for r, c in zip(row_vals, col_vals)], dtype=bool)
                if not np.any(cell_mask):
                    continue

                cell_vals = np.asarray(val_data)[cell_mask]
                cell_ok = stat_mask[cell_mask] if stat_mask is not None else np.ones(len(cell_vals), dtype=bool)
                cell_ids = [step_ids[idx] for idx, m in enumerate(cell_mask) if m]

                stat_pts = cell_vals[cell_ok]
                if len(stat_pts) > 0:
                    mean_val = float(np.nanmean(stat_pts))
                    frac = np.clip((mean_val - vmin) / max(1e-9, vmax - vmin), 0.0, 1.0)
                    face = interpolate_color(SEQ, frac)
                    ink = ink_on_color(face)
                    val_str = f"{mean_val:.2f}"
                    sub_str = f"{len(stat_pts)} Stufe(n)"
                    if self.spec.subtext:
                        # evaluate subtext template or expression
                        sub_str = self.spec.subtext
                else:
                    face = WASH
                    ink = INK2
                    lo = float(np.nanmin(cell_vals)) if len(cell_vals) else 0.0
                    hi = float(np.nanmax(cell_vals)) if len(cell_vals) else 0.0
                    val_str = f"{lo:.2f}*" if abs(hi - lo) < 0.005 else f"{lo:.2f}–{hi:.2f}*"
                    sub_str = "nicht stationär"

                rect = QGraphicsRectItem(j - 0.47, i - 0.45, 0.94, 0.90)
                rect.setBrush(pg.mkBrush(face))
                rect.setPen(pg.mkPen(None))
                rect.setData(0, cell_ids[0] if cell_ids else "")
                p.addItem(rect)

                html_text = (
                    f"<div style='text-align:center;'>"
                    f"<span style='font-size:13pt;font-weight:bold;color:{ink};'>{val_str}</span><br>"
                    f"<span style='font-size:7.5pt;color:{ink};'>{sub_str}</span>"
                    f"</div>"
                )
                txt = pg.TextItem(html=html_text, anchor=(0.5, 0.5))
                txt.setPos(j, i)
                p.addItem(txt)

        # Colorbar
        cb_plot = self.glw.addPlot(row=0, col=1)
        cb_plot.setMaximumWidth(60)
        cb_plot.getViewBox().setMouseEnabled(x=False, y=False)
        cb_plot.showGrid(x=False, y=False)
        cb_plot.getAxis("bottom").setStyle(showValues=False)
        cb_plot.getAxis("bottom").setPen(None)
        cb_plot.getAxis("left").setPen(None)
        cb_plot.getAxis("left").setStyle(showValues=False)
        cb_plot.getAxis("right").setPen(AXIS)
        cb_plot.getAxis("right").setTextPen(INK2)
        cb_plot.showAxis("right")

        # Draw vertical gradient rectangles
        n_grad = len(SEQ)
        for g_idx, c_hex in enumerate(SEQ):
            y_lo = vmin + (vmax - vmin) * (g_idx / n_grad)
            y_hi = vmin + (vmax - vmin) * ((g_idx + 1) / n_grad)
            g_rect = QGraphicsRectItem(0, y_lo, 1, y_hi - y_lo)
            g_rect.setBrush(pg.mkBrush(c_hex))
            g_rect.setPen(pg.mkPen(None))
            cb_plot.addItem(g_rect)

        cb_plot.setXRange(0, 1, padding=0)
        cb_plot.setYRange(vmin, vmax, padding=0)
        cb_ticks = [(vmin, f"{vmin:.1f}"), (0.5 * (vmin + vmax), f"{0.5*(vmin+vmax):.1f}"), (vmax, f"{vmax:.1f}")]
        cb_plot.getAxis("right").setTicks([cb_ticks])
        cb_plot.setLabel("right", f"{self.spec.value} (stationär)")

    def _on_export_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export Matrix ({self.spec.name})", f"{self.spec.name}.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if path:
            try:
                export_graphics_layout(self.glw, path, ExportOptions(width_mm=170, dpi=300))
                QMessageBox.information(self, "Export Matrix", f"Saved to {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", f"Cannot export matrix:\n{exc}")

    def export(self, path: str | Path, options: ExportOptions = ExportOptions()) -> Path:
        return export_graphics_layout(self.glw, path, options)


# ------------------------------------------------------------------ Results Window
class ResultsWindow(QMainWindow):
    """Main results views window with tabs for Table, XY and Matrix views."""

    step_inspected = Signal(str)

    def __init__(
        self,
        session: CampaignSession | None = None,
        results_df: pl.DataFrame | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._direct_df = results_df
        self._updating = False

        self.setWindowTitle("Results Views – Datualizer")
        self.resize(1200, 800)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Scenario:"))
        self.combo_scenario = QComboBox()
        self.combo_scenario.currentIndexChanged.connect(self._on_scenario_changed)
        toolbar.addWidget(self.combo_scenario)

        toolbar.addSpacing(16)
        self.btn_refresh = QPushButton("Recompute")
        self.btn_refresh.setToolTip("Re-evaluate results and refresh all views")
        self.btn_refresh.clicked.connect(self.refresh)
        toolbar.addWidget(self.btn_refresh)

        toolbar.addSpacing(8)
        self.btn_export = QPushButton("Export Current View...")
        self.btn_export.setToolTip("Export the current view to PNG, SVG, PDF or CSV")
        self.btn_export.clicked.connect(self.export_current_view)
        toolbar.addWidget(self.btn_export)

        toolbar.addStretch()
        self.status_label = ReadoutLabel("Initializing...")
        toolbar.addWidget(self.status_label)
        layout.addLayout(toolbar)

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        self._view_widgets: dict[str, QWidget] = {}
        self.refresh()

    def set_session(self, session: CampaignSession | None) -> None:
        self._session = session
        self._populate_scenarios()
        self.refresh()

    def set_results(self, df: pl.DataFrame | None) -> None:
        self._direct_df = df
        self.refresh()

    def current_scenario_name(self) -> str | None:
        txt = self.combo_scenario.currentText()
        if not txt or txt == "(Base)":
            return None
        return txt

    def _populate_scenarios(self) -> None:
        self.combo_scenario.blockSignals(True)
        self.combo_scenario.clear()
        self.combo_scenario.addItem("(Base)")
        if self._session is not None and self._session.campaign.evaluation:
            for s_name in self._session.campaign.evaluation.scenario_names():
                self.combo_scenario.addItem(s_name)
        self.combo_scenario.blockSignals(False)

    def _on_scenario_changed(self) -> None:
        if not self._updating:
            self.refresh()

    def refresh(self) -> None:
        self._updating = True
        try:
            if self._session is not None:
                ev = self._session.campaign.evaluation
                scenario = self.current_scenario_name()
                try:
                    df = self._session.results(scenario=scenario)
                except Exception as exc:
                    self.status_label.setText(f"Error computing results: {exc}")
                    return
                views = ev.views if ev and ev.views else []
            elif self._direct_df is not None:
                df = self._direct_df
                views = []
            else:
                self.status_label.setText("No campaign session or data.")
                return

            self._build_views(views, df)
            self.status_label.setText(f"Showing {len(df)} steps across {self.tabs.count()} view(s).")
        finally:
            self._updating = False

    def _build_views(self, views: list[ViewSpec], df: pl.DataFrame) -> None:
        curr_idx = self.tabs.currentIndex()
        curr_name = self.tabs.tabText(curr_idx) if curr_idx >= 0 else None

        self.tabs.clear()
        self._view_widgets.clear()

        # If no custom views configured, provide a default Table view
        if not views:
            default_spec = TableViewSpec(name="results", title="Results Table")
            views = [default_spec]

        for spec in views:
            w: QWidget
            if spec.kind == "table":
                tw = TableViewWidget(spec)
                tw.step_inspected.connect(self.step_inspected)
                tw.set_data(df)
                w = tw
            elif spec.kind == "xy":
                xyw = XYViewWidget(spec)
                xyw.step_inspected.connect(self.step_inspected)
                xyw.set_data(df)
                w = xyw
            elif spec.kind == "matrix":
                mw = MatrixViewWidget(spec)
                mw.step_inspected.connect(self.step_inspected)
                mw.set_data(df)
                w = mw
            else:
                continue

            tab_title = spec.title or spec.name
            self.tabs.addTab(w, tab_title)
            self._view_widgets[spec.name] = w

        # Restore previous tab if present
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == curr_name:
                self.tabs.setCurrentIndex(i)
                break

    def export_current_view(self, path: str | Path | None = None) -> Path | None:
        idx = self.tabs.currentIndex()
        if idx < 0:
            QMessageBox.information(self, "Export View", "No view to export.")
            return None

        w = self.tabs.currentWidget()
        if w is None:
            return None

        view_name = self.tabs.tabText(idx).replace(" ", "_")

        if isinstance(w, TableViewWidget):
            return w.export_csv(path)

        if not path:
            chosen, _ = QFileDialog.getSaveFileName(
                self, f"Export View ({view_name})", f"{view_name}.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
            )
            if not chosen:
                return None
            path = chosen

        try:
            if hasattr(w, "export"):
                out = w.export(path, ExportOptions(width_mm=170, dpi=300))
                QMessageBox.information(self, "Export View", f"Saved to {out}")
                return out
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Cannot export view:\n{exc}")
            return None
        return None


__all__ = [
    "MatrixViewWidget",
    "ResultsWindow",
    "TableViewWidget",
    "XYViewWidget",
]
