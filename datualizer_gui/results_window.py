"""Results views window: renders Table, XY and Matrix views with pyqtgraph (Etappe 2b, Schritt 8, PR 7).

Displays campaign evaluation results as configured in `Evaluation.views` (or default table view).
Supports:
1. `table`: Columns, filter expression, sort.
2. `xy`: Scatter & line plots, error bars (u_x, u_y), color_by (ordinal blue ramp or categorical palette),
   marker_by, filled/open points, connect_by lines, facet_by (small multiples with shared axes),
   ref_lines, secondary top x-axis (e.g. °C to 1000/T), side legend, tooltip & click-to-inspect.
3. `matrix`: Heatmap with rows, cols, value, status condition (grey wash + '*' if false),
   cell text (value + subtext), sequential blue color scale with colorbar, luminance-based text color.
4. `timeline`: time course of channels over master time with step bands and marks per step,
   one page per value of `page_by` (e.g. one per measuring day).
5. Export: PNG, SVG, PDF (mm, pt, dpi) via existing plot export, plus CSV for tables.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
    QSizePolicy,
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
    AlignedViewSpec,
    BarsViewSpec,
    MatrixViewSpec,
    RefLine,
    ReportViewSpec,
    SecondaryAxis,
    TableViewSpec,
    TimelinePanel,
    TimelineViewSpec,
    ViewSpec,
    XYViewSpec,
    eval_table_expr,
    filter_table,
    render_template,
    table_env,
)
from datualizer_gui.components.campaign_canvas import utc_offset_s
from datualizer_gui.components.curves import add_curve, curve_pen, light_grid
from datualizer_gui.components.interval_bands import ColorBandsItem
from datualizer_gui.components.time_axis import TimeAxisMode, make_time_axis
from datualizer_gui.components.export_dialog import ExportDialog
from datualizer_gui.components.plot_export import (
    ExportFormat,
    ExportOptions,
    export_graphics_layout,
    export_report,
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
LEGEND_W_PX = 175


class _HeadingSample(pg.ItemSample):
    """Empty sample: a legend row that carries only a heading."""

    def __init__(self) -> None:
        super().__init__(pg.ScatterPlotItem())

    def paint(self, p: Any, *args: Any) -> None:
        pass


def _legend_value(v: Any) -> str:
    return f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v)


def swatch(color: str) -> pg.ScatterPlotItem:
    """Legend sample: a filled square."""
    return pg.ScatterPlotItem(symbol="s", size=10, pen=None, brush=pg.mkBrush(color))


def marker_sample(symbol: str, filled: bool = False) -> pg.ScatterPlotItem:
    """Legend sample: a marker in the ink colour, filled or open."""
    return pg.ScatterPlotItem(symbol=symbol, size=8, pen=pg.mkPen(INK2, width=1.2),
                              brush=pg.mkBrush(INK2 if filled else SURFACE))


def add_layout_legend(
    glw: pg.GraphicsLayoutWidget,
    sections: Sequence[tuple[str | None, Sequence[tuple[Any, str]]]],
    row: int = 0,
    col: int = 1,
    rowspan: int = 1,
) -> pg.LegendItem | None:
    """A legend in its own column of the graphics layout (top aligned, fixed width).

    It is part of the scene, so the export and the report page show it (a Qt widget beside the plot
    would be missing there). `sections`: (heading or None, [(sample item, text), ...]); sections
    without entries are left out, without any entry there is no legend.
    """
    if not any(entries for _, entries in sections):
        return None
    legend = pg.LegendItem(labelTextColor=INK2, labelTextSize="8pt", verSpacing=-4)
    legend.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    for heading, entries in sections:
        if not entries:
            continue
        if heading:  # labels are rich text: a condition such as 'x <= 0.1' must not open a tag
            legend.addItem(_HeadingSample(), f"<b>{html.escape(heading)}</b>")
        for sample, text in entries:
            legend.addItem(sample, html.escape(text))
    glw.ci.addItem(legend, row=row, col=col, rowspan=rowspan)
    glw.ci.layout.setAlignment(legend, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    glw.ci.layout.setColumnFixedWidth(col, LEGEND_W_PX)
    return legend


def point_legend_sections(
    color_by: str | None, color_map: Mapping[Any, str], marker_by: str | None, marker_map: Mapping[Any, str],
    filled: str | None,
) -> list[tuple[str | None, list[tuple[Any, str]]]]:
    """Legend of the xy and aligned views: colours, marker symbols, filled vs open (the `filled` condition)."""
    sections: list[tuple[str | None, list[tuple[Any, str]]]] = [
        (color_by, [(swatch(c), _legend_value(v)) for v, c in color_map.items()]),
        (marker_by, [(marker_sample(m), _legend_value(v)) for v, m in marker_map.items()]),
    ]
    if filled:
        sections.append((filled, [(marker_sample("o", True), "erfüllt"), (marker_sample("o"), "nicht erfüllt")]))
    return sections


def facet_title(rows: pl.DataFrame, facet_by: str, value: Any, template: str | None) -> str:
    """Title of a facet: the template filled from its steps, else '<facet_by> = <value>'."""
    if template:
        return str(render_template(rows, template, reduce=True))
    return f"{facet_by} = {_legend_value(value)}"


def shared_range(plot: pg.PlotItem | None, xs: Sequence[np.ndarray], ys: Sequence[np.ndarray],
                 x_lim: tuple[float, float] | None, y_lim: tuple[float, float] | None) -> None:
    """Range of linked facets from the data of all of them (pyqtgraph would fit the first facet only)."""
    if plot is None:
        return
    for lim, arrays, setter in ((x_lim, xs, plot.setXRange), (y_lim, ys, plot.setYRange)):
        if lim is not None:
            setter(lim[0], lim[1], padding=0)
            continue
        vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays]) if arrays else np.array([])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            setter(float(vals.min()), float(vals.max()), padding=0.04)


def step_ids_of(table: pl.DataFrame) -> list[Any]:
    """The `step_id` column of a result table, or row numbers if there is none."""
    return table["step_id"].to_list() if "step_id" in table.columns else [f"step_{i}" for i in range(len(table))]


def fit_tick_text(axis: pg.AxisItem, labels: Sequence[str]) -> None:
    """Room for the given (multi-line) tick labels before the first paint.

    pyqtgraph drops labels that do not fit into the axis and widens it only after a paint, so a
    one-shot render (export, report page) showed none; the size is set from the text up front.
    """
    fm = QFontMetricsF(axis.style["tickFont"] or QApplication.font())
    lines = [line for t in labels for line in str(t).split("\n")]
    if axis.orientation in ("left", "right"):
        text_w = max((fm.horizontalAdvance(line) for line in lines), default=0.0)
        axis.setStyle(autoExpandTextSpace=False, tickTextWidth=int(text_w * 1.15) + 4)
    else:
        n_lines = max((str(t).count("\n") + 1 for t in labels), default=1)
        axis.setStyle(autoExpandTextSpace=False, tickTextHeight=int(fm.lineSpacing() * n_lines * 1.1) + 2)


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

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(SURFACE)
        self.glw.scene().setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        layout.addWidget(self.glw, stretch=1)
        self.status = ReadoutLabel("")
        layout.addWidget(self.status)

    def set_data(self, df: pl.DataFrame) -> None:
        self._df = df
        sub = filter_table(df, self.spec.filter)
        self.glw.clear()

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

        step_ids = step_ids_of(sub_valid)

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
            facets.append((_NO_FACET, np.ones(n_pts, dtype=bool)))
        facet_rows = sub_valid

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
            p.getAxis("left").enableAutoSIPrefix(False)

            if self.spec.secondary_x is not None:
                p.showAxis("top")
                p.getAxis("top").setPen(AXIS)
                p.getAxis("top").setTextPen(INK2)

            if f_name is not _NO_FACET:
                p.setTitle(facet_title(facet_rows.filter(pl.Series(f_mask)), self.spec.facet_by, f_name,
                                       self.spec.facet_label), color=INK)
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


        shared_range(first_plot, [x], [y], self.spec.x_lim, self.spec.y_lim)
        add_layout_legend(
            self.glw,
            point_legend_sections(self.spec.color_by, color_map, self.spec.marker_by, marker_map,
                                  self.spec.filled if filled_mask is not None else None),
            row=0, col=ncols, rowspan=-(-n_facets // ncols),
        )
        self.status.setText(f"{n_pts} points rendered across {n_facets} facet(s).")

    def _on_scatter_clicked(self, item: Any, points: Sequence[Any]) -> None:
        if points:
            step_id = points[0].data()
            if step_id:
                self.step_inspected.emit(str(step_id))

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

        step_ids = step_ids_of(sub)

        # Unique sorted rows (first row at the top; descending e.g. for temperature) and cols (ascending)
        rows_unique = sorted(list(set(row_vals)), reverse=self.spec.rows_descending, key=lambda v: (v is None, v))
        cols_unique = sorted(list(set(col_vals)), key=lambda v: (v is None, v))

        n_rows = len(rows_unique)
        n_cols = len(cols_unique)

        p = self.glw.addPlot(row=0, col=0)
        p.getViewBox().setMouseEnabled(x=False, y=False)
        p.invertY(True)  # first row at the top
        p.showGrid(x=False, y=False)
        # every row and column is labelled: pyqtgraph would drop all labels of a level that fill
        # more than 80 % of the axis (two-line row labels in a 105 mm export)
        for side in ("left", "bottom"):
            p.getAxis(side).setStyle(textFillLimits=[(0, 100.0)], autoExpandTextSpace=True)
        p.getAxis("bottom").setPen(AXIS)
        p.getAxis("left").setPen(AXIS)
        p.getAxis("bottom").setTextPen(INK2)
        p.getAxis("left").setTextPen(INK2)

        # X-Ticks
        x_ticks = [(j, f"{v:g}" if isinstance(v, (int, float)) else str(v)) for j, v in enumerate(cols_unique)]
        p.getAxis("bottom").setTicks([x_ticks])
        p.setLabel("bottom", self.spec.x_label or self.spec.cols)

        # Y-Ticks
        # Y-Ticks: the row value, or the template `row_label` filled from all steps of the row
        y_ticks = []
        for i, r_val in enumerate(rows_unique):
            if self.spec.row_label:
                rows_r = sub.filter(pl.Series([r == r_val for r in row_vals]))
                y_ticks.append((i, str(render_template(rows_r, self.spec.row_label, reduce=True))))
            else:
                y_ticks.append((i, f"{r_val:g}" if isinstance(r_val, (int, float)) else str(r_val)))
        p.getAxis("left").setTicks([y_ticks])
        fit_tick_text(p.getAxis("left"), [t for _, t in y_ticks])
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
                        rows_ok = sub.filter(pl.Series(cell_mask & (stat_mask if stat_mask is not None else True)))
                        sub_str = str(render_template(rows_ok, self.spec.subtext, reduce=True))
                        if len(stat_pts) > 1:
                            sub_str += f" · {len(stat_pts)} Stufen"
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


_NO_FACET = object()  # the one facet of a view without facet_by (None is a valid facet value)


def _to_epoch_array(series: pl.Series) -> np.ndarray:
    if series.dtype == pl.Datetime or series.dtype == pl.Date:
        return (series.cast(pl.Datetime("us", "UTC")).cast(pl.Int64) / 1e6).to_numpy()
    return series.cast(pl.Float64).to_numpy()


# ------------------------------------------------------------------ Aligned View Widget
class AlignedViewWidget(QWidget):
    """Render an AlignedViewSpec (e.g. breakthrough curve X(t) over relative time)."""

    step_inspected = Signal(str)

    def __init__(
        self,
        spec: AlignedViewSpec,
        session: CampaignSession | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self.session = session
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

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(SURFACE)
        self.glw.scene().setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        layout.addWidget(self.glw, stretch=1)
        self.status = ReadoutLabel("")
        layout.addWidget(self.status)

        if spec.footer:
            footer_lbl = QLabel(f"<small style='color: {INK2};'>{spec.footer}</small>")
            footer_lbl.setWordWrap(True)
            layout.addWidget(footer_lbl)

    def set_data(self, df: pl.DataFrame) -> None:
        self._df = df
        sub = filter_table(df, self.spec.filter)
        self.glw.clear()

        if len(sub) == 0:
            self.status.setText("No data matching filter.")
            return

        step_ids = step_ids_of(sub)

        if "step_start" not in sub.columns or "step_end" not in sub.columns:
            self.status.setText("Missing step_start or step_end columns.")
            return

        starts_epoch = np.asarray(_to_epoch_array(sub["step_start"]), dtype=float)
        ends_epoch = np.asarray(_to_epoch_array(sub["step_end"]), dtype=float)

        # Color mapping
        color_vals = None
        color_map: dict[Any, str] = {}
        if self.spec.color_by and self.spec.color_by in sub.columns:
            color_vals = sub[self.spec.color_by].to_list()
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
        if self.spec.marker_by and self.spec.marker_by in sub.columns:
            marker_vals = sub[self.spec.marker_by].to_list()
            distinct_markers = sorted(list(set(marker_vals)), key=lambda v: (v is None, v))
            for idx, v in enumerate(distinct_markers):
                marker_map[v] = MARKERS[idx % len(MARKERS)]

        # Filled condition mask
        filled_mask = None
        if self.spec.filled:
            f_raw = eval_table_expr(sub, self.spec.filled)
            if f_raw is not None:
                filled_mask = np.asarray(f_raw, dtype=bool)

        # Compute t0 for each step
        # t0: the step's start, or the first start of its sequence among `sequence_steps` of the full
        # table (steps of the sequence left out by the view's filter still set its start)
        t0_arr = starts_epoch.copy()
        if self.spec.t0 == "sequence_start":
            seq_col = self.spec.sequence_col
            if seq_col not in df.columns:
                self.status.setText(f"No column {seq_col!r} for the sequences.")
                return
            seq_rows = filter_table(df, self.spec.sequence_steps)
            all_starts = np.asarray(_to_epoch_array(seq_rows["step_start"]), dtype=float)
            first: dict[Any, float] = {}
            for v, t in zip(seq_rows[seq_col].to_list(), all_starts):
                if v is not None and np.isfinite(t):
                    first[v] = min(first.get(v, t), t)
            t0_arr = np.array([first.get(v, np.nan) for v in sub[seq_col].to_list()], dtype=float)

        # Channel lookup from session
        channel_names = []
        try:
            channel_names = compile_expr(self.spec.expr).names
        except Exception:
            pass

        # every source with the channel (e.g. one FID file per day); each step takes the one covering it
        ch_sources: dict[str, list[tuple[str, np.ndarray]]] = {}
        gap_rows: dict[tuple[str, str], np.ndarray] = {}  # rows starting a time gap, per source and channel
        if self.session is not None:
            for ch in channel_names:
                for s_id, ds in self.session.datasets.items():
                    m_t = self.session.master_time(s_id) if ch in ds.channels else None
                    if m_t is not None:
                        ch_sources.setdefault(ch, []).append((s_id, m_t))
                        gap_rows[(s_id, ch)] = np.asarray(self.session.track(s_id, ch).gap_rows, dtype=np.int64)

        # Faceting
        facets: list[tuple[Any, np.ndarray]] = []
        if self.spec.facet_by and self.spec.facet_by in sub.columns:
            facet_vals = sub[self.spec.facet_by].to_list()
            for f_val in sorted(list(set(facet_vals)), key=lambda v: (v is None, v)):
                mask = np.array([v == f_val for v in facet_vals], dtype=bool)
                facets.append((f_val, mask))
        else:
            facets.append((_NO_FACET, np.ones(len(sub), dtype=bool)))
        facet_rows = sub

        n_facets = len(facets)
        ncols = min(self.spec.facet_cols, n_facets)
        first_plot: pg.PlotItem | None = None

        total_curves = 0
        skipped = 0
        drawn_x: list[np.ndarray] = []
        drawn_y: list[np.ndarray] = []
        for k, (f_name, f_mask) in enumerate(facets):
            row = k // ncols
            col = k % ncols

            p = self.glw.addPlot(row=row, col=col)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("bottom").setPen(AXIS)
            p.getAxis("left").setPen(AXIS)
            p.getAxis("bottom").setTextPen(INK2)
            p.getAxis("left").setTextPen(INK2)
            p.getAxis("left").enableAutoSIPrefix(False)

            if f_name is not _NO_FACET:
                p.setTitle(facet_title(facet_rows.filter(pl.Series(f_mask)), self.spec.facet_by, f_name,
                                       self.spec.facet_label), color=INK)
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

            # Ref lines
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

            spot_pts: list[dict[str, Any]] = []

            step_indices = [int(i) for i in np.flatnonzero(f_mask)]
            for idx in step_indices:
                st = starts_epoch[idx]
                en = ends_epoch[idx]
                t0 = t0_arr[idx]
                s_id = step_ids[idx]
                c_val = color_vals[idx] if color_vals else None
                line_col = color_map.get(c_val, CAT[0])
                m_val = marker_vals[idx] if marker_vals else None
                marker_sym = marker_map.get(m_val, "o")
                is_filled = filled_mask[idx] if filled_mask is not None else True

                step_env = {c: sub[c][idx] for c in sub.columns}
                if self.session is not None and self.session.campaign.evaluation:
                    for ck, cv in self.session.campaign.evaluation.constants.items():
                        step_env[ck] = cv.value

                # channels from the source covering the step; names of the step table are constants
                # over the step; a name found nowhere gives no curve (nothing made up)
                t_slice = None
                ch_arrays = {}
                gaps = np.array([], dtype=np.int64)
                for ch, candidates in ch_sources.items():
                    best = max(((src_id, (m_t >= st) & (m_t <= en)) for src_id, m_t in candidates),
                               key=lambda c: int(c[1].sum()))
                    if best[1].sum() >= 2:
                        m_t = dict(candidates)[best[0]]
                        t_slice = m_t[best[1]]
                        ch_arrays[ch] = self.session.datasets[best[0]].df[ch].to_numpy()[best[1]]
                        rows = np.flatnonzero(best[1])
                        g = gap_rows[(best[0], ch)]
                        gaps = np.union1d(gaps, g[(g > rows[0]) & (g <= rows[-1])] - rows[0])
                missing = [ch for ch in channel_names if ch not in ch_arrays and ch not in step_env]
                if missing or not np.isfinite(t0) or (t_slice is None and ch_sources):
                    skipped += 1
                    continue
                if t_slice is None:
                    t_slice = np.array([st, en])

                t_rel_min = (t_slice - t0) / 60.0

                eval_env = dict(step_env)
                eval_env.update(ch_arrays)
                try:
                    res_expr = compile_expr(self.spec.expr, names=list(eval_env.keys())).evaluate(eval_env)
                    val_arr = res_expr.value if hasattr(res_expr, "value") else res_expr
                    y_arr = np.broadcast_to(np.asarray(val_arr, dtype=float), t_slice.shape).copy()
                except Exception:
                    skipped += 1
                    continue

                if np.isfinite(y_arr).any():
                    if self.spec.rolling_s and len(t_slice) > 2:
                        dt = np.nanmedian(np.diff(t_slice))
                        if dt > 0:
                            w_pts = max(1, round(self.spec.rolling_s / dt))
                            if w_pts > 1 and len(y_arr) >= w_pts:
                                kernel = np.ones(w_pts) / w_pts
                                y_arr = np.convolve(y_arr, kernel, mode="same")

                    valid_idx = np.isfinite(t_rel_min) & np.isfinite(y_arr)
                    if valid_idx.sum() >= 2:
                        # gaps of the source and missing values break the line (never bridged)
                        add_curve(p, t_rel_min, y_arr, gaps, curve_pen(line_col))
                        total_curves += 1
                        drawn_x.append(t_rel_min[valid_idx])
                        drawn_y.append(y_arr[valid_idx])

                        last_t = float(t_rel_min[valid_idx][-1])
                        last_y = float(y_arr[valid_idx][-1])
                        brush = pg.mkBrush(line_col) if is_filled else pg.mkBrush(SURFACE)
                        pen = pg.mkPen(line_col, width=1.5)
                        spot_pts.append({
                            "pos": (last_t, last_y),
                            "size": 8,
                            "symbol": marker_sym,
                            "brush": brush,
                            "pen": pen,
                            "data": s_id,
                        })

            if spot_pts:
                scatter = pg.ScatterPlotItem(
                    spots=spot_pts,
                    hoverable=True,
                    tip=lambda x, y, data: f"Step: {data}\nt: {x:.1f} min\nX: {y:.3f}",
                )
                scatter.sigClicked.connect(self._on_scatter_clicked)
                p.addItem(scatter)


        shared_range(first_plot, drawn_x, drawn_y, self.spec.x_lim, self.spec.y_lim)
        add_layout_legend(
            self.glw,
            point_legend_sections(self.spec.color_by, color_map, self.spec.marker_by, marker_map,
                                  self.spec.filled if filled_mask is not None else None),
            row=0, col=ncols, rowspan=-(-n_facets // ncols),
        )
        self.status.setText(f"{total_curves} aligned curve(s) rendered across {n_facets} facet(s)"
                            + (f"; {skipped} step(s) without data." if skipped else "."))

    def _on_scatter_clicked(self, item: Any, points: Sequence[Any]) -> None:
        if points:
            step_id = points[0].data()
            if step_id:
                self.step_inspected.emit(str(step_id))

    def _on_export_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export Aligned Plot ({self.spec.name})", f"{self.spec.name}.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if path:
            try:
                export_graphics_layout(self.glw, path, ExportOptions(width_mm=170, dpi=300))
                QMessageBox.information(self, "Export Plot", f"Saved to {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", f"Cannot export plot:\n{exc}")

    def export(self, path: str | Path, options: ExportOptions = ExportOptions()) -> Path:
        return export_graphics_layout(self.glw, path, options)


# ------------------------------------------------------------------ Bars View Widget
class BarsViewWidget(QWidget):
    """Render a BarsViewSpec (uncertainty budget bar chart with stacked variance shares)."""

    step_inspected = Signal(str)

    def __init__(self, spec: BarsViewSpec, parent: QWidget | None = None) -> None:
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

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(SURFACE)
        self.glw.scene().setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        layout.addWidget(self.glw, stretch=1)
        self.status = ReadoutLabel("")
        layout.addWidget(self.status)

        if spec.footer:
            footer_lbl = QLabel(f"<small style='color: {INK2};'>{spec.footer}</small>")
            footer_lbl.setWordWrap(True)
            layout.addWidget(footer_lbl)

    def set_data(self, df: pl.DataFrame) -> None:
        self._df = df
        sub = filter_table(df, self.spec.filter)
        self.glw.clear()

        if len(sub) == 0:
            self.status.setText("No data matching filter.")
            return

        sort_cols = [c for c in self.spec.sort_by if c in sub.columns]
        if sort_cols:
            sub = sub.sort(sort_cols)

        parts: list[tuple[str, str]] = []
        if self.spec.parts:
            for p in self.spec.parts:
                if isinstance(p, (tuple, list)) and len(p) >= 2:
                    parts.append((str(p[0]), str(p[1])))
                else:
                    parts.append((str(p), str(p)))
        else:  # the budget of the quantity: one column b_<quantity>__<input> per input
            prefix = f"b_{self.spec.quantity}__"
            parts = [(c, c[len(prefix):]) for c in sub.columns if c.startswith(prefix)]

        parts = [(c, lab) for c, lab in parts if c in sub.columns]
        if not parts:
            self.status.setText("No budget parts found in table.")
            return

        step_ids = step_ids_of(sub)

        u_col = self.spec.u_total or f"u_{self.spec.quantity}"
        if u_col in sub.columns:
            u_vals = sub[u_col].to_numpy().astype(float)
        else:
            u_vals = np.sqrt(np.sum([np.nan_to_num(sub[c].to_numpy().astype(float), nan=0.0) ** 2 for c, _ in parts], axis=0))

        n_steps = len(sub)
        x_indices = np.arange(n_steps)
        ok = np.isfinite(u_vals)  # a bar without total uncertainty is left out, not drawn as 0

        p0 = self.glw.addPlot(row=0, col=0)
        p1 = self.glw.addPlot(row=1, col=0)
        p0.setXLink(p1)
        for p in (p0, p1):  # same plot width in both rows (x linked in pixels), no SI prefix
            p.getAxis("left").setWidth(52)
            p.getAxis("left").enableAutoSIPrefix(False)
            p.getViewBox().setMouseEnabled(x=True, y=False)

        p0.showGrid(x=False, y=True, alpha=0.3)
        p0.getAxis("left").setPen(AXIS)
        p0.getAxis("bottom").setPen(AXIS)
        p0.getAxis("left").setTextPen(INK2)
        p0.getAxis("bottom").setTextPen(INK2)
        p0.setLabel("left", f"u({self.spec.quantity})")
        p0.getAxis("bottom").setTicks([])

        bg0 = pg.BarGraphItem(
            x=x_indices[ok],
            height=u_vals[ok],
            width=0.6,
            brush=pg.mkBrush(CAT[0]),
            pen=pg.mkPen(SURFACE, width=0.8),
        )
        p0.addItem(bg0)

        p1.showGrid(x=False, y=True, alpha=0.3)
        p1.getAxis("left").setPen(AXIS)
        p1.getAxis("bottom").setPen(AXIS)
        p1.getAxis("left").setTextPen(INK2)
        p1.getAxis("bottom").setTextPen(INK2)
        p1.setLabel("left", self.spec.y_label or f"Anteil an u²({self.spec.quantity})")
        if self.spec.x_label:
            p1.setLabel("bottom", self.spec.x_label)
        p1.setYRange(0, 1, padding=0)

        var_matrix = np.column_stack([np.nan_to_num(sub[c].to_numpy().astype(float), nan=0.0) ** 2 for c, _ in parts])
        var_sum = var_matrix.sum(axis=1, keepdims=True)
        var_sum[var_sum == 0] = 1.0
        shares = var_matrix / var_sum
        bottom = np.zeros(n_steps)

        colors = part_colors(len(parts))
        for k, (c, lab) in enumerate(parts):
            col_brush = pg.mkBrush(colors[k])
            bg1 = pg.BarGraphItem(
                x=x_indices[ok],
                y0=bottom[ok],
                height=shares[ok, k],
                width=0.6,
                brush=col_brush,
                pen=pg.mkPen(SURFACE, width=0.8),
            )
            p1.addItem(bg1)
            bottom += shares[:, k]

        tick_labels = (render_template(sub, self.spec.bar_label) if self.spec.bar_label
                       else [str(s) for s in step_ids])
        p1.getAxis("bottom").setTicks([list(zip(x_indices, tick_labels))])
        fit_tick_text(p1.getAxis("bottom"), tick_labels)
        p1.setXRange(-0.6, n_steps - 0.4, padding=0)  # room for the outer labels (cut off at the axis ends)

        add_layout_legend(
            self.glw,
            [("Quelle", [(swatch(colors[k]), lab) for k, (_, lab) in reversed(list(enumerate(parts)))])],
            row=1, col=1,
        )
        self.status.setText(f"Rendered uncertainty budget for {n_steps} steps across {len(parts)} parts.")

    def _on_export_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export Bars Plot ({self.spec.name})", f"{self.spec.name}.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if path:
            try:
                export_graphics_layout(self.glw, path, ExportOptions(width_mm=170, dpi=300))
                QMessageBox.information(self, "Export Plot", f"Saved to {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", f"Cannot export plot:\n{exc}")

    def export(self, path: str | Path, options: ExportOptions = ExportOptions()) -> Path:
        return export_graphics_layout(self.glw, path, options)


# ------------------------------------------------------------------ Report View Widget
# ------------------------------------------------------------------ Timeline View Widget
def part_colors(n: int) -> list[str]:
    """n distinct colours: the categorical palette, then its colours lightened (no repeats up to 16)."""
    out = list(CAT[:n])
    k = 0
    while len(out) < n:
        out.append(QColor(CAT[k % len(CAT)]).lighter(150 + 30 * (k // len(CAT))).name())
        k += 1
    return out


def ordinal_colors(values: Sequence[Any]) -> dict[Any, str]:
    """Colours of the distinct values: the ordinal blue ramp (numbers), else the categorical palette."""
    distinct = sorted({v for v in values if v is not None}, key=lambda v: (str(type(v)), v))
    if all(isinstance(v, (int, float, np.number)) for v in distinct):
        if len(distinct) <= 5:
            idx = RAMP_SUBSET.get(len(distinct), list(range(len(distinct))))
            return {v: RAMP[idx[k]] for k, v in enumerate(distinct)}
        return {v: interpolate_color(RAMP, k / max(1, len(distinct) - 1)) for k, v in enumerate(distinct)}
    return {v: CAT[k % len(CAT)] for k, v in enumerate(distinct)}


def _with_alpha(hex_color: str, alpha: int) -> QColor:
    c = QColor(hex_color)
    c.setAlpha(alpha)
    return c


BAND_ALPHA = 41  # 16 % as in the script's day pages


class TimelineViewWidget(QWidget):
    """Render a TimelineViewSpec: stacked panels over master time, one page per `page_by` value.

    Bands (one `ColorBandsItem` per panel), curves through `curves.add_curve` (gaps never
    bridged), marks per step from the result table. Legends sit in the graphics layout, so they
    are part of every export and report page. `pages()` / `set_page()` let the report draw one
    page per value.
    """

    step_inspected = Signal(str)

    def __init__(self, spec: TimelineViewSpec, session: CampaignSession | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.session = session
        self._df: pl.DataFrame | None = None
        self._sub: pl.DataFrame | None = None
        self._pages: list[Any] = [None]
        self._page: Any = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        header_layout = QHBoxLayout()
        title_text = spec.title or spec.name
        lbl = QLabel(f"<b>{title_text}</b>" + (f" – <small>{spec.subtitle}</small>" if spec.subtitle else ""))
        header_layout.addWidget(lbl)
        header_layout.addStretch()
        self.combo_page = QComboBox()
        self.combo_page.setVisible(spec.page_by is not None)
        self.combo_page.currentIndexChanged.connect(self._on_page_chosen)
        header_layout.addWidget(self.combo_page)
        self.btn_export = QPushButton("Export Plot...")
        self.btn_export.clicked.connect(self._on_export_clicked)
        header_layout.addWidget(self.btn_export)
        layout.addLayout(header_layout)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(SURFACE)
        self.glw.scene().setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        layout.addWidget(self.glw, stretch=1)
        self.status = ReadoutLabel("")
        layout.addWidget(self.status)
        if spec.footer:
            footer_lbl = QLabel(f"<small style='color: {INK2};'>{spec.footer}</small>")
            footer_lbl.setWordWrap(True)
            layout.addWidget(footer_lbl)
        self.plots: list[pg.PlotItem] = []

    # ---------------------------------------------------------------- pages
    def pages(self) -> list[Any]:
        """Values of `page_by` (sorted), or [None] for a single page."""
        return list(self._pages)

    def set_page(self, page: Any) -> None:
        self._page = page
        if page in self._pages:
            self.combo_page.blockSignals(True)
            self.combo_page.setCurrentIndex(self._pages.index(page))
            self.combo_page.blockSignals(False)
        self._render()

    def page_heading(self) -> tuple[str, str | None]:
        """Title and subtitle of the current page (the `page_title` template filled from its steps)."""
        title = self.spec.title or self.spec.name
        rows = self._page_rows()
        if self.spec.page_title and rows is not None:
            title = str(render_template(rows, self.spec.page_title, reduce=True))
        elif self._page is not None:
            title = f"{title} – {self._page}"
        return title, self.spec.subtitle

    def _on_page_chosen(self, index: int) -> None:
        if 0 <= index < len(self._pages):
            self._page = self._pages[index]
            self._render()

    def _page_rows(self) -> pl.DataFrame | None:
        if self._sub is None:
            return None
        if self.spec.page_by is None or self._page is None:
            return self._sub
        return self._sub.filter(pl.col(self.spec.page_by) == self._page)

    # ---------------------------------------------------------------- data
    def set_data(self, df: pl.DataFrame) -> None:
        self._df = df
        self._sub = filter_table(df, self.spec.filter)
        pages: list[Any] = [None]
        if self.spec.page_by is not None:
            if self.spec.page_by not in self._sub.columns:
                self._sub = None
                self.glw.clear()
                self.status.setText(f"No column {self.spec.page_by!r} for the pages.")
                return
            pages = sorted({v for v in self._sub[self.spec.page_by].to_list() if v is not None}, key=str) or [None]
        self._pages = pages
        self.combo_page.blockSignals(True)
        self.combo_page.clear()
        self.combo_page.addItems([str(p) for p in pages])
        self.combo_page.blockSignals(False)
        self._page = self._page if self._page in pages else pages[0]
        self.set_page(self._page)

    def _render(self) -> None:
        self.glw.clear()
        self.plots = []
        rows = self._page_rows()
        if rows is None or len(rows) == 0 or "step_start" not in rows.columns:
            self.status.setText("No steps matching filter.")
            return
        starts = np.asarray(_to_epoch_array(rows["step_start"]), dtype=float)
        ends = np.asarray(_to_epoch_array(rows["step_end"]), dtype=float)
        lo, hi = float(np.nanmin(starts)), float(np.nanmax(ends))
        step_ids = rows["step_id"].to_list() if "step_id" in rows.columns else [""] * len(rows)

        # bands, coloured consistently over all pages
        band_color: dict[str, str] = {}
        band_key: dict[str, Any] = {}
        bands: list[tuple[float, float, QColor, str]] = []
        colors: dict[Any, str] = {}
        if self.spec.bands:
            all_bands = filter_table(self._sub, self.spec.bands)
            colors = ordinal_colors(all_bands[self.spec.band_color_by].to_list()) if (
                self.spec.band_color_by and self.spec.band_color_by in all_bands.columns) else {}
            brows = filter_table(rows, self.spec.bands)
            b0 = np.asarray(_to_epoch_array(brows["step_start"]), dtype=float)
            b1 = np.asarray(_to_epoch_array(brows["step_end"]), dtype=float)
            keys = brows[self.spec.band_color_by].to_list() if colors else [None] * len(brows)
            texts = render_template(brows, self.spec.band_label) if self.spec.band_label else [""] * len(brows)
            for sid, a, b, k, t in zip(brows["step_id"].to_list(), b0, b1, keys, texts):
                c = colors.get(k, CAT[0])
                band_color[sid] = c
                band_key[sid] = k
                bands.append((a, b, _with_alpha(c, BAND_ALPHA), t))

        zone = self.session.campaign.timezone if self.session is not None else "UTC"
        offset = utc_offset_s(lo, zone)
        n = len(self.spec.panels)
        for i, panel in enumerate(self.spec.panels):
            axis = make_time_axis(TimeAxisMode.CLOCK, offset, None)
            p = self.glw.addPlot(row=i, col=0, axisItems={"bottom": axis})
            self.glw.ci.layout.setRowStretchFactor(i, max(1, round(panel.height * 10)))
            p.getViewBox().setMouseEnabled(x=True, y=False)
            for side in ("bottom", "left"):
                p.getAxis(side).setPen(AXIS)
                p.getAxis(side).setTextPen(INK2)
            p.getAxis("left").setWidth(52)  # same plot width in every panel (x linked in pixels)
            if i < n - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            if self.plots:
                p.setXLink(self.plots[0])
            if panel.y_label:
                p.setLabel("left", panel.y_label)
            if bands:
                p.addItem(ColorBandsItem(bands, QColor(INK2) if i == 0 else None), ignoreBounds=True)
            light_grid(p, 0.35)
            legend = pg.LegendItem(labelTextColor=INK2, labelTextSize="8pt", verSpacing=-4)
            legend.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.glw.ci.addItem(legend, row=i, col=1)
            self.glw.ci.layout.setAlignment(legend, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            ys = self._draw_traces(p, panel, lo, hi, legend, i)
            self._draw_marks(p, panel, rows, step_ids, ends, starts, band_color, legend)
            for r in panel.ref_lines:
                p.addItem(pg.InfiniteLine(pos=r.value, angle=0 if r.axis == "y" else 90,
                                          pen=pg.mkPen(r.color, width=r.width), label=r.label or None))
            if i == n - 1 and colors:
                used = sorted({band_key[sid] for sid in band_color}, key=lambda v: (str(type(v)), v))
                for k in used:
                    sample = pg.ScatterPlotItem(symbol="s", size=10, pen=None, brush=pg.mkBrush(_with_alpha(colors[k], 110)))
                    text = f"{k:g}" if isinstance(k, (int, float)) else str(k)
                    legend.addItem(sample, f"{text} {self.spec.band_legend}".strip())
            p.setXRange(lo, hi, padding=0)
            if panel.y_lim:
                p.setYRange(*panel.y_lim, padding=0)
            elif panel.y_robust and ys.size:
                top = float(np.nanpercentile(ys, panel.y_robust)) * 1.12
                if np.isfinite(top) and top > 0:
                    p.setYRange(0, top, padding=0)
            self.plots.append(p)
        self.glw.ci.layout.setColumnFixedWidth(1, LEGEND_W_PX)
        self.status.setText(f"{len(rows)} step(s), {len(bands)} band(s); page {self._page if self._page is not None else '–'}.")

    def _draw_traces(self, p: pg.PlotItem, panel: TimelinePanel, lo: float, hi: float,
                     legend: pg.LegendItem, panel_index: int) -> np.ndarray:
        """Curves of the panel inside [lo, hi]; returns their values (for the robust y range)."""
        values: list[np.ndarray] = []
        if self.session is None:
            return np.array([])
        for k, tr_spec in enumerate(panel.traces):
            color = tr_spec.color or (INK2 if len(panel.traces) == 1 else CAT[(k + 1) % len(CAT)])
            pen = curve_pen(color, 1.2)
            sources = tr_spec.sources or [s.id for s in self.session.campaign.sources]
            drawn = False
            for sid in sources:
                ds = self.session.datasets.get(sid)
                if ds is None or tr_spec.channel not in ds.channels or self.session.master_time(sid) is None:
                    continue
                tr = self.session.track(sid, tr_spec.channel)
                i0, i1 = np.searchsorted(tr.x, [lo, hi])
                if i1 - i0 < 2:
                    continue
                gap = tr.gap_rows[(tr.gap_rows >= i0) & (tr.gap_rows < i1)] - i0
                add_curve(p, tr.x[i0:i1], tr.y[i0:i1], gap, pen)
                values.append(tr.y[i0:i1])
                drawn = True
            if drawn:
                legend.addItem(pg.PlotDataItem(pen=pg.mkPen(color, width=1.4)), tr_spec.label or tr_spec.channel)
        return np.concatenate(values) if values else np.array([])

    def _draw_marks(self, p: pg.PlotItem, panel: TimelinePanel, rows: pl.DataFrame, step_ids: list,
                    ends: np.ndarray, starts: np.ndarray, band_color: dict[str, str], legend: pg.LegendItem) -> None:
        for mk in panel.marks:
            keep = np.ones(len(rows), dtype=bool)
            if mk.filter:
                f = eval_table_expr(rows, mk.filter)
                keep = np.broadcast_to(np.asarray(f, dtype=bool), (len(rows),)).copy()
            y = np.broadcast_to(np.asarray(eval_table_expr(rows, mk.y), dtype=float), (len(rows),))
            u = (np.broadcast_to(np.asarray(eval_table_expr(rows, mk.u_y), dtype=float), (len(rows),))
                 if mk.u_y else None)
            filled = (np.broadcast_to(np.asarray(eval_table_expr(rows, mk.filled), dtype=bool), (len(rows),))
                      if mk.filled else np.ones(len(rows), dtype=bool))
            idx = [j for j in range(len(rows)) if keep[j] and np.isfinite(y[j])]
            if mk.at == "window":
                for j in idx:
                    w = self._window(step_ids[j], mk.window)
                    if w is None:
                        continue
                    col = INK if filled[j] else MUTED
                    seg = pg.PlotCurveItem([w[0], w[1]], [y[j], y[j]], pen=pg.mkPen(col, width=3.5))
                    p.addItem(seg)
                if idx:
                    legend.addItem(pg.PlotDataItem(pen=pg.mkPen(INK, width=3.5)), mk.label or mk.y)
                    if mk.filled and mk.label_open:
                        legend.addItem(pg.PlotDataItem(pen=pg.mkPen(MUTED, width=3.5)), mk.label_open)
                continue
            xs = ends if mk.at == "end" else starts
            spots = []
            for j in idx:
                col = band_color.get(step_ids[j], INK2) if mk.color_by_band else INK2
                spots.append({
                    "pos": (xs[j], y[j]), "size": 8, "symbol": "o", "data": step_ids[j],
                    "brush": pg.mkBrush(col) if filled[j] else pg.mkBrush(SURFACE), "pen": pg.mkPen(col, width=1.3),
                })
                if u is not None and np.isfinite(u[j]):
                    p.addItem(pg.ErrorBarItem(x=np.array([xs[j]]), y=np.array([y[j]]), height=np.array([2 * u[j]]),
                                              pen=pg.mkPen(col, width=1)))
            if spots:
                sc = pg.ScatterPlotItem(spots=spots, hoverable=True)
                sc.sigClicked.connect(self._on_scatter_clicked)
                p.addItem(sc)
                legend.addItem(pg.ScatterPlotItem(symbol="o", size=8, brush=pg.mkBrush(INK2), pen=pg.mkPen(INK2)),
                               mk.label or mk.y)
                if mk.filled and mk.label_open:
                    legend.addItem(pg.ScatterPlotItem(symbol="o", size=8, brush=pg.mkBrush(SURFACE), pen=pg.mkPen(INK2)),
                                   mk.label_open)

    def _window(self, step_id: str, window: str) -> tuple[float, float] | None:
        if self.session is None or not step_id:
            return None
        try:
            sw = self.session.step_window(step_id, window)
        except KeyError:
            return None
        return sw.window if sw.applies else None

    def _on_scatter_clicked(self, item: Any, points: Sequence[Any]) -> None:
        if points:
            step_id = points[0].data()
            if step_id:
                self.step_inspected.emit(str(step_id))

    def _on_export_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export Plot ({self.spec.name})", f"{self.spec.name}.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if path:
            try:
                export_graphics_layout(self.glw, path, ExportOptions(width_mm=250, height_mm=150, dpi=300))
                QMessageBox.information(self, "Export Plot", f"Saved to {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", f"Cannot export plot:\n{exc}")

    def export(self, path: str | Path, options: ExportOptions = ExportOptions()) -> Path:
        return export_graphics_layout(self.glw, path, options)


def report_views(views: Sequence[ViewSpec], report: ReportViewSpec) -> list[ViewSpec]:
    """Views a report shows: those it lists, in its order; none listed = all other non-report views."""
    others = [v for v in views if v.kind != "report"]
    if not report.views:
        return others
    by_name = {v.name: v for v in others}
    return [by_name[n] for n in report.views if n in by_name]


class ReportViewWidget(QWidget):
    """Render a ReportViewSpec: lists included report pages and triggers PDF export."""

    def __init__(
        self,
        spec: ReportViewSpec,
        session: CampaignSession | None = None,
        views_dict: Mapping[str, QWidget] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self.session = session
        self.views_dict = views_dict or {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        header_layout = QHBoxLayout()
        title_text = spec.title or spec.name
        lbl = QLabel(f"<h2>{title_text}</h2>" + (f"<p style='color: {INK2};'>{spec.subtitle}</p>" if spec.subtitle else ""))
        header_layout.addWidget(lbl)
        header_layout.addStretch()

        self.btn_export = QPushButton("Export PDF Report...")
        self.btn_export.setStyleSheet("font-size: 11pt; padding: 6px 16px; font-weight: bold; background: #2a78d6; color: white; border-radius: 4px;")
        self.btn_export.clicked.connect(self._on_export_clicked)
        header_layout.addWidget(self.btn_export)
        layout.addLayout(header_layout)

        # Overview Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Page #", "Title / Content", "Type", "Page Size"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet(f"background: {SURFACE}; color: {INK}; gridline-color: {GRID};")
        layout.addWidget(self.table, stretch=1)

        self.status = ReadoutLabel("")
        layout.addWidget(self.status)

    def set_data(self, df: pl.DataFrame) -> None:
        self.table.setRowCount(0)
        row = 0
        if self.spec.cover:
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(f"Page {row + 1}"))
            self.table.setItem(row, 1, QTableWidgetItem("Cover Page (Title, Dates, KPI Tiles, Analyses)"))
            self.table.setItem(row, 2, QTableWidgetItem("Cover"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{self.spec.page_size} {self.spec.orientation.capitalize()}"))
            row += 1

        views_to_show = self.spec.views
        if not views_to_show and self.session and self.session.campaign.evaluation:
            views_to_show = [v.name for v in self.session.campaign.evaluation.views if v.name != self.spec.name]

        for v_name in views_to_show:
            w = self.views_dict.get(v_name)
            v_spec = getattr(w, "spec", None)
            v_title = getattr(v_spec, "title", "") or v_name
            v_kind = getattr(v_spec, "kind", "View")
            pages = w.pages() if w is not None and hasattr(w, "pages") else [None]
            for page in pages:
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(f"Page {row + 1}"))
                self.table.setItem(row, 1, QTableWidgetItem(v_title if page is None else f"{v_title} – {page}"))
                self.table.setItem(row, 2, QTableWidgetItem(v_kind.capitalize()))
                self.table.setItem(row, 3, QTableWidgetItem(f"{self.spec.page_size} {self.spec.orientation.capitalize()}"))
                row += 1

        self.table.resizeColumnsToContents()
        self.status.setText(f"Report contains {row} page(s). Click 'Export PDF Report...' to generate PDF.")

    def _on_export_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export Report ({self.spec.name})", f"{self.spec.name}.pdf", "PDF (*.pdf)"
        )
        if path:
            try:
                self.export(path)
                QMessageBox.information(self, "Export Report", f"Report saved to:\n{path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", f"Cannot export report:\n{exc}")

    def export(self, path: str | Path, options: ExportOptions | None = None) -> Path:
        opts = options or ExportOptions(width_mm=297, height_mm=210, dpi=300)
        views = []
        if self.session and self.session.campaign.evaluation:
            views = report_views(self.session.campaign.evaluation.views, self.spec)

        return export_report(
            session=self.session,
            views=views,
            view_widgets=self.views_dict,
            path=path,
            options=opts,
            cover=self.spec.cover,
            title=self.spec.title,
            subtitle=self.spec.subtitle,
            tiles=self.spec.tiles,
        )


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
            if self._direct_df is not None:
                df = self._direct_df
                views = (
                    self._session.campaign.evaluation.views
                    if (self._session and self._session.campaign.evaluation and self._session.campaign.evaluation.views)
                    else []
                )
            elif self._session is not None:
                ev = self._session.campaign.evaluation
                scenario = self.current_scenario_name()
                try:
                    df = self._session.results(scenario=scenario)
                except Exception as exc:
                    self.status_label.setText(f"Error computing results: {exc}")
                    return
                views = ev.views if ev and ev.views else []
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
            elif spec.kind == "aligned":
                aw = AlignedViewWidget(spec, session=self._session)
                aw.step_inspected.connect(self.step_inspected)
                aw.set_data(df)
                w = aw
            elif spec.kind == "bars":
                bw = BarsViewWidget(spec)
                bw.step_inspected.connect(self.step_inspected)
                bw.set_data(df)
                w = bw
            elif spec.kind == "timeline":
                tlw = TimelineViewWidget(spec, session=self._session)
                tlw.step_inspected.connect(self.step_inspected)
                tlw.set_data(df)
                w = tlw
            elif spec.kind == "report":
                rw = ReportViewWidget(spec, session=self._session, views_dict=self._view_widgets)
                rw.set_data(df)
                w = rw
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

        if isinstance(w, ReportViewWidget):
            if not path:
                chosen, _ = QFileDialog.getSaveFileName(
                    self, f"Export Report ({view_name})", f"{view_name}.pdf", "PDF (*.pdf)"
                )
                if not chosen:
                    return None
                path = chosen
            try:
                out = w.export(path)
                QMessageBox.information(self, "Export Report", f"Saved to {out}")
                return out
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", f"Cannot export report:\n{exc}")
                return None

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
    "AlignedViewWidget",
    "BarsViewWidget",
    "MatrixViewWidget",
    "ReportViewWidget",
    "ResultsWindow",
    "report_views",
    "TableViewWidget",
    "TimelineViewWidget",
    "XYViewWidget",
]
