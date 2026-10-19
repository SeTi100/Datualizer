"""Export the campaign plot as PNG, SVG or PDF for a report: light theme, size in mm, font size,
line width and resolution.

The plot is rebuilt off screen (`CampaignPlotCanvas.export_copy`: same tracks, subplots, colours,
time axis and ranges) and rendered into the file. Only the export gets wider lines and larger
markers; on screen the curves stay at 1 px (cost, see `curves`). Curves are reduced to the
columns of the target resolution, so a PNG at 300 dpi keeps the detail of a 300 dpi print.

Every device is painted at a logical 96 dpi and scaled by the painter: pyqtgraph draws axes into
a QPicture, which Qt rescales once more on a device of another resolution. The PNG gets its
dpi only afterwards; SVG and PDF are vector files, the dpi option applies to PNG only. Item
caches are switched off, so curves and axes stay vectors in SVG and PDF.
"""

from __future__ import annotations

from datetime import date, datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QMarginsF, QPointF, QRectF, QSize, QSizeF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPageSize, QPainter, QPdfWriter
from PySide6.QtWidgets import QApplication, QGraphicsItem, QTableWidget

from datualizer_gui.components.campaign_canvas import CampaignPlotCanvas
from datualizer_gui.components.coverage_strip import LEFT_AXIS_MIN_PX, lane_label, left_axis_width
from datualizer_gui.components.curves import EDGE_MARKER_SIZE_PX, POINT_SIZE_PX, curve_pen

SCENE_DPI = 96  # scene units are pixels at this resolution
MM_PER_INCH = 25.4
PT_PER_INCH = 72
TICK_EXAMPLE = "-0000.00"  # widest y tick label expected


class ExportFormat(str, Enum):
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"

    @classmethod
    def from_path(cls, path: str | Path) -> "ExportFormat":
        suffix = Path(path).suffix.lower().lstrip(".")
        try:
            return cls(suffix)
        except ValueError:
            raise ValueError(f"Cannot export to {Path(path).name!r}: use .png, .svg or .pdf.") from None


@dataclass(frozen=True)
class Theme:
    background: str
    foreground: str  # axes, tick labels, axis titles
    title: str
    grid_alpha: float
    legend_brush: tuple[int, int, int, int]
    hatch: str


LIGHT = Theme("#ffffff", "#222222", "#333333", 0.25, (255, 255, 255, 220), "#777777")
DARK = Theme("#18181c", "#c8c8c8", "#aaaaaa", 0.25, (24, 24, 28, 200), "#e8e8e8")


@dataclass(frozen=True)
class ExportOptions:
    width_mm: float = 170.0  # A4 text width
    height_mm: float = 0.0  # 0 = from the number of subplots
    dpi: int = 300  # PNG only (SVG and PDF are vector files)
    font_pt: float = 8.0
    line_width_pt: float = 0.75
    light: bool = True
    subplot_height_mm: float = 45.0  # per subplot when the height is automatic
    lane_height_mm: float = 5.0  # per coverage lane when the height is automatic

    def height_for(self, subplots: int, lanes: int) -> float:
        if self.height_mm > 0:
            return self.height_mm
        coverage = lanes * self.lane_height_mm + 12.0 if lanes else 0.0
        return max(subplots, 1) * self.subplot_height_mm + coverage


def mm_to_scene(mm: float) -> float:
    return mm / MM_PER_INCH * SCENE_DPI


def pt_to_px(pt: float, dpi: float) -> float:
    return pt / PT_PER_INCH * dpi


def export_plot(canvas: CampaignPlotCanvas, path: str | Path, options: ExportOptions = ExportOptions()) -> Path:
    """Write what the canvas shows (time range, y ranges) into a PNG, SVG or PDF file."""
    path = Path(path)
    fmt = ExportFormat.from_path(path)
    if not canvas.all_plots():
        raise ValueError("Nothing to export: select channels or show the coverage strip first.")
    copy = canvas.export_copy()
    lanes = len(copy.coverage) if copy.coverage_plot is not None else 0
    height_mm = options.height_for(len(copy.plot_items), lanes)
    w, h = mm_to_scene(options.width_mm), mm_to_scene(height_mm)
    device_dpi = options.dpi if fmt is ExportFormat.PNG else SCENE_DPI
    _style(copy, options, device_dpi)
    layout = copy.graphics_layout
    scene = layout.scene()
    for item in scene.items():
        item.setCacheMode(QGraphicsItem.CacheMode.NoCache)
    layout.ci.setGeometry(QRectF(0, 0, w, h))
    scene.setSceneRect(QRectF(0, 0, w, h))
    for _ in range(3):  # let the layout settle so curves reduce to the final widths
        layout.ci.layout.activate()
        QApplication.processEvents()
    source = QRectF(0, 0, w, h)
    if fmt is ExportFormat.PNG:
        scale = options.dpi / SCENE_DPI
        image = QImage(round(w * scale), round(h * scale), QImage.Format.Format_ARGB32)
        image.fill(QColor(_theme(options).background))
        _render(scene, image, QRectF(0, 0, image.width(), image.height()), source)
        dots_per_m = round(options.dpi / MM_PER_INCH * 1000)  # after painting, see module doc
        image.setDotsPerMeterX(dots_per_m)
        image.setDotsPerMeterY(dots_per_m)
        if not image.save(str(path)):
            raise OSError(f"Cannot write {path}")
    elif fmt is ExportFormat.SVG:
        from PySide6.QtSvg import QSvgGenerator

        gen = QSvgGenerator()
        gen.setFileName(str(path))
        gen.setSize(QSize(round(w), round(h)))
        gen.setViewBox(source)
        gen.setResolution(SCENE_DPI)
        gen.setTitle("Datualizer campaign plot")
        _render(scene, gen, source, source)
    else:
        writer = QPdfWriter(str(path))
        writer.setPageSize(QPageSize(QSizeF(options.width_mm, height_mm), QPageSize.Unit.Millimeter))
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        writer.setResolution(SCENE_DPI)
        writer.setTitle("Datualizer campaign plot")
        _render(scene, writer, QRectF(0, 0, writer.width(), writer.height()), source)
    copy.deleteLater()
    return path


def export_graphics_layout(
    layout_widget: pg.GraphicsLayoutWidget,
    path: str | Path,
    options: ExportOptions = ExportOptions(),
) -> Path:
    """Export any pg.GraphicsLayoutWidget into a PNG, SVG or PDF file."""
    path = Path(path)
    fmt = ExportFormat.from_path(path)
    height_mm = options.height_mm if options.height_mm > 0 else 120.0
    w = mm_to_scene(options.width_mm)
    h = mm_to_scene(height_mm)
    scene = layout_widget.scene()
    for item in scene.items():
        item.setCacheMode(QGraphicsItem.CacheMode.NoCache)

    orig_geo = layout_widget.ci.geometry()
    orig_rect = scene.sceneRect()

    layout_widget.ci.setGeometry(QRectF(0, 0, w, h))
    scene.setSceneRect(QRectF(0, 0, w, h))
    for _ in range(3):
        layout_widget.ci.layout.activate()
        QApplication.processEvents()
    source = QRectF(0, 0, w, h)

    if fmt is ExportFormat.PNG:
        scale = options.dpi / SCENE_DPI
        image = QImage(round(w * scale), round(h * scale), QImage.Format.Format_ARGB32)
        image.fill(QColor(_theme(options).background))
        _render(scene, image, QRectF(0, 0, image.width(), image.height()), source)
        dots_per_m = round(options.dpi / MM_PER_INCH * 1000)
        image.setDotsPerMeterX(dots_per_m)
        image.setDotsPerMeterY(dots_per_m)
        if not image.save(str(path)):
            raise OSError(f"Cannot write {path}")
    elif fmt is ExportFormat.SVG:
        from PySide6.QtSvg import QSvgGenerator

        gen = QSvgGenerator()
        gen.setFileName(str(path))
        gen.setSize(QSize(round(w), round(h)))
        gen.setViewBox(source)
        gen.setResolution(SCENE_DPI)
        gen.setTitle("Datualizer view export")
        _render(scene, gen, source, source)
    else:
        writer = QPdfWriter(str(path))
        writer.setPageSize(QPageSize(QSizeF(options.width_mm, height_mm), QPageSize.Unit.Millimeter))
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        writer.setResolution(SCENE_DPI)
        writer.setTitle("Datualizer view export")
        _render(scene, writer, QRectF(0, 0, writer.width(), writer.height()), source)

    layout_widget.ci.setGeometry(orig_geo)
    scene.setSceneRect(orig_rect)
    layout_widget.ci.layout.activate()
    QApplication.processEvents()
    return path


def export_report(
    session: Any,
    views: Sequence[Any],
    view_widgets: Mapping[str, Any],
    path: str | Path,
    options: ExportOptions = ExportOptions(width_mm=297, height_mm=210, dpi=300),
    cover: bool = True,
    title: str = "",
    subtitle: str | None = None,
) -> Path:
    """Export a multi-page PDF report (A4 landscape) containing a cover page and views."""
    path = Path(path)
    writer = QPdfWriter(str(path))
    page_w = options.width_mm if options.width_mm > 0 else 297.0
    page_h = options.height_mm if options.height_mm > 0 else 210.0
    writer.setPageSize(QPageSize(QSizeF(page_w, page_h), QPageSize.Unit.Millimeter))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0))
    writer.setResolution(SCENE_DPI)
    writer.setTitle(f"{title or 'Datualizer'} Report")

    w_px = writer.width()
    h_px = writer.height()

    painter = QPainter(writer)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # 1. Cover Page
        if cover:
            _draw_cover_page(painter, session, w_px, h_px, title, subtitle)
            if views:
                writer.newPage()

        # 2. Render each view
        renderable_views = [v for v in views if getattr(v, "kind", "") != "report"]
        for idx, v_spec in enumerate(renderable_views):
            v_name = getattr(v_spec, "name", str(idx))
            w = view_widgets.get(v_name)
            if w is not None:
                _draw_view_page(painter, v_spec, w, w_px, h_px, page_num=idx + (2 if cover else 1))
            if idx < len(renderable_views) - 1:
                writer.newPage()
    finally:
        painter.end()

    return path


def _draw_cover_page(
    painter: QPainter,
    session: Any,
    w_px: float,
    h_px: float,
    title: str = "",
    subtitle: str | None = None,
) -> None:
    painter.fillRect(QRectF(0, 0, w_px, h_px), QColor("#fcfcfb"))

    # Title
    camp_name = title or (session.campaign.name if session and hasattr(session, "campaign") else "Campaign")
    painter.setPen(QColor("#0b0b0b"))
    f_title = QFont()
    f_title.setPointSize(24)
    f_title.setBold(True)
    painter.setFont(f_title)
    painter.drawText(QPointF(50, 75), camp_name)

    # Subtitle with dates
    f_sub = QFont()
    f_sub.setPointSize(10.5)
    painter.setFont(f_sub)
    painter.setPen(QColor("#52514e"))
    if subtitle:
        sub_text = subtitle
    else:
        days_str = ""
        try:
            if session is not None:
                st = session.step_table()
                if "tag" in st.columns:
                    unique_days = sorted([str(d) for d in set(st["tag"].to_list()) if d is not None])
                    if unique_days:
                        days_str = f"{len(unique_days)} Messtage ({', '.join(unique_days)}) · "
        except Exception:
            pass
        sub_text = f"{days_str}Stand {date.today().strftime('%d.%m.%Y')}"
    painter.drawText(QPointF(50, 105), sub_text)

    # Gather KPIs
    n_steps = 0
    n_stat = 0
    try:
        if session is not None:
            results_df = session.results()
            n_steps = len(results_df)
            if "stationaer" in results_df.columns:
                n_stat = int((results_df["stationaer"] == True).sum())
            elif "stat_ok" in results_df.columns:
                n_stat = int((results_df["stat_ok"] == True).sum())
            else:
                n_stat = n_steps
    except Exception:
        pass

    an_res = {}
    try:
        if session is not None:
            an_res = session.analyses()
    except Exception:
        pass

    t50_str = "–"
    ea_str = "–"
    for a_name, a_val in an_res.items():
        if "T50" in a_name or getattr(a_val, "level", None) == 0.5:
            if hasattr(a_val, "value") and np.isfinite(getattr(a_val, "value", np.nan)):
                t50_str = f"{a_val.value:.0f} °C"
            elif isinstance(a_val, dict) and "T50" in a_val:
                t50_str = str(a_val["T50"])
        if "Arrhenius" in a_name or "fit" in a_name or "Ea" in a_name or hasattr(a_val, "derived"):
            if hasattr(a_val, "derived") and "Ea" in a_val.derived:
                ea_res = a_val.derived["Ea"]
                ea_str = f"{ea_res.value:.0f} ± {ea_res.u:.0f} kJ/mol"
            elif hasattr(a_val, "m") and np.isfinite(a_val.m):
                ea_str = f"{-a_val.m * 8.314:.0f} kJ/mol"

    tiles = [
        ("Dosierstufen", f"{n_steps}"),
        ("stationär auswertbar", f"{n_stat}"),
        ("T50", t50_str),
        ("Ea (scheinbar)", ea_str),
    ]

    # Draw KPI Tiles
    box_w = (w_px - 100 - 3 * 20) / 4.0
    box_h = 80.0
    box_y = 135.0
    for i, (lab, val) in enumerate(tiles):
        bx = 50.0 + i * (box_w + 20)
        rect = QRectF(bx, box_y, box_w, box_h)
        painter.setPen(QColor("#e1e0d9"))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(rect, 4, 4)

        f_lbl = QFont()
        f_lbl.setPointSize(9)
        painter.setFont(f_lbl)
        painter.setPen(QColor("#52514e"))
        painter.drawText(QRectF(bx + 12, box_y + 10, box_w - 24, 20), Qt.AlignmentFlag.AlignLeft, lab)

        f_val = QFont()
        f_val.setPointSize(18 if len(val) < 10 else 13)
        f_val.setBold(True)
        painter.setFont(f_val)
        painter.setPen(QColor("#0b0b0b"))
        painter.drawText(QRectF(bx + 12, box_y + 32, box_w - 24, 40), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, val)

    # Analyses and summary lines
    f_sect = QFont()
    f_sect.setPointSize(11.5)
    f_sect.setBold(True)
    painter.setFont(f_sect)
    painter.setPen(QColor("#0b0b0b"))
    curr_y = 245.0
    painter.drawText(QPointF(50, curr_y), "Auswertungen & Grundlagen")

    f_body = QFont()
    f_body.setPointSize(9.5)
    painter.setFont(f_body)
    painter.setPen(QColor("#0b0b0b"))
    curr_y += 24.0

    lines = []
    if an_res:
        lines.append("Analysen:")
        for a_name, a_val in an_res.items():
            if hasattr(a_val, "groups"):
                lines.append(f"  • {a_name} (Crossing): {len(a_val.groups)} Gruppe(n) ausgewertet")
            elif hasattr(a_val, "m"):
                lines.append(f"  • {a_name} (Fit): Steigung m = {a_val.m:.4g} ± {a_val.u_m:.4g}, b = {a_val.b:.4g} ± {a_val.u_b:.4g}, χ²_red = {a_val.chi2_red:.2f}")
            else:
                lines.append(f"  • {a_name}")

    if session and hasattr(session, "campaign"):
        ev = getattr(session.campaign, "evaluation", None)
        if ev and ev.constants:
            lines.append("")
            lines.append("Konstanten:")
            for ck, cv in ev.constants.items():
                u_str = f" ± {cv.u}" if cv.u else ""
                unit_str = f" {cv.unit}" if cv.unit else ""
                lines.append(f"  • {ck} = {cv.value}{u_str}{unit_str}" + (f" ({cv.note})" if cv.note else ""))

    if not lines:
        lines = [
            "  • Stationäre Umsatz- und Temperaturmessungen aus der Messkampagne",
            "  • Unsicherheiten nach GUM (Zufallsfehler und systematische Anteile)",
        ]

    for line in lines[:16]:
        painter.drawText(QPointF(50, curr_y), line)
        curr_y += 18.0

    # Cover Footer
    painter.setPen(QColor("#898781"))
    f_foot = QFont()
    f_foot.setPointSize(8.5)
    painter.setFont(f_foot)
    painter.drawText(QPointF(50, h_px - 30), f"Datualizer · Kampagnenbericht · {datetime.now().strftime('%d.%m.%Y %H:%M')}")


def _draw_view_page(
    painter: QPainter,
    v_spec: Any,
    widget: Any,
    w_px: float,
    h_px: float,
    page_num: int,
) -> None:
    painter.fillRect(QRectF(0, 0, w_px, h_px), QColor("#fcfcfb"))

    # Header
    v_title = getattr(v_spec, "title", "") or getattr(v_spec, "name", "View")
    v_sub = getattr(v_spec, "subtitle", None)

    f_title = QFont()
    f_title.setPointSize(14)
    f_title.setBold(True)
    painter.setFont(f_title)
    painter.setPen(QColor("#0b0b0b"))
    painter.drawText(QPointF(50, 42), v_title)

    if v_sub:
        f_sub = QFont()
        f_sub.setPointSize(9.5)
        painter.setFont(f_sub)
        painter.setPen(QColor("#52514e"))
        painter.drawText(QPointF(50, 62), v_sub)

    # Plot or Table content
    content_rect = QRectF(50, 72, w_px - 100, h_px - 125)

    if hasattr(widget, "glw"):
        glw = widget.glw
        scene = glw.scene()
        for item in scene.items():
            item.setCacheMode(QGraphicsItem.CacheMode.NoCache)

        orig_geo = glw.ci.geometry()
        orig_rect = scene.sceneRect()

        target_w = content_rect.width()
        target_h = content_rect.height()

        glw.ci.setGeometry(QRectF(0, 0, target_w, target_h))
        scene.setSceneRect(QRectF(0, 0, target_w, target_h))
        for _ in range(3):
            glw.ci.layout.activate()
            QApplication.processEvents()

        scene.render(painter, content_rect, QRectF(0, 0, target_w, target_h))

        glw.ci.setGeometry(orig_geo)
        scene.setSceneRect(orig_rect)
        glw.ci.layout.activate()
        QApplication.processEvents()
    elif hasattr(widget, "table"):
        table = widget.table
        _draw_table_content(painter, table, content_rect)

    # Footer
    footer_text = getattr(v_spec, "footer", None)
    painter.setPen(QColor("#52514e"))
    f_foot = QFont()
    f_foot.setPointSize(8.5)
    painter.setFont(f_foot)
    if footer_text:
        painter.drawText(QRectF(50, h_px - 45, w_px - 200, 35), Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap, footer_text)
    painter.setPen(QColor("#898781"))
    painter.drawText(QRectF(w_px - 150, h_px - 45, 100, 35), Qt.AlignmentFlag.AlignRight, f"Seite {page_num}")


def _draw_table_content(painter: QPainter, table: QTableWidget, rect: QRectF) -> None:
    """Draw a table widget onto the painter within rect."""
    n_rows = min(table.rowCount(), 25)
    n_cols = min(table.columnCount(), 12)
    if n_rows == 0 or n_cols == 0:
        return

    col_w = rect.width() / n_cols
    row_h = min(22.0, (rect.height() - 30.0) / (n_rows + 1))

    # Header row
    f_hdr = QFont()
    f_hdr.setPointSize(8.5)
    f_hdr.setBold(True)
    painter.setFont(f_hdr)
    painter.setPen(QColor("#0b0b0b"))

    for c in range(n_cols):
        hdr_item = table.horizontalHeaderItem(c)
        hdr_text = hdr_item.text() if hdr_item else f"Col {c}"
        cell_rect = QRectF(rect.x() + c * col_w, rect.y(), col_w, row_h)
        painter.fillRect(cell_rect, QColor("#f0efec"))
        painter.drawRect(cell_rect)
        painter.drawText(cell_rect.adjusted(4, 2, -4, -2), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, hdr_text)

    # Data rows
    f_cell = QFont()
    f_cell.setPointSize(8)
    painter.setFont(f_cell)
    for r in range(n_rows):
        y = rect.y() + (r + 1) * row_h
        bg_col = QColor("#ffffff") if r % 2 == 0 else QColor("#fcfcfb")
        for c in range(n_cols):
            cell_rect = QRectF(rect.x() + c * col_w, y, col_w, row_h)
            painter.fillRect(cell_rect, bg_col)
            painter.setPen(QColor("#e1e0d9"))
            painter.drawRect(cell_rect)

            item = table.item(r, c)
            txt = item.text() if item else ""
            painter.setPen(QColor("#0b0b0b"))
            painter.drawText(cell_rect.adjusted(4, 2, -4, -2), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)



def _theme(options: ExportOptions) -> Theme:
    return LIGHT if options.light else DARK


def _render(scene, device, target: QRectF, source: QRectF) -> None:
    painter = QPainter(device)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        scene.render(painter, target, source)
    finally:
        painter.end()


def _style(copy: CampaignPlotCanvas, options: ExportOptions, device_dpi: float) -> None:
    """Theme, fonts and line widths for the file. Pens stay cosmetic (a data-space pen would be
    stretched by the axis scales), so their widths are given in device pixels."""
    theme = _theme(options)
    font_px = pt_to_px(options.font_pt, SCENE_DPI)
    font = QFont()
    css_px = max(round(font_px), 1)  # Qt rich text only takes whole pixels
    font.setPixelSize(css_px)
    axis_pen = pg.mkPen(theme.foreground, width=pt_to_px(0.5, device_dpi))
    css = {"color": theme.foreground, "font-size": f"{css_px}px"}
    copy.graphics_layout.setBackground(theme.background)
    marker_scale = device_dpi / SCENE_DPI
    for p in copy.all_plots():
        for name in ("left", "bottom"):
            axis = p.getAxis(name)
            axis.setPen(axis_pen)
            axis.setTextPen(theme.foreground)
            axis.setStyle(tickFont=font)
            if axis.labelText or axis.labelUnits:
                axis.setLabel(text=axis.labelText, units=axis.labelUnits or None, **css)
        p.showGrid(x=True, y=True, alpha=theme.grid_alpha)
        if p.titleLabel.text:
            p.titleLabel.setText(p.titleLabel.text, size=f"{css_px}px", color=theme.title)
        if p.legend is not None:
            p.legend.setBrush(pg.mkBrush(*theme.legend_brush))
            for _, label in p.legend.items:  # setLabelTextSize only affects entries added later
                label.setText(label.text, size=f"{css_px}px", color=theme.foreground)
    for key, curve in copy.curves.items():
        curve.lod_scale = device_dpi / SCENE_DPI
        curve.set_pen(curve_pen(copy.colors[key], width=pt_to_px(options.line_width_pt, device_dpi)))
        if curve.markers is not None:
            curve.markers.lod_scale = curve.lod_scale
            curve.markers.setSize(POINT_SIZE_PX * marker_scale)
    for markers in copy.edge_markers.values():
        markers.setSize(EDGE_MARKER_SIZE_PX * marker_scale)
    if copy.coverage_item is not None:
        copy.coverage_item.hatch_color = theme.hatch
    # equal left axes keep the linked time axes aligned: tick labels plus the rotated axis title
    needed = [left_axis_width(p, [TICK_EXAMPLE]) + round(1.6 * font_px) for p in copy.plot_items.values()]
    if copy.coverage_plot is not None:
        needed.append(left_axis_width(copy.coverage_plot, [lane_label(c) for c in copy.coverage]))
    width = max(needed + [LEFT_AXIS_MIN_PX])
    for p in copy.all_plots():
        p.getAxis("left").setWidth(width)
