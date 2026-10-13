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

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QMarginsF, QRectF, QSize, QSizeF
from PySide6.QtGui import QColor, QFont, QImage, QPageSize, QPainter, QPdfWriter
from PySide6.QtWidgets import QApplication, QGraphicsItem

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
