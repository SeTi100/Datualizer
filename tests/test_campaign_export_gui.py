"""Campaign window: plot export as PNG, SVG, PDF for a report (light theme, size, font, line width, dpi)."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import pyqtgraph as pg
import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from datualizer_core.campaign import ChannelDisplay
from datualizer_core.schema import PlotStyle
from datualizer_gui.campaign_window import CampaignWindow
from datualizer_gui.components.export_dialog import ExportDialog, options_from_json, options_to_json
from datualizer_gui.components.plot_export import (
    ExportFormat,
    ExportOptions,
    _style,
    export_plot,
    mm_to_scene,
)
from datualizer_gui.settings import EXPORT_OPTIONS, get_str

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
FID_DAT = SA / "fid_ifid_12_10_extract.dat"
PICO = SA / "picolog_12_10_extract.csv"
FID_KEY = f"{FID_DAT.stem}/conc"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture()
def window(qapp: QApplication, tmp_path: Path) -> CampaignWindow:
    pico = tmp_path / PICO.name
    shutil.copyfile(PICO, pico)
    saved = datetime(2026, 10, 12, 16, 0, 21, tzinfo=timezone.utc).timestamp()
    os.utime(pico, (saved, saved))
    win = CampaignWindow()
    win.add_sources([FID_DAT, pico])
    win.set_coverage_visible(True)
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()
    yield win
    win.close()


def test_png_has_the_size_and_resolution_asked_for(window: CampaignWindow, tmp_path: Path) -> None:
    out = window.export_plot(tmp_path / "plot.png", ExportOptions(width_mm=160, height_mm=100, dpi=200))
    image = QImage(str(out))
    assert (image.width(), image.height()) == (round(160 / 25.4 * 200), round(100 / 25.4 * 200))
    assert round(image.dotsPerMeterX() * 0.0254) == 200
    assert QColor(image.pixel(2, 2)).name() == "#ffffff"  # light theme for print


def test_dark_theme_on_request(window: CampaignWindow, tmp_path: Path) -> None:
    out = window.export_plot(tmp_path / "dark.png", ExportOptions(dpi=96, light=False))
    assert QColor(QImage(str(out)).pixel(2, 2)).name() == "#18181c"


def test_svg_and_pdf_are_vector_files(window: CampaignWindow, tmp_path: Path) -> None:
    svg = window.export_plot(tmp_path / "plot.svg", ExportOptions(width_mm=170, height_mm=120)).read_text(encoding="utf-8")
    assert "<svg" in svg and "<path" in svg and "<image" not in svg  # no item cached as a bitmap
    assert "conc" in svg  # axis titles as text
    pdf = window.export_plot(tmp_path / "plot.pdf", ExportOptions(width_mm=170, height_mm=120)).read_bytes()
    assert pdf.startswith(b"%PDF")
    box = pdf.split(b"/MediaBox [")[1].split(b"]")[0].split()
    assert [round(float(v)) for v in box[2:]] == [round(170 / 25.4 * 72), round(120 / 25.4 * 72)]  # page in pt
    assert b"/Subtype /Image" not in pdf


def curve_pixels(path: Path, color: str) -> int:
    image = QImage(str(path))
    target = QColor(color)
    count = 0
    for y in range(0, image.height(), 2):
        for x in range(0, image.width(), 2):
            c = QColor(image.pixel(x, y))
            count += abs(c.red() - target.red()) + abs(c.green() - target.green()) + abs(c.blue() - target.blue()) < 60
    return count


def test_line_width_only_in_the_file(window: CampaignWindow, tmp_path: Path) -> None:
    color = window.canvas.colors[FID_KEY]
    thin = window.export_plot(tmp_path / "thin.png", ExportOptions(dpi=150, line_width_pt=0.3, height_mm=80))
    thick = window.export_plot(tmp_path / "thick.png", ExportOptions(dpi=150, line_width_pt=2.0, height_mm=80))
    assert curve_pixels(thick, color) > 2 * curve_pixels(thin, color)
    assert pg.mkPen(window.canvas.curves[FID_KEY].opts["pen"]).widthF() == 1  # screen stays at 1 px


def test_export_shows_the_view_and_keeps_the_data(qapp: QApplication, window: CampaignWindow) -> None:
    x = window.canvas.curves[FID_KEY].getOriginalDataset()[0]
    window.canvas.set_view_range(x[20], x[120])
    y_screen = window.canvas.plot_items[FID_KEY].getViewBox().viewRange()[1]
    before = window.session.datasets[FID_DAT.stem].df.clone()
    copy = window.canvas.export_copy()
    assert copy.view_x_range() == pytest.approx((x[20], x[120]))
    assert copy.plot_items[FID_KEY].getViewBox().viewRange()[1] == pytest.approx(y_screen)
    assert list(copy.plot_items) == list(window.canvas.plot_items)
    assert window.session.datasets[FID_DAT.stem].df.equals(before)
    copy.deleteLater()


def test_all_left_axes_share_one_width(window: CampaignWindow) -> None:
    """Linked time axes line up only with equal left axes, also with print fonts."""
    copy = window.canvas.export_copy()
    _style(copy, ExportOptions(font_pt=11), 300)
    widths = {p.getAxis("left").fixedWidth for p in copy.all_plots()}
    assert len(widths) == 1
    copy.deleteLater()


def test_points_and_markers_scale_with_the_resolution(qapp: QApplication, window: CampaignWindow) -> None:
    window.sources_panel.apply_display(FID_DAT.stem, "conc", ChannelDisplay(style=PlotStyle.POINTS))
    copy = window.canvas.export_copy()
    _style(copy, ExportOptions(), 300)
    curve = copy.curves[FID_KEY]
    assert curve.markers.opts["size"] == pytest.approx(4 * 300 / 96)
    assert curve.lod_scale == pytest.approx(300 / 96)
    copy.deleteLater()


def test_auto_height_follows_the_subplots() -> None:
    opts = ExportOptions(subplot_height_mm=40, lane_height_mm=5)
    assert opts.height_for(3, 0) == 120
    assert opts.height_for(2, 2) == 80 + 10 + 12
    assert ExportOptions(height_mm=99).height_for(5, 3) == 99
    assert mm_to_scene(25.4) == 96


def test_formats() -> None:
    assert ExportFormat.from_path("a/b.PDF") is ExportFormat.PDF
    with pytest.raises(ValueError):
        ExportFormat.from_path("plot.jpg")


def test_nothing_to_export(qapp: QApplication, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export_plot(CampaignWindow().canvas, tmp_path / "x.png")


def test_options_are_remembered(window: CampaignWindow, tmp_path: Path) -> None:
    opts = ExportOptions(width_mm=120, dpi=150, font_pt=10, line_width_pt=1.0, light=True)
    window.export_plot(tmp_path / "p.png", opts)
    assert options_from_json(get_str(EXPORT_OPTIONS)) == opts
    assert options_from_json("{broken") == ExportOptions()
    assert options_from_json('{"width_mm": 99, "unknown": 1}').width_mm == 99
    assert options_from_json(options_to_json(opts)) == opts


def test_dialog(window: CampaignWindow) -> None:
    dialog = ExportDialog(ExportOptions(), ExportFormat.SVG, window)
    assert not dialog.dpi_spin.isEnabled()  # vector file
    dialog.width_spin.setValue(80)
    dialog.line_spin.setValue(1.5)
    dialog.light_check.setChecked(False)
    opts = dialog.options()
    assert (opts.width_mm, opts.line_width_pt, opts.light) == (80, 1.5, False)
    assert ExportDialog(ExportOptions(), ExportFormat.PNG, window).dpi_spin.isEnabled()


def test_campaign_menu_has_export(window: CampaignWindow) -> None:
    menus = {a.text(): a.menu() for a in window.menuBar().actions()}
    assert "&Export Plot..." in [a.text() for a in menus["&Campaign"].actions()]


def test_legend_uses_the_print_font(qapp: QApplication, window: CampaignWindow) -> None:
    from datualizer_gui.components.plot_groups import GroupBy

    window.sources_panel.set_checked(PICO.stem, "kanal_8_heizung")
    window.sources_panel.set_checked(PICO.stem, "kanal_2_kat_unten")
    window.set_grouping(group_by=GroupBy.UNIT)
    copy = window.canvas.export_copy()
    _style(copy, ExportOptions(font_pt=12), 300)
    labels = [label for _, label in copy.plot_items["unit:C"].legend.items]
    assert len(labels) == 3 and all(label.opts["size"] == "16px" for label in labels)  # 12 pt at 96 px/in
    copy.deleteLater()
