"""Bericht ohne fest verdrahtete Namen: Legenden im Grafik-Layout, Deckblatt-Kacheln und
Matrix-Zeilen als Vorlagen.

Die Legenden der Ansichten `xy`, `aligned` und `bars` waren Qt-Widgets neben dem Plot und fehlten
deshalb im Bericht und im Einzelexport (die Bilanz-Seite der SA-Kampagne hatte keine Legende).
Das Deckblatt zählte alle Schritte als „Dosierstufen“ und suchte T50/Ea über Namen, die
Matrix-Zeilen nahmen die Spanne aus einer Spalte `T_bett` mit festem „°C“. Jetzt ist beides
Konfiguration (`ReportViewSpec.tiles`, `MatrixViewSpec.row_label`) über `render_template` mit
Aggregaten `{formel|min}`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pyqtgraph as pg
import pytest
from pydantic import ValidationError
from PySide6.QtWidgets import QApplication

from datualizer_core.calc import ExprError
from datualizer_core.calc.analyses import CrossingAnalysis
from datualizer_core.campaign import (
    AlignedViewSpec,
    BarsViewSpec,
    Evaluation,
    MatrixViewSpec,
    ReportViewSpec,
    XYViewSpec,
)
from datualizer_core.campaign.views import ReportTile, check_template, render_template, report_tiles
from datualizer_gui.components.plot_export import ExportOptions
from datualizer_gui.results_window import AlignedViewWidget, BarsViewWidget, MatrixViewWidget, XYViewWidget


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


def _steps() -> pl.DataFrame:
    t = [datetime(2026, 10, 12, 10, tzinfo=timezone.utc) + timedelta(minutes=m) for m in (0, 30, 35, 65, 70, 100, 105, 135)]
    return pl.DataFrame({
        "step_id": ["s1", "s2", "s3", "s4"],
        "step_start": t[0::2],
        "step_end": t[1::2],
        "klasse": ["dosierung", "null", "dosierung", "dosierung"],
        "tag": ["a", "a", "b", "b"],
        "T_klasse": [175.0, 175.0, 200.0, 200.0],
        "T_bett": [174.5, 176.0, 199.8, 203.1],
        "v_mfc": [10.0, 0.0, 20.0, 10.0],
        "X": [0.38, None, 0.57, 0.92],
        "u_X": [0.04, None, 0.05, 0.03],
        "b_X__S": [0.02, None, 0.015, 0.005],
        "b_X__f": [0.025, None, 0.035, 0.02],
        "stationaer": [True, False, True, False],
    })


# ------------------------------------------------------------------ templates with aggregates
class TestAggregates:
    def test_min_max_n_over_all_rows(self) -> None:
        t = _steps()
        assert render_template(t, "({T_bett|min:.0f}–{T_bett|max:.0f})", reduce=True) == "(174–203)"
        assert render_template(t, "{step_id|n} Schritte, {X|n} mit X") == ["4 Schritte, 3 mit X"] * 4
        assert render_template(t, "{X|sum:.2f} {X|median:.2f} {X|mean:.2f}", reduce=True) == "1.87 0.57 0.62"
        assert render_template(t, "{tag|min}–{tag|max}", reduce=True) == "a–b"  # texts sorted
        # without aggregate, a row keeps its own value; with one, every row gets the same
        assert render_template(t, "{T_bett:.0f}/{T_bett|max:.0f}")[0] == "174/203"

    def test_empty_table(self) -> None:
        t = _steps().clear()
        assert render_template(t, "{step_id|n}: {T_bett|max:.0f}", reduce=True) == "0: –"

    def test_unknown_aggregate_fails_on_load(self) -> None:
        with pytest.raises(ValueError, match="Unknown aggregate 'maximum'"):
            check_template("{T_bett|maximum}")
        with pytest.raises(ValidationError, match="Unknown aggregate"):
            MatrixViewSpec(name="m", rows="T_klasse", cols="v_mfc", value="X", row_label="{T_bett|top}")


# ------------------------------------------------------------------ cover tiles
class TestTiles:
    def test_counts_and_analysis_results(self) -> None:
        crossing = pl.DataFrame({"v_mfc": [10.0, 20.0], "n": [6, 5], "T50": ["178 °C", "203 °C"]})
        fit = pl.DataFrame({"n": [8], "Ea_kJ_mol": [21.4], "u_Ea_kJ_mol": [4.2]})
        tiles = [
            ReportTile(label="Dosierstufen", value="{step_id|n}", filter="klasse == 'dosierung'"),
            ReportTile(label="stationär", value="{step_id|n}", filter="klasse == 'dosierung' and stationaer"),
            ReportTile(label="T50 bei 10 Nl/min", value="{T50}", analysis="T50", filter="v_mfc == 10"),
            ReportTile(label="Ea", value="{Ea_kJ_mol:.0f} ± {u_Ea_kJ_mol:.0f} kJ/mol", analysis="fit"),
        ]
        # analyses give result objects with a `.df` table or plain tables
        res = type("Res", (), {"df": crossing})()
        assert report_tiles(_steps(), {"T50": res, "fit": fit}, tiles) == [
            ("Dosierstufen", "3"), ("stationär", "2"), ("T50 bei 10 Nl/min", "178 °C"), ("Ea", "21 ± 4 kJ/mol"),
        ]

    def test_missing_analysis_or_name_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="Tile 'T50'.*no result of analysis 'T50'"):
            report_tiles(_steps(), {}, [ReportTile(label="T50", value="{T50}", analysis="T50")])
        with pytest.raises(ValueError, match="Tile 'n'") as err:
            report_tiles(_steps(), {}, [ReportTile(label="n", value="{schritt|n}")])
        assert isinstance(err.value.__cause__, ExprError)

    def test_tile_on_unknown_analysis_fails_on_load(self) -> None:
        tile = ReportTile(label="T50", value="{T50}", analysis="T5O")
        with pytest.raises(ValidationError, match="unknown analysis 'T5O'"):
            Evaluation(analyses=[CrossingAnalysis(name="T50", x="T_bett", y="X", level=0.5)],
                       views=[ReportViewSpec(name="bericht", tiles=[tile])])


# ------------------------------------------------------------------ legends in the graphics layout
def _legend(w) -> pg.LegendItem:
    items = [it for it in w.glw.ci.items if isinstance(it, pg.LegendItem)]
    assert len(items) == 1
    return items[0]


def _texts(legend: pg.LegendItem) -> list[str]:
    return [label.text for _, label in legend.items]


def _exports(w, tmp_path: Path) -> None:
    """The export renders the scene of `glw`: the legend is in it."""
    assert _legend(w).scene() is w.glw.scene()
    out = w.export(tmp_path / f"{w.spec.name}.png", ExportOptions(width_mm=170, height_mm=105, dpi=100))
    assert out.stat().st_size > 1000


def test_xy_legend_in_layout_right_of_facets(qapp, tmp_path: Path) -> None:
    spec = XYViewSpec(name="bilanz", x="T_bett", y="X", color_by="v_mfc", color_mode="ordinal", marker_by="tag",
                      filled="stationaer", facet_by="T_klasse", facet_cols=3)
    w = XYViewWidget(spec)
    w.resize(900, 500)
    w.set_data(_steps())
    legend = _legend(w)
    assert w.glw.ci.layout.itemAt(0, 2) is legend  # two facets: legend in the third column
    assert _texts(legend) == ["<b>v_mfc</b>", "10", "20", "<b>tag</b>", "a", "b",  # the drawn points only
                              "<b>stationaer</b>", "erfüllt", "nicht erfüllt"]
    assert not hasattr(w, "legend_panel")
    _exports(w, tmp_path)
    w.set_data(_steps())  # redrawn: still one legend
    _legend(w)
    w.deleteLater()
    qapp.processEvents()


def test_xy_without_groups_has_no_legend(qapp) -> None:
    w = XYViewWidget(XYViewSpec(name="xy", x="T_bett", y="X"))
    w.set_data(_steps())
    assert not [it for it in w.glw.ci.items if isinstance(it, pg.LegendItem)]
    w.deleteLater()
    qapp.processEvents()


def test_aligned_and_bars_legends_in_layout(qapp, tmp_path: Path) -> None:
    aligned = AlignedViewWidget(AlignedViewSpec(name="durchbruch", expr="X", t0="step_start", color_by="v_mfc",
                                                filled="X <= 0.6"))
    aligned.set_data(_steps())
    # the condition is the heading, escaped: '<' would open a tag in the rich-text label
    assert _texts(_legend(aligned)) == ["<b>v_mfc</b>", "0", "10", "20", "<b>X &lt;= 0.6</b>", "erfüllt", "nicht erfüllt"]
    _exports(aligned, tmp_path)

    bars = BarsViewWidget(BarsViewSpec(name="unsicherheit", quantity="X", sort_by=["T_bett"], filter="X > 0"))
    bars.set_data(_steps())
    legend = _legend(bars)
    assert bars.glw.ci.layout.itemAt(1, 1) is legend  # beside the shares
    assert _texts(legend) == ["<b>Quelle</b>", "f", "S"]  # top layer first
    _exports(bars, tmp_path)
    for w in (aligned, bars):
        w.deleteLater()
    qapp.processEvents()


def _drawn_labels(axis: pg.AxisItem) -> list[str]:
    """Tick labels the axis draws on its first paint (as `AxisItem.paint` does, into a QPicture)."""
    from PySide6.QtGui import QPainter, QPicture

    picture = QPicture()
    painter = QPainter(picture)
    try:
        return [text for _, _, text in axis.generateDrawSpecs(painter)[2]]
    finally:
        painter.end()


def test_matrix_row_labels_fit_on_the_first_paint(qapp) -> None:
    """A one-shot render (export, report page) must show the labels: pyqtgraph drops labels wider than
    the axis and widens it only after a paint, so the matrix sizes its left axis up front."""
    from PySide6.QtCore import QRectF

    spec = MatrixViewSpec(name="m", rows="T_klasse", cols="v_mfc", value="X", status="stationaer",
                          filter="klasse == 'dosierung'", row_label="Klasse {T_klasse:.0f} °C\n({T_bett|min:.1f} bis {T_bett|max:.1f} °C)")
    w = MatrixViewWidget(spec)
    w.set_data(_steps())
    w.glw.ci.setGeometry(QRectF(0, 0, 643, 397))  # 170 x 105 mm
    for _ in range(3):
        w.glw.ci.layout.activate()
        qapp.processEvents()
    left = w.glw.ci.getItem(0, 0).getAxis("left")
    assert _drawn_labels(left) == ["Klasse 200 °C\n(199.8 bis 203.1 °C)", "Klasse 175 °C\n(174.5 bis 174.5 °C)"]
    w.deleteLater()
    qapp.processEvents()


def test_matrix_rows_without_label_show_the_value(qapp) -> None:
    w = MatrixViewWidget(MatrixViewSpec(name="m", rows="T_klasse", cols="v_mfc", value="X", status="stationaer",
                                        filter="klasse == 'dosierung'"))
    w.set_data(_steps())
    ticks = dict(w.glw.ci.getItem(0, 0).getAxis("left")._tickLevels[0])
    assert ticks == {0: "200", 1: "175"}  # no unit, no T_bett range made up
    w.deleteLater()
    qapp.processEvents()
