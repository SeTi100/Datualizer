"""Berichtsseiten (PR 11): Tagesverlauf als Ansicht `timeline`, Vorlagen-Texte, Matrix-Beschriftungen.

Der Tagesverlauf zeichnet echte FID-Daten (Ausschnitt 12.10., mit der 11-s-Lücke der Datei) über
Master-Zeit, Schritte als Bänder, Marken pro Schritt aus der Ergebnistabelle, eine Seite je Wert
von `page_by`. Die Matrix beschriftet jede Zeile auch im 105-mm-Export, die heisseste Zeile steht
oben, und der Untertext ist eine Vorlage wie im Skript (c_ein je Zelle).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyqtgraph as pg
import pytest
from pydantic import ValidationError
from PySide6.QtWidgets import QApplication

from datualizer_core.calc import ExprError
from datualizer_core.campaign import (
    Campaign,
    CampaignSession,
    Evaluation,
    MatrixViewSpec,
    Measure,
    NamedWindow,
    ReportViewSpec,
    StepClass,
    TimelineMarks,
    TimelinePanel,
    TimelineTrace,
    TimelineViewSpec,
    WindowRule,
    XYViewSpec,
    master_datetime,
)
from datualizer_core.campaign.views import render_template, template_fields
from datualizer_gui.components.curves import FastCurve
from datualizer_gui.components.interval_bands import ColorBandsItem
from datualizer_gui.components.plot_export import ExportOptions, export_report, report_pages
from datualizer_gui.results_window import MatrixViewWidget, ResultsWindow, TimelineViewWidget, report_views

FID = Path(__file__).parent / "fixtures" / "sa_12_10" / "fid_ifid_12_10_extract.dat"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


# ------------------------------------------------------------------ templates
class TestTemplates:
    def test_fields_and_rows(self) -> None:
        t = pl.DataFrame({"c_in": [147.34, 152.0], "tag": ["a", "b"], "X": [0.5, None]})
        assert template_fields("{c_in:.0f} mg/Nm³ · {tag}") == [("c_in", ".0f"), ("tag", "")]
        assert render_template(t, "{c_in:.0f} mg/Nm³ ({tag})") == ["147 mg/Nm³ (a)", "152 mg/Nm³ (b)"]
        assert render_template(t, "{c_in:.0f}", reduce=True) == "150"  # averaged over the rows
        assert render_template(t, "{tag}", reduce=True) == "a, b"
        assert render_template(t, "X = {X:.2f}") == ["X = 0.50", "X = –"]  # no value: a dash, not 0
        assert render_template(t, "{{fest}} {c_in / 2:.1f}")[0] == "{fest} 73.7"  # formulas and literal braces

    def test_wrong_templates_fail_loudly(self) -> None:
        with pytest.raises(ValidationError, match="not a number format"):
            MatrixViewSpec(name="m", rows="a", cols="b", value="X", subtext="{c_in:.0q}")
        with pytest.raises(ValidationError):
            MatrixViewSpec(name="m", rows="a", cols="b", value="X", subtext="{c_in +}")
        with pytest.raises(ExprError, match="c_ein"):
            render_template(pl.DataFrame({"c_in": [1.0]}), "{c_ein}")


# ------------------------------------------------------------------ matrix
def _matrix_rows() -> pl.DataFrame:
    rows = []
    for i, t in enumerate([100.0, 150.0, 175.0, 200.0, 225.0, 250.0]):
        for v in (10.0, 20.0):
            rows.append({"step_id": f"s{i}{v:g}", "T_klasse": t, "v_mfc": v, "T_bett": t + 1, "X": t / 300,
                         "c_in": 100.0 + v, "stationaer": True})
    rows.append({"step_id": "extra", "T_klasse": 250.0, "v_mfc": 10.0, "T_bett": 251.0, "X": 0.8, "c_in": 130.0,
                 "stationaer": True})
    return pl.DataFrame(rows)


def test_matrix_labels_every_row_with_hottest_on_top(qapp, tmp_path: Path) -> None:
    spec = MatrixViewSpec(name="m", rows="T_klasse", cols="v_mfc", value="X", status="stationaer",
                          subtext="{c_in:.0f} mg/Nm³", row_label="{T_klasse:.0f} °C\n({T_bett|min:.0f}–{T_bett|max:.0f})")
    w = MatrixViewWidget(spec)
    w.resize(900, 600)
    w.set_data(_matrix_rows())
    p = w.glw.ci.getItem(0, 0)
    assert p.getViewBox().yInverted()
    left = p.getAxis("left")
    ticks = dict(left._tickLevels[0])
    assert ticks[0].startswith("250 °C") and ticks[5].startswith("100 °C")  # first row = top
    assert ticks[0] == "250 °C\n(251–251)" and ticks[5] == "100 °C\n(101–101)"  # row_label over the row's steps
    # two-line labels of six rows fill more than 80 % of a 105 mm axis: pyqtgraph would drop them all
    assert left.style["textFillLimits"] == [(0, 100.0)]
    texts = [it.textItem.toPlainText() for it in w.glw.scene().items() if isinstance(it, pg.TextItem)]
    assert any(t.endswith("120 mg/Nm³ · 2 Stufen") for t in texts)  # 250 °C, 10 Nl/min: mean of 110 and 130
    assert any(t.endswith("120 mg/Nm³") and "Stufen" not in t for t in texts)
    out = w.export(tmp_path / "m.png", ExportOptions(width_mm=170, height_mm=105, dpi=100))
    assert out.stat().st_size > 1000
    w.deleteLater()
    qapp.processEvents()


# ------------------------------------------------------------------ timeline
CLASSES = [StepClass(name="dosierung", when="v_mfc > 0"), StepClass(name="null", when="v_mfc == 0")]


@pytest.fixture(scope="module")
def session() -> CampaignSession:
    """FID extract with four steps by hand on two 'days' (a, b): dose, zero, dose, zero."""
    s = CampaignSession(Campaign(name="Timeline"))
    s.add_source(FID, source_id="fid")
    t0 = s.track("fid", "conc").x[0]
    zone = s.campaign.timezone
    for (a, b), v, tag in zip([(0, 60), (60, 150), (150, 200), (200, 290)], (10.0, 0.0, 20.0, 0.0), "aabb"):
        s.add_interval(master_datetime(t0 + a, zone), master_datetime(t0 + b, zone), params={"v_mfc": v, "tag": tag})
    s.set_evaluation(Evaluation(
        step_classes=CLASSES,
        windows=[NamedWindow(name="blank", rule=WindowRule(length_s=30), classes=["null"])],
        measures=[Measure(alias="S", source="fid", channel="conc"),
                  Measure(alias="S_b", source="fid", channel="conc", window="blank")],
        views=[TIMELINE, ReportViewSpec(name="bericht", views=["verlauf"])],
    ))
    return s


TIMELINE = TimelineViewSpec(
    name="verlauf", title="Messtag", page_by="tag", page_title="Messtag {tag}",
    bands="klasse == 'dosierung'", band_color_by="v_mfc", band_label="{v_mfc:g}", band_legend="Nl/min",
    panels=[
        TimelinePanel(traces=[TimelineTrace(channel="conc", label="FID")], y_robust=99.7, height=2,
                      marks=[TimelineMarks(y="S_b", at="window", window="blank", filter="klasse == 'null'",
                                           label="Blindwert")]),
        TimelinePanel(marks=[TimelineMarks(y="S", filter="klasse == 'dosierung'", filled="S > 0", label="S")],
                      y_label="S"),
    ],
)


def test_timeline_one_page_per_day(qapp, session: CampaignSession) -> None:
    df = session.results()
    w = TimelineViewWidget(TIMELINE, session=session)
    w.set_data(df)
    assert w.pages() == ["a", "b"] and w.combo_page.count() == 2
    assert w.page_heading() == ("Messtag a", None)
    assert len(w.plots) == 2
    first = w.plots[0]
    curves = [it for it in first.items if isinstance(it, FastCurve)]
    assert len(curves) == 1
    x, _ = curves[0].getOriginalDataset()
    starts = df["step_start"].dt.epoch("us").to_numpy() / 1e6
    ends = df["step_end"].dt.epoch("us").to_numpy() / 1e6
    assert x[0] >= starts[0] - 1 and x[-1] <= ends[1] + 1  # only the page's time range
    bands = [it for it in first.items if isinstance(it, ColorBandsItem)]
    assert len(bands) == 1 and len(bands[0].bands) == 1 and bands[0].bands[0][3] == "10"
    # the blank segment spans the named window of the zero phase
    segs = [it for it in first.items if isinstance(it, pg.PlotCurveItem) and not isinstance(it, FastCurve)]
    blank = session.step_window(df["step_id"][1], "blank").window
    assert len(segs) == 1 and segs[0].getData()[0].tolist() == pytest.approx(list(blank))
    dots = [it for it in w.plots[1].items if isinstance(it, pg.ScatterPlotItem)]
    assert dots and dots[0].points()[0].pos().x() == pytest.approx(ends[0])  # at the step end

    w.set_page("b")
    assert w.page_heading()[0] == "Messtag b"
    bands = [it for it in w.plots[0].items if isinstance(it, ColorBandsItem)]
    assert bands[0].bands[0][3] == "20"
    # colours over all pages: the 20 Nl/min band of page b is not the colour of 10 Nl/min on page a
    w.set_page("a")
    first_color = [it for it in w.plots[0].items if isinstance(it, ColorBandsItem)][0].bands[0][2].name()
    assert bands[0].bands[0][2].name() != first_color
    w.deleteLater()
    qapp.processEvents()


def _record_pages(monkeypatch) -> list[str]:
    """Titles of the view pages a report draws (PDF text needs fonts, which CI may not have)."""
    import datualizer_gui.components.plot_export as pe

    seen: list[str] = []
    draw = pe._draw_view_page

    def spy(painter, v_spec, widget, *args, **kwargs):
        seen.append(widget.page_heading()[0] if hasattr(widget, "page_heading") else v_spec.title or v_spec.name)
        return draw(painter, v_spec, widget, *args, **kwargs)

    monkeypatch.setattr(pe, "_draw_view_page", spy)
    return seen


def test_report_has_a_page_per_day(qapp, session: CampaignSession, tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtPdf import QPdfDocument

    seen = _record_pages(monkeypatch)
    win = ResultsWindow(session=session)
    views = report_views(session.campaign.evaluation.views, session.campaign.evaluation.views[-1])
    assert [v.name for v in views] == ["verlauf"]
    assert [p for _, _, p in report_pages(views, win._view_widgets)] == ["a", "b"]
    assert win._view_widgets["bericht"].table.rowCount() == 3  # cover + two days
    out = export_report(session, views, win._view_widgets, tmp_path / "r.pdf")
    doc = QPdfDocument()
    doc.load(str(out))
    assert doc.pageCount() == 3 and seen == ["Messtag a", "Messtag b"]
    assert win._view_widgets["verlauf"].page_heading()[0] == "Messtag a"  # back on the first page
    doc.close()
    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_campaign_window_report_follows_the_report_view(
    qapp, session: CampaignSession, tmp_path: Path, monkeypatch
) -> None:
    from PySide6.QtPdf import QPdfDocument

    import datualizer_gui.components.plot_export as pe
    from datualizer_gui.campaign_window import CampaignWindow

    ev = session.campaign.evaluation
    session.campaign.evaluation = ev.model_copy(update={"views": [
        XYViewSpec(name="xy", title="Erst XY", x="S", y="S"), TIMELINE,
        ReportViewSpec(name="bericht", title="Mein Bericht", views=["verlauf", "xy"]),
    ]})
    seen = _record_pages(monkeypatch)
    covers: list[str] = []
    cover = pe._draw_cover_page
    monkeypatch.setattr(pe, "_draw_cover_page", lambda p, s, w, h, title="", sub=None, **kw: (covers.append(title), cover(p, s, w, h, title, sub, **kw)))
    try:
        win = CampaignWindow(session)
        out = win.export_report(tmp_path / "k.pdf")
        doc = QPdfDocument()
        doc.load(str(out))
        assert doc.pageCount() == 4
        doc.close()
        assert covers == ["Mein Bericht"] and seen == ["Messtag a", "Messtag b", "Erst XY"]  # the report's order
        win.close()
        win.deleteLater()
        qapp.processEvents()
    finally:
        session.campaign.evaluation = ev


def test_timeline_marks_need_a_known_window() -> None:
    bad = TIMELINE.model_copy(update={"panels": [TimelinePanel(marks=[TimelineMarks(y="S", at="window", window="weg")])]})
    with pytest.raises(ValidationError, match="unknown window 'weg'"):
        Evaluation(step_classes=CLASSES, views=[bad])
