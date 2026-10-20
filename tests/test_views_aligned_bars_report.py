"""Tests for aligned, bars, report views and multi-page PDF report export (PR 8)."""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import polars as pl
import pyqtgraph as pg
import pytest
from PySide6.QtWidgets import QApplication

from datualizer_core.calc.uncertain import Constant
from datualizer_core.campaign import (
    AlignedViewSpec,
    BarsViewSpec,
    Campaign,
    CampaignSession,
    Evaluation,
    MatrixViewSpec,
    RefLine,
    ReportViewSpec,
    TableViewSpec,
    ViewSpec,
    XYViewSpec,
)
from datualizer_gui.components.plot_export import ExportOptions, export_report
from datualizer_gui.results_window import (
    AlignedViewWidget,
    BarsViewWidget,
    ReportViewWidget,
    ResultsWindow,
)


@pytest.fixture
def sample_results_df() -> pl.DataFrame:
    t0_1 = datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc)
    t1_1 = datetime(2026, 10, 12, 10, 30, tzinfo=timezone.utc)
    t0_2 = datetime(2026, 10, 12, 10, 35, tzinfo=timezone.utc)
    t1_2 = datetime(2026, 10, 12, 11, 5, tzinfo=timezone.utc)
    t0_3 = datetime(2026, 10, 12, 11, 10, tzinfo=timezone.utc)
    t1_3 = datetime(2026, 10, 12, 11, 40, tzinfo=timezone.utc)

    return pl.DataFrame({
        "step_id": ["s1", "s2", "s3"],
        "step_start": [t0_1, t0_2, t0_3],
        "step_end": [t1_1, t1_2, t1_3],
        "tag": ["12.10.", "12.10.", "12.10."],
        "sequenz": [1, 1, 2],
        "T_klasse": [175.0, 175.0, 200.0],
        "T_bett": [174.5, 175.2, 199.8],
        "V_MFC": [10.0, 20.0, 10.0],
        "fid_mgC": [25.0, 18.0, 5.0],
        "B": [2.0, 2.0, 2.0],
        "f": [2.701, 2.701, 2.701],
        "c_in": [100.0, 100.0, 100.0],
        "X": [0.38, 0.57, 0.92],
        "u_X": [0.04, 0.05, 0.03],
        "b_X__fid_mgC": [0.02, 0.015, 0.005],
        "b_X__B": [0.01, 0.01, 0.01],
        "b_X__f": [0.025, 0.035, 0.02],
        "b_X__c_in": [0.02, 0.03, 0.015],
        "stationaer": [True, True, False],
    })


class TestViewSpecModels:
    def test_aligned_view_spec(self):
        spec = AlignedViewSpec(
            name="durchbruch",
            title="Durchbruchskurve",
            expr="1 - (fid_mgC - B) * f / c_in",
            t0="sequence_start",
            sequence_col="sequenz",
            facet_by="T_klasse",
            color_by="V_MFC",
            marker_by="tag",
            filled="stationaer",
            rolling_s=10.0,
            ref_lines=[RefLine(axis="y", value=1.0)],
        )
        assert spec.kind == "aligned"
        assert spec.name == "durchbruch"
        assert spec.rolling_s == 10.0
        assert len(spec.ref_lines) == 1

        # JSON round-trip
        data = spec.model_dump()
        restored = AlignedViewSpec.model_validate(data)
        assert restored == spec

    def test_bars_view_spec(self):
        spec = BarsViewSpec(
            name="unsicherheit",
            title="Unsicherheitsbudget",
            quantity="X",
            sort_by=["T_bett"],
            filter="stationaer",
        )
        assert spec.kind == "bars"
        assert spec.quantity == "X"
        assert spec.u_total is None

        data = spec.model_dump()
        restored = BarsViewSpec.model_validate(data)
        assert restored == spec

    def test_report_view_spec(self):
        spec = ReportViewSpec(
            name="bericht",
            title="Kampagnenbericht",
            views=["durchbruch", "unsicherheit"],
            cover=True,
            orientation="landscape",
        )
        assert spec.kind == "report"
        assert spec.cover is True
        assert len(spec.views) == 2

        data = spec.model_dump()
        restored = ReportViewSpec.model_validate(data)
        assert restored == spec

    def test_view_spec_discriminated_union(self):
        ev = Evaluation(
            views=[
                TableViewSpec(name="tab"),
                XYViewSpec(name="plot", x="T_bett", y="X"),
                MatrixViewSpec(name="mat", rows="T_klasse", cols="V_MFC", value="X"),
                AlignedViewSpec(name="align", expr="fid_mgC"),
                BarsViewSpec(name="bars", quantity="X"),
                ReportViewSpec(name="rep"),
            ]
        )
        assert len(ev.views) == 6
        assert ev.views[3].kind == "aligned"
        assert ev.views[4].kind == "bars"
        assert ev.views[5].kind == "report"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


class TestAlignedViewWidget:
    def test_render_and_export(self, qapp, sample_results_df, tmp_path):
        spec = AlignedViewSpec(
            name="breakthrough",
            title="Durchbruch",
            expr="1 - (fid_mgC - B) * f / c_in",
            facet_by="T_klasse",
            color_by="V_MFC",
            marker_by="tag",
            filled="stationaer",
            rolling_s=5.0,
            ref_lines=[RefLine(axis="y", value=1.0)],
        )
        widget = AlignedViewWidget(spec)
        widget.set_data(sample_results_df)

        assert widget.glw is not None
        assert "aligned curve(s) rendered" in widget.status.text()

        # Test export
        export_file = tmp_path / "aligned_test.png"
        out = widget.export(export_file, ExportOptions(width_mm=160, dpi=150))
        assert Path(out).exists()
        assert Path(out).stat().st_size > 500


class TestBarsViewWidget:
    def test_render_and_export(self, qapp, sample_results_df, tmp_path):
        spec = BarsViewSpec(
            name="budget",
            title="Budget",
            quantity="X",
            sort_by=["T_bett"],
        )
        widget = BarsViewWidget(spec)
        widget.set_data(sample_results_df)

        assert widget.glw is not None
        assert "Rendered uncertainty budget" in widget.status.text()

        export_file = tmp_path / "bars_test.svg"
        out = widget.export(export_file, ExportOptions(width_mm=160))
        assert Path(out).exists()
        assert Path(out).stat().st_size > 500


class TestReportViewWidgetAndExport:
    def test_report_export(self, qapp, sample_results_df, tmp_path):
        session = CampaignSession(Campaign(name="Test Catalysis"))

        align_spec = AlignedViewSpec(
            name="breakthrough",
            title="Durchbruch",
            expr="1 - (fid_mgC - B) * f / c_in",
            facet_by="T_klasse",
        )
        bars_spec = BarsViewSpec(
            name="budget",
            title="Unsicherheitsbudget",
            quantity="X",
        )
        rep_spec = ReportViewSpec(
            name="report",
            title="Catalysis Report",
            views=["breakthrough", "budget"],
            cover=True,
        )

        w_align = AlignedViewWidget(align_spec, session=session)
        w_align.set_data(sample_results_df)

        w_bars = BarsViewWidget(bars_spec)
        w_bars.set_data(sample_results_df)

        view_widgets = {"breakthrough": w_align, "budget": w_bars}
        w_rep = ReportViewWidget(rep_spec, session=session, views_dict=view_widgets)
        w_rep.set_data(sample_results_df)

        assert w_rep.table.rowCount() == 3  # Cover + 2 views

        # Export multi-page report
        pdf_path = tmp_path / "catalysis_report.pdf"
        out = w_rep.export(pdf_path)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 2000  # Multi-page PDF


class TestResultsWindowIntegration:
    def test_results_window_with_all_view_types(self, qapp, sample_results_df):
        session = CampaignSession(Campaign(name="Integration Campaign"))
        views = [
            TableViewSpec(name="t1", title="Table View"),
            XYViewSpec(name="xy1", title="XY Plot", x="T_bett", y="X"),
            AlignedViewSpec(name="al1", title="Breakthrough", expr="1 - (fid_mgC - B) * f / c_in"),
            BarsViewSpec(name="b1", title="Budget", quantity="X"),
            ReportViewSpec(name="r1", title="Report"),
        ]
        session.campaign.evaluation = Evaluation(views=views)

        win = ResultsWindow(session=session)
        win._direct_df = sample_results_df
        win.refresh()

        assert win.tabs.count() == 5
        assert win.tabs.tabText(0) == "Table View"
        assert win.tabs.tabText(1) == "XY Plot"
        assert win.tabs.tabText(2) == "Breakthrough"
        assert win.tabs.tabText(3) == "Budget"
        assert win.tabs.tabText(4) == "Report"


class TestCampaignWindowReportExport:
    def test_campaign_window_export_report(self, qapp, sample_results_df, tmp_path):
        from datualizer_gui.campaign_window import CampaignWindow

        session = CampaignSession(Campaign(name="Catalysis Campaign"))
        views = [
            TableViewSpec(name="results", title="Results Table"),
            AlignedViewSpec(name="al", title="Breakthrough", expr="1 - (fid_mgC - B) * f / c_in"),
            BarsViewSpec(name="b", title="Budget", quantity="X"),
        ]
        session.campaign.evaluation = Evaluation(views=views)

        win = CampaignWindow(session)
        # Populate results window with sample df
        win.results_panel._direct_df = sample_results_df
        if win._results_window is None:
            win._results_window = ResultsWindow(session=session, parent=win)
        win._results_window._direct_df = sample_results_df
        win._results_window.refresh()

        out_pdf = tmp_path / "full_campaign_report.pdf"
        res = win.export_report(out_pdf)
        assert Path(res).exists()
        assert Path(res).stat().st_size > 1500


class TestAlignedSignalEmission:
    def test_step_inspected_signal(self, qapp, sample_results_df):
        spec = AlignedViewSpec(
            name="breakthrough",
            expr="1 - (fid_mgC - B) * f / c_in",
        )
        widget = AlignedViewWidget(spec)
        widget.set_data(sample_results_df)

        emitted = []
        widget.step_inspected.connect(lambda sid: emitted.append(sid))

        # Find scatter plot item
        scatters = [item for item in widget.glw.scene().items() if isinstance(item, pg.ScatterPlotItem)]
        assert len(scatters) > 0
        # Simulate scatter clicked
        pts = scatters[0].points()
        if len(pts) > 0:
            scatters[0].sigClicked.emit(scatters[0], [pts[0]])
            assert len(emitted) == 1
            assert emitted[0] == "s1"


