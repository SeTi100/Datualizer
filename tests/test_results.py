"""Tests for Etappe 2b, step 8, PR 5: Rechengrössen, Kriterien, Szenarien, Evaluation, Results Dock."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from datualizer_core.calc import ExprError
from datualizer_core.calc.uncertain import Constant
from datualizer_core.campaign import (
    Campaign,
    CampaignSession,
    Evaluation,
    Measure,
    NamedWindow,
    StepClass,
)
from datualizer_core.campaign.evaluation import Calculation, Criterion, Rating, Scenario
from datualizer_gui.campaign_window import CampaignWindow
from datualizer_gui.components.results_panel import ResultsPanel


@pytest.fixture
def sa_constants() -> dict[str, Constant]:
    return {
        "f": Constant(value=2.701, u=0.135, kind="syst", note="mgC -> mg Ethanol (RF 0.71)"),
        "V_kat": Constant(value=0.0135, unit="m3"),
        "luft_m3_h": Constant(value=100.0, u=7.2, kind="syst", scope="tag"),
        "T_ref": Constant(value=24.4, u=3.0, kind="syst"),
    }


@pytest.fixture
def sa_calculations() -> list[Calculation]:
    return [
        Calculation(name="V_N", expr="luft_m3_h * 273.15 / (T_ref + 273.15)", unit="Nm3/h"),
        Calculation(name="c_in", expr="-mdot * 3600 * 1000 / V_N", unit="mg/Nm3"),
        Calculation(name="X", expr="1 - (S - B) * f / c_in"),
        Calculation(name="dXdt", expr="-dS * 60 * f / c_in", unit="1/min"),
        Calculation(name="T_bett", expr="(K1 + K2) / 2", unit="°C"),
        Calculation(name="k", expr="V_N * (T_bett + 273.15) / 273.15 / V_kat / 3600 * -log(1 - X)", unit="1/s"),
    ]


@pytest.fixture
def sa_criteria() -> list[Criterion]:
    return [
        Criterion(name="c_ok", expr="c_in >= 60", text="c_in < 60"),
        Criterion(name="flach", expr="abs(dS * 60) <= 0.1 or abs(dXdt) <= 0.004", text="nicht stationär: FID nicht flach"),
        Criterion(name="t_ok", expr="abs(dK1) <= 5", text="nicht stationär: Temperatur driftet"),
        Criterion(name="geladen", expr="beladung >= 3", text="nicht stationär: Speicherphase (< 3 g dosiert)"),
    ]


@pytest.fixture
def sa_ratings() -> list[Rating]:
    return [
        Rating(label="gut", all_of=["c_ok", "flach", "t_ok", "geladen"]),
        Rating(label="brauchbar", any_of=["c_ok"]),
        Rating(label="unsicher"),
    ]


@pytest.fixture
def sa_scenarios() -> list[Scenario]:
    return [
        Scenario(name="Luft 115 m³/h", constants={"luft_m3_h": Constant(value=115.0, u=7.2, kind="syst")}),
    ]


@pytest.fixture
def sa_evaluation(sa_constants, sa_calculations, sa_criteria, sa_ratings, sa_scenarios) -> Evaluation:
    return Evaluation(
        constants=sa_constants,
        calculations=sa_calculations,
        criteria=sa_criteria,
        ratings=sa_ratings,
        scenarios=sa_scenarios,
    )


class TestEvaluationModels:
    def test_constant_scope_and_naming(self):
        c = Constant(value=100.0, u=7.2, scope="tag")
        assert c.name == ""
        assert c.scope == "tag"
        assert c.value == 100.0

    def test_evaluation_roundtrip_json(self, sa_evaluation: Evaluation):
        data = sa_evaluation.model_dump()
        ev2 = Evaluation.model_validate(data)
        assert len(ev2.constants) == 4
        assert ev2.constants["luft_m3_h"].scope == "tag"
        assert ev2.constants["f"].name == "f"
        assert len(ev2.calculations) == 6
        assert len(ev2.criteria) == 4
        assert len(ev2.ratings) == 3
        assert len(ev2.scenarios) == 1
        assert ev2.scenario("Luft 115 m³/h").constants["luft_m3_h"].value == 115.0

    def test_validation_duplicate_names(self):
        with pytest.raises(ValueError, match="Duplicate calculation"):
            Evaluation(
                calculations=[
                    Calculation(name="X", expr="1 + 1"),
                    Calculation(name="X", expr="2 + 2"),
                ]
            )

        with pytest.raises(ValueError, match="Duplicate criterion"):
            Evaluation(
                criteria=[
                    Criterion(name="c1", expr="1 > 0", text="err1"),
                    Criterion(name="c1", expr="2 > 0", text="err2"),
                ]
            )

        with pytest.raises(ValueError, match="Duplicate scenario"):
            Evaluation(
                scenarios=[
                    Scenario(name="s1"),
                    Scenario(name="s1"),
                ]
            )

    def test_validation_unknown_criterion_in_rating(self):
        with pytest.raises(ValueError, match="unknown criterion"):
            Evaluation(
                criteria=[Criterion(name="c_ok", expr="1 > 0", text="err")],
                ratings=[Rating(label="gut", all_of=["c_ok", "unknown_crit"])],
            )

    def test_validation_column_clash(self):
        with pytest.raises(ValueError, match="clashes with the rating column"):
            Evaluation(
                constants={"bewertung": Constant(value=1.0)}
            )

        with pytest.raises(ValueError, match="clashes with the reason column"):
            Evaluation(
                constants={"grund": Constant(value=1.0)}
            )

        with pytest.raises(ValueError, match="is already a step table column"):
            Evaluation(
                calculations=[Calculation(name="dauer_s", expr="1.0")]
            )


class TestResultsComputation:
    def test_z16_recalculation_and_scenario(self, sa_evaluation: Evaluation):
        # Step table with 17.10. step Z16 and another step Z27 with violated criteria
        df_steps = pl.DataFrame({
            "step_id": ["Z16", "Z27"],
            "step_label": ["200°C 10Nl", "220°C 20Nl"],
            "step_start": [1729163056.0, 1729175083.0],
            "step_end": [1729166170.0, 1729175468.0],
            "dauer_s": [3114.0, 385.0],
            "klasse": ["dosierung", "dosierung"],
            "klasse_hinweis": [None, None],
            "T_soll": [200.0, 220.0],
            "V_MFC": [10.0, 20.0],
            # Z16 values: S=25.807, B=2.2664, mdot=-13.2523 g/h
            # Z27 values: c_in < 60 (mdot=-2.0 g/h), dK1=16.0 (> 5 K)
            "S": [25.807, 34.2589],
            "u_S": [0.1418, 0.5598],
            "B": [2.2664, 2.09],
            "u_B": [1.0, 2.0],
            "dS": [-0.0482 / 60, -1.8217 / 60],
            "u_dS": [0.001, 0.02],
            "mdot": [-13.2523 / 3600, -2.0 / 3600],
            "u_mdot": [0.0428 / 3600, 0.08 / 3600],
            "K1": [209.9984, 227.7944],
            "u_K1": [0.5, 0.5],
            "K2": [188.4857, 199.3312],
            "u_K2": [0.5, 0.5],
            "dK1": [0.2109, 16.0357],
            "u_dK1": [0.1, 0.2],
            "beladung": [11.4633, 1.6422],
        })

        session = CampaignSession()
        session.step_table = lambda spec=None, config=None: df_steps

        # 1. Base results
        res = session.results(spec=sa_evaluation)

        # Check Z16 (row 0)
        assert res["V_N"][0] == pytest.approx(91.80, abs=0.01)
        assert res["c_in"][0] == pytest.approx(144.36, abs=0.01)
        assert res["X"][0] == pytest.approx(0.560, abs=0.001)
        assert res["u_X"][0] == pytest.approx(0.047, abs=0.005)
        assert res["u_X_zufall"][0] == pytest.approx(0.019, abs=0.005)
        assert res["u_X_syst"][0] == pytest.approx(0.039, abs=0.005)

        # Check budget columns exist and have non-zero signed contributions
        assert "b_X__S" in res.columns
        assert "b_X__B" in res.columns
        assert "b_X__f" in res.columns
        assert "b_X__luft_m3_h" in res.columns
        assert "b_X__T_ref" in res.columns
        assert "b_X__mdot" in res.columns
        assert float(res["b_X__S"][0]) < 0  # higher S reduces conversion
        assert float(res["b_X__B"][0]) > 0  # higher B increases conversion

        # Criteria and rating for Z16
        assert res["c_ok"][0] is True
        assert res["flach"][0] is True
        assert res["t_ok"][0] is True
        assert res["geladen"][0] is True
        assert res["bewertung"][0] == "gut"
        assert res["grund"][0] == ""

        # Check Z27 (row 1): violations and reason
        assert res["c_ok"][1] is False  # c_in is ~21.8 < 60
        assert res["t_ok"][1] is False  # dK1 = 16 > 5
        assert res["geladen"][1] is False  # beladung = 1.64 < 3
        assert res["bewertung"][1] == "unsicher"
        grund_z27 = res["grund"][1]
        assert "c_in < 60" in grund_z27
        assert "nicht stationär: Temperatur driftet" in grund_z27
        assert "nicht stationär: Speicherphase (< 3 g dosiert)" in grund_z27
        assert "; " in grund_z27

        # 2. Scenario "Luft 115 m³/h"
        res_scen = session.results(scenario="Luft 115 m³/h", spec=sa_evaluation)
        assert res_scen["V_N"][0] == pytest.approx(105.57, abs=0.01)
        assert res_scen["c_in"][0] == pytest.approx(125.53, abs=0.01)
        assert res_scen["X"][0] == pytest.approx(0.494, abs=0.001)

    def test_empty_step_table(self, sa_evaluation: Evaluation):
        df_empty = pl.DataFrame({
            "step_id": pl.Series([], dtype=pl.String),
            "dauer_s": pl.Series([], dtype=pl.Float64),
            "mdot": pl.Series([], dtype=pl.Float64),
            "u_mdot": pl.Series([], dtype=pl.Float64),
            "S": pl.Series([], dtype=pl.Float64),
            "u_S": pl.Series([], dtype=pl.Float64),
            "B": pl.Series([], dtype=pl.Float64),
            "u_B": pl.Series([], dtype=pl.Float64),
            "dS": pl.Series([], dtype=pl.Float64),
            "K1": pl.Series([], dtype=pl.Float64),
            "K2": pl.Series([], dtype=pl.Float64),
            "dK1": pl.Series([], dtype=pl.Float64),
            "beladung": pl.Series([], dtype=pl.Float64),
        })
        session = CampaignSession()
        session.step_table = lambda spec=None, config=None: df_empty

        res = session.results(spec=sa_evaluation)
        assert len(res) == 0
        assert "X" in res.columns
        assert "u_X" in res.columns
        assert "bewertung" in res.columns
        assert "grund" in res.columns

    def test_chained_calculations_and_scenarios(self):
        ev = Evaluation(
            constants={
                "a": Constant(value=10.0, u=1.0, kind="zufall"),
                "b": Constant(value=2.0, u=0.2, kind="syst"),
            },
            calculations=[
                Calculation(name="c1", expr="a + b"),
                Calculation(name="c2", expr="c1 * 3"),
                Calculation(name="c3", expr="c2 / 2"),
            ],
            criteria=[
                Criterion(name="is_big", expr="c3 > 15", text="too small"),
            ],
            ratings=[
                Rating(label="high", all_of=["is_big"]),
                Rating(label="low", none_of=["is_big"]),
            ],
            scenarios=[
                Scenario(name="double_b", constants={"b": Constant(value=4.0, u=0.2, kind="syst")}),
            ],
        )

        df = pl.DataFrame({"step_id": ["S1"], "dauer_s": [100.0]})
        session = CampaignSession()
        session.step_table = lambda spec=None, config=None: df

        # Base: a=10, b=2 -> c1=12, c2=36, c3=18 > 15
        res = session.results(spec=ev)
        assert res["c1"][0] == 12.0
        assert res["c2"][0] == 36.0
        assert res["c3"][0] == 18.0
        # Sensitivities: dc3/da = 1.5, dc3/db = 1.5
        assert res["b_c3__a"][0] == pytest.approx(1.5 * 1.0)
        assert res["b_c3__b"][0] == pytest.approx(1.5 * 0.2)
        assert res["is_big"][0] is True
        assert res["bewertung"][0] == "high"
        assert res["grund"][0] == ""

        # Scenario: b=4 -> c1=14, c2=42, c3=21
        res_scen = session.results(scenario="double_b", spec=ev)
        assert res_scen["c3"][0] == 21.0

        # Scenario with dict override
        res_dict = session.results(scenario={"a": Constant(value=2.0, u=0.5)}, spec=ev)
        # a=2, b=2 -> c1=4, c2=12, c3=6 <= 15
        assert res_dict["c3"][0] == 6.0
        assert res_dict["is_big"][0] is False
        assert res_dict["bewertung"][0] == "low"
        assert res_dict["grund"][0] == "too small"

        # Unknown scenario raises KeyError
        with pytest.raises(KeyError, match="No scenario 'nonexistent'"):
            session.results(scenario="nonexistent", spec=ev)

    def test_uncertainty_from_a_formula(self):
        # PR 10c: 17.10. Z16, mass flow from the journal readings (GUM: 0.01 g/√12, ±5 s per time)
        # and the residual drift |dX/dt| · 10 min of the script as inputs of their own
        ev = Evaluation(
            constants={"u_masse": Constant(value=0.01 / 12**0.5), "u_zeit": Constant(value=5.0)},
            calculations=[
                Calculation(
                    name="mdot", expr="-(masse_0 - masse_1) / dauer_s", unit="g/s",
                    u_expr="sqrt((sqrt(2) * u_masse / dauer_s)**2 + ((masse_0 - masse_1) * sqrt(2) * u_zeit / dauer_s**2)**2)",
                ),
                Calculation(name="drift", expr="0", u_expr="abs(dXdt) * 10", kind="zufall"),
                Calculation(name="X", expr="0.5685 + drift"),
            ],
        )
        df = pl.DataFrame({
            "step_id": ["Z16", "Z99"], "dauer_s": [3114.0, 3114.0],
            "masse_0": [9650.0, 9650.0], "masse_1": [9638.3, 9638.3], "dXdt": [0.00088, float("nan")],
        })
        session = CampaignSession()
        session.step_table = lambda spec=None, config=None: df
        res = session.results(spec=ev)
        assert res["mdot"][0] * 3600 == pytest.approx(-13.526, abs=0.001)
        assert res["u_mdot"][0] * 3600 == pytest.approx(0.0311, abs=0.0001)  # as the script
        assert res["u_mdot_zufall"][0] == res["u_mdot"][0]
        assert res["u_X"][0] == pytest.approx(0.0088, abs=1e-6)
        assert res["b_X__drift"][0] == pytest.approx(0.0088, abs=1e-6)
        assert np.isnan(res["u_X"][1])  # no drift known: u unknown, not 0

        ev_syst = Evaluation(calculations=[Calculation(name="a", expr="2", u_expr="0.1", kind="syst", group="g")])
        res = session.results(spec=ev_syst)
        assert res["u_a_syst"][0] == pytest.approx(0.1) and res["u_a_zufall"][0] == 0.0

        ev_neg = Evaluation(calculations=[Calculation(name="a", expr="1", u_expr="-dauer_s")])
        with pytest.raises(ValueError, match="negative"):
            session.results(spec=ev_neg)

    def test_unknown_variable_in_calculation_fails_loudly(self):
        ev = Evaluation(
            calculations=[Calculation(name="bad", expr="nonexistent_var * 2")],
        )
        df = pl.DataFrame({"step_id": ["S1"], "dauer_s": [10.0]})
        session = CampaignSession()
        session.step_table = lambda spec=None, config=None: df
        with pytest.raises(ExprError, match="(?i)unknown name 'nonexistent_var'"):
            session.results(spec=ev)


class TestResultsPanelGui:
    @pytest.fixture(autouse=True)
    def ensure_qapp(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    def test_results_panel_components(self, sa_evaluation: Evaluation):
        panel = ResultsPanel()
        df_steps = pl.DataFrame({
            "step_id": ["Z16"],
            "dauer_s": [3114.0],
            "mdot": [-13.2523 / 3600],
            "u_mdot": [0.0428 / 3600],
            "S": [25.807],
            "u_S": [0.1418],
            "B": [2.2664],
            "u_B": [1.0],
            "dS": [-0.0482 / 60],
            "K1": [209.9984],
            "K2": [188.4857],
            "dK1": [0.2109],
            "beladung": [11.4633],
        })

        camp = Campaign(evaluation=sa_evaluation)
        session = CampaignSession(camp)
        session.step_table = lambda spec=None, config=None: df_steps

        panel.set_session(session)

        # Main table populated
        assert panel.table.rowCount() == 1
        assert panel.table.columnCount() > 10

        # Scenario combo populated
        assert panel.scenario_combo.count() == 2
        assert panel.scenario_combo.itemText(0) == "(Base)"
        assert panel.scenario_combo.itemText(1) == "Luft 115 m³/h"

        # Switch scenario
        panel.scenario_combo.setCurrentIndex(1)
        assert panel.current_scenario_name() == "Luft 115 m³/h"

        # Check live syntax validation
        panel._on_calc_expr_changed("1 - (S - B) * f / c_in")
        assert "valid" in panel.calc_syntax_label.text().lower()

        panel._on_calc_expr_changed("1 - * 2")
        assert "✗" in panel.calc_syntax_label.text()

        # An own uncertainty (PR 10c) is shown, kept when saving and can be set in the form
        x_row = [c.name for c in session.campaign.evaluation.calculations].index("X")
        panel.calc_table.selectRow(x_row)
        assert panel.calc_u_edit.text() == ""
        panel.calc_u_edit.setText("abs(dS) * 600")
        panel._save_calc()
        x = next(c for c in session.campaign.evaluation.calculations if c.name == "X")
        assert x.u_expr == "abs(dS) * 600" and x.kind == "zufall"
        assert panel.calc_table.item(x_row, 4).text() == "abs(dS) * 600 (zufall)"
        panel.calc_table.clearSelection()
        panel.calc_table.selectRow(x_row)
        assert panel.calc_u_edit.text() == "abs(dS) * 600"

        # Check filter
        panel.filter_edit.setText("Z16")
        assert not panel.table.isRowHidden(0)
        panel.filter_edit.setText("nonexistent")
        assert panel.table.isRowHidden(0)
        panel.filter_edit.setText("")
        assert not panel.table.isRowHidden(0)

        # Check double-click emits step_double_clicked
        received_step = []
        panel.step_double_clicked.connect(received_step.append)
        panel._on_cell_double_clicked(0, 0)
        assert received_step == ["Z16"]

        # Check Tab 4: Analyses
        assert panel.tabs.count() == 5
        assert panel.tabs.tabText(4) == "Analyses"

    def test_results_panel_analyses_tab(self, ensure_qapp):
        from datualizer_core.calc.analyses import CrossingAnalysis
        eval_with_ana = Evaluation(
            analyses=[CrossingAnalysis(name="T50", x="T_bett", y="X", level=0.5, unit="°C")]
        )
        camp = Campaign(evaluation=eval_with_ana)
        session = CampaignSession(camp)
        df_steps = pl.DataFrame({
            "step_id": ["Z16", "Z19"],
            "T_bett": [150.0, 200.0],
            "X": [0.3, 0.7],
            "u_X": [0.05, 0.05],
        })
        session.step_table = lambda spec=None, config=None: df_steps

        panel = ResultsPanel()
        panel.set_session(session)

        assert panel.analysis_combo.count() == 1
        assert panel.analysis_combo.currentText() == "T50"
        assert panel.analysis_table.rowCount() == 1
        assert panel.analysis_table.columnCount() > 0
        assert "T50" in panel.analysis_status.text()

    def test_campaign_window_dock_results(self, sa_evaluation: Evaluation):
        camp = Campaign(evaluation=sa_evaluation)
        session = CampaignSession(camp)
        win = CampaignWindow(session)
        assert hasattr(win, "dock_results")
        assert hasattr(win, "results_panel")
        assert win.dock_results.widget() is win.results_panel
        win.dock_results.show()
        assert not win.dock_results.isHidden()
        win.close()
