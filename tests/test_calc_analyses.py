"""Tests for cross-step analyses (Etappe 2b, step 8, PR 6).

Covers:
1. `crossing`: linear interpolation, range from y ± u_y, "< x_min" / "> x_max" without crossing,
   and `bereich_syst` with systematic covariance.
2. `fit_line`: line fit with effective variance (sigma^2 = u_y^2 + m^2 * u_x^2), reduced chi^2,
   Birge factor, GLS covariance from Constant.scope, derived quantities (Ea = -m * 8.314).
   Scope test: systematic factor on all k of a single day shifts only b, u_m unchanged;
   factor hitting only one day increases u_m when multiple days are fitted.
3. `aggregate`: 1D aggregation and 2D pivot matrix (mean, median, n, min, max).
4. SA Real data:
   - Light-off 10 Nl/min: T50 = 179 °C [162 °C … 195 °C].
   - Arrhenius Ea = 21 ± 4 kJ/mol from 8 points.
   - Comparison with scipy.odr when available.
5. CampaignSession.analyses() and Evaluation.analyses integration.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import numpy as np
import polars as pl

from datualizer_core.calc.analyses import (
    AggregateAnalysis,
    CrossingAnalysis,
    FitLineAnalysis,
    aggregate,
    build_row_covariance,
    crossing,
    fit_line,
)
from datualizer_core.calc.uncertain import Constant
from datualizer_core.campaign import Campaign, CampaignSession, Evaluation


# ------------------------------------------------------------------ 1. Crossing tests
def test_crossing_basic() -> None:
    # Linear ramp from x=100..200, y=0.0..1.0
    df = pl.DataFrame({
        "x": [100.0, 150.0, 200.0],
        "y": [0.0, 0.5, 1.0],
        "u_y": [0.1, 0.1, 0.1],
    })
    res = crossing(df, x="x", y="y", level=0.5, u_y="u_y")
    assert res.groups[None].x_cross == pytest.approx(150.0)
    assert res.groups[None].wert == "150"
    # y + u_y reaches 0.5 at x = 140; y - u_y reaches 0.5 at x = 160
    assert res.groups[None].x_lo == pytest.approx(140.0)
    assert res.groups[None].x_hi == pytest.approx(160.0)
    assert res.groups[None].bereich == "140 … 160"


def test_crossing_no_crossing_above_and_below() -> None:
    # All points below level
    df_below = pl.DataFrame({"x": [10.0, 20.0, 30.0], "y": [0.1, 0.2, 0.3], "u_y": [0.05, 0.05, 0.05]})
    res_b = crossing(df_below, x="x", y="y", level=0.5, unit="°C")
    assert res_b.groups[None].x_cross is None
    assert res_b.groups[None].wert == "> 30 °C"

    # All points above level
    df_above = pl.DataFrame({"x": [10.0, 20.0, 30.0], "y": [0.6, 0.7, 0.8], "u_y": [0.05, 0.05, 0.05]})
    res_a = crossing(df_above, x="x", y="y", level=0.5, unit="°C")
    assert res_a.groups[None].x_cross is None
    assert res_a.groups[None].wert == "< 10 °C"


def test_crossing_grouped() -> None:
    df = pl.DataFrame({
        "x": [100.0, 200.0, 100.0, 200.0],
        "y": [0.2, 0.8, 0.1, 0.5],
        "u_y": [0.05, 0.05, 0.05, 0.05],
        "group": ["A", "A", "B", "B"],
    })
    res = crossing(df, x="x", y="y", level=0.5, group="group")
    assert len(res.groups) == 2
    # Group A: y crosses 0.5 at 100 + (0.5 - 0.2)/(0.8 - 0.2)*100 = 150
    assert res.groups["A"].x_cross == pytest.approx(150.0)
    # Group B: y crosses 0.5 at 200
    assert res.groups["B"].x_cross == pytest.approx(200.0)
    assert len(res.table) == 2


# ------------------------------------------------------------------ 2. Line Fit tests
def test_fit_line_basic() -> None:
    # Perfect line y = 2 x + 3
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x + 3.0
    ux = np.full(5, 0.05)
    uy = np.full(5, 0.1)
    df = pl.DataFrame({"x": x, "y": y, "ux": ux, "uy": uy})

    res = fit_line(df, x_expr="x", y_expr="y", u_x="ux", u_y="uy")
    assert res.m == pytest.approx(2.0, abs=1e-6)
    assert res.b == pytest.approx(3.0, abs=1e-6)
    assert res.u_m > 0
    assert res.u_b > 0
    assert res.chi2_red == pytest.approx(0.0, abs=1e-5)


def test_fit_line_derived_quantities() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = -2.5 * x + 6.0
    df = pl.DataFrame({"x": x, "y": y, "ux": [0.05]*5, "uy": [0.1]*5})

    res = fit_line(
        df,
        x_expr="x",
        y_expr="y",
        u_x="ux",
        u_y="uy",
        derived={"Ea": "-m * 8.314"},
    )
    assert "Ea" in res.derived
    val_ea, u_ea, u_ea_rand = res.derived["Ea"]
    assert val_ea == pytest.approx(2.5 * 8.314, rel=1e-4)
    assert u_ea == pytest.approx(res.u_m * 8.314, rel=1e-4)


# ------------------------------------------------------------------ 3. Scope & Covariance tests
def test_fit_line_scope_same_day_vs_multi_day() -> None:
    """Requirement 4:
    A systematic factor on all k of one day (e.g. air flow) shifts only b, u_m remains unchanged.
    A factor that only hits one day affects m when points of multiple days are fitted.
    """
    # 1. Single day: 5 points
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ux = np.full(5, 0.05)
    uy_rand = np.full(5, 0.1)
    y = 2.0 * x + 1.0

    df_1day = pl.DataFrame({
        "x": x,
        "y": y,
        "ux": ux,
        "u_y_zufall": uy_rand,
        "tag": ["2026-10-12"] * 5,
        "b_y__luft": [0.2] * 5,  # Systematic contribution of air flow
    })
    consts = {"luft": Constant(name="luft", value=100.0, u=7.2, kind="syst", scope="tag")}

    res_1day = fit_line(df_1day, x_expr="x", y_expr="y", u_x="ux", constants=consts)
    # Slope uncertainty with correlation MUST equal random-only slope uncertainty!
    assert res_1day.u_m == pytest.approx(res_1day.u_m_zufall, rel=1e-4)
    # Intercept uncertainty MUST be larger with correlation!
    assert res_1day.u_b > res_1day.u_m

    # 2. Multiple days: Day 1 (pts 0, 1, 2) and Day 2 (pts 3, 4)
    df_2days = pl.DataFrame({
        "x": x,
        "y": y,
        "ux": ux,
        "u_y_zufall": uy_rand,
        "tag": ["2026-10-12", "2026-10-12", "2026-10-12", "2026-10-17", "2026-10-17"],
        "b_y__luft": [0.2] * 5,
    })

    res_2days = fit_line(df_2days, x_expr="x", y_expr="y", u_x="ux", constants=consts)
    # When multiple days are fitted, the day-to-day error tilts the line between days:
    # u_m with correlation MUST be strictly larger than random-only!
    assert res_2days.u_m > res_2days.u_m_zufall * 1.2


# ------------------------------------------------------------------ 4. Aggregate tests
def test_aggregate_1d_and_2d() -> None:
    df = pl.DataFrame({
        "T_klasse": [150.0, 150.0, 200.0, 200.0, 250.0],
        "V_MFC": [10.0, 20.0, 10.0, 20.0, 10.0],
        "X": [0.39, 0.37, 0.56, 0.42, 0.93],
    })

    # 1D aggregate
    agg1 = aggregate(df, rows="T_klasse", value="X", how="mean")
    assert agg1.columns == ["T_klasse", "X_mean"]
    assert len(agg1) == 3
    assert agg1["X_mean"][0] == pytest.approx(0.38)

    # 2D pivot matrix
    agg2 = aggregate(df, rows="T_klasse", cols="V_MFC", value="X", how="mean")
    assert agg2.columns == ["T_klasse", "10.0", "20.0"]
    assert len(agg2) == 3
    assert agg2["10.0"][0] == pytest.approx(0.39)
    assert agg2["20.0"][0] == pytest.approx(0.37)


# ------------------------------------------------------------------ 5. Real SA Data tests (Plan 7)
def test_sa_light_off_t50_v10() -> None:
    """SA Light-off 10 Nl/min from Plan 7: T50 = 179 °C, range 162 °C … 195 °C."""
    # Data from kampagne_ergebnisse.csv / schritte.csv for V_MFC == 10.0 stationary
    df_v10 = pl.DataFrame({
        "tag": ["2026-10-17", "2026-10-17", "2026-10-17", "2026-10-17", "2026-10-15", "2026-10-15"],
        "schritt": ["Z23", "Z19", "Z16", "Z29", "Run41", "Run48"],
        "V_MFC": [10.0] * 6,
        "T_bett": [151.4747, 178.8838, 199.242, 220.6682, 246.0096, 246.046],
        "u_T_bett": [3.8295, 5.3659, 6.3888, 7.4702, 8.348, 8.8354],
        "X": [0.3925, 0.5009, 0.5596, 0.642, 0.9414, 0.9149],
        "u_X": [0.0724, 0.0558, 0.0469, 0.062, 0.0128, 0.0265],
        "u_X_zufall": [0.0434, 0.0292, 0.021, 0.0518, 0.0115, 0.0253],
        "u_X_syst": [0.0579, 0.0476, 0.042, 0.0341, 0.0056, 0.0081],
        "stationaer": [True] * 6,
        "bewertung": ["gut"] * 6,
    })

    res = crossing(
        df_v10,
        x="T_bett",
        y="X",
        level=0.5,
        group="V_MFC",
        u_y="u_X",
        unit="°C",
        name="T50",
    )
    grp = res.groups[10.0]
    assert grp.x_cross == pytest.approx(178.7, abs=0.5)
    assert grp.wert == "179 °C"
    assert grp.bereich == "162 °C … 195 °C"
    assert "165 °C" in grp.bereich_syst


def test_sa_arrhenius_8_points() -> None:
    """Arrhenius Ea 21 ± 4 kJ/mol from 8 points in SA data, and control against scipy.odr."""
    # The 8 stationary catalytic oxidation points with 0.05 < X < 0.80
    df_arr = pl.DataFrame({
        "tag": [
            "2026-10-17", "2026-10-17", "2026-10-17", "2026-10-12",
            "2026-10-17", "2026-10-17", "2026-10-17", "2026-10-17",
        ],
        "schritt": ["Z24", "Z23", "Z19", "Z28", "Z26", "Z16", "Z28", "Z29"],
        "V_MFC": [20.0, 10.0, 10.0, 25.0, 20.0, 10.0, 20.0, 10.0],
        "T_bett": [149.9456, 151.4747, 178.8838, 185.3595, 195.1669, 199.242, 219.3087, 220.6682],
        "u_T_bett": [3.9697, 3.8295, 5.3659, 5.7974, 6.99, 6.3888, 7.7479, 7.4702],
        "k_1_s": [1.3513, 1.4637, 2.1721, 2.2479, 1.7848, 2.6786, 3.7106, 3.5082],
        "u_lnk_zufall": [0.0787, 0.1434, 0.0841, 0.117, 0.0556, 0.0581, 0.1072, 0.1409],
        "b_k__luft_m3_h": [0.072 * 1.3513, 0.072 * 1.4637, 0.072 * 2.1721, 0.072 * 2.2479,
                           0.072 * 1.7848, 0.072 * 2.6786, 0.072 * 3.7106, 0.072 * 3.5082],
    })

    consts = {"luft_m3_h": Constant(name="luft_m3_h", value=100.0, u=7.2, kind="syst", scope="tag")}

    res = fit_line(
        df_arr,
        x_expr="1000 / (T_bett + 273.15)",
        y_expr="log(k_1_s)",
        u_x="1000 * u_T_bett / (T_bett + 273.15)**2",
        u_y="u_lnk_zufall",
        derived={"Ea": "-m * 8.314462"},
        constants=consts,
    )

    assert res.n == 8
    # Slope m ≈ -2.57, Birge ≈ sqrt(2.06) ≈ 1.44
    assert res.m == pytest.approx(-2.57, abs=0.05)
    val_ea, u_ea, u_ea_rand = res.derived["Ea"]
    # Ea = 21 ± 4 kJ/mol!
    assert val_ea == pytest.approx(21.4, abs=0.5)
    assert u_ea == pytest.approx(4.3, abs=0.5)

    # Control comparison against scipy.odr if scipy is available
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from scipy.odr import ODR, Model, RealData
            T = df_arr["T_bett"].to_numpy() + 273.15
            x = 1000.0 / T
            ux = 1000.0 * df_arr["u_T_bett"].to_numpy() / (T**2)
            y = np.log(df_arr["k_1_s"].to_numpy())
            uy = df_arr["u_lnk_zufall"].to_numpy()

            def f(B, x): return B[0] * x + B[1]
            data = RealData(x, y, sx=ux, sy=uy)
            odr = ODR(data, Model(f), beta0=[-2.5, 6.0])
            out = odr.run()
        # ODR and effective variance slope should be within 20%
        assert res.m == pytest.approx(out.beta[0], rel=0.20)
    except ImportError:
        pass


# ------------------------------------------------------------------ 6. CampaignSession.analyses integration
def test_session_analyses_full_workflow() -> None:
    ev = Evaluation(
        analyses=[
            CrossingAnalysis(name="T50", x="T_bett", y="X", level=0.5, unit="°C"),
            FitLineAnalysis(
                name="arrhenius",
                x_expr="1000 / (T_bett + 273.15)",
                y_expr="log(k_1_s)",
                derived={"Ea": "-m * 8.314"},
            ),
            AggregateAnalysis(name="matrix", rows="T_klasse", cols="V_MFC", value="X", how="mean"),
        ]
    )
    # Check duplicate analysis names validation
    with pytest.raises(ValueError, match="Duplicate analysis name"):
        Evaluation(
            analyses=[
                CrossingAnalysis(name="T50", x="T", y="X", level=0.5),
                CrossingAnalysis(name="T50", x="T", y="X", level=0.9),
            ]
        )

    assert ev.analysis_names() == ["T50", "arrhenius", "matrix"]
    assert ev.analysis("T50").kind == "crossing"

    # End-to-end call through session.analyses()
    camp = Campaign(evaluation=ev)
    session = CampaignSession(camp)
    df_steps = pl.DataFrame({
        "step_id": ["Z16", "Z19", "Z23", "Z24"],
        "T_bett": [199.0, 179.0, 151.0, 150.0],
        "u_T_bett": [6.0, 5.0, 4.0, 4.0],
        "X": [0.56, 0.50, 0.39, 0.37],
        "u_X": [0.05, 0.05, 0.07, 0.07],
        "k_1_s": [2.67, 2.17, 1.46, 1.35],
        "u_lnk_zufall": [0.06, 0.08, 0.14, 0.08],
        "T_klasse": [200.0, 175.0, 150.0, 150.0],
        "V_MFC": [10.0, 10.0, 10.0, 20.0],
    })
    session.step_table = lambda spec=None, config=None: df_steps
    res_dict = session.analyses()
    assert "T50" in res_dict
    assert "arrhenius" in res_dict
    assert "matrix" in res_dict
    assert res_dict["T50"].groups[None].x_cross is not None
    assert res_dict["arrhenius"].m < 0
    assert "Ea" in res_dict["arrhenius"].derived
    assert len(res_dict["matrix"]) == 3


def test_fit_line_monte_carlo() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = -2.0 * x + 5.0
    df = pl.DataFrame({"x": x, "y": y, "ux": [0.05]*5, "uy": [0.1]*5})

    res = fit_line(
        df,
        x_expr="x",
        y_expr="y",
        u_x="ux",
        u_y="uy",
        derived={"Ea": "-m * 8.314"},
        mc_samples=500,
    )
    assert "mc_m_mean" in res.mc_results
    assert res.mc_results["mc_m_mean"] == pytest.approx(-2.0, abs=0.1)
    assert res.mc_results["mc_m_std"] == pytest.approx(res.u_m, rel=0.3)
    assert "mc_Ea_mean" in res.mc_results
