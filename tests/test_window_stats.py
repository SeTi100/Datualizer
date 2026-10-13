"""Window statistics of a measurement step (Etappe 2b, step 6, package A; package B in test_window_uncertainty.py).

Synthetic series with a known answer, then the scale (OCR, SQLite) in the journal step
"T Einstellung 155": the mass falls while ethanol is dosed.
"""

from __future__ import annotations

from datetime import date
import math
from pathlib import Path

import numpy as np
import pytest

from datualizer_core.campaign import Campaign, CampaignSession, ClockModel
from datualizer_core.ingestion.profiles import get_profile
from datualizer_core.ingestion.source_loader import with_choices
from datualizer_core.stats import StatsConfig, WindowSpec, window_stats

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
JOURNAL = SA / "labjournal_12_10_extract.xlsx"
SCALE = SA / "instrument_reader_12_10_extract.db"

NO_GAPS = np.array([], dtype=np.int64)


def line(n: int = 101, a: float = 2.0, b: float = 0.5, t0: float = 1.79e9):
    """y = a + b·t on a 1 s grid, t counted from t0 (epoch-sized to catch cancellation)."""
    t = np.arange(n, dtype=float)
    return t0 + t, a + b * t


class TestLine:
    """A straight line: every estimator must give the exact value."""

    def test_location_and_slope(self) -> None:
        x, y = line()
        r = window_stats(x, y, NO_GAPS, WindowSpec(x[10], x[90]), None, StatsConfig())
        assert r.n == 81 and r.n_excluded == 0 and r.n_missing == 0
        assert r.mean == pytest.approx(2 + 0.5 * 50)
        assert r.median == pytest.approx(2 + 0.5 * 50)
        assert r.sd == pytest.approx(0.5 * np.std(np.arange(10, 91), ddof=1))
        assert r.sem == pytest.approx(r.sd / math.sqrt(81))
        assert (r.min, r.max) == (pytest.approx(7.0), pytest.approx(47.0))
        assert r.slope == pytest.approx(0.5, rel=1e-9)
        assert r.slope_se == pytest.approx(0.0, abs=1e-9)
        assert r.intercept == pytest.approx(2 + 0.5 * 10)  # fit line at the window start
        assert r.r2 == pytest.approx(1.0)
        assert r.residual_sd == pytest.approx(0.0, abs=1e-9)

    def test_two_point_and_integral(self) -> None:
        x, y = line()
        w = WindowSpec(x[0] + 10.5, x[0] + 90.5)  # edges between samples
        r = window_stats(x, y, NO_GAPS, w, None, StatsConfig())
        assert r.two_point == pytest.approx(0.5)  # nearest points inside: 11 and 90, on the line
        # ∫ (2 + 0.5 t) dt from 10.5 to 90.5, edge values interpolated
        assert r.integral == pytest.approx(2 * 80 + 0.25 * (90.5**2 - 10.5**2))
        assert r.coverage == pytest.approx(1.0)
        assert r.integral_gap_s == pytest.approx(0.0)
        assert r.window_s == pytest.approx(80.0)

    def test_robust_slope_and_light(self) -> None:
        """Package B on the line: Theil-Sen exact, interval collapsed, a clean slope is drift."""
        x, y = line()
        r = window_stats(x, y, NO_GAPS, WindowSpec(x[0], x[-1]), None, StatsConfig())
        assert r.slope_theil_sen == pytest.approx(0.5)
        assert r.slope_ts_ci == (pytest.approx(0.5), pytest.approx(0.5))
        assert r.slope_se_hac == pytest.approx(0.0, abs=1e-9)
        assert r.stationary is False and any("not stationary: slope" in c for c in r.checks)
        assert not any("Theil-Sen" in c for c in r.checks)


def test_classic_slope_se_matches_the_textbook_formula() -> None:
    rng = np.random.default_rng(7)
    x = 1.79e9 + np.arange(200, dtype=float)
    y = 5.0 - 0.01 * (x - x[0]) + rng.normal(0, 0.3, len(x))
    r = window_stats(x, y, NO_GAPS, WindowSpec(x[0], x[-1]), None, StatsConfig())
    (b, a), cov = np.polyfit(x - x[0], y, 1, cov=True)  # cov scaled by residual variance / (n - 2)
    assert r.slope == pytest.approx(b, rel=1e-9)
    assert r.slope_se == pytest.approx(math.sqrt(cov[0, 0]), rel=1e-9)
    res = y - (a + b * (x - x[0]))
    assert r.residual_sd == pytest.approx(math.sqrt(res @ res / (len(x) - 2)))
    assert r.r2 == pytest.approx(1 - res @ res / np.sum((y - y.mean()) ** 2))
    assert r.quantiles == (pytest.approx(np.quantile(y, 0.05)), pytest.approx(np.quantile(y, 0.95)))


class TestGaps:
    def test_flagged_time_gap_is_not_bridged(self) -> None:
        """Samples 0..40 and 60..100 (a logger pause): the 20 s are reported, not interpolated."""
        x, y = line()
        keep = np.r_[0:41, 60:101]
        x, y = x[keep], y[keep]
        gap_rows = np.array([41])  # row where the time gap starts (first row after the pause)
        w = WindowSpec(x[0], x[-1])
        r = window_stats(x, y, gap_rows, w, None, StatsConfig())
        assert r.integral_gap_s == pytest.approx(20.0)
        assert r.coverage == pytest.approx(80 / 100)
        expected = (2 * 40 + 0.25 * 40**2) + (2 * 40 + 0.25 * (100**2 - 60**2))
        assert r.integral == pytest.approx(expected)
        assert not any("coverage" in c for c in r.checks)  # 0.8 is not below the default 0.8
        r_strict = window_stats(x, y, gap_rows, w, None, StatsConfig(min_coverage=0.9))
        assert any("coverage 80" in c for c in r_strict.checks)

    def test_empty_values_split_the_integral(self) -> None:
        """Time runs on, values are missing (invalid readings): same as a gap, counted as missing."""
        x, y = line()
        y = y.copy()
        y[41:60] = np.nan
        r = window_stats(x, y, NO_GAPS, WindowSpec(x[0], x[-1]), None, StatsConfig())
        assert r.n_missing == 19 and r.n == 82
        assert r.integral_gap_s == pytest.approx(20.0)  # from sample 40 to sample 60

    def test_excluded_points_are_counted_and_not_used(self) -> None:
        x, y = line()
        y = y.copy()
        y[50] = 1e6  # an OCR misreading
        flags = np.zeros(len(x), dtype=bool)
        flags[50] = True
        r = window_stats(x, y, NO_GAPS, WindowSpec(x[0], x[-1]), flags, StatsConfig())
        assert r.n_excluded == 1 and r.n == 100
        assert r.slope == pytest.approx(0.5)
        assert r.integral_gap_s == pytest.approx(2.0)  # 49..51 has no usable value in between
        assert any("excluded" in c for c in r.checks)

    def test_gap_outside_the_window_does_not_count(self) -> None:
        x, y = line()
        r = window_stats(x, y, np.array([5]), WindowSpec(x[20], x[80]), None, StatsConfig())
        assert r.coverage == pytest.approx(1.0)


class TestTwoPoint:
    def test_gum_uncertainty(self) -> None:
        """u² = (u²(y1) + u²(y2))/Δt² + (Δy/Δt²)²·u²(Δt), u(Δt) = √2 · u_clock."""
        x, y = line(b=0.2)
        r = window_stats(x, y, NO_GAPS, WindowSpec(x[0], x[100]), None, StatsConfig(), reading_u=0.05, clock_u_s=1.0)
        dt, dy = 100.0, 20.0
        u2 = (2 * 0.05**2) / dt**2 + (dy / dt**2) ** 2 * (2 * 1.0**2)
        assert r.two_point == pytest.approx(0.2)
        assert r.two_point_u == pytest.approx(math.sqrt(u2))
        assert r.two_point_dt == pytest.approx(100.0)

    def test_nearest_point_distance_is_reported(self) -> None:
        x, y = line()
        r = window_stats(x, y, NO_GAPS, WindowSpec(x[0] + 10.4, x[0] + 90.0), None, StatsConfig())
        # only points inside the window: 11 s for the edge at 10.4 s; the largest distance is reported
        assert r.two_point_offset_s == pytest.approx(0.6)

    def test_median_around_the_edges(self) -> None:
        """With a half-width the value at an edge is the median of the points in ±h."""
        x, y = line(b=0.0)
        y = y.copy()
        y[0] = 100.0  # a spike right at the start would ruin the nearest point
        cfg = StatsConfig(point_halfwidth_s=3.0)
        r = window_stats(x, y, NO_GAPS, WindowSpec(x[0], x[100]), None, cfg)
        assert r.two_point == pytest.approx(0.0)
        nearest = window_stats(x, y, NO_GAPS, WindowSpec(x[0], x[100]), None, StatsConfig())
        assert nearest.two_point == pytest.approx(-98 / 100)

    def test_no_clock_uncertainty_is_said(self) -> None:
        x, y = line()
        r = window_stats(x, y, NO_GAPS, WindowSpec(x[0], x[-1]), None, StatsConfig())
        assert any("clock uncertainty not given" in a for a in r.assumptions)


class TestTooLittle:
    def test_empty_window(self) -> None:
        x, y = line()
        r = window_stats(x, y, NO_GAPS, WindowSpec(x[-1] + 10, x[-1] + 20), None, StatsConfig())
        assert r.n == 0 and r.coverage == 0.0 and r.integral_gap_s == pytest.approx(10.0)
        assert r.mean is None and r.slope is None and r.two_point is None and r.integral == 0.0
        assert any("no data" in c for c in r.checks)

    def test_few_points(self) -> None:
        x, y = line()
        r = window_stats(x, y, NO_GAPS, WindowSpec(x[0], x[2]), None, StatsConfig(min_points=5))
        assert r.n == 3 and r.slope == pytest.approx(0.5)
        assert any("3 points" in c for c in r.checks)

    def test_constant_values(self) -> None:
        """A quantized scale can show the same reading for a whole window: no R², said so."""
        x, _ = line()
        r = window_stats(x, np.full(len(x), 9683.8), NO_GAPS, WindowSpec(x[0], x[-1]), None, StatsConfig())
        assert r.slope == 0.0 and r.r2 is None and r.sd == 0.0
        assert any("constant" in c for c in r.checks)

    def test_window_must_be_ordered(self) -> None:
        with pytest.raises(ValueError):
            WindowSpec(10.0, 10.0)


def test_config_is_serializable() -> None:
    cfg = StatsConfig(exclude_flags=["spike"], point_halfwidth_s=5, reading_u={"scale/masse": 0.01 / math.sqrt(12)})
    assert StatsConfig.model_validate_json(cfg.model_dump_json()) == cfg
    c = Campaign(stats=cfg)
    assert Campaign.model_validate_json(c.model_dump_json()).stats == cfg
    assert Campaign().stats == StatsConfig()


# ------------------------------------------------------------------ real data
@pytest.fixture()
def session() -> CampaignSession:
    s = CampaignSession()
    s.add_source(JOURNAL, with_choices(JOURNAL, get_profile("labjournal_xlsx"), clock_date=date(2026, 10, 12)),
                 source_id="journal")
    s.add_source(SCALE, source_id="scale")
    s.set_mapping(s.suggest_mapping("journal"))
    return s


class TestScale:
    def test_mass_falls_in_step_155(self, session: CampaignSession) -> None:
        table = session.window_table([("scale", "masse")])
        assert len(table) == len(session.intervals().intervals)  # one row per step × channel
        step = table.filter(table["step_id"] == "journal:19").row(0, named=True)
        assert step["step_label"] == "T Einstellung 155" and step["v_mfc"] == 1.0
        assert (step["source"], step["channel"]) == ("scale", "masse")
        # the scale reads every ~1 s over the 194 s step; empty OCR readings are missing, not values
        assert step["n"] > 150 and step["n_missing"] > 0
        # mass falls: roughly 2.4 g/h of ethanol (journal: 0.15 g in 194 s ≈ 2.8 g/h)
        slope_g_h = step["slope"] * 3600
        assert -5.0 < slope_g_h < -1.0
        assert step["slope_se"] > 0 and step["slope_se"] * 3600 < 0.5
        assert step["two_point"] < 0
        # 36 empty readings (~49 s) are not bridged: coverage 75 %, said in the checks
        assert 0.7 < step["coverage"] < 0.8 and 40 < step["integral_gap_s"] < 60
        assert "coverage 75% below 80%" in step["checks"] and "36 rows without a value" in step["checks"]
        assert "excluded flags: spike, dropout" in step["assumptions"]

    def test_steps_without_scale_data(self, session: CampaignSession) -> None:
        table = session.window_table([("scale", "masse")])
        first = table.filter(table["step_id"] == "journal:0").row(0, named=True)
        assert first["n"] == 0 and first["coverage"] == 0.0 and first["slope"] is None
        assert "no data" in first["checks"]

    def test_clock_uncertainty_enters_the_two_point(self, session: CampaignSession) -> None:
        before = session.window_table([("scale", "masse")]).filter(lambda_id("journal:19"))
        session.set_clock("scale", ClockModel(uncertainty_s=5.0, method="manual"))
        after = session.window_table([("scale", "masse")]).filter(lambda_id("journal:19"))
        assert after["two_point_u"][0] > before["two_point_u"][0]
        assert after["slope"][0] == before["slope"][0]

    def test_config_from_the_campaign(self, session: CampaignSession, tmp_path: Path) -> None:
        assert session.window_table([("scale", "massenstrom_waage")]).filter(lambda_id("journal:19"))["n_excluded"][0] == 1
        resolution_u = 0.01 / math.sqrt(12)  # 0.01 g display step, an input from the device data
        session.set_stats_config(StatsConfig(exclude_flags=[], min_points=500, reading_u={"scale/masse": resolution_u}))
        row = session.window_table([("scale", "massenstrom_waage")]).filter(lambda_id("journal:19")).row(0, named=True)
        assert row["n_excluded"] == 0 and "only 186 points (min 500)" in row["checks"]  # the spike counts again
        mass = session.window_table([("scale", "masse")]).filter(lambda_id("journal:19")).row(0, named=True)
        assert mass["two_point_u"] == pytest.approx(math.sqrt(2) * resolution_u / mass["two_point_dt"])
        session.save(tmp_path / "campaign.json")
        assert CampaignSession.open(tmp_path / "campaign.json").campaign.stats.min_points == 500

    def test_unknown_channel_fails_loudly(self, session: CampaignSession) -> None:
        with pytest.raises(KeyError, match="nope"):
            session.window_table([("scale", "nope")])


def lambda_id(step_id: str):
    import polars as pl

    return pl.col("step_id") == step_id
