"""Uncertainty with autocorrelation, robust slope and stationarity light (Etappe 2b, step 6, package B).

Monte-Carlo with simulated AR(1) series (fixed seed): the intervals must cover the true value about
as often as they claim. Then the OCR scale with misreadings: OLS and Theil-Sen part, the warning shows.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from datualizer_core.ingestion.source_loader import load_source
from datualizer_core.stats import StatsConfig, WindowSpec, window_stats
from datualizer_core.stats import window as wmod

OCR = Path(__file__).parent / "fixtures" / "sa_12_10" / "ocr_recording_105012_extract.csv"
NO_GAPS = np.array([], dtype=np.int64)
T0 = 1.79e9


def ar1(rng: np.random.Generator, n: int, phi: float, sigma: float = 1.0) -> np.ndarray:
    """Stationary AR(1): x_i = φ·x_{i−1} + e_i, started from the stationary distribution."""
    e = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(1 - phi**2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def stats_of(y: np.ndarray, cfg: StatsConfig | None = None):
    x = T0 + np.arange(len(y), dtype=float)
    return window_stats(x, y, NO_GAPS, WindowSpec(x[0], x[-1]), None, cfg or StatsConfig())


# ------------------------------------------------------------------ formulas
class TestEffectiveSampleSize:
    def test_n_eff_and_sem_eff_follow_the_ar1_formula(self) -> None:
        y = 5 + ar1(np.random.default_rng(3), 400, 0.6)
        r = stats_of(y)
        d = y - y.mean()
        rho = float(d[:-1] @ d[1:]) / float(d @ d)
        assert r.rho1 == pytest.approx(rho)
        assert r.n_eff == pytest.approx(400 * (1 - rho) / (1 + rho))
        assert r.sem_eff == pytest.approx(r.sd / math.sqrt(r.n_eff))
        assert r.sem_eff > 1.5 * r.sem  # autocorrelation: the classic SEM is far too small
        assert r.integral_u == pytest.approx(r.sem_eff * r.covered_s)
        assert any("AR(1) approximation" in a for a in r.assumptions)

    def test_negative_autocorrelation_keeps_n(self) -> None:
        y = 5 + np.tile([1.0, -1.0], 50) + np.random.default_rng(1).normal(0, 0.1, 100)
        r = stats_of(y)
        assert r.rho1 < 0 and r.n_eff == 100 and r.sem_eff == pytest.approx(r.sem)

    def test_drift_makes_n_eff_tiny_and_says_so(self) -> None:
        y = 0.05 * np.arange(300) + np.random.default_rng(2).normal(0, 0.1, 300)
        r = stats_of(y)
        assert r.rho1 > 0.9 and r.n_eff < 10
        assert any(c.startswith("n_eff = ") for c in r.checks)

    def test_constant_values(self) -> None:
        r = stats_of(np.full(50, 9683.8))
        assert r.sem_eff == 0.0 and r.rho1 is None and r.n_eff == 50
        assert r.slope_theil_sen == 0.0 and r.stationary is True


class TestSlopeUncertainty:
    def test_hac_with_lag_0_is_whites_estimator(self) -> None:
        rng = np.random.default_rng(4)
        y = 2 - 0.01 * np.arange(150) + rng.normal(0, 1, 150) * np.linspace(0.5, 2, 150)
        r = stats_of(y, StatsConfig(hac_lag=0))
        t = np.arange(150.0)
        dt = t - t.mean()
        res = y - (r.intercept + r.slope * t)
        white = math.sqrt(np.sum((res * dt) ** 2) * 150 / 148) / float(dt @ dt)
        assert r.slope_se_hac == pytest.approx(white)
        assert "slope_se_hac: Newey-West, Bartlett weights, lag 0, small-sample factor n/(n−2)" in r.assumptions

    def test_default_lag_rule(self) -> None:
        r = stats_of(np.random.default_rng(5).normal(size=200))
        assert any("lag 4," in a for a in r.assumptions)  # ⌊4·(200/100)^(2/9)⌋ = 4

    def test_durbin_watson_and_ar1_factor(self) -> None:
        y = 0.02 * np.arange(500) + ar1(np.random.default_rng(6), 500, 0.7)
        r = stats_of(y)
        t = np.arange(500.0)
        res = y - (r.intercept + r.slope * t)
        rho = float(res[:-1] @ res[1:]) / float(res @ res)
        assert r.durbin_watson == pytest.approx(np.sum(np.diff(res) ** 2) / (res @ res))
        assert r.durbin_watson == pytest.approx(2 * (1 - rho), abs=0.02)
        assert r.slope_se_ar1 == pytest.approx(r.slope_se * math.sqrt((1 + rho) / (1 - rho)))
        assert r.slope_se_hac > r.slope_se


class TestTheilSen:
    def test_matches_scipy_with_ties(self) -> None:
        """Quantized readings (0.01 g steps) give many ties in y; tie correction as in scipy."""
        rng = np.random.default_rng(8)
        t = np.arange(120.0)
        y = np.round(9674 - 0.004 * t + rng.normal(0, 0.02, 120), 2)
        b, (lo, hi) = wmod.theil_sen(t, y)
        ref = stats.theilslopes(y, t, alpha=0.95)
        assert b == pytest.approx(ref.slope)
        assert (lo, hi) == (pytest.approx(ref.low_slope), pytest.approx(ref.high_slope))

    def test_autocorrelation_widens_the_interval(self) -> None:
        y = 0.01 * np.arange(300) + ar1(np.random.default_rng(9), 300, 0.8)
        r = stats_of(y)
        _, (lo, hi) = wmod.theil_sen(np.arange(300.0), y)
        assert r.slope_ts_ci[1] - r.slope_ts_ci[0] > 2 * (hi - lo)
        assert any("variance widened by (1+ρ)/(1−ρ)" in a for a in r.assumptions)

    def test_large_window_uses_a_fixed_sample(self) -> None:
        n = 3 * wmod.THEIL_SEN_MAX_POINTS
        rng = np.random.default_rng(10)
        y = 1 + 0.003 * np.arange(n) + rng.normal(0, 1, n)
        r1, r2 = stats_of(y), stats_of(y)
        assert r1.slope_theil_sen == r2.slope_theil_sen and r1.slope_ts_ci == r2.slope_ts_ci  # reproducible
        full = stats.theilslopes(y, np.arange(float(n)), alpha=0.95)
        assert r1.slope_theil_sen == pytest.approx(full.slope, abs=0.02 * (full.high_slope - full.low_slope))
        assert r1.slope_ts_ci[0] == pytest.approx(full.low_slope, abs=0.05 * (full.high_slope - full.low_slope))
        assert any("fixed-seed sample" in a for a in r1.assumptions)
        assert not any("fixed-seed sample" in a for a in stats_of(y[: wmod.THEIL_SEN_MAX_POINTS]).assumptions)

    def test_can_be_switched_off(self) -> None:
        r = stats_of(np.random.default_rng(11).normal(size=50), StatsConfig(robust_slope=False))
        assert r.slope_theil_sen is None and r.slope_ts_ci is None


class TestStationarity:
    def test_step_in_the_middle_is_seen_by_the_halves(self) -> None:
        """A step change leaves the slope test weak but the halves differ."""
        rng = np.random.default_rng(12)
        y = np.r_[np.zeros(100), np.ones(100)] + rng.normal(0, 0.2, 200)
        r = stats_of(y)
        assert r.stationary is False
        assert r.halves_diff == pytest.approx(1.0, abs=0.1)
        assert any("second half differs" in c for c in r.checks)

    def test_noise_is_stationary(self) -> None:
        r = stats_of(3 + np.random.default_rng(13).normal(0, 1, 200))
        assert r.stationary is True and not any("not stationary" in c for c in r.checks)
        assert any(a.startswith("stationary: slope test") and "R statistic not computed" in a for a in r.assumptions)

    def test_too_few_points_gives_no_light(self) -> None:
        r = stats_of(np.array([1.0, 2.0]))
        assert r.stationary is None and r.sem_eff is None


# ------------------------------------------------------------------ Monte-Carlo
@pytest.mark.parametrize("phi, max_false_alarm", [(0.5, 0.10), (0.8, 0.15)])
def test_interval_coverage_on_ar1_series(phi: float, max_false_alarm: float) -> None:
    """300 AR(1) series of 200 points per case (fixed seed); nominal 95 %.

    Measured here: φ = 0.5: mean 0.94, AR(1) slope 0.93, Theil-Sen 0.95, HAC 0.89, classic 0.79,
    light false alarm 0.07; φ = 0.8: mean 0.91, AR(1) 0.92, Theil-Sen 0.93, HAC 0.75, classic 0.53,
    false alarm 0.12. The Newey-West lag rule (lag 4 for n = 200) is too short for strong
    autocorrelation; that is the known limit of the rule, so HAC is only checked against classic.
    """
    rng = np.random.default_rng(1)
    reps, n, true_slope = 300, 200, 0.01
    z = stats.norm.ppf(0.975)
    hit = dict(mean=0, ar1=0, ts=0, hac=0, classic=0, false_alarm=0)
    for _ in range(reps):
        flat = stats_of(3 + ar1(rng, n, phi))
        hit["mean"] += abs(flat.mean - 3) <= stats.t.ppf(0.975, flat.n_eff - 1) * flat.sem_eff
        hit["false_alarm"] += flat.stationary is False
        r = stats_of(3 + true_slope * np.arange(n) + ar1(rng, n, phi))
        hit["ar1"] += abs(r.slope - true_slope) <= z * r.slope_se_ar1
        hit["hac"] += abs(r.slope - true_slope) <= z * r.slope_se_hac
        hit["classic"] += abs(r.slope - true_slope) <= z * r.slope_se
        hit["ts"] += r.slope_ts_ci[0] <= true_slope <= r.slope_ts_ci[1]
    cover = {k: round(float(v) / reps, 3) for k, v in hit.items()}
    assert 0.89 <= cover["mean"] <= 0.99, cover
    assert 0.89 <= cover["ar1"] <= 0.99, cover
    assert 0.90 <= cover["ts"] <= 0.99, cover
    assert cover["classic"] < 0.85, cover  # the problem package B solves
    assert cover["hac"] > cover["classic"] + 0.05, cover
    assert cover["false_alarm"] <= max_false_alarm, cover  # stationary series rarely called drifting


# ------------------------------------------------------------------ real data
class TestOcrScale:
    """OCR of the scale display: misreadings up to 67405 g (instead of ≈ 9672 g), flagged as spikes (P31)."""

    @pytest.fixture(scope="class")
    @staticmethod
    def ocr():
        ds = load_source(OCR)
        t = ds["seconds_from_start"].to_numpy().astype(float)
        y = ds["mass_g"].to_numpy().astype(float)
        spikes = np.asarray(ds.flag_mask(["spike"], channel="mass_g"))
        return t, y, spikes, WindowSpec(t[0], t[-1])

    def test_misreadings_part_ols_and_theil_sen(self, ocr) -> None:
        t, y, _, w = ocr
        r = window_stats(t, y, NO_GAPS, w, None, StatsConfig())
        assert r.n == 298
        assert r.slope * 3600 < -1000  # OLS pulled by the misreadings
        assert -20 < r.slope_theil_sen * 3600 < -15  # robust: the scale loses ≈ 17 g/h
        assert "OLS and Theil-Sen slope differ by more than 2 SE (outliers or a kink in the window?)" in r.checks

    def test_without_the_spikes_they_agree(self, ocr) -> None:
        t, y, spikes, w = ocr
        r = window_stats(t, y, NO_GAPS, w, spikes, StatsConfig())
        assert r.n_excluded == 33
        assert r.slope * 3600 == pytest.approx(r.slope_theil_sen * 3600, abs=1.0)
        assert not any("Theil-Sen" in c for c in r.checks)
        # the mass falls: not stationary, with the reason; the residuals are far from independent
        assert r.stationary is False and any(c.startswith("not stationary: slope") for c in r.checks)
        assert r.durbin_watson < 0.5 and r.slope_se_hac > r.slope_se


def test_window_table_has_the_package_b_columns() -> None:
    """Scale (SQLite) in step "T Einstellung 155": robust and OLS mass flow agree, HAC SE is larger."""
    from datetime import date

    from datualizer_core.campaign import CampaignSession
    from datualizer_core.ingestion.profiles import get_profile
    from datualizer_core.ingestion.source_loader import with_choices

    journal = OCR.parent / "labjournal_12_10_extract.xlsx"
    s = CampaignSession()
    s.add_source(journal, with_choices(journal, get_profile("labjournal_xlsx"), clock_date=date(2026, 10, 12)),
                 source_id="journal")
    s.add_source(OCR.parent / "instrument_reader_12_10_extract.db", source_id="scale")
    s.set_mapping(s.suggest_mapping("journal"))
    table = s.window_table([("scale", "masse")])
    for col in ("sem_eff", "n_eff", "slope_se_hac", "slope_se_ar1", "durbin_watson", "slope_theil_sen",
                "ts_ci_low", "ts_ci_high", "halves_diff", "stationary", "integral_u"):
        assert col in table.columns, col
    row = table.filter(table["step_id"] == "journal:19").row(0, named=True)
    assert row["slope_theil_sen"] * 3600 == pytest.approx(row["slope"] * 3600, abs=0.2)  # ≈ −2.4 g/h
    assert row["ts_ci_low"] < row["slope_theil_sen"] < row["ts_ci_high"]
    assert row["slope_se_hac"] > row["slope_se"]
    assert row["stationary"] is False and "not stationary: slope" in row["checks"]
