"""Statistics of one evaluation window of a channel (Etappe 2b, step 6, package A).

What a measurement step needs from a channel: location and spread, the OLS slope (mass flow from
a scale, drift), a two-point difference with a GUM uncertainty (the lab journal's control) and the
integral (dosed amount). Every result carries the checks that failed and the assumptions it rests
on, in plain text, so nothing is judged silently (see docs/RESEARCH_FENSTER_STATISTIK.md).

Rules:

- Only finite values that are not excluded by a flag are used; excluded and empty values are counted.
- Time gaps (`gap_rows`) and empty or excluded values are never bridged: the integral is taken
  per gap-free stretch and the uncovered time is reported, not interpolated.
- `sem` and `slope_se` assume independent values. Package B adds what densely sampled, autocorrelated
  sensors need: n_eff and sem_eff (AR(1) approximation), the Newey-West (HAC) and AR(1)-corrected
  slope SE, Durbin-Watson, the robust Theil-Sen slope with a Kendall confidence interval and a
  stationarity light (slope test and comparison of the two window halves).
- Practical stationarity (PR 3): with 1-Hz data almost every slope is significant, what matters is
  whether it is small. With a tolerance δ per channel (`StatsConfig.equivalence`) an equivalence
  test (TOST) asks whether the 90 % interval of the slope lies inside ±δ (`stationary_practical`).

The module knows nothing about campaigns or the GUI; `CampaignSession.window_table` calls it.
scipy only for the t and normal quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as _sps

# Theil-Sen takes all n(n-1)/2 pairwise slopes. Measured (numpy, one core): n = 1000 ≈ 23 ms and
# 4 MB, n = 2000 ≈ 0.2 s and 16 MB, n = 5000 ≈ 1 s and 100 MB. Above this many points a fixed-seed
# random sample of THEIL_SEN_MAX_PAIRS pairs is taken instead (said in the assumptions).
THEIL_SEN_MAX_POINTS = 1000
THEIL_SEN_MAX_PAIRS = THEIL_SEN_MAX_POINTS * (THEIL_SEN_MAX_POINTS - 1) // 2
CONFIDENCE = 0.95  # level of the intervals behind the stationarity light and the Theil-Sen interval
TOST_ALPHA = 0.05  # each one-sided test of the equivalence test; together a 1 − 2α = 90 % interval


class StatsConfig(BaseModel):
    """Settings of the window statistics; serializable, stored in campaign.json."""

    model_config = ConfigDict(extra="forbid")

    exclude_flags: list[str] = Field(
        default_factory=lambda: ["spike", "dropout"],
        description="Quality flags whose points are left out (counted in n_excluded).",
    )
    point_halfwidth_s: float = Field(
        0.0, ge=0.0,
        description="Two-point value at an edge: 0 = nearest usable point, else median of the points within ±h.",
    )
    hac_lag: int | None = Field(None, ge=0, description="Newey-West lag; None = ⌊4·(n/100)^(2/9)⌋.")
    robust_slope: bool = Field(True, description="Also compute the Theil-Sen slope and its interval.")
    min_points: int = Field(5, ge=1, description="Fewer usable points in a window are reported in checks.")
    min_coverage: float = Field(0.8, ge=0.0, le=1.0, description="Lower coverage is reported in checks.")
    reading_u: dict[str, float] = Field(
        default_factory=dict,
        description="Standard uncertainty of one reading per 'source/channel' (type B, e.g. resolution/√12); "
        "an input from the device data, never a code constant. Missing = 0, said in the assumptions.",
    )
    equivalence: dict[str, float] = Field(
        default_factory=dict,
        description="Tolerance δ of the slope per 'source/channel' in unit per second (e.g. 0.1 mgC/min = "
        "0.1/60): practically stationary if the 90 % interval of the slope lies inside ±δ (TOST). "
        "Missing = not tested.",
    )


class Check(str):
    """One finding in `WindowResult.checks`: plain text with a traffic-light `level`.

    A `str`, so joined tables and text searches keep working; `level` is "fail" when the window
    cannot be evaluated (no or too few points), otherwise "warn".
    """

    level: str

    def __new__(cls, text: str, level: str = "warn") -> "Check":
        obj = super().__new__(cls, text)
        obj.level = level
        return obj


@dataclass(frozen=True)
class WindowSpec:
    """Evaluated time span in master epoch seconds (e.g. `ResolvedInterval.window`)."""

    start: float
    end: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", float(self.start))
        object.__setattr__(self, "end", float(self.end))
        if not (self.end > self.start):
            raise ValueError(f"Window end {self.end} is not after start {self.start}.")

    @property
    def length_s(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class WindowResult:
    """Characteristic values of one window. None = cannot be computed (reason in `checks`)."""

    window_s: float
    n: int  # usable points (finite, not excluded)
    n_excluded: int  # finite points left out because of `exclude_flags`
    n_missing: int  # rows in the window without a value
    coverage: float  # covered time / window length (gaps, empty and excluded values not covered)
    covered_s: float
    # location and spread
    mean: float | None
    median: float | None
    sd: float | None
    sem: float | None  # sd/√n: assumes independent values
    min: float | None
    max: float | None
    quantiles: tuple[float, float] | None  # 5 % and 95 %
    # OLS line over time (seconds)
    slope: float | None  # unit per second
    slope_se: float | None  # classic standard error
    intercept: float | None  # value of the fit line at the window start
    r2: float | None
    residual_sd: float | None
    # two-point difference between the window edges
    two_point: float | None  # unit per second
    two_point_u: float | None  # standard uncertainty (GUM)
    two_point_dt: float | None  # time between the two support points
    two_point_offset_s: float | None  # largest distance of a support point from its window edge
    # integral (trapezoid, gaps not bridged)
    integral: float  # unit · seconds over the covered time
    integral_gap_s: float  # window time without coverage, not integrated
    # autocorrelation, robust slope, stationarity (package B)
    sem_eff: float | None = None  # sd/√n_eff
    rho1: float | None = None  # lag-1 autocorrelation of the deviations from the mean
    n_eff: float | None = None  # n·(1−ρ₁)/(1+ρ₁), AR(1) approximation; n if ρ₁ ≤ 0
    slope_se_hac: float | None = None  # Newey-West standard error of the OLS slope
    slope_se_ar1: float | None = None  # slope_se·√((1+ρ)/(1−ρ)), ρ = lag-1 autocorrelation of the residuals
    durbin_watson: float | None = None  # of the OLS residuals; ≈ 2 = no lag-1 autocorrelation
    slope_theil_sen: float | None = None  # median of the pairwise slopes
    slope_ts_ci: tuple[float, float] | None = None  # 95 % interval (Kendall, AR(1)-widened)
    integral_u: float | None = None  # sem_eff · covered time (approximation)
    halves_diff: float | None = None  # mean of the second half minus mean of the first half
    stationary: bool | None = None  # light: no significant slope and no significant halves difference
    stationary_practical: bool | None = None  # TOST: 90 % interval of the slope inside ±δ; None = no δ
    equivalence_delta: float | None = None  # the δ tested (unit per second)
    checks: list[Check] = field(default_factory=list)  # what failed or needs a look (with a level)
    assumptions: list[str] = field(default_factory=list)  # what the values rest on
    # for drawing (window inspector); not table columns
    intercept_theil_sen: float | None = None  # Theil-Sen line at the window start: median(y − b·t)
    used_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64), metadata={"table": False})
    excluded_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64), metadata={"table": False})
    two_point_points: tuple[tuple[float, float], tuple[float, float]] | None = field(
        default=None, metadata={"table": False}
    )  # (time, value) of both support points

    @property
    def level(self) -> str:
        """Traffic light of the whole window: "fail", "warn" (any finding) or "ok"."""
        levels = {getattr(c, "level", "warn") for c in self.checks}
        return "fail" if "fail" in levels else "warn" if levels else "ok"

    @property
    def slope_se_eff(self) -> float | None:
        """Larger of `slope_se_hac` and `slope_se_ar1` (what the stationarity light uses)."""
        ses = [v for v in (self.slope_se_hac, self.slope_se_ar1) if v is not None]
        return max(ses) if ses else None


def window_stats(
    x: np.ndarray,
    y: np.ndarray,
    gap_rows: np.ndarray,
    window: WindowSpec,
    flags: np.ndarray | None,
    config: StatsConfig,
    *,
    reading_u: float = 0.0,
    clock_u_s: float | None = None,
    equivalence: float | None = None,
) -> WindowResult:
    """Statistics of `y` over `x` inside `window`.

    `x`: time per row (master epoch seconds; NaN rows cannot be placed and are ignored).
    `y`: values (NaN = no value). `gap_rows`: rows where a time gap starts (the step from the
    previous row into this one is not data). `flags`: True where a row is excluded (e.g. spike).
    `reading_u`: standard uncertainty of one reading. `clock_u_s`: uncertainty of the source clock
    (`ClockModel.uncertainty_s`); it enters u(Δt) of the two-point difference. `equivalence`:
    tolerance δ of the slope (unit per second) for `stationary_practical`; None = not tested.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    excluded = np.zeros(len(x), dtype=bool) if flags is None else np.asarray(flags, dtype=bool)
    gap = np.zeros(len(x), dtype=bool)
    g = np.asarray(gap_rows, dtype=np.int64)
    gap[g[(g >= 0) & (g < len(x))]] = True

    # Placeable rows in time order; a gap row breaks the link into it even if rows are reordered
    rows = np.flatnonzero(np.isfinite(x))
    rows = rows[np.argsort(x[rows], kind="stable")]
    t, v, ex, br = x[rows], y[rows], excluded[rows], gap[rows]
    usable = np.isfinite(v) & ~ex

    a, b = window.start, window.end
    inside = (t >= a) & (t <= b)
    n_excluded = int(np.sum(inside & np.isfinite(v) & ex))
    n_missing = int(np.sum(inside & ~np.isfinite(v)))
    tw, yw = t[inside & usable], v[inside & usable]
    n = len(yw)
    used_rows = rows[inside & usable]
    excluded_rows = rows[inside & np.isfinite(v) & ex]

    checks: list[str] = []
    assumptions = [
        "excluded flags: " + (", ".join(config.exclude_flags) if config.exclude_flags else "none"),
        "sem and slope_se assume independent values; sem_eff, slope_se_hac and slope_se_ar1 allow for autocorrelation",
        "integral: trapezoid per gap-free stretch, gaps and empty or excluded values not bridged",
    ]

    covered, integral = _covered_and_integral(t, v, usable, br, a, b)
    coverage = covered / window.length_s
    if n == 0:
        checks.append(Check("no data in the window", "fail"))
    elif n < config.min_points:
        checks.append(Check(f"only {n} points (min {config.min_points})", "fail"))
    if n and coverage < config.min_coverage:
        checks.append(f"coverage {coverage:.0%} below {config.min_coverage:.0%}")
    if n_excluded:
        checks.append(f"{n_excluded} points excluded by flags")
    if n_missing:
        checks.append(f"{n_missing} rows without a value")

    loc = _location(yw)
    if n and loc["sd"] == 0.0:
        checks.append("constant values in the window (quantized or frozen?)")
    fit = _ols(tw - a, yw)

    tp = _two_point(t, v, usable, window, config.point_halfwidth_s, reading_u, clock_u_s)
    if tp is None and n:
        checks.append("two-point difference: no usable value at an edge")
    if clock_u_s is None:
        assumptions.append("clock uncertainty not given (u(Δt) from the clock = 0)")
    else:
        assumptions.append(f"u(Δt) = √2 · {clock_u_s:g} s clock uncertainty (both points independent)")
    if reading_u == 0.0:
        assumptions.append("reading uncertainty not given (u(y) of a single reading = 0)")
    if tp is not None and fit["slope"] is not None and fit["slope_se"] is not None:
        u = math.hypot(tp[1], fit["slope_se"])
        if u > 0 and abs(tp[0] - fit["slope"]) > 2 * u:
            checks.append("two-point and OLS slope differ by more than 2 u (outlier at an edge or a kink?)")

    auto = _autocorrelation(tw - a, yw, loc, fit, config.hac_lag, checks, assumptions)
    ts = _theil_sen(tw - a, yw, fit, checks, assumptions) if config.robust_slope else {}
    st = _stationarity(tw - a, yw, window.length_s, loc, fit, auto, checks, assumptions)
    st.update(_equivalence(fit["slope"], auto, equivalence, checks, assumptions))
    integral_u = None if auto.get("sem_eff") is None else auto["sem_eff"] * covered
    if integral_u is not None:
        assumptions.append("integral_u = sem_eff · covered time (approximation; gap time not included)")

    return WindowResult(
        window_s=window.length_s, n=n, n_excluded=n_excluded, n_missing=n_missing,
        coverage=coverage, covered_s=covered, **loc, **fit,
        two_point=None if tp is None else tp[0], two_point_u=None if tp is None else tp[1],
        two_point_dt=None if tp is None else tp[2], two_point_offset_s=None if tp is None else tp[3],
        two_point_points=None if tp is None else tp[4],
        integral=integral, integral_gap_s=window.length_s - covered,
        **auto, **ts, **st, integral_u=integral_u,
        checks=[c if isinstance(c, Check) else Check(c) for c in checks], assumptions=assumptions,
        used_rows=used_rows, excluded_rows=excluded_rows,
    )


# ------------------------------------------------------------------ pieces
def _location(y: np.ndarray) -> dict:
    n = len(y)
    if n == 0:
        return dict(mean=None, median=None, sd=None, sem=None, min=None, max=None, quantiles=None)
    constant = np.ptp(y) == 0  # exact: no rounding residue from the mean of equal values
    sd = (0.0 if constant else float(np.std(y, ddof=1))) if n > 1 else None
    return dict(
        mean=float(np.mean(y)), median=float(np.median(y)), sd=sd,
        sem=None if sd is None else sd / math.sqrt(n),
        min=float(np.min(y)), max=float(np.max(y)),
        quantiles=(float(np.quantile(y, 0.05)), float(np.quantile(y, 0.95))),
    )


def _ols(t: np.ndarray, y: np.ndarray) -> dict:
    """y = intercept + slope·t (t from the window start); classic SE with n − 2 degrees of freedom."""
    none = dict(slope=None, slope_se=None, intercept=None, r2=None, residual_sd=None)
    n = len(t)
    if n < 2:
        return none
    tm = t.mean()
    ym = y[0] if np.ptp(y) == 0 else y.mean()
    dt, dy = t - tm, y - ym
    sxx = float(dt @ dt)
    if sxx == 0.0:
        return none
    slope = float(dt @ dy) / sxx
    res = dy - slope * dt
    sse, sst = float(res @ res), float(dy @ dy)
    s_res = math.sqrt(sse / (n - 2)) if n > 2 else None
    return dict(
        slope=slope,
        slope_se=None if s_res is None else s_res / math.sqrt(sxx),
        intercept=float(ym - slope * tm),
        r2=None if sst == 0.0 else 1.0 - sse / sst,
        residual_sd=s_res,
    )


def _covered_and_integral(
    t: np.ndarray, v: np.ndarray, usable: np.ndarray, gap: np.ndarray, a: float, b: float
) -> tuple[float, float]:
    """Covered time and trapezoid integral inside [a, b].

    A link between two neighbouring rows counts if both have a usable value and no time gap
    starts at the second one. Links crossing an edge are cut there, the edge value interpolated.
    """
    if len(t) < 2:
        return 0.0, 0.0
    t0, t1, y0, y1 = t[:-1], t[1:], v[:-1], v[1:]
    ok = usable[:-1] & usable[1:] & ~gap[1:] & (t1 > t0)
    lo, hi = np.maximum(t0, a), np.minimum(t1, b)
    ok &= hi > lo
    if not ok.any():
        return 0.0, 0.0
    t0, t1, y0, y1, lo, hi = t0[ok], t1[ok], y0[ok], y1[ok], lo[ok], hi[ok]
    k = (y1 - y0) / (t1 - t0)
    ylo, yhi = y0 + k * (lo - t0), y0 + k * (hi - t0)
    return float(np.sum(hi - lo)), float(np.sum((hi - lo) * (ylo + yhi) / 2))


def _edge_value(
    t: np.ndarray, v: np.ndarray, usable: np.ndarray, w: WindowSpec, at: float, h: float, reading_u: float
) -> tuple[float, float, float, float] | None:
    """(value, u(value), time, distance) at window edge `at`, from usable points inside the window.

    h = 0: the nearest point. h > 0: the median of the points within ±h of the edge (inside the
    window that is the half towards the middle); its time is the mean time of those points.
    """
    keep = usable & (t >= w.start) & (t <= w.end)
    tu, vu = t[keep], v[keep]
    if not len(tu):
        return None
    if h > 0:
        near = np.abs(tu - at) <= h
        if not near.any():
            return None
        vals, times = vu[near], tu[near]
        m = len(vals)
        # standard error of a median ≈ 1.2533·σ/√m (normal values)
        ua = 1.2533 * float(np.std(vals, ddof=1)) / math.sqrt(m) if m > 1 else 0.0
        tm = float(np.mean(times))
        return float(np.median(vals)), math.hypot(ua, reading_u), tm, abs(tm - at)
    i = int(np.argmin(np.abs(tu - at)))
    return float(vu[i]), reading_u, float(tu[i]), float(abs(tu[i] - at))


def _two_point(
    t: np.ndarray, v: np.ndarray, usable: np.ndarray, w: WindowSpec, h: float,
    reading_u: float, clock_u_s: float | None,
) -> tuple[float, float, float, float] | None:
    """(Δy/Δt, u, Δt, largest edge distance, ((t1, y1), (t2, y2))); u² = (u1² + u2²)/Δt² + (Δy/Δt²)²·u²(Δt)."""
    p1 = _edge_value(t, v, usable, w, w.start, h, reading_u)
    p2 = _edge_value(t, v, usable, w, w.end, h, reading_u)
    if p1 is None or p2 is None:
        return None
    dt = p2[2] - p1[2]
    if dt <= 0:
        return None
    dy = p2[0] - p1[0]
    u_dt2 = 2 * (clock_u_s or 0.0) ** 2
    u2 = (p1[1] ** 2 + p2[1] ** 2) / dt**2 + (dy / dt**2) ** 2 * u_dt2
    return dy / dt, math.sqrt(u2), dt, max(p1[3], p2[3]), ((p1[2], p1[0]), (p2[2], p2[0]))


# ------------------------------------------------------------------ package B
def _lag1(d: np.ndarray) -> float | None:
    """Lag-1 autocorrelation of deviations `d` (consecutive values, equal spacing assumed)."""
    ss = float(d @ d)
    if len(d) < 3 or ss == 0.0:
        return None
    return float(d[:-1] @ d[1:]) / ss


def _n_eff(n: int, rho: float | None) -> float:
    """n·(1−ρ)/(1+ρ) (AR(1)); n for ρ ≤ 0 or unknown, never below 1."""
    if rho is None or rho <= 0:
        return float(n)
    return max(n * (1 - rho) / (1 + rho), 1.0)


def _residuals(t: np.ndarray, y: np.ndarray, fit: dict) -> np.ndarray | None:
    if fit["slope"] is None or len(t) < 3:
        return None
    return y - (fit["intercept"] + fit["slope"] * t)


def _autocorrelation(
    t: np.ndarray, y: np.ndarray, loc: dict, fit: dict, hac_lag: int | None,
    checks: list[str], assumptions: list[str],
) -> dict:
    """n_eff, sem_eff and the autocorrelation-aware SE of the OLS slope."""
    n = len(y)
    out = dict(sem_eff=None, rho1=None, n_eff=None, slope_se_hac=None, slope_se_ar1=None, durbin_watson=None)
    if n < 3 or loc["sd"] is None:
        return out
    rho = None if loc["sd"] == 0.0 else _lag1(y - loc["mean"])
    n_eff = _n_eff(n, rho)
    out.update(rho1=rho, n_eff=n_eff, sem_eff=loc["sd"] / math.sqrt(n_eff))
    assumptions.append("n_eff, sem_eff: AR(1) approximation from consecutive usable points (equal spacing assumed)")
    if n_eff < 10 <= n:
        checks.append(f"n_eff = {n_eff:.1f} of {n} points: strong autocorrelation, the mean is hardly reliable")

    res = _residuals(t, y, fit)
    if res is None:
        return out
    sse = float(res @ res)
    if sse > 0:
        out["durbin_watson"] = float(np.sum(np.diff(res) ** 2)) / sse
    rho_res = _lag1(res)
    if fit["slope_se"] is not None:
        if rho_res is None or rho_res <= 0:
            out["slope_se_ar1"] = fit["slope_se"]
        elif rho_res < 1:
            out["slope_se_ar1"] = fit["slope_se"] * math.sqrt((1 + rho_res) / (1 - rho_res))

    # Newey-West: OLS point estimate, HAC variance with Bartlett weights
    lag = hac_lag if hac_lag is not None else int(math.floor(4 * (n / 100) ** (2 / 9)))
    lag = min(lag, n - 1)
    dt = t - t.mean()
    sxx = float(dt @ dt)
    u = res * dt
    s = float(u @ u)
    for j in range(1, lag + 1):
        s += 2 * (1 - j / (lag + 1)) * float(u[j:] @ u[:-j])
    out["slope_se_hac"] = math.sqrt(max(s, 0.0) * n / (n - 2)) / sxx
    assumptions.append(f"slope_se_hac: Newey-West, Bartlett weights, lag {lag}, small-sample factor n/(n−2)")
    assumptions.append("slope_se_ar1: AR(1) approximation from the lag-1 autocorrelation of the residuals")
    return out


def _pairwise_slopes(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """(slopes, number of all pairs with distinct times). Above THEIL_SEN_MAX_POINTS a sample."""
    n = len(t)
    if n <= THEIL_SEN_MAX_POINTS:
        parts = []
        for k in range(1, n):
            d = t[k:] - t[:-k]
            ok = d != 0
            parts.append((y[k:][ok] - y[:-k][ok]) / d[ok])
        s = np.concatenate(parts)
        return s, len(s)
    rng = np.random.default_rng(0)  # fixed seed: the same window gives the same result
    i = rng.integers(0, n, THEIL_SEN_MAX_PAIRS)
    j = rng.integers(0, n, THEIL_SEN_MAX_PAIRS)
    d = t[j] - t[i]
    ok = d != 0
    _, tc = np.unique(t, return_counts=True)
    n_pairs = n * (n - 1) // 2 - int(np.sum(tc * (tc - 1) // 2))
    return (y[j][ok] - y[i][ok]) / d[ok], n_pairs


def _tie_term(v: np.ndarray) -> float:
    _, c = np.unique(v, return_counts=True)
    c = c[c > 1].astype(float)
    return float(np.sum(c * (c - 1) * (2 * c + 5)))


def theil_sen(
    t: np.ndarray, y: np.ndarray, confidence: float = CONFIDENCE, var_factor: float = 1.0
) -> tuple[float, tuple[float, float]] | None:
    """Theil-Sen slope and its interval from the Kendall distribution (normal approximation).

    Var(S) = [n(n−1)(2n+5) − Σ ties in t − Σ ties in y]/18, times `var_factor` (≥ 1 widens the
    interval for autocorrelated values). The interval bounds are the pairwise slopes at the ranks
    (N ∓ z·√Var(S))/2 (as scipy.stats.theilslopes). None with fewer than 2 distinct times.
    """
    if len(t) < 2:
        return None
    slopes, n_pairs = _pairwise_slopes(t, y)
    if not len(slopes):
        return None
    return float(np.median(slopes)), _kendall_interval(t, y, slopes, n_pairs, confidence, var_factor)


def _kendall_interval(
    t: np.ndarray, y: np.ndarray, slopes: np.ndarray, n_pairs: int, confidence: float, var_factor: float
) -> tuple[float, float]:
    n, m = len(t), len(slopes)
    var_s = (n * (n - 1) * (2 * n + 5) - _tie_term(t) - _tie_term(y)) / 18 * var_factor
    c = float(_sps.norm.ppf(0.5 + confidence / 2)) * math.sqrt(max(var_s, 0.0))
    hi = min(int(round((n_pairs + c) / 2)), n_pairs - 1)
    lo = max(int(round((n_pairs - c) / 2)) - 1, 0)
    if m != n_pairs:  # sample: same quantiles of the sampled slopes
        lo = int(round(lo / max(n_pairs - 1, 1) * (m - 1)))
        hi = int(round(hi / max(n_pairs - 1, 1) * (m - 1)))
    part = np.partition(slopes, sorted({lo, hi}))
    return float(part[lo]), float(part[hi])


def _theil_sen(t: np.ndarray, y: np.ndarray, fit: dict, checks: list[str], assumptions: list[str]) -> dict:
    if len(t) < 3:
        return {}
    slopes, n_pairs = _pairwise_slopes(t, y)
    if not len(slopes):
        return {}
    b = float(np.median(slopes))
    res = y - b * t
    rho = _lag1(res - np.median(res))
    factor = (1 + rho) / (1 - rho) if rho is not None and 0 < rho < 1 else 1.0
    ci = _kendall_interval(t, y, slopes, n_pairs, CONFIDENCE, factor)
    note = "slope_ts_ci: Kendall normal approximation with tie correction"
    if factor != 1.0:
        note += f", variance widened by (1+ρ)/(1−ρ) = {factor:.2g} for autocorrelated residuals (approximation)"
    assumptions.append(note)
    if len(t) > THEIL_SEN_MAX_POINTS:
        assumptions.append(
            f"Theil-Sen from a fixed-seed sample of {THEIL_SEN_MAX_PAIRS} pairs (n = {len(t)} > {THEIL_SEN_MAX_POINTS})"
        )
    # the OLS SE would not do: the very outliers this is meant to reveal inflate it
    se_ts = (ci[1] - ci[0]) / (2 * float(_sps.norm.ppf(0.5 + CONFIDENCE / 2)))
    if fit["slope"] is not None and abs(fit["slope"] - b) > 2 * se_ts:
        checks.append("OLS and Theil-Sen slope differ by more than 2 SE (outliers or a kink in the window?)")
    return dict(slope_theil_sen=b, slope_ts_ci=ci, intercept_theil_sen=float(np.median(res)))


def _significant(diff: float, u: float) -> tuple[bool, float]:
    """(|diff| > z · u, |diff|/u); each of the two light tests at half the error level.

    Normal, not t quantiles: u already allows for autocorrelation, and t with n_eff degrees of
    freedom punished it twice (a clear drift of 6 SE on the OCR scale went unnoticed).
    """
    if u == 0.0:
        return diff != 0.0, math.inf if diff else 0.0
    level = 1 - (1 - CONFIDENCE) / 2  # Bonferroni: the light as a whole errs in ≈ 5 % of stationary windows
    return abs(diff) > float(_sps.norm.ppf(0.5 + level / 2)) * u, abs(diff) / u


def _stationarity(
    t: np.ndarray, y: np.ndarray, length_s: float, loc: dict, fit: dict, auto: dict,
    checks: list[str], assumptions: list[str],
) -> dict:
    """Light from a slope test (HAC) and a comparison of the two window halves."""
    verdicts = []
    halves_diff = None
    b = fit["slope"]
    ses = [v for v in (auto["slope_se_hac"], auto["slope_se_ar1"]) if v is not None]
    se = max(ses) if ses else None  # the HAC lag rule undercovers at strong autocorrelation
    if b is not None and se is not None:
        sig, ratio = _significant(b, se)
        verdicts.append(not sig)
        if sig:
            drift = b * length_s
            rel = f" = {abs(drift) / loc['sd']:.1f} sd" if loc["sd"] else ""
            checks.append(
                f"not stationary: slope {b:.3g}/s significant ({ratio:.1f} SE), drift over the window {drift:.3g}{rel}"
            )
    first, second = y[t < length_s / 2], y[t >= length_s / 2]
    if len(first) >= 3 and len(second) >= 3:
        u2 = 0.0
        for h in (first, second):
            sd = float(np.std(h, ddof=1)) if np.ptp(h) else 0.0
            u2 += sd**2 / _n_eff(len(h), _lag1(h - h.mean()) if sd else None)  # sem_eff² of the half
        halves_diff = float(second.mean() - first.mean())
        sig, ratio = _significant(halves_diff, math.sqrt(u2))
        verdicts.append(not sig)
        if sig:
            checks.append(f"not stationary: second half differs from the first by {halves_diff:.3g} ({ratio:.1f} u_eff)")
    if verdicts:
        assumptions.append(
            f"stationary: slope test (larger of slope_se_hac and slope_se_ar1) and halves comparison, "
            f"together at {CONFIDENCE:.0%} (Bonferroni), normal quantiles; R statistic not computed"
        )
    return dict(halves_diff=halves_diff, stationary=all(verdicts) if verdicts else None)


def _equivalence(slope: float | None, auto: dict, delta: float | None, checks: list[str], assumptions: list[str]) -> dict:
    """Practical stationarity: two one-sided tests (TOST) that the slope lies inside ±δ.

    Equivalent to the 1 − 2α interval slope ± z₁₋α · SE lying inside ±δ. SE is the larger of
    slope_se_hac and slope_se_ar1, normal quantiles as in the light. Unlike the light, a large n
    makes this test more lenient, not stricter: a tiny but certain drift is practically stationary.
    """
    if delta is None:
        return dict(stationary_practical=None, equivalence_delta=None)
    delta = abs(float(delta))
    ses = [v for v in (auto["slope_se_hac"], auto["slope_se_ar1"]) if v is not None]
    if slope is None or not ses:
        checks.append("practical stationarity: no slope with an autocorrelation-aware SE, not tested")
        return dict(stationary_practical=None, equivalence_delta=delta)
    z = float(_sps.norm.ppf(1 - TOST_ALPHA))
    lo, hi = slope - z * max(ses), slope + z * max(ses)
    ok = -delta < lo and hi < delta
    assumptions.append(
        f"stationary_practical: TOST at {TOST_ALPHA:.0%} per side, {1 - 2 * TOST_ALPHA:.0%} interval of the slope "
        f"[{lo:.3g}, {hi:.3g}]/s (larger of slope_se_hac and slope_se_ar1, normal quantiles) "
        f"{'inside' if ok else 'not inside'} ±δ = ±{delta:.3g}/s"
    )
    if not ok:
        checks.append(
            f"not practically stationary: {1 - 2 * TOST_ALPHA:.0%} interval of the slope [{lo:.3g}, {hi:.3g}]/s "
            f"not inside ±{delta:.3g}/s"
        )
    return dict(stationary_practical=ok, equivalence_delta=delta)
