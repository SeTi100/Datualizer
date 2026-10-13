"""Statistics of one evaluation window of a channel (Etappe 2b, step 6, package A).

What a measurement step needs from a channel: location and spread, the OLS slope (mass flow from
a scale, drift), a two-point difference with a GUM uncertainty (the lab journal's control) and the
integral (dosed amount). Every result carries the checks that failed and the assumptions it rests
on, in plain text, so nothing is judged silently (see docs/RESEARCH_FENSTER_STATISTIK.md).

Rules:

- Only finite values that are not excluded by a flag are used; excluded and empty values are counted.
- Time gaps (`gap_rows`) and empty or excluded values are never bridged: the integral is taken
  per gap-free stretch and the uncovered time is reported, not interpolated.
- Autocorrelation is not handled yet (package B): `sem` and `slope_se` assume independent
  values and are too small for densely sampled sensors. The fields of package B are None.

The module knows nothing about campaigns or the GUI; `CampaignSession.window_table` calls it.
Only numpy, no scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


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
    hac_lag: int | None = Field(None, ge=0, description="Newey-West lag (package B); None = ⌊4·(n/100)^(2/9)⌋.")
    robust_slope: bool = Field(True, description="Also compute the Theil-Sen slope (package B).")
    min_points: int = Field(5, ge=1, description="Fewer usable points in a window are reported in checks.")
    min_coverage: float = Field(0.8, ge=0.0, le=1.0, description="Lower coverage is reported in checks.")
    reading_u: dict[str, float] = Field(
        default_factory=dict,
        description="Standard uncertainty of one reading per 'source/channel' (type B, e.g. resolution/√12); "
        "an input from the device data, never a code constant. Missing = 0, said in the assumptions.",
    )


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
    """Characteristic values of one window. None = cannot be computed (reason in `checks`) or package B."""

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
    # package B (autocorrelation, robust slope, stationarity): not computed yet
    sem_eff: float | None = None
    rho1: float | None = None
    n_eff: float | None = None
    slope_se_hac: float | None = None
    durbin_watson: float | None = None
    slope_theil_sen: float | None = None
    slope_ts_ci: tuple[float, float] | None = None
    integral_u: float | None = None
    stationary: bool | None = None
    checks: list[str] = field(default_factory=list)  # what failed or needs a look
    assumptions: list[str] = field(default_factory=list)  # what the values rest on


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
) -> WindowResult:
    """Statistics of `y` over `x` inside `window`.

    `x`: time per row (master epoch seconds; NaN rows cannot be placed and are ignored).
    `y`: values (NaN = no value). `gap_rows`: rows where a time gap starts (the step from the
    previous row into this one is not data). `flags`: True where a row is excluded (e.g. spike).
    `reading_u`: standard uncertainty of one reading. `clock_u_s`: uncertainty of the source clock
    (`ClockModel.uncertainty_s`); it enters u(Δt) of the two-point difference.
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

    checks: list[str] = []
    assumptions = [
        "excluded flags: " + (", ".join(config.exclude_flags) if config.exclude_flags else "none"),
        "sem and slope_se assume independent values; autocorrelation (sem_eff, n_eff, HAC) not computed yet",
        "integral: trapezoid per gap-free stretch, gaps and empty or excluded values not bridged",
    ]

    covered, integral = _covered_and_integral(t, v, usable, br, a, b)
    coverage = covered / window.length_s
    if n == 0:
        checks.append("no data in the window")
    elif n < config.min_points:
        checks.append(f"only {n} points (min {config.min_points})")
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

    return WindowResult(
        window_s=window.length_s, n=n, n_excluded=n_excluded, n_missing=n_missing,
        coverage=coverage, covered_s=covered, **loc, **fit,
        two_point=None if tp is None else tp[0], two_point_u=None if tp is None else tp[1],
        two_point_dt=None if tp is None else tp[2], two_point_offset_s=None if tp is None else tp[3],
        integral=integral, integral_gap_s=window.length_s - covered,
        checks=checks, assumptions=assumptions,
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
    """(Δy/Δt, u, Δt, largest edge distance); u² = (u1² + u2²)/Δt² + (Δy/Δt²)²·u²(Δt)."""
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
    return dy / dt, math.sqrt(u2), dt, max(p1[3], p2[3])
