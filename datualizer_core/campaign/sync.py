"""Clock alignment tools (Etappe 2b, step 7): offsets from written clock pairs and cross-correlation.

Both only *suggest* a clock model; the user applies it, and the model states how it was found
(`method`, `note`, `uncertainty_s`).

- Cell pairs: the lab journal holds the same events on two clocks ("Start FID" on the analyzer
  clock, "Start Real" on the PC clock). Each pair is one `ClockAnchor`; the offset is the median
  of the pair offsets, the spread their median absolute deviation.
- Cross-correlation: two tracks resampled onto one grid inside a chosen time span, compared at
  every lag of a bounded search range (Pearson r over the samples both have). Gaps of either
  track are never bridged; they only reduce the overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import numpy as np

from datualizer_core.campaign.clock import ClockAnchor, ClockCells, ClockModel, to_epoch_s
from datualizer_core.ingestion.readers.excel import read_cells
from datualizer_core.stats.window import Check

JOURNAL_PAIRS = "journal_pairs"
CROSS_CORRELATION = "cross_correlation"

# Cross-correlation checks (stated with the result, not hidden)
R_WARN = 0.7   # peak r below: weak similarity
R_FAIL = 0.3   # peak r below: no usable similarity
AMBIGUITY = 0.05  # another peak within this r of the best one: ambiguous lag
MIN_OVERLAP = 0.5  # lags where both tracks share fewer than this share of the reference samples are left out
MAX_GRID = 400_000  # grid points (span + search range); a finer step is coarsened and a note says so


class ClockCellsError(ValueError):
    """A configured cell is empty or holds no readable time."""


# ------------------------------------------------------------------ cell pairs
def _excel_day_fraction(value: float) -> time:
    seconds = round(value * 86400.0, 6)
    whole = int(seconds)
    return time(whole // 3600, whole // 60 % 60, whole % 60, int(round((seconds - whole) * 1e6)))


def cell_datetime(value: Any, day: date | None, ref: str) -> datetime:
    """Date-time of one cell value; times without date get `day` (an error names the cell)."""
    if isinstance(value, datetime):
        if value.date() in (date(1899, 12, 30), date(1899, 12, 31), date(1900, 1, 1)):  # Excel time-only
            value = value.time()
        else:
            return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value < 1.0:
        value = _excel_day_fraction(float(value))  # time stored as fraction of a day
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%H:%M:%S", "%H:%M", "%H:%M:%S.%f"):
            try:
                value = datetime.strptime(text, fmt).time()
                break
            except ValueError:
                pass
        else:
            try:
                return datetime.fromisoformat(text)
            except ValueError:
                pass
    if isinstance(value, time):
        if day is None:
            raise ClockCellsError(f"Cell {ref} holds a time without date ({value}); set the date of the pairs.")
        return datetime.combine(day, value)
    if value is None:
        raise ClockCellsError(f"Cell {ref} is empty.")
    raise ClockCellsError(f"Cell {ref} holds no time: {value!r}.")


def anchors_from_cells(path: str | Path, cells: ClockCells) -> list[ClockAnchor]:
    """One anchor per configured cell pair (source time, master time) of the workbook at `path`."""
    refs = [ref for p in cells.pairs for ref in (p.source_cell, p.master_cell)]
    values = read_cells(path, cells.sheet, refs)
    anchors = []
    for p in cells.pairs:
        note = p.note or f"{p.source_cell} → {p.master_cell}"
        anchors.append(ClockAnchor(
            source_time=cell_datetime(values[p.source_cell], cells.day, p.source_cell),
            master_time=cell_datetime(values[p.master_cell], cells.day, p.master_cell),
            note=note,
        ))
    return anchors


@dataclass(frozen=True)
class PairOffsets:
    """Offsets (master − source) of the anchors and what follows from them."""

    offsets_s: np.ndarray
    offset_s: float  # median
    spread_s: float  # median absolute deviation (unscaled)
    uncertainty_s: float  # max(spread, half the reading step)


def pair_offsets(anchors: list[ClockAnchor], zone: str, resolution_s: float = 1.0) -> PairOffsets:
    if not anchors:
        raise ClockCellsError("No clock pairs.")
    offs = np.array([to_epoch_s(a.master_time, zone) - to_epoch_s(a.source_time, zone) for a in anchors])
    med = float(np.median(offs))
    spread = float(np.median(np.abs(offs - med)))
    return PairOffsets(offs, med, spread, max(spread, resolution_s / 2.0))


def clock_from_cells(path: str | Path, cells: ClockCells, zone: str) -> tuple[ClockModel, list[ClockAnchor]]:
    """Pure offset from cell pairs: median of the pair offsets, spread as uncertainty.

    The anchors are returned for display; the model keeps only the offset (one wrong pair must
    not bend the mapping, as anchors would) and the cell configuration for replay.
    """
    anchors = anchors_from_cells(path, cells)
    po = pair_offsets(anchors, zone, cells.resolution_s)
    listed = ", ".join(f"{o:g}" for o in po.offsets_s)
    sheet = f" on sheet {cells.sheet}" if cells.sheet else ""
    note = (
        f"Median of {len(anchors)} clock pair(s){sheet} in {Path(path).name}: offsets {listed} s; "
        f"± = max(median abs. deviation {po.spread_s:g} s, half the reading step {cells.resolution_s:g} s)."
    )
    model = ClockModel(
        offset_s=po.offset_s,
        uncertainty_s=po.uncertainty_s,
        method=JOURNAL_PAIRS,
        note=note,
        cells=cells.model_copy(deep=True),
    )
    return model, anchors


# ----------------------------------------------------------- cross-correlation
def resample(x: np.ndarray, y: np.ndarray, grid: np.ndarray, gap_rows: np.ndarray | None = None) -> np.ndarray:
    """Linear interpolation of a track onto `grid`; NaN outside it, in time gaps and next to missing values.

    `gap_rows`: rows where a time gap starts (no line is drawn into them), as in `Track.gap_rows`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    starts_gap = np.zeros(len(x), dtype=bool)
    if gap_rows is not None and len(gap_rows):
        starts_gap[np.asarray(gap_rows, dtype=int)] = True
    keep = np.isfinite(x)
    x, y, starts_gap = x[keep], y[keep], starts_gap[keep]
    order = np.argsort(x, kind="stable")
    x, y, starts_gap = x[order], y[order], starts_gap[order]
    out = np.full(len(grid), np.nan)
    if len(x) == 0:
        return out
    right = np.searchsorted(x, grid, side="left")
    exact = (right < len(x)) & (x[np.minimum(right, len(x) - 1)] == grid)
    out[exact] = y[right[exact]]
    inner = ~exact & (right > 0) & (right < len(x))
    r = right[inner]
    l = r - 1
    w = (grid[inner] - x[l]) / (x[r] - x[l])
    vals = y[l] + w * (y[r] - y[l])
    vals[starts_gap[r]] = np.nan  # never across a gap
    out[inner] = vals
    return out


def _corr(b: np.ndarray, a: np.ndarray, nfft: int, count: int) -> np.ndarray:
    """c[k] = Σ_i a[i] · b[i + k] for k = 0 … count − 1 (FFT, a shorter than b)."""
    fb = np.fft.rfft(b, nfft)
    fa = np.fft.rfft(a, nfft)
    return np.fft.irfft(fb * np.conj(fa), nfft)[:count]


def _median_step(x: np.ndarray, start: float, end: float) -> float:
    inside = np.sort(x[np.isfinite(x) & (x >= start) & (x <= end)])
    d = np.diff(inside)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else np.nan


@dataclass
class XCorrResult:
    """Best shift of the moving track onto the reference, with the whole correlation curve.

    `lag_s` is added to the moving source's master time (`ClockModel.shifted(lag_s)`).
    """

    lag_s: float          # best lag, refined between grid steps
    r: float              # Pearson r at the best grid lag
    lags_s: np.ndarray
    r_curve: np.ndarray   # NaN where the overlap is too small
    overlap: np.ndarray   # samples both tracks have, per lag
    step_s: float
    span: tuple[float, float]
    lag_range: tuple[float, float]
    derivative: bool
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        levels = {c.level for c in self.checks}
        return "fail" if "fail" in levels else "warn" if levels else "ok"

    @property
    def uncertainty_s(self) -> float:
        """Half the grid step: the resolution of the search, not the risk of a wrong peak."""
        return self.step_s / 2.0

    def summary(self) -> str:
        kind = "derivative" if self.derivative else "signal"
        return (
            f"Cross-correlation ({kind}, grid {self.step_s:g} s, search {self.lag_range[0]:g} … "
            f"{self.lag_range[1]:g} s): shift {self.lag_s:+.3f} s, r = {self.r:.3f}"
        )


def cross_correlate(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    mov_x: np.ndarray,
    mov_y: np.ndarray,
    start: float,
    end: float,
    lag_min: float,
    lag_max: float,
    *,
    step_s: float | None = None,
    derivative: bool = False,
    ref_gaps: np.ndarray | None = None,
    mov_gaps: np.ndarray | None = None,
) -> XCorrResult:
    """Find the shift L in [lag_min, lag_max] for which mov(t − L) best matches ref(t) on [start, end].

    Both tracks go onto one grid (default step: the coarser median sampling step inside the
    span); r(L) is Pearson's r over the grid points both have. Lags where they share less than
    MIN_OVERLAP of the reference's points are left out. `derivative` compares the first
    differences instead (removes offsets and slow drifts that differ between the devices).
    """
    if not end > start:
        raise ValueError("The time span is empty.")
    if not lag_max > lag_min:
        raise ValueError("The search range is empty (lag_max must be above lag_min).")
    notes: list[str] = []
    if step_s is None:
        step_s = np.nanmax([_median_step(np.asarray(ref_x, float), start, end),
                            _median_step(np.asarray(mov_x, float), start - lag_max, end - lag_min)])
        if not np.isfinite(step_s):
            raise ValueError("One of the tracks has fewer than two samples in the time span.")
    if not step_s > 0:
        raise ValueError("The grid step must be positive.")
    total = (end - start + lag_max - lag_min) / step_s
    if total > MAX_GRID:
        new = (end - start + lag_max - lag_min) / MAX_GRID
        notes.append(f"Grid step coarsened from {step_s:g} s to {new:g} s (at most {MAX_GRID} points).")
        step_s = new
    lo, hi = int(np.ceil(lag_min / step_s)), int(np.floor(lag_max / step_s))
    if hi <= lo:
        raise ValueError("The search range is shorter than one grid step.")
    n = int(np.floor((end - start) / step_s)) + 1
    grid_a = start + step_s * np.arange(n)
    grid_b = start - hi * step_s + step_s * np.arange(n + hi - lo)
    a = resample(ref_x, ref_y, grid_a, ref_gaps)
    b = resample(mov_x, mov_y, grid_b, mov_gaps)
    if derivative:
        a = np.diff(a) / step_s
        b = np.diff(b) / step_s
    ma, mb = np.isfinite(a), np.isfinite(b)
    if ma.sum() < 3 or mb.sum() < 3:
        raise ValueError("Too few samples of a track in the time span (or only gaps).")
    a0 = np.where(ma, a - np.nanmean(a), 0.0)  # centring keeps the sums small; r does not change
    b0 = np.where(mb, b - np.nanmean(b), 0.0)
    count = hi - lo + 1
    nfft = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    fa, fma = ma.astype(float), mb.astype(float)
    cnt = np.rint(_corr(fma, fa, nfft, count))
    sa = _corr(fma, a0, nfft, count)
    sb = _corr(b0, fa, nfft, count)
    saa = _corr(fma, a0 * a0, nfft, count)
    sbb = _corr(b0 * b0, fa, nfft, count)
    sab = _corr(b0, a0, nfft, count)
    with np.errstate(invalid="ignore", divide="ignore"):
        va = cnt * saa - sa * sa
        vb = cnt * sbb - sb * sb
        r = (cnt * sab - sa * sb) / np.sqrt(va * vb)
    tiny = (va <= 1e-12 * cnt * np.abs(saa)) | (vb <= 1e-12 * cnt * np.abs(sbb))  # no variation in the overlap
    r[(cnt < max(3.0, MIN_OVERLAP * ma.sum())) | tiny] = np.nan
    r = np.clip(r, -1.0, 1.0)
    # k = hi − lag index; reverse so the lags increase
    lags = (hi - np.arange(count))[::-1] * step_s
    r, cnt = r[::-1], cnt[::-1]
    if not np.isfinite(r).any():
        raise ValueError("No lag with enough overlap and variation in both tracks; widen the span or the search range.")

    best = int(np.nanargmax(r))
    lag = float(lags[best])
    checks: list[Check] = []
    if best in (0, count - 1) or not (np.isfinite(r[best - 1]) and np.isfinite(r[best + 1])):
        checks.append(Check(f"Best shift at the edge of the search range ({lag:+g} s); widen it.", "fail"))
    else:
        y0, y1, y2 = r[best - 1], r[best], r[best + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            lag += float(0.5 * (y0 - y2) / den) * step_s  # parabola through the peak
    rb = float(r[best])
    if rb < R_FAIL:
        checks.append(Check(f"Peak r = {rb:.2f} < {R_FAIL}: the tracks do not look alike in this span.", "fail"))
    elif rb < R_WARN:
        checks.append(Check(f"Peak r = {rb:.2f} < {R_WARN}: weak similarity.", "warn"))
    # main lobe: from the peak down to the next local minima on both sides
    left = best
    while left > 0 and np.isfinite(r[left - 1]) and r[left - 1] <= r[left]:
        left -= 1
    right = best
    while right < count - 1 and np.isfinite(r[right + 1]) and r[right + 1] <= r[right]:
        right += 1
    outside = np.ones(count, dtype=bool)
    outside[left:right + 1] = False
    rest = np.where(outside & np.isfinite(r), r, -np.inf)
    if np.isfinite(rest).any() and rest.max() >= rb - AMBIGUITY:
        other = int(np.argmax(rest))
        checks.append(Check(
            f"Ambiguous: a second peak at {lags[other]:+g} s has r = {r[other]:.2f} (periodic signal?).", "warn"
        ))
    share = cnt[best] / ma.sum()
    if share < 0.8:
        checks.append(Check(f"Only {share:.0%} of the reference samples overlap at the best shift.", "warn"))
    return XCorrResult(
        lag_s=lag, r=rb, lags_s=lags, r_curve=r, overlap=cnt, step_s=float(step_s), span=(start, end),
        lag_range=(lag_min, lag_max), derivative=derivative, checks=checks, notes=notes,
    )
