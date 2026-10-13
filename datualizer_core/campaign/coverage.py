"""Coverage of a campaign: when each source has data on master time, and where it has gaps (step 4).

A source covers the span from its first to its last time stamp. Inside that span it has a gap
where it delivers no values:

- `time_gap`: the time step is flagged as `gap` (P12, threshold from `QualityConfig.gap_factor`),
  e.g. a device that stopped writing for 11 s.
- `no_values`: rows whose time runs on but whose channels are all empty, e.g. a logger outage
  (P24) or invalid OCR readings (P28). This also covers empty rows at the start or end.

Nothing is changed or dropped; the intervals only describe the loaded data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl

from datualizer_core.dataset import DualModeDataset

TIME_GAP = "time_gap"
NO_VALUES = "no_values"


@dataclass(frozen=True)
class CoverageGap:
    """A stretch of master time inside a source's span without values."""

    start: float  # master epoch seconds: last sample before the gap (or first row of the span)
    end: float  # first sample after the gap (or last row of the span)
    reason: str  # TIME_GAP or NO_VALUES
    rows: tuple[int, int]  # first and last row of the dataset touching the gap

    @property
    def duration_s(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SourceCoverage:
    """Covered intervals and gaps of one source on master time."""

    source_id: str
    label: str
    span: tuple[float, float] | None  # None: no absolute time, the source cannot be placed
    segments: list[tuple[float, float]] = field(default_factory=list)  # intervals with values
    gaps: list[CoverageGap] = field(default_factory=list)

    @property
    def covered_s(self) -> float:
        return float(sum(b - a for a, b in self.segments))

    def gap_at(self, t: float) -> CoverageGap | None:
        """The gap containing master time `t` (bounds included), if any."""
        for g in self.gaps:
            if g.start <= t <= g.end:
                return g
        return None


def value_rows(ds: DualModeDataset, channels: Sequence[str] | None = None) -> np.ndarray:
    """Boolean mask: the row has a value in at least one of `channels` (default: all channels)."""
    chans = list(ds.channels if channels is None else channels)
    if not chans:
        return np.zeros(len(ds.df), dtype=bool)
    return ds.df.select(pl.any_horizontal([pl.col(c).is_not_null() for c in chans]))[:, 0].to_numpy()


def coverage_intervals(
    t: np.ndarray, has_value: np.ndarray, gap_rows: Sequence[int] | np.ndarray = ()
) -> tuple[list[tuple[float, float]], list[CoverageGap]]:
    """Split a time axis into covered segments and gaps.

    `t`: master time per row (NaN rows cannot be placed and are skipped), in row order.
    `has_value`: the row carries at least one value. `gap_rows`: rows where a time gap starts.
    """
    t = np.asarray(t, dtype=float)
    rows = np.flatnonzero(np.isfinite(t))
    if not len(rows):
        return [], []
    valued = rows[np.asarray(has_value, dtype=bool)[rows]]
    first_row, last_row = int(rows[0]), int(rows[-1])
    if not len(valued):
        return [], [CoverageGap(float(t[first_row]), float(t[last_row]), NO_VALUES, (first_row, last_row))]

    # Between two consecutive valued rows: placeable empty rows (no values) or a flagged time gap
    pos = np.searchsorted(rows, valued)
    empty_between = np.diff(pos) > 1
    is_gap = np.zeros(len(t), dtype=np.int64)
    gap_rows = np.asarray(gap_rows, dtype=np.int64)
    is_gap[gap_rows[(gap_rows >= 0) & (gap_rows < len(t))]] = 1
    gap_count = np.cumsum(is_gap)
    time_gap = gap_count[valued[1:]] - gap_count[valued[:-1]] > 0
    breaks = np.flatnonzero(empty_between | time_gap)

    segments: list[tuple[float, float]] = []
    gaps: list[CoverageGap] = []
    if valued[0] > first_row:
        gaps.append(CoverageGap(float(t[first_row]), float(t[valued[0]]), NO_VALUES, (first_row, int(valued[0]))))
    seg_start = int(valued[0])
    for k in breaks:
        prev, cur = int(valued[k]), int(valued[k + 1])
        reason = NO_VALUES if empty_between[k] else TIME_GAP
        segments.append((float(t[seg_start]), float(t[prev])))
        gaps.append(CoverageGap(float(t[prev]), float(t[cur]), reason, (prev, cur)))
        seg_start = cur
    segments.append((float(t[seg_start]), float(t[valued[-1]])))
    if valued[-1] < last_row:
        gaps.append(CoverageGap(float(t[valued[-1]]), float(t[last_row]), NO_VALUES, (int(valued[-1]), last_row)))
    return segments, gaps
