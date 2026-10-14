"""Measurement steps (intervals) of a campaign: from a step table or set by hand (Etappe 2b, step 5).

Two origins, one result:

- `IntervalMapping`: column roles of a source that lists steps (a lab journal). Which column is
  start, end, name or set point is configuration (`StepColumns`, suggested by the source profile),
  never code. Steps are derived anew on every load, so a corrected journal is picked up.
- `Interval`: a step set by hand (right click → marker), stored in `campaign.json`.

`WindowRule` cuts the part of each step that is evaluated (skip the settling, keep the last
minutes). Rows that cannot become a step are reported in `IntervalSet.problems`, not dropped
silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import re
from typing import Any, Literal, Sequence

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from datualizer_core.campaign.clock import ClockModel, epoch_seconds, to_epoch_s
from datualizer_core.dataset import DualModeDataset
from datualizer_core.ingestion.steps import EndKind, EndRule, StepColumns
from datualizer_core.pipeline.operators import clean_column_name
from datualizer_core.schema import ColumnKind
from datualizer_core.timebase import localize, parse_datetime

_CLOCK = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?\s*$")


class IntervalConfigError(ValueError):
    """A step mapping names a column the source does not have, or a source that does not exist."""


class IntervalMapping(StepColumns):
    """Column roles of one source that lists measurement steps."""

    source_id: str
    clock_source: str | None = Field(
        None,
        description="Source whose clock the times are read on (e.g. FID times in the journal). None = this source.",
    )


class Interval(BaseModel):
    """A measurement step set by hand. Times are master wall-clock times in the campaign zone."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str = ""
    start: datetime
    end: datetime
    params: dict[str, Any] = Field(default_factory=dict)
    note: str = ""
    enabled: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> "Interval":
        if self.end <= self.start:
            raise ValueError(f"Interval {self.id!r}: end {self.end} is not after start {self.start}.")
        return self


class WindowRule(BaseModel):
    """Which part of each step is evaluated."""

    model_config = ConfigDict(extra="forbid")

    skip_start_s: float = Field(0.0, ge=0.0, description="Settling time cut from the start.")
    skip_end_s: float = Field(0.0, ge=0.0, description="Time cut from the end.")
    length_s: float | None = Field(None, gt=0.0, description="Keep only this long a window (None = all).")
    anchor: Literal["start", "end"] = Field("end", description="Which end of the step the window sticks to.")

    def is_identity(self) -> bool:
        return self.skip_start_s == 0 and self.skip_end_s == 0 and self.length_s is None

    def apply(self, start: float, end: float) -> tuple[float, float] | None:
        """Window inside [start, end] (epoch seconds), or None if nothing is left."""
        lo, hi = start + self.skip_start_s, end - self.skip_end_s
        if self.length_s is not None:
            if self.anchor == "end":
                lo = max(lo, hi - self.length_s)
            else:
                hi = min(hi, lo + self.length_s)
        return (lo, hi) if hi > lo else None


class WindowOverride(BaseModel):
    """Evaluated window of one step set by hand (dragged in the window inspector).

    Replaces what the `WindowRule` gives for this step only. Times are master wall-clock times in
    the campaign zone, like manual steps. `label` is the step's name when the override was set:
    if a corrected step table gives the step id another name, this is reported, not ignored.
    """

    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime
    label: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> "WindowOverride":
        if self.end <= self.start:
            raise ValueError(f"Window override: end {self.end} is not after start {self.start}.")
        return self


@dataclass(frozen=True)
class ResolvedInterval:
    """A step on master time (epoch seconds), ready to draw and evaluate."""

    id: str
    label: str
    start: float
    end: float
    window: tuple[float, float] | None  # after the WindowRule; None if nothing is left
    params: dict[str, Any] = field(default_factory=dict)
    origin: str = "manual"  # "manual" or "<source>:row <n>"
    manual: bool = True
    note: str = ""
    window_override: bool = False  # window set by hand for this step (WindowOverride)

    @property
    def duration_s(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class IntervalSet:
    intervals: list[ResolvedInterval] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def get(self, interval_id: str) -> ResolvedInterval:
        for iv in self.intervals:
            if iv.id == interval_id:
                return iv
        raise KeyError(interval_id)


# ------------------------------------------------------------------ columns
def resolve_column(ds: DualModeDataset, name: str) -> str:
    """Loaded column for a raw or cleaned name; fails loudly if there is none."""
    if name in ds.df.columns:
        return name
    clean = clean_column_name(name)
    if clean in ds.df.columns:
        return clean
    raise IntervalConfigError(f"Column {name!r} not found; the source has {ds.df.columns}.")


def _clock_seconds(text: str) -> float | None:
    m = _CLOCK.match(text)
    if m is None:
        return None
    h, mi, s, frac = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0), m.group(4)
    if h > 23 or mi > 59 or s > 59:
        return None
    return h * 3600 + mi * 60 + s + (float(f"0.{frac}") if frac else 0.0)


def _time_epochs(
    values: pl.Series, ref_dates: Sequence[date | None], zone: str, formats: Sequence[str],
    after: np.ndarray | None = None,
) -> np.ndarray:
    """Epoch seconds of a time column: date-time values, date-time texts or clock texts (NaN if unreadable).

    Clock texts get the date of `ref_dates` (the row's own date). If `after` is given and a clock
    time lies before it, the next day is used (a step running over midnight). Wall-clock values
    are read in `zone`; texts with an offset are exact instants.
    """
    if values.dtype in (pl.Datetime, pl.Date):
        s = values.cast(pl.Datetime("us"))
        if s.dtype.time_zone is None:  # type: ignore[union-attr]
            s = s.dt.replace_time_zone(zone, ambiguous="earliest", non_existent="null")
        return (s.dt.epoch("us").cast(pl.Float64) / 1e6).fill_null(np.nan).to_numpy().astype(float)

    texts = values.cast(pl.String).to_list()
    naive: list[datetime | None] = [None] * len(texts)
    exact: dict[int, float] = {}
    is_clock = np.zeros(len(texts), dtype=bool)
    for i, v in enumerate(texts):
        if v is None:
            continue
        secs = _clock_seconds(v)
        if secs is not None:
            is_clock[i] = True
            if ref_dates[i] is not None:
                naive[i] = datetime.combine(ref_dates[i], datetime.min.time()) + timedelta(seconds=secs)
            continue
        dt = parse_datetime(v.strip(), formats)
        if dt is None:
            continue
        if dt.tzinfo is None:
            naive[i] = dt
        else:
            exact[i] = to_epoch_s(dt, zone)
    if after is not None:
        for i in np.flatnonzero(is_clock):
            if naive[i] is not None and np.isfinite(after[i]):
                naive[i] = _next_day_if_before(naive[i], after[i], zone)
    out = np.array([np.nan if u is None else u / 1e6 for u in localize(naive, zone).epoch_us], dtype=float)
    for i, t in exact.items():
        out[i] = t
    return out


def _next_day_if_before(value: datetime, after_epoch: float, zone: str) -> datetime:
    t = localize([value], zone).epoch_us[0]
    return value + timedelta(days=1) if t is not None and t / 1e6 < after_epoch else value


def _row_dates(ds: DualModeDataset, zone: str) -> list[date | None]:
    abs_time = ds.time_abs
    if abs_time is None:
        return [None] * len(ds.df)
    if abs_time.dtype.time_zone is not None:  # type: ignore[union-attr]
        abs_time = abs_time.dt.convert_time_zone(zone)
    return abs_time.dt.date().to_list()


def _fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _ranges(rows: Sequence[int]) -> str:
    """'3, 18, 40–49' for rows 3, 18, 40..49."""
    out, start, prev = [], None, None
    for r in list(rows) + [None]:
        if start is None:
            start = prev = r
        elif r is not None and r == prev + 1:
            prev = r
        else:
            out.append(f"{start}" if start == prev else f"{start}–{prev}")
            start = prev = r
    return ", ".join(out)


# ----------------------------------------------------------------- derive
def derive_intervals(
    ds: DualModeDataset,
    mapping: IntervalMapping,
    zone: str,
    clock: ClockModel,
    rule: WindowRule,
    datetime_formats: Sequence[str] = (),
) -> tuple[list[ResolvedInterval], list[str]]:
    """Steps of one source on master time, plus the rows that could not become a step."""
    df = ds.df
    sid = mapping.source_id
    problems: list[str] = []
    dates = _row_dates(ds, zone)
    if mapping.start_column is None:
        start = epoch_seconds(ds, zone)
        if start is None:
            return [], [f"{sid}: no absolute time (set the date of clock times or an origin)."]
    else:
        start = _time_epochs(df[resolve_column(ds, mapping.start_column)], dates, zone, datetime_formats)

    end_rule: EndRule = mapping.end
    if end_rule.kind is EndKind.COLUMN:
        end = _time_epochs(df[resolve_column(ds, end_rule.column)], dates, zone, datetime_formats, after=start)
    elif end_rule.kind is EndKind.DURATION:
        dur = df[resolve_column(ds, end_rule.column)].cast(pl.Float64, strict=False).fill_null(np.nan).to_numpy()
        end = start + dur
    elif end_rule.kind is EndKind.FIXED:
        end = start + float(end_rule.seconds)
    else:  # NEXT_START: the following row with a start, in time order
        end = np.full(len(start), np.nan)
        rows = np.flatnonzero(np.isfinite(start))
        ordered = rows[np.argsort(start[rows], kind="stable")]
        end[ordered[:-1]] = start[ordered[1:]]
        if len(ordered):
            problems.append(f"{sid}: row {int(ordered[-1])}: last step has no next start, no end.")

    labels = []
    for name in mapping.label_columns:
        col = df[resolve_column(ds, name)]
        if name in mapping.fill_down:
            col = col.forward_fill()
        labels.append(col.to_list())
    params = {resolve_column(ds, n): df[resolve_column(ds, n)].to_list() for n in mapping.parameter_columns}

    master_start = clock.apply(start, zone)
    master_end = clock.apply(end, zone)
    out: list[ResolvedInterval] = []
    no_start, no_end, reversed_, empty = [], [], [], []
    for i in range(len(df)):
        if not np.isfinite(start[i]):
            no_start.append(i)
            continue
        if not np.isfinite(end[i]):
            if end_rule.kind is not EndKind.NEXT_START:
                no_end.append(i)
            continue
        if end[i] <= start[i]:
            reversed_.append(i)
            continue
        parts = [_fmt(col[i]) for col in labels if col[i] is not None and str(col[i]).strip()]
        p = {k: v[i] for k, v in params.items()}
        s, e = float(master_start[i]), float(master_end[i])
        window = rule.apply(s, e)
        if window is None:
            empty.append(i)
        out.append(ResolvedInterval(
            id=f"{sid}:{i}", label=" · ".join(parts) or f"{sid} row {i}", start=s, end=e,
            window=window, params=p, origin=f"{sid}:row {i}", manual=False,
        ))
    for rows, what in [
        (no_start, "no start time"), (no_end, "no readable end"),
        (reversed_, "end not after start"), (empty, "window rule leaves nothing"),
    ]:
        if rows:
            problems.append(f"{sid}: {'row' if len(rows) == 1 else 'rows'} {_ranges(rows)}: {what}" + (" (skipped)." if what != "window rule leaves nothing" else "."))
    return out, problems


def resolve_manual(iv: Interval, zone: str, rule: WindowRule) -> ResolvedInterval:
    s, e = to_epoch_s(iv.start, zone), to_epoch_s(iv.end, zone)
    return ResolvedInterval(
        id=iv.id, label=iv.label or iv.id, start=s, end=e, window=rule.apply(s, e),
        params=dict(iv.params), origin="manual", manual=True, note=iv.note,
    )


def master_datetime(epoch_s: float, zone: str) -> datetime:
    """Wall-clock date-time in `zone` for master epoch seconds (how manual steps are stored)."""
    s = pl.Series([int(round(epoch_s * 1e6))]).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    return s.dt.convert_time_zone(zone).dt.replace_time_zone(None)[0]


# ---------------------------------------------------------------- suggest
def suggest_mapping(ds: DualModeDataset, source_id: str, hints: StepColumns | None = None) -> IntervalMapping:
    """Column roles for a step table: the profile's hints if they fit, otherwise guessed from data shape.

    The guess never looks at column names: a text column whose values read as clock or date-times
    is the end, a sparse text column (filled in some rows only) is a block name, the other text
    columns are labels, and numeric columns with few distinct values are set points.
    """
    if hints is not None:
        try:
            for name in [hints.start_column, hints.end.column, *hints.label_columns, *hints.parameter_columns]:
                if name is not None:
                    resolve_column(ds, name)
            return IntervalMapping(source_id=source_id, **hints.model_dump())
        except IntervalConfigError:
            pass  # profile does not fit this file: fall back to the data shape

    df = ds.df
    time_col = next((c for c, k in ds.column_kinds.items() if k is ColumnKind.TIME), None)
    has_start = df[time_col].is_not_null() if time_col else pl.Series([True] * len(df))
    rows = df.filter(has_start)
    end_col, labels, fill = None, [], []
    for c in df.columns:
        if c == time_col or rows[c].dtype != pl.String:
            continue
        vals = [v for v in rows[c].to_list() if v is not None and v.strip()]
        if not vals:
            continue
        if end_col is None and all(_clock_seconds(v) is not None for v in vals) and len(vals) >= 0.8 * len(rows):
            end_col = c
        elif all(_clock_seconds(v) is None for v in vals):
            labels.append(c)
            if len(vals) < 0.5 * len(rows):
                fill.append(c)
    params = [
        c for c in ds.channels
        if 1 < rows[c].drop_nulls().n_unique() <= max(3, len(rows) // 3)
    ][:3]
    end = EndRule(kind=EndKind.COLUMN, column=end_col) if end_col else EndRule(kind=EndKind.NEXT_START)
    # the leftmost sparse text column is taken as block name; otherwise the leftmost text column
    label = fill[:1] or labels[:1]
    return IntervalMapping(
        source_id=source_id, end=end, label_columns=label, fill_down=fill[:1], parameter_columns=params,
    )
