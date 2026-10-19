"""Measurement steps (intervals) of a campaign: from a step table or set by hand (Etappe 2b, step 5).

Two origins, one result:

- `IntervalMapping`: column roles of a source that lists steps (a lab journal). Which column is
  start, end, name or set point is configuration (`StepColumns`, suggested by the source profile),
  never code. Steps are derived anew on every load, so a corrected journal is picked up.
- `Interval`: a step set by hand (right click → marker), stored in `campaign.json`.

Steps are read from the raw cells of the source (`Grid`): for a workbook the whole sheet as in
Excel, for other sources the loaded table with its header in row 1. Columns are found by their
header text only (`ingestion.grid.match_header`); the header row is the first row that has all
headers of the mapping. `RowSelection` picks the rows (range, stop rule, formula, rows left out).
A missing header fails loudly with a suggestion; every row that does not become a step is in
`StepDerivation.rows` and `problems` with its reason, never dropped silently.

`WindowRule` cuts the part of each step that is evaluated (skip the settling, keep the last
minutes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
import math
import re
from typing import Any, Literal, Sequence

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from datualizer_core.calc.expr import ExprError, compile_expr
from datualizer_core.campaign.clock import ClockModel, to_epoch_s
from datualizer_core.dataset import DualModeDataset
from datualizer_core.ingestion.grid import (
    Grid,
    GridError,
    HeaderMatch,
    display_text,
    excel_time_only,
    find_header_row,
    is_empty,
)
from datualizer_core.ingestion.steps import CellRef, EndKind, EndRule, StepColumns, StopRule
from datualizer_core.pipeline.operators import clean_column_name
from datualizer_core.schema import ColumnKind
from datualizer_core.timebase import localize, parse_datetime

DEFAULT_WINDOW = "default"  # name of the campaign's window rule among the named windows

_CLOCK = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?\s*$")


class IntervalConfigError(ValueError):
    """A step mapping names a column the source does not have, or a source that does not exist."""


class CellValue(CellRef):
    """A value written once in a sheet (e.g. 'Baseline mgC'), found by its label, kept with every step.

    It becomes the parameter `name` of each step of the mapping, so formulas and relations use it
    like a set point (e.g. the room air baseline of that measuring day).
    """

    name: str = Field(description="Parameter name in the steps (cleaned style, e.g. 'raumluft').")
    source_id: str | None = Field(None, description="Source with the cell; None = the mapping's source.")
    sheet: str | None = Field(None, description="Sheet; None = the sheet of the mapping (or of the source).")


class StepStamp(BaseModel):
    """One step as it was when the campaign was saved (to report what a changed file changes)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str = ""
    start: datetime
    end: datetime
    row: int | None = None


class MappingSnapshot(BaseModel):
    """What a mapping gave when the campaign was last saved; a changed file is compared with it."""

    model_config = ConfigDict(extra="forbid")

    header_row: int | None = None
    columns: list[str] = Field(default_factory=list)
    last_row: int | None = None
    steps: list[StepStamp] = Field(default_factory=list)


class IntervalMapping(StepColumns):
    """Column roles of one source that lists measurement steps."""

    source_id: str
    clock_source: str | None = Field(
        None,
        description="Source whose clock the times are read on (e.g. FID times in the journal). None = this source.",
    )
    sheet: str | None = Field(None, description="Sheet of a workbook; None = the sheet the source was loaded from.")
    day: date | None = Field(
        None, description="Date of clock times without date; None = the source's clock date (TimeConfig.clock_date)."
    )
    cell_values: list[CellValue] = Field(default_factory=list, description="Labelled cells kept with every step.")
    resolved: MappingSnapshot | None = Field(
        None, description="Header, columns and steps when last saved (written by the session, compared on open)."
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
    the campaign zone, like manual steps. `label` and `step_start` describe the step when the
    override was set: if a changed step table gives the step another id, the override follows the
    step with the same name and start (reported); if the id now names another step, this is
    reported, not ignored. `window`: the named window it replaces (`Evaluation.windows`);
    None = the window rule (`default`). Stored under `override_key(step id, window)`.
    """

    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime
    label: str = ""
    step_start: datetime | None = Field(None, description="Start of the step when the override was set.")
    window: str | None = Field(None, description="Named window it replaces; None = the window rule (default).")

    @field_validator("window")
    @classmethod
    def _default_is_none(cls, v: str | None) -> str | None:
        return None if v == DEFAULT_WINDOW else v

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


RowStatus = Literal["above", "header", "step", "excluded", "problem"]


@dataclass(frozen=True)
class RowState:
    """What became of one row of the table: a step, left out (grey) or a problem (orange), and why."""

    status: RowStatus
    reason: str = ""
    step_id: str | None = None


@dataclass(frozen=True)
class StepDerivation:
    """Steps of one mapping plus everything needed to show how they were read."""

    intervals: list[ResolvedInterval]
    problems: list[str]
    grid: Grid
    header: HeaderMatch | None = None
    roles: dict[int, str] = field(default_factory=dict)  # 0-based column -> role text
    rows: dict[int, RowState] = field(default_factory=dict)  # 1-based row -> state
    last_row: int | None = None

    def snapshot(self, zone: str) -> MappingSnapshot:
        return MappingSnapshot(
            header_row=self.header.row if self.header else None,
            columns=list(self.header.headers) if self.header else [],
            last_row=self.last_row,
            steps=[
                StepStamp(id=iv.id, label=iv.label, start=master_datetime(iv.start, zone),
                          end=master_datetime(iv.end, zone), row=_origin_row(iv.origin))
                for iv in self.intervals
            ],
        )


def _origin_row(origin: str) -> int | None:
    m = re.search(r":row (\d+)$", origin)
    return int(m.group(1)) if m else None


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


def _next_day_if_before(value: datetime, after_epoch: float, zone: str) -> datetime:
    t = localize([value], zone).epoch_us[0]
    return value + timedelta(days=1) if t is not None and t / 1e6 < after_epoch else value


def _fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (datetime, date, time)):
        return display_text(value)
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


def _rows_text(rows: Sequence[int]) -> str:
    return f"{'row' if len(rows) == 1 else 'rows'} {_ranges(rows)}"


def _number(value: Any) -> float | None:
    """A cell as number, or None (empty or not a number)."""
    if is_empty(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        for text in (value.strip(), value.strip().replace(",", ".")):
            try:
                return float(text)
            except ValueError:
                pass
    return None


# ------------------------------------------------------------------- grids
def grid_from_dataset(ds: DualModeDataset, zone: str, time_name: str | None = None) -> Grid:
    """A loaded table as grid: header in row 1, data from row 2, the time column as absolute date-times.

    The time column is named `time_name` (the source's original time header), so a mapping finds
    it like any other column; the date of each row is kept for clock-time texts in other columns.
    """
    df = ds.df
    tcol = next((c for c, k in ds.column_kinds.items() if k is ColumnKind.TIME), None)
    abs_time = ds.time_abs
    headers = [(time_name or c) if c == tcol else c for c in df.columns]
    times: list[Any] | None = None
    days: dict[int, date] = {}
    if abs_time is not None and tcol is not None:
        local = abs_time
        if local.dtype.time_zone is not None:  # type: ignore[union-attr]
            local = local.dt.convert_time_zone(zone)
        times = abs_time.to_list()
        days = {i + 2: d for i, d in enumerate(local.dt.date().to_list()) if d is not None}
    rows: list[list[Any]] = [list(headers)]
    for i, row in enumerate(df.iter_rows()):
        values = list(row)
        if times is not None:
            values[df.columns.index(tcol)] = times[i]
        rows.append(values)
    return Grid(rows=rows, kind="table", row_days=days)


# ------------------------------------------------------------------- times
def _cell_times(
    grid: Grid, rows: Sequence[int], col: int, day: date | None, zone: str, formats: Sequence[str],
    after: dict[int, float] | None = None,
) -> tuple[dict[int, float], list[int]]:
    """Epoch seconds of a time column for `rows` (missing or unreadable: NaN), and rows with a clock time but no date.

    Cells may hold date-times, times, or texts (clock time or date-time). Clock times get the
    row's date (tables) or `day`. With `after` (the step starts), a clock time before the start
    is on the next day (a step over midnight). Wall-clock values are read in `zone`.
    """
    naive: dict[int, datetime] = {}
    exact: dict[int, float] = {}
    clock: set[int] = set()
    no_day: list[int] = []
    for r in rows:
        v = grid.value(r, col)
        if is_empty(v):
            continue
        if isinstance(v, datetime) and excel_time_only(v):
            v = v.time()
        t_of_day: float | None = None
        if isinstance(v, time):
            t_of_day = v.hour * 3600 + v.minute * 60 + v.second + v.microsecond / 1e6
        elif isinstance(v, datetime):
            if v.tzinfo is not None:
                exact[r] = to_epoch_s(v, zone)
            else:
                naive[r] = v
            continue
        elif isinstance(v, date):
            naive[r] = datetime.combine(v, time())
            continue
        elif isinstance(v, str):
            t_of_day = _clock_seconds(v)
            if t_of_day is None:
                dt = parse_datetime(v.strip(), formats)
                if dt is None:
                    continue
                if dt.tzinfo is None:
                    naive[r] = dt
                else:
                    exact[r] = to_epoch_s(dt, zone)
                continue
        else:
            continue
        d = grid.row_days.get(r, day)
        if d is None:
            no_day.append(r)
            continue
        clock.add(r)
        naive[r] = datetime.combine(d, time()) + timedelta(seconds=t_of_day)
    if after is not None:
        for r in clock:
            a = after.get(r, math.nan)
            if np.isfinite(a):
                naive[r] = _next_day_if_before(naive[r], a, zone)
    keys = list(naive)
    out = {r: math.nan for r in rows}
    for r, us in zip(keys, localize([naive[k] for k in keys], zone).epoch_us):
        out[r] = math.nan if us is None else us / 1e6
    out.update(exact)
    return out, no_day


# ------------------------------------------------------------------ derive
def required_columns(mapping: StepColumns, time_name: str | None) -> tuple[list[str], list[str]]:
    """Header names a mapping needs, and the optional ones (parameters with a default)."""
    start = mapping.start_column or time_name
    defaults = {clean_column_name(k) for k in mapping.parameter_defaults}
    required = [n for n in [start, *mapping.end.columns(), *mapping.label_columns, mapping.id_column] if n]
    optional = []
    for n in mapping.parameter_columns:
        (optional if clean_column_name(n) in defaults else required).append(n)
    return list(dict.fromkeys(required)), optional


def _row_env(grid: Grid, header: HeaderMatch, rows: Sequence[int]) -> tuple[dict[str, np.ndarray], set[str]]:
    """Values of every column for a formula over rows: cleaned header names, numbers or text."""
    env: dict[str, np.ndarray] = {"row": np.array(rows, dtype=float)}
    text: set[str] = set()
    seen: dict[str, int] = {}
    for c, h in enumerate(header.headers):
        name = clean_column_name(h, c) or f"col_{c}"
        n = seen.get(name, 0) + 1
        seen[name] = n
        name = name if n == 1 else f"{name}_{n}"
        values = [grid.value(r, c) for r in rows]
        present = [v for v in values if not is_empty(v)]
        if present and all(_number(v) is not None and not isinstance(v, str) for v in present):
            env[name] = np.array([math.nan if is_empty(v) else float(v) for v in values], dtype=float)
        elif present:
            env[name] = np.array([None if is_empty(v) else display_text(v) for v in values], dtype=object)
            text.add(name)
        else:
            env[name] = np.full(len(rows), math.nan)
    return env, text


def row_names(grid: Grid, header: HeaderMatch) -> tuple[list[str], set[str]]:
    """Names a `where` formula may use (cleaned headers and `row`) and which of them hold text."""
    last = grid.last_used_row()
    env, text = _row_env(grid, header, list(range(header.row + 1, last + 1)))
    return list(env), text


def derive_steps(
    grid: Grid,
    mapping: IntervalMapping,
    zone: str,
    clock: ClockModel,
    rule: WindowRule,
    *,
    day: date | None = None,
    datetime_formats: Sequence[str] = (),
    time_name: str | None = None,
    extra_params: dict[str, Any] | None = None,
    extra_notes: Sequence[str] = (),
) -> StepDerivation:
    """Steps of one source on master time, plus the state of every row and the problems.

    `day`: date of clock times without date. `time_name`: header of the source's time column
    (used when the mapping has no start column). `extra_params`: values kept with every step
    (resolved `cell_values`). Raises IntervalConfigError if a header is missing (with a suggestion)
    or the `where` formula does not fit the table.
    """
    sid = mapping.source_id
    sel = mapping.rows
    day = mapping.day or day
    if mapping.start_column is None and time_name is None:
        raise IntervalConfigError(f"{sid}: no time column; choose the start column.")
    required, optional = required_columns(mapping, time_name)
    try:
        header = find_header_row(grid, required, optional, sel.header_row)
    except GridError as exc:
        raise IntervalConfigError(f"{sid}: {exc}") from None
    col = header.columns
    start_name = mapping.start_column or time_name
    roles: dict[int, list[str]] = {}

    def role(name: str | None, text: str) -> None:
        if name is not None and name in col:
            roles.setdefault(col[name], []).append(text)

    role(start_name, "start")
    end_rule: EndRule | None = mapping.end
    while end_rule is not None:
        role(end_rule.column, "end" if end_rule.kind is EndKind.COLUMN else "duration")
        end_rule = end_rule.fallback
    for n in mapping.label_columns:
        role(n, "name")
    for n in mapping.parameter_columns:
        role(n, "parameter")
    role(mapping.id_column, "id")

    problems: list[str] = list(extra_notes)
    states: dict[int, RowState] = {r: RowState("above") for r in range(1, header.row)}
    states[header.row] = RowState("header")
    last_used = grid.last_used_row()
    data_rows = list(range(header.row + 1, last_used + 1))
    first = sel.first_row or header.row + 1
    last = min(sel.last_row or last_used, last_used)

    def exclude(rows: Sequence[int], reason: str, report: bool = True, what: str | None = None) -> None:
        for r in rows:
            states[r] = RowState("excluded", reason)
        if rows and report:
            problems.append(f"{sid}: {_rows_text(rows)}: {what or reason} (not read).")

    exclude([r for r in data_rows if r < first], f"above the first row {first}")
    exclude([r for r in data_rows if r > last], f"below the last row {last}")
    candidates = [r for r in data_rows if first <= r <= last]
    skip = set(sel.skip_rows)
    exclude([r for r in candidates if r in skip], "left out by hand")
    exclude([r for r in candidates if r not in skip and grid.row_empty(r)], "empty row", report=False)
    candidates = [r for r in candidates if r not in skip and not grid.row_empty(r)]

    start, no_day = _cell_times(grid, candidates, col[start_name], day, zone, datetime_formats)

    # stop rule: the rows from the stop on are not read
    stop_at, stop_why = None, ""
    if sel.stop is StopRule.FIRST_EMPTY_START:
        stop_at = next((r for r in candidates if not np.isfinite(start[r])), None)
        stop_why = f"stop: row {stop_at} is the first without start time"
    elif sel.stop is StopRule.TIME_JUMP:
        prev_r = None
        for r in candidates:
            if not np.isfinite(start[r]):
                continue
            if prev_r is not None and start[r] < start[prev_r] - sel.time_jump_min * 60:
                stop_at = r
                back = (start[prev_r] - start[r]) / 60
                stop_why = (
                    f"stop: start in row {r} ({_wall(start[r], zone)}) is {back:.0f} min before row {prev_r} "
                    f"({_wall(start[prev_r], zone)}): another day or block"
                )
                break
            prev_r = r
    if stop_at is not None:
        after = [r for r in candidates if r >= stop_at]
        exclude(after, stop_why, report=False)
        problems.append(f"{sid}: {stop_why}; {_rows_text([r for r in after])} not read.")
        candidates = [r for r in candidates if r < stop_at]

    # formula over the row values
    if sel.where and sel.where.strip() and candidates:
        env, text = _row_env(grid, header, candidates)
        try:
            expr = compile_expr(sel.where, names=list(env), text_names=text)
            value = np.broadcast_to(np.asarray(expr.evaluate(env).value), (len(candidates),))
        except ExprError as exc:
            raise IntervalConfigError(f"{sid}: where: {exc.message}" + (
                f" (known: {', '.join(sorted(env))})" if "unknown" in exc.message else "")) from None
        keep = value.astype(bool) if value.dtype == bool else (value.astype(float) != 0) & ~np.isnan(value.astype(float))
        exclude([r for r, k in zip(candidates, keep) if not k], f"where {sel.where!r} is not true")
        candidates = [r for r, k in zip(candidates, keep) if k]

    no_start = [r for r in candidates if not np.isfinite(start[r])]
    candidates = [r for r in candidates if np.isfinite(start[r])]

    # end: the rule, then its fallbacks for rows it leaves without end
    end = {r: math.nan for r in candidates}
    notes: dict[int, str] = {}
    rule_end: EndRule | None = mapping.end
    level = 0
    fallback_rows: dict[str, list[int]] = {}
    while rule_end is not None:
        todo = [r for r in candidates if not np.isfinite(end[r])]
        if not todo:
            break
        got = _end_times(grid, rule_end, todo, candidates, start, col, day, zone, datetime_formats)
        for r in todo:
            if np.isfinite(got.get(r, math.nan)):
                end[r] = got[r]
                if level:
                    notes[r] = f"no end from {_rule_text(mapping.end)}: {_rule_text(rule_end)} used"
                    fallback_rows.setdefault(notes[r], []).append(r)
        rule_end = rule_end.fallback
        level += 1
    for note, rows in fallback_rows.items():
        problems.append(f"{sid}: {_rows_text(rows)}: {note}.")

    labels = []
    for name in mapping.label_columns:
        values: dict[int, Any] = {}
        last_value = None
        for r in data_rows:
            v = grid.value(r, col[name])
            if name in mapping.fill_down:
                last_value = v if not is_empty(v) else last_value
                v = last_value
            values[r] = v
        labels.append(values)

    # parameters: typed per column; defaults for missing columns or empty cells
    defaults = {clean_column_name(k): v for k, v in mapping.parameter_defaults.items()}
    params: dict[str, dict[int, Any]] = {}
    for name in mapping.parameter_columns:
        key = clean_column_name(name)
        if name not in col:
            params[key] = {r: defaults[key] for r in candidates}
            problems.append(f"{sid}: no column '{name}': parameter '{key}' = {_fmt(defaults[key])} for every step.")
            continue
        raw = {r: grid.value(r, col[name]) for r in candidates}
        numeric = all(_number(v) is not None for v in raw.values() if not is_empty(v))
        vals = {r: (None if is_empty(v) else (_number(v) if numeric else display_text(v))) for r, v in raw.items()}
        if key in defaults:
            empty = [r for r, v in vals.items() if v is None]
            for r in empty:
                vals[r] = defaults[key]
            if empty:
                problems.append(f"{sid}: {_rows_text(empty)}: '{name}' empty, {_fmt(defaults[key])} used.")
        params[key] = vals
    for key, v in defaults.items():
        if key not in params:
            params[key] = {r: v for r in candidates}
    for key, v in (extra_params or {}).items():
        params[key] = {r: v for r in candidates}

    # ids
    ids: dict[int, str] = {}
    no_id, dup = [], []
    if mapping.id_column is not None:
        seen: dict[str, int] = {}
        for r in candidates:
            v = grid.value(r, col[mapping.id_column])
            if is_empty(v):
                no_id.append(r)
                continue
            key = f"{sid}:{_fmt(_number(v) if _number(v) is not None else display_text(v))}"
            if key in seen:
                dup.append((r, seen[key], key))
                continue
            seen[key] = r
            ids[r] = key
    else:
        ids = {r: f"{sid}:{r - header.row - 1}" for r in candidates}

    master_start = clock.apply(np.array([start[r] for r in candidates], dtype=float), zone)
    master_end = clock.apply(np.array([end[r] for r in candidates], dtype=float), zone)
    out: list[ResolvedInterval] = []
    no_end, reversed_, empty = [], [], []
    for k, r in enumerate(candidates):
        if r not in ids:
            continue
        if not np.isfinite(end[r]):
            no_end.append(r)
            continue
        if end[r] <= start[r]:
            reversed_.append(r)
            continue
        parts = [_fmt(lab[r]) for lab in labels if not is_empty(lab[r])]
        p = {key: vals[r] for key, vals in params.items()}
        s, e = float(master_start[k]), float(master_end[k])
        window = rule.apply(s, e)
        if window is None:
            empty.append(r)
        out.append(ResolvedInterval(
            id=ids[r], label=" · ".join(parts) or f"{sid} row {r}", start=s, end=e,
            window=window, params=p, origin=f"{sid}:row {r}", manual=False, note=notes.get(r, ""),
        ))
        states[r] = RowState("step", notes.get(r, ""), ids[r])

    def problem(rows: Sequence[int], reason: str, text: str | None = None) -> None:
        for r in rows:
            states[r] = RowState("problem", reason)
        if rows:
            problems.append(f"{sid}: {_rows_text(rows)}: {text or reason} (skipped).")

    no_day_set = set(no_day)
    problem([r for r in no_start if r in no_day_set], "clock time without date", "clock time without date (set the date)")
    problem([r for r in no_start if r not in no_day_set], "no start time")
    last_next = [r for r in no_end if _rule_has(mapping.end, EndKind.NEXT_START) and r == candidates[-1]]
    if last_next:
        for r in last_next:
            states[r] = RowState("problem", "last step has no next start, no end")
        problems.append(f"{sid}: {_rows_text(last_next)}: last step has no next start, no end.")
    problem([r for r in no_end if r not in last_next], "no readable end")
    problem(reversed_, "end not after start")
    problem(no_id, f"no id in '{mapping.id_column}'")
    for r, other, key in dup:
        states[r] = RowState("problem", f"id {key} already in row {other}")
        problems.append(f"{sid}: row {r}: id {key} already in row {other} (skipped).")
    if empty:
        problems.append(f"{sid}: {_rows_text(empty)}: window rule leaves nothing.")
    for r in empty:
        states[r] = RowState("step", "window rule leaves nothing", ids.get(r))
    role_text = {c: ", ".join(v) for c, v in roles.items()}
    return StepDerivation(out, problems, grid, header, role_text, states, last_used)


def _rule_has(rule: EndRule | None, kind: EndKind) -> bool:
    while rule is not None:
        if rule.kind is kind:
            return True
        rule = rule.fallback
    return False


def _rule_text(rule: EndRule) -> str:
    if rule.kind is EndKind.COLUMN:
        return f"'{rule.column}'"
    if rule.kind is EndKind.DURATION:
        return f"duration '{rule.column}'"
    if rule.kind is EndKind.FIXED:
        return f"fixed {rule.seconds:g} s"
    return "next start"


def _wall(epoch_s: float, zone: str) -> str:
    return master_datetime(epoch_s, zone).strftime("%H:%M:%S")


def _end_times(
    grid: Grid, rule: EndRule, todo: list[int], candidates: list[int], start: dict[int, float],
    col: dict[str, int], day: date | None, zone: str, formats: Sequence[str],
) -> dict[int, float]:
    if rule.kind is EndKind.COLUMN:
        got, _ = _cell_times(grid, todo, col[rule.column], day, zone, formats, after=start)
        return got
    if rule.kind is EndKind.DURATION:
        out = {}
        for r in todo:
            d = _number(grid.value(r, col[rule.column]))
            out[r] = start[r] + d if d is not None else math.nan
        return out
    if rule.kind is EndKind.FIXED:
        return {r: start[r] + float(rule.seconds) for r in todo}
    ordered = sorted(candidates, key=lambda r: start[r])  # the following step in time order
    nxt = {a: start[b] for a, b in zip(ordered, ordered[1:])}
    return {r: nxt.get(r, math.nan) for r in todo}


def derive_intervals(
    source: DualModeDataset | Grid,
    mapping: IntervalMapping,
    zone: str,
    clock: ClockModel,
    rule: WindowRule,
    datetime_formats: Sequence[str] = (),
    **kwargs,
) -> tuple[list[ResolvedInterval], list[str]]:
    """Steps of one source on master time, plus the rows that could not become a step.

    `source`: a grid, or a loaded dataset (read as a table with its time column as start).
    """
    if isinstance(source, DualModeDataset):
        time_name = kwargs.pop("time_name", None) or "time"
        source = grid_from_dataset(source, zone, time_name)
        kwargs["time_name"] = time_name
    d = derive_steps(source, mapping, zone, clock, rule, datetime_formats=datetime_formats, **kwargs)
    return d.intervals, d.problems


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


# ------------------------------------------------------------------ changes
def _names(items: Sequence[str], limit: int = 8) -> str:
    items = list(items)
    return ", ".join(items[:limit]) + (f" … (+{len(items) - limit})" if len(items) > limit else "")


def diff_snapshot(sid: str, old: MappingSnapshot, new: MappingSnapshot) -> list[str]:
    """What changed in a step table since it was saved: columns, header, rows, steps (one text each)."""
    out = []
    if old.columns and new.columns:
        added = [c for c in new.columns if c not in old.columns and not c.startswith("col_")]
        gone = [c for c in old.columns if c not in new.columns and not c.startswith("col_")]
        if added:
            out.append(f"{sid}: new columns: {_names(added)}.")
        if gone:
            out.append(f"{sid}: columns gone: {_names(gone)}.")
    if old.header_row is not None and new.header_row is not None and old.header_row != new.header_row:
        out.append(f"{sid}: header now in row {new.header_row} (was {old.header_row}).")
    if old.last_row is not None and new.last_row is not None and new.last_row != old.last_row:
        n = new.last_row - old.last_row
        out.append(f"{sid}: table has {abs(n)} {'more' if n > 0 else 'fewer'} rows (last row {new.last_row}, was {old.last_row}).")
    # the same step (name, start, end) is unchanged even if it moved to another row or id
    content = lambda s: (s.label, s.start, s.end)  # noqa: E731
    old_left = {content(s): s for s in old.steps}
    moved = 0
    after_left = []
    for s in new.steps:
        o = old_left.pop(content(s), None)
        if o is None:
            after_left.append(s)
        elif o.id != s.id:
            moved += 1
    before = {s.id: s for s in old_left.values()}
    changed = [(before.pop(s.id), s) for s in after_left if s.id in before]
    added = [s for s in after_left if s.id not in {b.id for _, b in changed}]
    if added:
        out.append(f"{sid}: new steps: {_names([_stamp(s) for s in added])}.")
    if before:
        out.append(f"{sid}: steps gone: {_names([_stamp(s) for s in before.values()])}.")
    if changed:
        out.append(f"{sid}: steps changed: {_names([_change(a, b) for a, b in changed])}.")
    if moved:
        out.append(f"{sid}: {moved} unchanged steps have a new id (rows moved; an id column keeps ids stable).")
    return out


def _stamp(s: StepStamp) -> str:
    return f"{s.id} '{s.label}' {s.start:%H:%M:%S}"


def _change(a: StepStamp, b: StepStamp) -> str:
    parts = []
    if a.label != b.label:
        parts.append(f"'{a.label}' → '{b.label}'")
    if a.start != b.start:
        parts.append(f"start {a.start:%H:%M:%S} → {b.start:%H:%M:%S}")
    if a.end != b.end:
        parts.append(f"end {a.end:%H:%M:%S} → {b.end:%H:%M:%S}")
    return f"{b.id} ({', '.join(parts)})"


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
