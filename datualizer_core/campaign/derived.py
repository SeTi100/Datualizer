"""Calculated channels and masks of a campaign source (Etappe 2b, step 8, PR 2).

`DerivedChannel`: a new channel computed with a formula (`datualizer_core.calc`) from the
channels and parameters of the same source, optionally piecewise in master time (`Piece`), e.g.
a FID scale that changed with a recalibration at 09:24. `Mask`: a master time span whose values
are flagged `masked` (e.g. a test gas window); nothing is deleted.

Both are stored per source in `campaign.json`. `apply_derived` builds a new dataset from the
loaded one: the raw channels and their values stay exactly as loaded, calculated channels are
appended, masks become flag events. A calculated channel inherits the flags of every channel
its formulas use (a spike in the raw signal is a spike in the scaled one), except `missing`:
its own nulls are detected as `missing_calculated`.

`calibration_channel` turns a device calibration (`ingestion.calibration.suggest_linear_calibration`,
e.g. from Testa iFID `Settings*.xml`) into a calculated channel. It is a suggestion; the user
confirms it (principle 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from datualizer_core.calc import ExprError, compile_expr
from datualizer_core.campaign.clock import to_epoch_s
from datualizer_core.dataset import DualModeDataset
from datualizer_core.ingestion.calibration import LinearCalibration
from datualizer_core.quality import FLAGS_SCHEMA

MASKED = "masked"
# flags a calculated channel does not inherit: its own nulls are detected as missing_calculated
NOT_INHERITED = ("missing", "missing_calculated")
_T = "__master_t"


class Piece(BaseModel):
    """A formula that replaces the channel's default formula in a master time span.

    Times are master wall-clock times in the campaign zone (like manual steps); None = open.
    The span includes its start and excludes its end, so two pieces can meet at one instant.
    """

    model_config = ConfigDict(extra="forbid")

    start: datetime | None = None
    end: datetime | None = None
    expr: str
    note: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> "Piece":
        if self.start is not None and self.end is not None:
            try:
                ok = self.end > self.start
            except TypeError:  # one with, one without UTC offset
                ok = True
            if not ok:
                raise ValueError(f"Piece {self.expr!r}: end {self.end} is not after start {self.start}.")
        return self


class DerivedChannel(BaseModel):
    """A calculated channel of a source: `expr`, or the formula of the piece whose span holds the sample."""

    model_config = ConfigDict(extra="forbid")

    name: str
    expr: str = Field("", description="Default formula; empty = no value outside the pieces.")
    unit: str = ""
    pieces: list[Piece] = Field(default_factory=list, description="Formulas for master time spans (piecewise).")
    note: str = Field("", description="Origin, e.g. the calibration it was suggested from.")

    @model_validator(mode="after")
    def _has_formula(self) -> "DerivedChannel":
        if not self.name.strip():
            raise ValueError("A calculated channel needs a name.")
        if not self.expr.strip() and not self.pieces:
            raise ValueError(f"Calculated channel {self.name!r} has no formula.")
        return self

    def formulas(self) -> list[str]:
        return ([self.expr] if self.expr.strip() else []) + [p.expr for p in self.pieces]

    def inputs(self) -> list[str]:
        """Names the formulas use, in order of appearance (raises ExprError for a broken formula)."""
        out: list[str] = []
        for text in self.formulas():
            out += [n for n in compile_expr(text).names if n not in out]
        return out

    def describe(self) -> str:
        """One line per formula, for tooltips."""
        lines = [f"= {self.expr}"] if self.expr.strip() else ["no value outside the pieces"]
        for p in self.pieces:
            span = f"{_fmt(p.start) or '…'} – {_fmt(p.end) or '…'}"
            lines.append(f"{span}: = {p.expr}" + (f"  ({p.note})" if p.note else ""))
        return "\n".join(lines)


class Mask(BaseModel):
    """A master time span whose values are flagged `masked` (start and end included).

    Times are master wall-clock times in the campaign zone. `channels`: the channels concerned,
    empty = every channel of the source (then the flag is row level, without channel).
    """

    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime
    channels: list[str] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> "Mask":
        try:
            ok = self.end > self.start
        except TypeError:
            ok = True
        if not ok:
            raise ValueError(f"Mask: end {self.end} is not after start {self.start}.")
        return self


@dataclass(frozen=True)
class MaskSpan:
    """A mask on master time (epoch seconds), ready to draw."""

    source_id: str
    index: int  # position in CampaignSource.masks
    start: float
    end: float
    channels: tuple[str, ...]  # empty = all channels
    reason: str

    def applies_to(self, channel: str) -> bool:
        return not self.channels or channel in self.channels


def _fmt(t: datetime | None) -> str:
    return "" if t is None else t.strftime("%d.%m.%Y %H:%M:%S")


def _events(rows: np.ndarray, channel: str | None, flag: str) -> pl.DataFrame:
    return pl.DataFrame(
        {"row": rows.astype(np.int64), "channel": [channel] * len(rows), "flag": [flag] * len(rows)},
        schema=FLAGS_SCHEMA,
    )


def apply_derived(
    ds: DualModeDataset,
    master_t: np.ndarray | None,
    derived: Sequence[DerivedChannel],
    masks: Sequence[Mask],
    zone: str,
) -> tuple[DualModeDataset, list[str]]:
    """Dataset with the calculated channels and mask flags of one source, plus the problems found.

    `master_t`: master epoch seconds of every row (None if the source has no absolute time).
    A calculated channel that cannot be built (unknown name, broken formula, pieces that overlap
    or need a time the source does not have) is left out and reported; so is a mask that names
    an unknown channel or needs a time. Formulas may use the source's channels and parameters and
    the calculated channels listed before them. Without calculated channels and masks, `ds`
    itself is returned.
    """
    if not derived and not masks:
        return ds, []
    problems: list[str] = []
    df = ds.df
    t = pl.Series(_T, master_t if master_t is not None else np.full(len(df), np.nan), dtype=pl.Float64)
    work = df.with_columns(t.fill_nan(None))
    known = set(ds.channels) | set(ds.parameters)
    taken = set(df.columns)
    added: list[str] = []
    units: dict[str, str] = {}
    attrs: dict[str, dict[str, Any]] = {}
    inputs: dict[str, list[str]] = {}
    for dc in derived:
        try:
            expr, used = _build(dc, known, taken, master_t is not None, zone)
        except (ExprError, ValueError) as exc:
            problems.append(f"calculated channel {dc.name!r} left out: {exc}")
            continue
        work = work.with_columns(expr.cast(pl.Float64).alias(dc.name))
        known.add(dc.name)
        taken.add(dc.name)
        added.append(dc.name)
        inputs[dc.name] = used
        if dc.unit:
            units[dc.name] = dc.unit
        attrs[dc.name] = {"is_calculated": True, "formula": dc.describe(), "inputs": used}

    flags = [f for i, m in enumerate(masks) if (f := _mask_events(m, i, master_t, known, zone, problems)) is not None]
    extra = pl.concat(flags) if flags else None
    if added:
        pool = ds.quality_flags if extra is None else pl.concat([ds.quality_flags, extra])
        pool = pool.filter(~pl.col("flag").is_in(NOT_INHERITED))
        inherited = []
        for name in added:  # in order, so a channel built on a calculated one inherits its flags too
            got = pool.filter(pl.col("channel").is_in(inputs[name])).with_columns(pl.lit(name).alias("channel"))
            inherited.append(got)
            pool = pl.concat([pool, got])
        extra = pl.concat(([extra] if extra is not None else []) + inherited)
    return ds.with_calculated(work.drop(_T), units, attrs, extra), problems


def _build(
    dc: DerivedChannel, known: set[str], taken: set[str], has_time: bool, zone: str
) -> tuple[pl.Expr, list[str]]:
    if dc.name in taken:
        raise ValueError("the source already has a column with this name")
    default = pl.lit(None, dtype=pl.Float64)
    if dc.expr.strip():
        default = compile_expr(dc.expr, known).to_polars()
    if not dc.pieces:
        return default, dc.inputs()
    if not has_time:
        raise ValueError("pieces need master time, the source has no absolute time")
    spans = []
    for p in dc.pieces:
        lo = to_epoch_s(p.start, zone) if p.start is not None else -np.inf
        hi = to_epoch_s(p.end, zone) if p.end is not None else np.inf
        if hi <= lo:
            raise ValueError(f"piece {p.expr!r} ends before it starts")
        spans.append((lo, hi, compile_expr(p.expr, known).to_polars(), p.expr))
    spans.sort(key=lambda s: s[0])
    for a, b in zip(spans, spans[1:]):
        if b[0] < a[1]:
            raise ValueError(f"pieces {a[3]!r} and {b[3]!r} overlap")
    t = pl.col(_T)
    out = pl.when(t.is_null()).then(pl.lit(None, dtype=pl.Float64))
    for lo, hi, expr, _text in spans:
        cond = pl.lit(True)
        if np.isfinite(lo):
            cond = cond & (t >= lo)
        if np.isfinite(hi):
            cond = cond & (t < hi)
        out = out.when(cond).then(expr)
    return out.otherwise(default), dc.inputs()


def _mask_events(
    m: Mask, index: int, master_t: np.ndarray | None, known: set[str], zone: str, problems: list[str]
) -> pl.DataFrame | None:
    what = f"mask {index + 1} ({m.reason or 'no reason'}) left out"
    unknown = [c for c in m.channels if c not in known]
    if unknown:
        problems.append(f"{what}: unknown channels {unknown}")
        return None
    if master_t is None:
        problems.append(f"{what}: the source has no absolute time")
        return None
    rows = mask_rows(master_t, to_epoch_s(m.start, zone), to_epoch_s(m.end, zone))
    if not m.channels:
        return _events(rows, None, MASKED)
    return pl.concat([_events(rows, ch, MASKED) for ch in m.channels])


def mask_rows(master_t: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Rows whose master time lies in [start_s, end_s]."""
    with np.errstate(invalid="ignore"):
        return np.flatnonzero((master_t >= start_s) & (master_t <= end_s))


# ------------------------------------------------------------------ calibration
def calibration_channel(
    cal: LinearCalibration, channel: str, name: str, unit: str | None = None, source: str = ""
) -> DerivedChannel:
    """The calculated channel a calibration suggests, over the whole time (add pieces for a span)."""
    return DerivedChannel(
        name=name, expr=cal.formula(channel), unit=cal.fields.result_unit if unit is None else unit,
        note=cal.note(source),
    )
