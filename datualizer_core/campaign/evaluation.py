"""What a campaign evaluates per measurement step (Etappe 2b, step 8, PR 3).

- `StepClass`: a kind of step ("dosing", "zero phase"), given by a formula over the step
  parameters (`V_MFC > 0`, `Probe == 'Eingang'`). The first class whose formula is true wins;
  a step without a class or with several is reported, never guessed.
- `NamedWindow`: a window rule with a name, optionally only for some classes (the blank value of
  a zero phase, the last 120 s of a dosing step). The campaign's `window_rule` is the window
  `default`, so older campaign.json files read unchanged.
- `Measure`: one number per step: a statistic (`stat`, a field of `WindowResult`) of one channel
  in one named window, under a short name (`alias`) that later formulas use. Its standard
  uncertainty comes with it (`MEASURE_U`).

`Evaluation` holds all three and is stored in campaign.json; `CampaignSession.step_table` turns
it into one row per step. Later PRs add constants, relations, calculations and criteria to it.
Nothing here names a channel, a parameter or a class: that is configuration (principle 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from datualizer_core.calc.expr import FUNCTIONS, ExprError, compile_expr
from datualizer_core.calc.relations import Relation
from datualizer_core.campaign.intervals import DEFAULT_WINDOW, ResolvedInterval, WindowRule
from datualizer_core.stats.window import WindowResult

# Columns of the step table before the parameters; parameters and aliases must not take these names
STEP_COLUMNS = ("step_id", "step_label", "step_start", "step_end", "dauer_s", "klasse", "klasse_hinweis")
# Names a class formula knows besides the parameters (text: step_id, step_label)
STEP_NAMES = ("step_id", "step_label", "dauer_s")

# Statistics a measure can take, and where its standard uncertainty comes from (None = none given)
MEASURE_U: dict[str, str | None] = {
    "mean": "sem_eff ⊕ reading_u",
    "slope": "slope_se_eff",
    "two_point": "two_point_u",
    "integral": "integral_u",
}
_NOT_A_NUMBER = {"quantiles", "slope_ts_ci", "checks", "assumptions"}
MEASURE_STATS: tuple[str, ...] = tuple(
    [f.name for f in fields(WindowResult) if f.metadata.get("table", True) and f.name not in _NOT_A_NUMBER]
    + ["slope_se_eff"]
)


def measure_value(r: WindowResult, stat: str, reading_u: float = 0.0) -> tuple[float | None, float | None]:
    """(value, standard uncertainty) of a statistic; truth values as 1/0, u None where none is defined.

    mean: √(sem_eff² + reading_u²) – the reading uncertainty (type B, e.g. the zero point) does not
    average out; slope: the larger of the HAC and AR(1) SE; two_point: its GUM u; integral: integral_u.
    """
    value = getattr(r, stat)
    if isinstance(value, bool):
        value = float(value)
    elif value is not None:
        value = float(value)
    if value is None:
        return None, None
    if stat == "mean":
        u = None if r.sem_eff is None else math.hypot(r.sem_eff, reading_u)
    elif stat == "slope":
        u = r.slope_se_eff
    elif stat == "two_point":
        u = r.two_point_u
    elif stat == "integral":
        u = r.integral_u
    else:
        u = None
    return value, u


def _name(value: str, what: str) -> str:
    if not value.isidentifier() or value in FUNCTIONS:
        raise ValueError(f"{what} {value!r} must be a name usable in formulas (letters, digits, '_', no function name).")
    return value


class StepClass(BaseModel):
    """A kind of step: `when` is a formula over the step parameters (numbers and text)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    when: str = Field(description="Formula over step parameters, e.g. \"V_MFC > 0\" or \"Probe == 'Eingang'\".")

    @field_validator("when")
    @classmethod
    def _parses(cls, v: str) -> str:
        compile_expr(v)  # syntax and text use; names are checked against the steps when classifying
        return v


class NamedWindow(BaseModel):
    """A window rule with a name; `classes` = only for steps of these classes (empty = all steps)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    rule: WindowRule = Field(default_factory=WindowRule)
    classes: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _not_default(cls, v: str) -> str:
        if v == DEFAULT_WINDOW:
            raise ValueError(f"'{DEFAULT_WINDOW}' is the campaign's window rule; give this window another name.")
        return v


class Measure(BaseModel):
    """One number per step: `stat` of `source/channel` in window `window`, called `alias`."""

    model_config = ConfigDict(extra="forbid")

    alias: str
    source: str
    channel: str
    window: str = DEFAULT_WINDOW
    stat: str = "mean"

    @field_validator("alias")
    @classmethod
    def _alias(cls, v: str) -> str:
        return _name(v, "Alias")

    @field_validator("stat")
    @classmethod
    def _stat(cls, v: str) -> str:
        if v not in MEASURE_STATS:
            raise ValueError(f"Unknown statistic {v!r}; possible are {', '.join(MEASURE_STATS)}.")
        return v

    def columns(self) -> tuple[str, str, str]:
        """Its columns in the step table: value, standard uncertainty, checks."""
        return self.alias, f"u_{self.alias}", f"checks_{self.alias}"


class Evaluation(BaseModel):
    """Step classes, named windows and measures of a campaign (stored in campaign.json)."""

    model_config = ConfigDict(extra="forbid")

    step_classes: list[StepClass] = Field(default_factory=list)
    windows: list[NamedWindow] = Field(default_factory=list)
    measures: list[Measure] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "Evaluation":
        classes = [c.name for c in self.step_classes]
        windows = [w.name for w in self.windows]
        for what, names in (("step class", classes), ("window", windows)):
            dup = sorted({n for n in names if names.count(n) > 1})
            if dup:
                raise ValueError(f"Duplicate {what} name(s): {', '.join(dup)}.")
        for w in self.windows:
            unknown = [c for c in w.classes if c not in classes]
            if unknown:
                raise ValueError(f"Window {w.name!r}: unknown class(es) {', '.join(unknown)}; defined are {classes}.")
        taken: dict[str, str] = {c: "a step table column" for c in STEP_COLUMNS}
        for m in self.measures:
            if m.window != DEFAULT_WINDOW and m.window not in windows:
                raise ValueError(f"Measure {m.alias!r}: unknown window {m.window!r}; defined are {[DEFAULT_WINDOW, *windows]}.")
            for col in m.columns():
                if col in taken:
                    raise ValueError(f"Measure {m.alias!r}: column {col!r} is already {taken[col]}.")
                taken[col] = f"a column of measure {m.alias!r}"
        for r in self.relations:
            for col in r.columns():
                if col in taken:
                    raise ValueError(f"Relation {r.name!r}: column {col!r} is already {taken[col]}.")
                taken[col] = f"a column of relation {r.name!r}"
        return self

    def window(self, name: str) -> NamedWindow | None:
        """The named window, None for `default` (the campaign's window rule); KeyError if unknown."""
        if name == DEFAULT_WINDOW:
            return None
        for w in self.windows:
            if w.name == name:
                return w
        raise KeyError(f"No window {name!r}; defined are {[DEFAULT_WINDOW, *[w.name for w in self.windows]]}.")

    def window_names(self) -> list[str]:
        return [DEFAULT_WINDOW, *[w.name for w in self.windows]]


# ------------------------------------------------------------------ step parameters
def parameter_columns(steps: Sequence[ResolvedInterval], taken: Sequence[str] = STEP_COLUMNS) -> dict[str, str]:
    """Column name of every step parameter: its own, or `param_<name>` if that is taken."""
    out: dict[str, str] = {}
    for iv in steps:
        for k in iv.params:
            out.setdefault(k, f"param_{k}" if k in taken else k)
    return out


def _column(values: list[Any]) -> np.ndarray:
    """Numbers as float (missing = NaN); a column with any text as text (object array, missing = None)."""
    present = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if any(isinstance(v, str) for v in present):
        return np.array([None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v) for v in values], dtype=object)
    out = np.full(len(values), np.nan)
    for i, v in enumerate(values):
        try:
            out[i] = np.nan if v is None else float(v)
        except (TypeError, ValueError):  # neither number nor text (a date): not usable in a formula
            return np.array([None if v is None else str(v) for v in values], dtype=object)
    return out


def step_env(steps: Sequence[ResolvedInterval]) -> dict[str, np.ndarray]:
    """What a class formula sees: step_id, step_label (text), dauer_s and every parameter, one value per step."""
    env: dict[str, np.ndarray] = {
        "step_id": np.array([iv.id for iv in steps], dtype=object),
        "step_label": np.array([iv.label for iv in steps], dtype=object),
        "dauer_s": np.array([iv.duration_s for iv in steps], dtype=float),
    }
    for key, col in parameter_columns(steps).items():
        env[col] = _column([iv.params.get(key) for iv in steps])
    return env


# ------------------------------------------------------------------ classes
@dataclass(frozen=True)
class StepClassification:
    """Class per step id (None = no class), a hint per step where it is not clear-cut, all hints as problems."""

    classes: dict[str, str | None] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def of(self, step_id: str) -> str | None:
        return self.classes.get(step_id)


def _truth(value: np.ndarray, n: int) -> np.ndarray:
    v = np.broadcast_to(np.asarray(value), (n,))
    if v.dtype == bool:
        return v.copy()
    v = v.astype(float)
    return (v != 0) & ~np.isnan(v)  # NaN (a missing parameter) is not "true"


def classify_steps(steps: Sequence[ResolvedInterval], classes: Sequence[StepClass]) -> StepClassification:
    """Class of every step: the first class whose `when` is true.

    No class matches → None and a hint; several match → the first and a hint naming all.
    A formula naming an unknown parameter or misusing text raises `ExprError` (with the class
    name and the known names): a wrong configuration fails loudly instead of leaving steps unclassified.
    """
    if not classes or not steps:
        return StepClassification({iv.id: None for iv in steps})
    env = step_env(steps)
    text = [k for k, v in env.items() if v.dtype == object]
    n = len(steps)
    hits = []
    for c in classes:
        try:
            expr = compile_expr(c.when, env.keys(), text_names=text)
            hits.append(_truth(expr.evaluate(env).value, n))
        except ExprError as e:
            raise ExprError(f"step class {c.name!r}: {e.message}", e.text, e.pos, e.known) from None
    hits_arr = np.array(hits)
    out: dict[str, str | None] = {}
    notes: dict[str, str] = {}
    for i, iv in enumerate(steps):
        names = [c.name for c, hit in zip(classes, hits_arr[:, i]) if hit]
        out[iv.id] = names[0] if names else None
        if not names:
            notes[iv.id] = "no class matches"
        elif len(names) > 1:
            notes[iv.id] = f"matches {', '.join(names)}; '{names[0]}' taken (first in the list)"
    problems = [f"{sid}: {note}." for sid, note in notes.items()]
    return StepClassification(out, notes, problems)


# ------------------------------------------------------------------ windows
@dataclass(frozen=True)
class StepWindow:
    """Window `name` of one step on master time.

    `window` is None if the rule leaves nothing or the window does not apply to the step's class
    (`applies` False, the reason in `note`). `override`: set by hand for this step.
    """

    name: str
    window: tuple[float, float] | None
    applies: bool = True
    override: bool = False
    note: str = ""


def override_key(step_id: str, window: str | None = None) -> str:
    """Key of a `WindowOverride` in `Campaign.window_overrides`: the step id, `<id>@<window>` for a named window."""
    return step_id if window in (None, DEFAULT_WINDOW) else f"{step_id}@{window}"


__all__ = [
    "DEFAULT_WINDOW",
    "MEASURE_STATS",
    "MEASURE_U",
    "STEP_COLUMNS",
    "Evaluation",
    "Measure",
    "NamedWindow",
    "StepClass",
    "StepClassification",
    "StepWindow",
    "classify_steps",
    "measure_value",
    "override_key",
    "parameter_columns",
    "step_env",
]
