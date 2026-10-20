"""Relations between measurement steps (Etappe 2b, step 8, PR 4).

Works on the step table (Polars) and formulas from `calc/expr.py`:

- `nearest_reference`: looks up the nearest matching reference step before and/or after each step
  (where = condition formula, match = columns with identical values). If both are found: mean,
  uncertainty u = max(|delta| / (2 * sqrt(3)), u_min), kind = "eingeklammert", quelle = step IDs.
  If one found: u = u_one, kind = "einseitig". If none: fallback (value, u) or NaN, kind = "keiner".
  Produces 6 columns: <name>, u_<name>, <name>_quelle, <name>_art, <name>_vor, <name>_nach.
- `cumulative`: running sum in time order, resetting when `reset_when` is true (formula; function
  `changed(col)` = value differs from previous dosing row) and at time gaps >= gap_s between steps.
- `bin`: rounds a value or column to the nearest multiple of `width`: round(x / width) * width.
- Pydantic models: `NearestReferenceRelation`, `CumulativeRelation`, `BinRelation`, and `Relation`
  (discriminated union by `kind`).

No GUI or campaign imports (import principle from AGENTS.md).
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Mapping, Sequence, Union

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from datualizer_core.calc.expr import FUNCTIONS, NUMBER_NAMES, Expr, ExprError, compile_expr
from datualizer_core.calc.uncertain import UArray

# Function available in relation expressions
RELATION_FUNCTIONS: dict[str, tuple[int, int | None, str]] = {
    "changed": (1, 1, "changed(col): true if col differs from previous dosing step"),
}


def _name(value: str, what: str) -> str:
    if not value.isidentifier() or value in FUNCTIONS or value in NUMBER_NAMES:
        raise ValueError(
            f"{what} {value!r} must be a name usable in formulas (letters, digits, '_', no function name, not nan or inf)."
        )
    return value


# ------------------------------------------------------------------ Pydantic models
class FallbackValue(BaseModel):
    """Fallback reference value and standard uncertainty when no reference step is found.

    Each is a number or a formula over the step table (PR 10e), e.g. the parameter ``raumluft``
    that a journal cell (`CellValue`) or a mapping's `parameter_defaults` gives every step of a
    measuring day. Where the formula is NaN the step has no reference (kind "keiner", value NaN).
    """

    model_config = ConfigDict(extra="forbid")

    value: float | str
    u: float | str = 0.0

    @field_validator("value", "u")
    @classmethod
    def _formula(cls, v: float | str) -> float | str:
        if isinstance(v, str):
            compile_expr(v)
        return v


class NearestReferenceRelation(BaseModel):
    """Reference step lookup before/after (e.g. blank value for zero phases)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["nearest_reference"] = "nearest_reference"
    value: str = Field(min_length=1, description="Column/alias of the reference value.")
    where: str = Field(min_length=1, description="Filter condition formula for candidate reference steps.")
    match: list[str] = Field(default_factory=list, description="Columns that must match between reference and target.")
    side: Literal["both", "before", "after"] = "both"
    combine: Literal["mean_rect", "mean", "linear"] = "mean_rect"
    u_min: float = Field(default=0.3, ge=0.0)
    u_one: float = Field(default=1.0, ge=0.0)
    fallback: FallbackValue | None = None

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "Relation name")

    @field_validator("value")
    @classmethod
    def _val_valid(cls, v: str) -> str:
        return _name(v, "Value column")

    @field_validator("where")
    @classmethod
    def _where_valid(cls, v: str) -> str:
        compile_expr(v)
        return v

    @field_validator("match")
    @classmethod
    def _match_valid(cls, v: list[str]) -> list[str]:
        for col in v:
            _name(col, "Match column")
        return v

    def columns(self) -> tuple[str, str, str, str, str, str]:
        """Names of the 6 columns added to the step table."""
        return (
            self.name,
            f"u_{self.name}",
            f"{self.name}_quelle",
            f"{self.name}_art",
            f"{self.name}_vor",
            f"{self.name}_nach",
        )


class CumulativeRelation(BaseModel):
    """Running sum across steps (e.g. catalyst loading)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["cumulative"] = "cumulative"
    expr: str = Field(min_length=1, description="Formula for increment per step.")
    reset_when: str | None = Field(default=None, description="Reset condition formula; may use changed(col).")
    gap_s: float = Field(default=300.0, ge=0.0, description="Time gap in seconds that triggers reset.")

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "Relation name")

    @field_validator("expr")
    @classmethod
    def _expr_valid(cls, v: str) -> str:
        compile_expr(v)
        return v

    @field_validator("reset_when")
    @classmethod
    def _reset_valid(cls, v: str | None) -> str | None:
        if v is not None:
            compile_expr(v, extra_functions=RELATION_FUNCTIONS)
        return v

    def columns(self) -> tuple[str, str]:
        """The running sum and the run number of each step."""
        return (self.name, f"{self.name}_lauf")


class BinRelation(BaseModel):
    """Rounding to a grid (e.g. temperature classes)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["bin"] = "bin"
    expr: str = Field(min_length=1, description="Formula or column to bin.")
    width: float = Field(gt=0.0, description="Bin grid width.")

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "Relation name")

    @field_validator("expr")
    @classmethod
    def _expr_valid(cls, v: str) -> str:
        compile_expr(v)
        return v

    def columns(self) -> tuple[str]:
        return (self.name,)


Relation = Annotated[
    Union[NearestReferenceRelation, CumulativeRelation, BinRelation],
    Field(discriminator="kind"),
]


# ------------------------------------------------------------------ table helpers
def _table_env(table: pl.DataFrame) -> tuple[dict[str, Any], list[str]]:
    """Build evaluation environment and list of text column names from a step table."""
    text_names = [c for c in table.columns if table[c].dtype in (pl.String, pl.Categorical)]
    env: dict[str, Any] = {}
    for c in table.columns:
        if c in text_names:
            env[c] = np.array(
                [None if v is None else str(v) for v in table[c].to_list()],
                dtype=object,
            )
        else:
            try:
                env[c] = UArray(table[c].to_numpy().astype(float))
            except (ValueError, TypeError):
                text_names.append(c)
                env[c] = np.array(
                    [None if v is None else str(v) for v in table[c].to_list()],
                    dtype=object,
                )
    return env, text_names


def _time_columns(table: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract start and end times in epoch seconds (or numeric seconds)."""
    start_col = next((c for c in ("step_start", "start") if c in table.columns), None)
    end_col = next((c for c in ("step_end", "ende", "end") if c in table.columns), None)
    if start_col is None or end_col is None:
        raise ValueError(
            f"Step table must have start and end time columns ('step_start'/'step_end' or 'start'/'ende'); "
            f"found columns: {table.columns}."
        )

    def _to_seconds(col: str) -> np.ndarray:
        s = table[col]
        if s.dtype in (pl.Datetime, pl.Date, pl.Time):
            # Convert datetime to epoch seconds
            dt_s = s.cast(pl.Datetime("us", "UTC"))
            return (dt_s.to_physical().to_numpy() / 1e6).astype(float)
        return s.to_numpy().astype(float)

    return _to_seconds(start_col), _to_seconds(end_col)


def _step_ids(table: pl.DataFrame) -> list[str]:
    """Extract step IDs from table."""
    id_col = next((c for c in ("step_id", "id", "schritt") if c in table.columns), None)
    if id_col is not None:
        return [str(v) if v is not None else "" for v in table[id_col].to_list()]
    return [str(i) for i in range(len(table))]


# ------------------------------------------------------------------ nearest_reference
def nearest_reference(
    table: pl.DataFrame,
    *args: Any,
    name: str | None = None,
    value: str | None = None,
    where: str | None = None,
    match: Sequence[str] | None = None,
    side: str = "both",
    combine: str = "mean_rect",
    u_min: float = 0.3,
    u_one: float = 1.0,
    fallback: FallbackValue | dict[str, Any] | tuple[float, float] | float | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
    """Find the nearest matching reference step before and/or after each step.

    Parameters
    ----------
    table : pl.DataFrame
        Step table containing time columns and candidate parameters.
    name : str, optional
        Prefix for the generated columns. Defaults to `value`.
    value : str
        Column name holding the reference measurement value.
    where : str
        Expression identifying usable reference steps, e.g. ``"klasse == 'null' and abs(dS*60) <= 0.1"``.
    match : sequence of str, optional
        Columns that must have identical values between target and reference steps (e.g. ``["temperatur", "luft"]``).
    side : {"both", "before", "after"}, default "both"
        Look before, after, or both.
    combine : {"mean_rect", "mean", "linear"}, default "mean_rect"
        Combination method when bracketed by two reference steps.
    u_min : float, default 0.3
        Minimum standard uncertainty when bracketed.
    u_one : float, default 1.0
        Standard uncertainty when reference is found on one side only.
    fallback : FallbackValue, dict, tuple, float or None, optional
        Fallback (value, u) when no matching reference step is found.

    Returns
    -------
    pl.DataFrame
        6 columns: `<name>`, `u_<name>`, `<name>_quelle`, `<name>_art`, `<name>_vor`, `<name>_nach`.
    """
    # Flexible positional argument parsing:
    # Option A: nearest_reference(table, "B", "S_b", "where...", match, ...)
    # Option B: nearest_reference(table, "S_b", "where...", match, ...)
    pos_args = list(args)
    if value is None and pos_args:
        # Check if first pos_arg is name or value
        if len(pos_args) >= 3 and any(op in str(pos_args[2]) for op in ("==", "!=", "<", ">", " and ", " or ")):
            # pattern: table, name, value, where, [match]
            name = str(pos_args[0])
            value = str(pos_args[1])
            where = str(pos_args[2])
            pos_args = pos_args[3:]
        else:
            # pattern: table, value, where, [match]
            value = str(pos_args[0])
            pos_args = pos_args[1:]
    if where is None and pos_args:
        where = str(pos_args[0])
        pos_args = pos_args[1:]
    if match is None and pos_args:
        match = pos_args[0]
        pos_args = pos_args[1:]
    if "side" in kwargs:
        side = kwargs.pop("side")
    if "combine" in kwargs:
        combine = kwargs.pop("combine")
    if "u_min" in kwargs:
        u_min = float(kwargs.pop("u_min"))
    if "u_one" in kwargs:
        u_one = float(kwargs.pop("u_one"))
    if "fallback" in kwargs:
        fallback = kwargs.pop("fallback")

    if value is None:
        raise ValueError("Missing required argument 'value' for nearest_reference.")
    if where is None:
        raise ValueError("Missing required argument 'where' for nearest_reference.")
    if name is None:
        name = value

    # Normalize fallback (numbers, or formulas evaluated over the table below)
    fb_val: float | str | None = None
    fb_u: float | str | None = None
    if fallback is not None:
        if isinstance(fallback, FallbackValue):
            fb_val, fb_u = fallback.value, fallback.u
        elif isinstance(fallback, dict):
            fb = FallbackValue.model_validate(fallback)
            fb_val, fb_u = fb.value, fb.u
        elif isinstance(fallback, (tuple, list)):
            fb_val = float(fallback[0])
            fb_u = float(fallback[1]) if len(fallback) > 1 else 0.0
        elif isinstance(fallback, (int, float)):
            fb_val, fb_u = float(fallback), 0.0

    n = len(table)
    if n == 0:
        return pl.DataFrame(
            {
                name: pl.Series([], dtype=pl.Float64),
                f"u_{name}": pl.Series([], dtype=pl.Float64),
                f"{name}_quelle": pl.Series([], dtype=pl.String),
                f"{name}_art": pl.Series([], dtype=pl.String),
                f"{name}_vor": pl.Series([], dtype=pl.Float64),
                f"{name}_nach": pl.Series([], dtype=pl.Float64),
            }
        )

    if value not in table.columns:
        raise ValueError(f"Value column {value!r} not found in table; available: {table.columns}.")

    match_cols = list(match or [])
    for mc in match_cols:
        if mc not in table.columns:
            raise ValueError(f"Match column {mc!r} not found in table; available: {table.columns}.")

    # Evaluate where condition
    env, text_names = _table_env(table)
    expr = compile_expr(where, names=table.columns, text_names=text_names)
    where_res = expr.evaluate(env)
    where_arr = (where_res.value != 0) & ~np.isnan(where_res.value.astype(float))

    def _per_row(spec: float | str | None, what: str) -> np.ndarray | None:
        if spec is None:
            return None
        if not isinstance(spec, str):
            return np.full(n, float(spec))
        try:
            res = compile_expr(spec, names=table.columns, text_names=text_names).evaluate(env)
        except ExprError as e:
            raise ExprError(f"fallback {what} of {name!r}: {e.message}", e.text, e.pos, e.known) from None
        return np.broadcast_to(np.asarray(res.value, dtype=float), (n,)).copy()

    fb_vals, fb_us = _per_row(fb_val, "value"), _per_row(fb_u, "u")

    # Times and IDs
    t_start, t_end = _time_columns(table)
    ids = _step_ids(table)
    val_arr = table[value].to_numpy().astype(float)

    # Pre-extract match columns as lists for equality comparison
    match_data = [table[mc].to_list() for mc in match_cols]

    def _matches(idx_a: int, idx_b: int) -> bool:
        for col_vals in match_data:
            va, vb = col_vals[idx_a], col_vals[idx_b]
            if va is None or vb is None:
                if va != vb:
                    return False
            elif isinstance(va, float) and isinstance(vb, float):
                if not (math.isclose(va, vb, rel_tol=1e-5) or (math.isnan(va) and math.isnan(vb))):
                    return False
            elif va != vb:
                return False
        return True

    # Usable candidate mask: where condition is true and value is finite
    usable = where_arr & np.isfinite(val_arr)

    out_val = np.full(n, np.nan)
    out_u = np.full(n, np.nan)
    out_quelle: list[str] = []
    out_art: list[str] = []
    out_vor = np.full(n, np.nan)
    out_nach = np.full(n, np.nan)

    for i in range(n):
        s_i = t_start[i]
        e_i = t_end[i]

        b_vor: float | None = None
        id_vor: str | None = None
        best_vor_end: float = -math.inf

        b_nach: float | None = None
        id_nach: str | None = None
        best_nach_start: float = math.inf

        if side in ("both", "before"):
            # Search candidate before step i: r ended before or at s_i
            for r in range(n):
                if r != i and usable[r] and t_end[r] <= s_i and _matches(i, r):
                    # Nearest: maximum end time (latest before i)
                    if t_end[r] > best_vor_end:
                        best_vor_end = t_end[r]
                        b_vor = val_arr[r]
                        id_vor = ids[r]

        if side in ("both", "after"):
            # Search candidate after step i: r started at or after e_i
            for r in range(n):
                if r != i and usable[r] and t_start[r] >= e_i and _matches(i, r):
                    # Nearest: minimum start time (earliest after i)
                    if t_start[r] < best_nach_start:
                        best_nach_start = t_start[r]
                        b_nach = val_arr[r]
                        id_nach = ids[r]

        if b_vor is not None:
            out_vor[i] = b_vor
        if b_nach is not None:
            out_nach[i] = b_nach

        has_vor = b_vor is not None
        has_nach = b_nach is not None

        if has_vor and has_nach:
            # Bracketing (both sides found)
            if combine == "linear" and best_nach_start > best_vor_end:
                t_mid = 0.5 * (s_i + e_i)
                w = (t_mid - best_vor_end) / (best_nach_start - best_vor_end)
                w = max(0.0, min(1.0, w))
                ref_v = (1.0 - w) * b_vor + w * b_nach
            else:
                ref_v = 0.5 * (b_vor + b_nach)
            delta = abs(b_vor - b_nach)
            ref_u = max(delta / (2.0 * math.sqrt(3.0)), u_min)
            out_val[i] = ref_v
            out_u[i] = ref_u
            out_quelle.append(f"{id_vor}; {id_nach}")
            out_art.append("eingeklammert")
        elif has_vor or has_nach:
            # One-sided
            ref_v = b_vor if has_vor else b_nach
            used_id = id_vor if has_vor else id_nach
            out_val[i] = ref_v  # type: ignore[assignment]
            out_u[i] = u_one
            out_quelle.append(str(used_id))
            out_art.append("einseitig")
        else:
            # None found
            if fb_vals is not None and np.isfinite(fb_vals[i]):
                out_val[i] = fb_vals[i]
                out_u[i] = fb_us[i] if fb_us is not None else 0.0
            out_quelle.append("")
            out_art.append("keiner")

    return pl.DataFrame(
        {
            name: pl.Series(name, out_val, dtype=pl.Float64),
            f"u_{name}": pl.Series(f"u_{name}", out_u, dtype=pl.Float64),
            f"{name}_quelle": pl.Series(f"{name}_quelle", out_quelle, dtype=pl.String),
            f"{name}_art": pl.Series(f"{name}_art", out_art, dtype=pl.String),
            f"{name}_vor": pl.Series(f"{name}_vor", out_vor, dtype=pl.Float64),
            f"{name}_nach": pl.Series(f"{name}_nach", out_nach, dtype=pl.Float64),
        }
    )


# ------------------------------------------------------------------ cumulative
def _extract_changed_columns(expr_text: str) -> list[str]:
    """Extract column names wrapped in changed(...) within an expression."""
    import re
    return re.findall(r"\bchanged\s*\(\s*([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\)", expr_text)


def cumulative(
    table: pl.DataFrame,
    *args: Any,
    name: str = "cumulative",
    expr: str | None = None,
    reset_when: str | None = None,
    gap_s: float = 300.0,
    keep_nan: bool = True,
    runs: bool = False,
    **kwargs: Any,
) -> pl.Series | tuple[pl.Series, pl.Series]:
    """Running sum across steps in time order with reset conditions.

    Resets to 0.0 when `reset_when` evaluates to true (before adding the current increment)
    or when the time gap between consecutive steps is >= `gap_s`.
    The function `changed(col)` inside `reset_when` is true when `col` has changed value
    since the previous dosing step.

    Parameters
    ----------
    table : pl.DataFrame
        Step table.
    name : str, default "cumulative"
        Name of the returned Polars Series.
    expr : str
        Formula for the increment per step (e.g. ``"-mdot * dauer_s"``).
    reset_when : str, optional
        Condition formula that triggers reset.
    gap_s : float, default 300.0
        Time gap threshold in seconds.
    keep_nan : bool, default True
        If True, rows where `expr` evaluates to NaN get NaN in the output.
    runs : bool, default False
        Also return the run number of each step (`<name>_lauf`): 1 for the first step in time order,
        one more at every reset. Steps of one run share it, e.g. to align curves on the start of a
        loading sequence.

    Returns
    -------
    pl.Series
        Cumulative sum Series (and the run numbers with `runs`).
    """
    pos_args = list(args)
    if expr is None and pos_args:
        # Check if first is name or expr
        if len(pos_args) >= 2 and any(op in str(pos_args[1]) for op in ("+", "-", "*", "/", " ")):
            name = str(pos_args[0])
            expr = str(pos_args[1])
            pos_args = pos_args[2:]
        else:
            expr = str(pos_args[0])
            pos_args = pos_args[1:]
    if reset_when is None and pos_args:
        reset_when = str(pos_args[0])
        pos_args = pos_args[1:]
    if pos_args:
        gap_s = float(pos_args[0])
    if "reset_when" in kwargs:
        reset_when = kwargs.pop("reset_when")
    if "gap_s" in kwargs:
        gap_s = float(kwargs.pop("gap_s"))
    if "name" in kwargs:
        name = str(kwargs.pop("name"))

    if expr is None:
        raise ValueError("Missing required argument 'expr' for cumulative.")

    n = len(table)
    if n == 0:
        empty = pl.Series(name, [], dtype=pl.Float64)
        return (empty, pl.Series(f"{name}_lauf", [], dtype=pl.Int64)) if runs else empty

    # Time order
    has_time = any(c in table.columns for c in ("step_start", "start"))
    if has_time:
        t_start, t_end = _time_columns(table)
        order = np.argsort(t_start, kind="stable")
    else:
        t_start, t_end = np.zeros(n), np.zeros(n)
        order = np.arange(n)

    env, text_names = _table_env(table)

    # Compile and evaluate increment expression
    inc_expr = compile_expr(expr, names=table.columns, text_names=text_names)
    inc_arr = inc_expr.evaluate(env).value.astype(float)

    # Reset condition
    reset_arr = np.zeros(n, dtype=bool)
    if reset_when is not None and reset_when.strip():
        # Precompute changed(col) if present
        changed_cols = _extract_changed_columns(reset_when)
        for col in changed_cols:
            if col not in table.columns:
                raise ValueError(f"changed({col!r}): column not found in table; available: {table.columns}.")
            # A dosing row is any row where klasse != "null" (if klasse is present)
            if "klasse" in table.columns:
                is_dosing = np.array(
                    [v is not None and str(v) != "null" for v in table["klasse"].to_list()],
                    dtype=bool,
                )
            else:
                is_dosing = np.ones(n, dtype=bool)

            ch_arr = np.zeros(n, dtype=bool)
            prev_dosing_val = None
            col_vals = table[col].to_list()
            for idx in order:
                if not is_dosing[idx]:
                    ch_arr[idx] = False
                else:
                    v = col_vals[idx]
                    if prev_dosing_val is None:
                        ch_arr[idx] = True
                        prev_dosing_val = v
                    elif isinstance(v, float) and isinstance(prev_dosing_val, float):
                        if math.isclose(v, prev_dosing_val, rel_tol=1e-5):
                            ch_arr[idx] = False
                        else:
                            ch_arr[idx] = True
                            prev_dosing_val = v
                    elif v != prev_dosing_val:
                        ch_arr[idx] = True
                        prev_dosing_val = v
                    else:
                        ch_arr[idx] = False
            env[f"__changed__{col}"] = UArray(ch_arr.astype(float))

        reset_expr = compile_expr(
            reset_when,
            names=table.columns,
            text_names=text_names,
            extra_functions=RELATION_FUNCTIONS,
        )
        r_val = reset_expr.evaluate(env).value
        reset_arr = (r_val != 0) & ~np.isnan(r_val.astype(float))

    # Calculate cumulative sum
    out = np.full(n, np.nan)
    run = np.zeros(n, dtype=np.int64)
    run_no = 1
    total = 0.0
    last_end: float | None = None

    for k, idx in enumerate(order):
        reset = bool(reset_arr[idx])
        # Gap reset
        if last_end is not None and has_time and gap_s is not None and gap_s >= 0:
            if (t_start[idx] - last_end) >= gap_s:
                reset = True
        if reset:
            total = 0.0
            run_no += 1 if k > 0 else 0
        run[idx] = run_no

        inc = inc_arr[idx]
        if np.isfinite(inc):
            total += inc
            out[idx] = total
        else:
            out[idx] = np.nan if keep_nan else total

        if has_time:
            last_end = t_end[idx] if last_end is None else max(last_end, t_end[idx])

    if runs:
        return pl.Series(name, out, dtype=pl.Float64), pl.Series(f"{name}_lauf", run, dtype=pl.Int64)
    return pl.Series(name, out, dtype=pl.Float64)


# ------------------------------------------------------------------ bin
def bin(
    table_or_values: pl.DataFrame | pl.Series | np.ndarray | Sequence[float] | float,
    expr_or_width: str | float,
    width: float | None = None,
    *,
    name: str | None = None,
) -> pl.Series | float:
    """Round value(s) to the nearest multiple of `width`: round(x / width) * width.

    Can be called either as:
    - ``bin(table, "T_bett", 25) -> pl.Series``
    - ``bin(series, 25) -> pl.Series``
    - ``bin(149.5, 25) -> float``
    """
    if width is None:
        # bin(values, width)
        values = table_or_values
        w = float(expr_or_width)
        col_name = name or "bin"
        if isinstance(values, (int, float)):
            if w <= 0:
                raise ValueError(f"Bin width must be positive, got {w}.")
            return round(float(values) / w) * w
        if isinstance(values, pl.Series):
            col_name = name or values.name
            arr = values.to_numpy().astype(float)
        elif isinstance(values, np.ndarray):
            arr = values.astype(float)
        else:
            arr = np.asarray(values, dtype=float)

        if w <= 0:
            raise ValueError(f"Bin width must be positive, got {w}.")
        res = np.round(arr / w) * w
        return pl.Series(col_name, res, dtype=pl.Float64)
    else:
        # bin(table, expr, width)
        if not isinstance(table_or_values, pl.DataFrame):
            raise TypeError(f"Expected pl.DataFrame as first argument, got {type(table_or_values).__name__}.")
        table = table_or_values
        expr = str(expr_or_width)
        w = float(width)
        if w <= 0:
            raise ValueError(f"Bin width must be positive, got {w}.")
        col_name = name or expr
        env, text_names = _table_env(table)
        compiled = compile_expr(expr, names=table.columns, text_names=text_names)
        arr = compiled.evaluate(env).value.astype(float)
        res = np.round(arr / w) * w
        return pl.Series(col_name, res, dtype=pl.Float64)


# ------------------------------------------------------------------ apply relations
def apply_relation(table: pl.DataFrame, rel: Relation) -> pl.DataFrame:
    """Apply one relation to the step table, returning a new DataFrame with the column(s) added."""
    if isinstance(rel, NearestReferenceRelation):
        cols = nearest_reference(
            table,
            name=rel.name,
            value=rel.value,
            where=rel.where,
            match=rel.match,
            side=rel.side,
            combine=rel.combine,
            u_min=rel.u_min,
            u_one=rel.u_one,
            fallback=rel.fallback,
        )
        return table.with_columns(cols)
    elif isinstance(rel, CumulativeRelation):
        s = cumulative(
            table,
            name=rel.name,
            expr=rel.expr,
            reset_when=rel.reset_when,
            gap_s=rel.gap_s,
            runs=True,
        )
        return table.with_columns(*s)
    elif isinstance(rel, BinRelation):
        s = bin(table, rel.expr, rel.width, name=rel.name)
        return table.with_columns(s)
    raise TypeError(f"Unknown relation type {type(rel).__name__}.")


def apply_relations(table: pl.DataFrame, relations: Sequence[Relation]) -> pl.DataFrame:
    """Apply a sequence of relations in order to a step table."""
    cur = table
    for rel in relations:
        cur = apply_relation(cur, rel)
    return cur


__all__ = [
    "BinRelation",
    "CumulativeRelation",
    "FallbackValue",
    "NearestReferenceRelation",
    "RELATION_FUNCTIONS",
    "Relation",
    "apply_relation",
    "apply_relations",
    "bin",
    "cumulative",
    "nearest_reference",
]
