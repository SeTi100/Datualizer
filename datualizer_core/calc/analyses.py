"""Cross-step analyses: level crossings, line fits with GLS covariance, and aggregations (Etappe 2b, step 8, PR 6).

Works on the results table (from `session.results()`) or step table, formulas from `calc/expr.py`,
and uncertain constants (`calc/uncertain.py`).

1. `crossing`: linear interpolation for threshold crossings (e.g. light-off T50/T90), range from
   y ± u_y, without crossing text "< x_min" or "> x_max", and systematic range `bereich_syst`
   with all systematic contributions of the same scope shifted together.
2. `fit_line`: weighted line fit y = m x + b with errors on both coordinates via effective variance
   (sigma^2 = u_y^2 + m^2 * u_x^2, 5 iterations), reduced chi^2, Birge factor max(1, sqrt(chi^2_red)),
   and Generalized Least Squares (GLS) using row-to-row covariance from `Constant.scope`.
   Outputs: m and u_m (random only), u_m with correlation, u_b, and derived quantities (e.g. Ea = -m * 8.314).
3. `aggregate`: pivot table or 1D aggregation (mean, median, n, min, max).
4. Between-row correlation: `Constant.scope` (e.g. "tag" = constant is identical for all steps of the
   same measurement day; "campaign" = constant applies across the entire campaign).
   Random contributions only on the diagonal; systematic contributions of the same constant within the
   same scope are fully correlated (c_i * c_j), independent across scope boundaries. `group` from PR 1
   remains within-row correlation and is not repurposed.
5. Monte Carlo (optional, `mc_samples`): draws from multivariate normal distribution with the same covariance.

No GUI or campaign imports (principle from AGENTS.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Annotated, Any, Literal, Mapping, Sequence, Union

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from datualizer_core.calc.expr import FUNCTIONS, Expr, ExprError, compile_expr
from datualizer_core.calc.uncertain import Constant, UArray


def _name(value: str, what: str) -> str:
    if not value.isidentifier() or value in FUNCTIONS:
        raise ValueError(
            f"{what} {value!r} must be a name usable in formulas (letters, digits, '_', no function name)."
        )
    return value


# ------------------------------------------------------------------ Pydantic models
class CrossingAnalysis(BaseModel):
    """Threshold crossing interpolation (e.g. Light-off T50/T90)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["crossing"] = "crossing"
    x: str = Field(min_length=1, description="x column or expression (e.g. 'T_bett').")
    y: str = Field(min_length=1, description="y column or expression (e.g. 'X').")
    level: float = Field(description="Crossing level (e.g. 0.5 for T50).")
    group: str | None = Field(default=None, description="Grouping column (e.g. 'v_mfc').")
    u_y: str | None = Field(default=None, description="Standard uncertainty column or expression for y.")
    where: str | None = Field(default=None, description="Filter condition.")
    unit: str = Field(default="", description="Unit of x for formatted text output (e.g. '°C').")
    scope: str = Field(default="tag", description="Scope for between-row correlation (e.g. 'tag').")
    mc_samples: int | None = Field(default=None, ge=1, description="Optional Monte Carlo sample count.")

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "Analysis name")

    @field_validator("where")
    @classmethod
    def _where_valid(cls, v: str | None) -> str | None:
        if v is not None:
            compile_expr(v)
        return v


class FitLineAnalysis(BaseModel):
    """Line fit y = m x + b with effective variance and GLS covariance."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["fit_line"] = "fit_line"
    x_expr: str = Field(min_length=1, description="x formula or column (e.g. '1000 / (T_bett + 273.15)').")
    y_expr: str = Field(min_length=1, description="y formula or column (e.g. 'log(k)').")
    u_x: str | None = Field(default=None, description="Standard uncertainty of x (formula or column).")
    u_y: str | None = Field(default=None, description="Standard uncertainty of y (formula or column).")
    where: str | None = Field(default=None, description="Filter condition.")
    method: Literal["effective_variance"] = "effective_variance"
    derived: dict[str, str] = Field(
        default_factory=dict, description="Derived formulas, e.g. {'Ea': '-m * 8.314462'}."
    )
    scope: str = Field(default="tag", description="Scope for between-row correlation (e.g. 'tag').")
    mc_samples: int | None = Field(default=None, ge=1, description="Optional Monte Carlo sample count.")

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "Analysis name")

    @field_validator("x_expr", "y_expr")
    @classmethod
    def _expr_valid(cls, v: str) -> str:
        compile_expr(v)
        return v

    @field_validator("where", "u_x", "u_y")
    @classmethod
    def _opt_expr_valid(cls, v: str | None) -> str | None:
        if v is not None:
            compile_expr(v)
        return v

    @field_validator("derived")
    @classmethod
    def _derived_valid(cls, v: dict[str, str]) -> dict[str, str]:
        for k, expr in v.items():
            _name(k, "Derived quantity")
            compile_expr(expr)
        return v


class AggregateAnalysis(BaseModel):
    """Aggregation or pivot matrix."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["aggregate"] = "aggregate"
    rows: str = Field(min_length=1, description="Row grouping column.")
    cols: str | None = Field(default=None, description="Optional column grouping for 2D pivot matrix.")
    value: str = Field(min_length=1, description="Value column to aggregate.")
    where: str | None = Field(default=None, description="Filter condition.")
    how: Literal["mean", "median", "n", "min", "max", "sum", "std"] = "mean"

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "Analysis name")

    @field_validator("where")
    @classmethod
    def _where_valid(cls, v: str | None) -> str | None:
        if v is not None:
            compile_expr(v)
        return v


Analysis = Annotated[
    Union[CrossingAnalysis, FitLineAnalysis, AggregateAnalysis],
    Field(discriminator="kind"),
]


# ------------------------------------------------------------------ Results Dataclasses
@dataclass
class CrossingGroup:
    """Crossing result for one group."""

    group_value: Any
    n: int
    wert: str
    bereich: str
    bereich_syst: str
    x_cross: float | None
    x_lo: float | None
    x_hi: float | None
    x_lo_syst: float | None
    x_hi_syst: float | None
    text: str = ""


@dataclass
class CrossingResult:
    """Full result of a crossing analysis."""

    name: str
    table: pl.DataFrame
    groups: dict[Any, CrossingGroup]
    mc_results: dict[str, Any] = field(default_factory=dict)

    @property
    def df(self) -> pl.DataFrame:
        return self.table

    @property
    def wert(self) -> str:
        if None in self.groups:
            return self.groups[None].wert
        if len(self.groups) == 1:
            return next(iter(self.groups.values())).wert
        return f"{len(self.groups)} groups"

    @property
    def bereich(self) -> str:
        if None in self.groups:
            return self.groups[None].bereich
        if len(self.groups) == 1:
            return next(iter(self.groups.values())).bereich
        return ""

    @property
    def bereich_syst(self) -> str:
        if None in self.groups:
            return self.groups[None].bereich_syst
        if len(self.groups) == 1:
            return next(iter(self.groups.values())).bereich_syst
        return ""


@dataclass
class FitLineResult:
    """Result of a line fit y = m x + b with GLS and Birge factor."""

    name: str
    m: float
    u_m: float           # with correlation (GLS)
    u_m_zufall: float    # random only
    b: float
    u_b: float
    chi2_red: float
    birge: float
    n: int
    derived: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # name -> (val, u_corr, u_rand)
    table: pl.DataFrame = field(default_factory=pl.DataFrame)
    mc_results: dict[str, float] = field(default_factory=dict)
    # the fit with random uncertainties only (effective variance, Birge factor): no correlation between rows
    m_zufall: float = math.nan
    b_zufall: float = math.nan
    chi2_red_zufall: float = math.nan
    derived_zufall: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (val, u) of that fit

    @property
    def df(self) -> pl.DataFrame:
        return self.table


# ------------------------------------------------------------------ Table helpers
def _table_env(table: pl.DataFrame) -> tuple[dict[str, Any], list[str]]:
    """Build evaluation environment and list of text column names from a step/results table."""
    text_names = [c for c in table.columns if table[c].dtype in (pl.String, pl.Categorical)]
    env: dict[str, Any] = {}
    for c in table.columns:
        if c in text_names:
            env[c] = np.array(
                [None if v is None else str(v) for v in table[c].to_list()],
                dtype=object,
            )
        elif table[c].dtype == pl.Boolean:
            env[c] = table[c].to_numpy().astype(bool)
        else:
            try:
                vals = table[c].to_numpy().astype(float)
                u_col = f"u_{c}"
                if u_col in table.columns and table[u_col].dtype not in (pl.String, pl.Categorical):
                    u_vals = table[u_col].to_numpy().astype(float)
                    env[c] = UArray.input(c, vals, u_vals, kind="zufall")
                else:
                    env[c] = UArray(vals)
            except (ValueError, TypeError):
                text_names.append(c)
                env[c] = np.array(
                    [None if v is None else str(v) for v in table[c].to_list()],
                    dtype=object,
                )
    return env, text_names


def _step_dates(table: pl.DataFrame) -> list[str]:
    """Extract YYYY-MM-DD measurement day strings per row."""
    if "tag" in table.columns:
        return [str(v) if v is not None else "" for v in table["tag"].to_list()]
    start_col = next((c for c in ("step_start", "start") if c in table.columns), None)
    if start_col is not None:
        s = table[start_col]
        if s.dtype in (pl.Datetime, pl.Date):
            return [str(d)[:10] if d is not None else "" for d in s.to_list()]
        # Physical epoch seconds or floats
        try:
            import datetime
            nums = s.to_numpy().astype(float)
            dates = []
            for num in nums:
                if np.isnan(num):
                    dates.append("")
                else:
                    # Treat as epoch seconds
                    dt = datetime.datetime.fromtimestamp(num, tz=datetime.timezone.utc)
                    dates.append(dt.strftime("%Y-%m-%d"))
            return dates
        except Exception:
            pass
    return ["" for _ in range(len(table))]


def _filter_table(table: pl.DataFrame, where: str | None) -> pl.DataFrame:
    """Filter table by formula condition."""
    if where is None or not where.strip():
        return table
    env, text_names = _table_env(table)
    expr = compile_expr(where, names=list(env.keys()), text_names=text_names)
    res = expr.evaluate(env)
    val = res.value
    if val.dtype == bool:
        mask = val
    else:
        v = val.astype(float)
        mask = (v != 0) & ~np.isnan(v)
    return table.filter(mask)


# ------------------------------------------------------------------ Covariance Builder
def build_row_covariance(
    table: pl.DataFrame,
    y_expr: str,
    constants: Mapping[str, Constant] | None = None,
    u_y: str | None = None,
    scope_default: str = "tag",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build covariance matrix across rows for a quantity `y_expr`.

    Parameters
    ----------
    table : pl.DataFrame
        Filtered results table with budget columns.
    y_expr : str
        Formula or column for y.
    constants : Mapping[str, Constant] | None
        Campaign constants with their scope and kind.
    u_y : str | None
        Optional override for standard uncertainty of y.
    scope_default : str
        Default scope if constant scope is not set.

    Returns
    -------
    y : ndarray
        Evaluated y values per row.
    u_y_rand : ndarray
        Random standard uncertainty per row (diagonal).
    V_y : ndarray
        (N, N) full covariance matrix of y.
    budget_contributions : dict[str, ndarray]
        Signed sensitivity contribution per input and row.
    """
    n = len(table)
    if n == 0:
        return np.array([]), np.array([]), np.zeros((0, 0)), {}

    env, text_names = _table_env(table)
    # Add constants to env
    const_dict = dict(constants) if constants else {}
    for k, c in const_dict.items():
        if k not in env:
            env[k] = c.to_uarray()

    expr = compile_expr(y_expr, names=list(env.keys()), text_names=text_names)
    res = expr.evaluate(env)
    y_val = np.broadcast_to(np.asarray(res.value, dtype=float), (n,)).copy()

    # Determine gradients ∂y/∂col for columns that appear in y_expr
    diff_env: dict[str, Any] = {}
    for k in expr.names:
        if k in table.columns and table[k].dtype not in (pl.String, pl.Categorical):
            try:
                diff_env[k] = UArray.input(k, table[k].to_numpy().astype(float), 1.0)
            except Exception:
                pass
        elif k in const_dict:
            diff_env[k] = const_dict[k].to_uarray()
        elif k in env:
            diff_env[k] = env[k]

    try:
        diff_res = expr.evaluate(diff_env)
    except Exception:
        diff_res = res

    grads: dict[str, np.ndarray] = {}
    if isinstance(diff_res, UArray):
        for k, g in diff_res.grads.items():
            grads[k] = np.broadcast_to(np.asarray(g, dtype=float), (n,)).copy()
    if y_expr in table.columns and y_expr not in grads:
        grads[y_expr] = np.ones(n, dtype=float)

    # 1. Systematic contributions per input
    budget_contributions: dict[str, np.ndarray] = {}
    for col, g in grads.items():
        prefix = f"b_{col}__"
        for c_name in table.columns:
            if c_name.startswith(prefix):
                inp = c_name[len(prefix):]
                b_val = table[c_name].to_numpy().astype(float)
                c_inp = g * b_val
                if inp in budget_contributions:
                    budget_contributions[inp] = budget_contributions[inp] + c_inp
                else:
                    budget_contributions[inp] = c_inp
        # Also check for direct constant dependency
        if col in const_dict and const_dict[col].kind == "syst":
            c_inp = g * const_dict[col].u
            if col in budget_contributions:
                budget_contributions[col] = budget_contributions[col] + c_inp
            else:
                budget_contributions[col] = c_inp

    # If y_expr is a direct column and has uX_* columns (e.g. legacy/fixture columns)
    if y_expr in table.columns:
        for c_name in table.columns:
            if c_name.startswith("uX_") and c_name.endswith("_syst"):
                inp = c_name[3:-5]
                b_val = table[c_name].to_numpy().astype(float)
                budget_contributions[inp] = b_val

    # Also check if u_<col>_syst exists but no individual budget columns
    for col, g in grads.items():
        if not budget_contributions and f"u_{col}_syst" in table.columns:
            u_sys = table[f"u_{col}_syst"].to_numpy().astype(float)
            if np.any(u_sys > 0):
                budget_contributions[f"{col}_syst"] = g * u_sys

    # 2. Random uncertainty
    if u_y is not None:
        u_expr = compile_expr(u_y, names=list(env.keys()), text_names=text_names)
        u_res = u_expr.evaluate(env)
        u_input = np.broadcast_to(np.asarray(u_res.value, dtype=float), (n,)).copy()
        if "zufall" in u_y.lower() or "rand" in u_y.lower():
            u_y_rand = u_input
        else:
            # u_y is total uncertainty; subtract systematic variance so total variance = u_input^2
            var_sys_total = np.zeros(n, dtype=float)
            for c_vec in budget_contributions.values():
                var_sys_total += c_vec**2
            var_rand = np.maximum(0.0, u_input**2 - var_sys_total)
            u_y_rand = np.sqrt(var_rand)
    else:
        var_rand = np.zeros(n, dtype=float)
        for col, g in grads.items():
            if f"u_{col}_zufall" in table.columns:
                u_z = table[f"u_{col}_zufall"].to_numpy().astype(float)
                var_rand += (g * u_z) ** 2
            elif f"u_{col}" in table.columns:
                u_tot = table[f"u_{col}"].to_numpy().astype(float)
                if f"u_{col}_syst" in table.columns:
                    u_s = table[f"u_{col}_syst"].to_numpy().astype(float)
                    var_rand += np.maximum(0.0, (g * u_tot) ** 2 - (g * u_s) ** 2)
                else:
                    var_rand += (g * u_tot) ** 2
            elif col in const_dict and const_dict[col].kind == "zufall":
                var_rand += (g * const_dict[col].u) ** 2
        u_y_rand = np.sqrt(var_rand)

    # 3. Group systematic inputs that share a `group` within a row
    # In PR 1: systematic inputs with the same `group` add linearly: c_group = sum_{i in g} c_i
    grouped_contributions: dict[str, tuple[str, np.ndarray]] = {}  # key -> (scope, c_vec)
    step_dates = _step_dates(table)

    # Map each input to group and scope
    group_map: dict[str, list[str]] = {}
    standalone: list[str] = []
    for inp in budget_contributions:
        const = const_dict.get(inp)
        grp = const.group if const else None
        if grp:
            group_map.setdefault(grp, []).append(inp)
        else:
            standalone.append(inp)

    # For standalone inputs
    for inp in standalone:
        const = const_dict.get(inp)
        sc = const.scope if const and const.scope else scope_default
        grouped_contributions[inp] = (sc, budget_contributions[inp])

    # For grouped inputs
    for grp, inps in group_map.items():
        c_sum = np.zeros(n, dtype=float)
        sc = scope_default
        for inp in inps:
            c_sum += budget_contributions[inp]
            const = const_dict.get(inp)
            if const and const.scope:
                sc = const.scope
        grouped_contributions[f"group_{grp}"] = (sc, c_sum)

    # 4. Build Covariance Matrix V_y
    V_y = np.diag(u_y_rand**2)

    for key, (sc, c_vec) in grouped_contributions.items():
        # Determine partition per row
        if sc in ("tag", "day", "date"):
            partitions = step_dates
        elif sc in ("campaign", "global", "all"):
            partitions = ["global" for _ in range(n)]
        elif sc in table.columns:
            partitions = [str(v) for v in table[sc].to_list()]
        elif sc:
            partitions = [sc for _ in range(n)]
        else:
            # Empty scope -> independent between rows (only diagonal)
            partitions = [f"row_{i}" for i in range(n)]

        for i in range(n):
            for j in range(n):
                if partitions[i] == partitions[j] and partitions[i] != "":
                    V_y[i, j] += c_vec[i] * c_vec[j]

    return y_val, u_y_rand, V_y, budget_contributions


# ------------------------------------------------------------------ 1. Crossing
def _interpolate_crossing(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """Find first threshold crossing in sorted x points."""
    for i in range(len(x) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return None


def _format_crossing(
    x: np.ndarray, y: np.ndarray, level: float, x_cross: float | None, unit: str = ""
) -> str:
    """Format crossing value or '< x_min' / '> x_max' without crossing."""
    u_str = f" {unit}" if unit else ""
    if x_cross is not None:
        return f"{x_cross:.0f}{u_str}".strip()
    if len(x) == 0:
        return "–"
    # General direction of curve
    is_increasing = y[-1] >= y[0] if len(y) > 1 else True
    if is_increasing:
        if float(np.nanmax(y)) < level:
            return f"> {np.nanmax(x):.0f}{u_str}".strip()
        if float(np.nanmin(y)) >= level:
            return f"< {np.nanmin(x):.0f}{u_str}".strip()
    else:
        if float(np.nanmin(y)) > level:
            return f"> {np.nanmax(x):.0f}{u_str}".strip()
        if float(np.nanmax(y)) <= level:
            return f"< {np.nanmin(x):.0f}{u_str}".strip()
    return f"> {np.nanmax(x):.0f}{u_str}".strip()


def crossing(
    table: pl.DataFrame,
    x: str,
    y: str,
    level: float,
    group: str | None = None,
    u_y: str | None = None,
    where: str | None = None,
    unit: str = "",
    scope: str = "tag",
    constants: Mapping[str, Constant] | None = None,
    mc_samples: int | None = None,
    name: str = "crossing",
) -> CrossingResult:
    """Linear threshold crossing interpolation (e.g. Light-off T50/T90).

    Parameters
    ----------
    table : pl.DataFrame
        Step or results table.
    x : str
        Column or formula for x (e.g. 'T_bett').
    y : str
        Column or formula for y (e.g. 'X').
    level : float
        Crossing threshold (e.g. 0.5 for T50).
    group : str | None
        Optional column to group by (e.g. 'v_mfc').
    u_y : str | None
        Standard uncertainty for y. Default: 'u_{y}'.
    where : str | None
        Filter condition expression.
    unit : str
        Unit of x for formatting (e.g. '°C').
    scope : str
        Scope for between-row correlation of systematic uncertainties (default 'tag').
    constants : Mapping[str, Constant] | None
        Campaign constants.
    mc_samples : int | None
        Optional Monte Carlo samples count.
    name : str
        Name of the analysis.
    """
    sub = _filter_table(table, where)
    if len(sub) == 0:
        return CrossingResult(
            name=name,
            table=pl.DataFrame(
                schema={
                    (group or "group"): pl.String,
                    "n": pl.Int64,
                    name: pl.String,
                    f"{name}_bereich": pl.String,
                    f"{name}_bereich_syst": pl.String,
                }
            ),
            groups={},
        )

    # Determine groups
    if group is not None and group in sub.columns:
        group_keys = sub[group].unique().sort().to_list()
    else:
        group_keys = [None]

    groups_res: dict[Any, CrossingGroup] = {}
    table_rows: list[dict[str, Any]] = []

    for g_key in group_keys:
        if g_key is not None:
            g_table = sub.filter(pl.col(group) == g_key)
        else:
            g_table = sub

        if len(g_table) == 0:
            continue

        # Evaluate x, y, and covariance for this group
        g_env, g_text = _table_env(g_table)
        x_expr = compile_expr(x, names=list(g_env.keys()), text_names=g_text)
        x_vals = np.broadcast_to(np.asarray(x_expr.evaluate(g_env).value, dtype=float), (len(g_table),)).copy()

        y_vals, u_y_rand, V_y, budget = build_row_covariance(
            g_table, y_expr=y, constants=constants, u_y=u_y, scope_default=scope
        )
        u_y_tot = np.sqrt(np.diag(V_y))

        # Filter out NaN in x or y
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_g = x_vals[valid]
        y_g = y_vals[valid]
        u_tot_g = u_y_tot[valid]
        V_y_g = V_y[np.ix_(valid, valid)]

        # Sort by x ascending
        order = np.argsort(x_g)
        x_s = x_g[order]
        y_s = y_g[order]
        u_tot_s = u_tot_g[order]
        V_y_s = V_y_g[np.ix_(order, order)]

        n_pts = len(x_s)
        if n_pts == 0:
            continue

        # 1. Base crossing
        x_mid = _interpolate_crossing(x_s, y_s, level)
        mid_text = _format_crossing(x_s, y_s, level, x_mid, unit)

        # 2. Total uncertainty range (point-by-point ± u)
        # For increasing curve, y + u reaches level at lower x (lo), y - u at higher x (hi)
        x_lo_val = _interpolate_crossing(x_s, y_s + u_tot_s, level)
        x_hi_val = _interpolate_crossing(x_s, y_s - u_tot_s, level)
        txt_lo = _format_crossing(x_s, y_s + u_tot_s, level, x_lo_val, unit)
        txt_hi = _format_crossing(x_s, y_s - u_tot_s, level, x_hi_val, unit)
        bereich = f"{txt_lo} … {txt_hi}"

        # 3. Systematic range (all systematic contributions in scope shifted together)
        # Extract total systematic uncertainty vector per point: sqrt(diag(V_y) - u_rand^2)
        u_rand_s = u_y_rand[valid][order]
        var_syst_s = np.maximum(0.0, u_tot_s**2 - u_rand_s**2)
        u_sys_s = np.sqrt(var_syst_s)

        if np.any(u_sys_s > 0):
            x_lo_sys = _interpolate_crossing(x_s, y_s + u_sys_s, level)
            x_hi_sys = _interpolate_crossing(x_s, y_s - u_sys_s, level)
            txt_lo_sys = _format_crossing(x_s, y_s + u_sys_s, level, x_lo_sys, unit)
            txt_hi_sys = _format_crossing(x_s, y_s - u_sys_s, level, x_hi_sys, unit)
            bereich_syst = f"{txt_lo_sys} … {txt_hi_sys}"
        else:
            x_lo_sys = x_mid
            x_hi_sys = x_mid
            bereich_syst = "–"

        # Combined display text
        display_text = f"{mid_text} [{bereich}]" if bereich and bereich != "–" else mid_text

        grp_obj = CrossingGroup(
            group_value=g_key,
            n=n_pts,
            wert=mid_text,
            bereich=bereich,
            bereich_syst=bereich_syst,
            x_cross=x_mid,
            x_lo=x_lo_val,
            x_hi=x_hi_val,
            x_lo_syst=x_lo_sys,
            x_hi_syst=x_hi_sys,
            text=display_text,
        )
        groups_res[g_key] = grp_obj

        row_dict: dict[str, Any] = {
            (group or "group"): g_key if g_key is not None else "all",
            "n": n_pts,
            name: mid_text,
            f"{name}_bereich": bereich,
            f"{name}_bereich_syst": bereich_syst,
        }
        table_rows.append(row_dict)

    res_table = pl.DataFrame(table_rows) if table_rows else pl.DataFrame()

    # Optional Monte Carlo
    mc_res: dict[str, Any] = {}
    if mc_samples is not None and mc_samples > 0 and None in groups_res:
        rng = np.random.default_rng(42)
        # MC for the first or single group
        valid_x = []
        try:
            dy_samples = rng.multivariate_normal(np.zeros(len(x_s)), V_y_s, size=mc_samples)
            for s_idx in range(mc_samples):
                y_sim = y_s + dy_samples[s_idx]
                sim_x = _interpolate_crossing(x_s, y_sim, level)
                if sim_x is not None:
                    valid_x.append(sim_x)
            if valid_x:
                mc_res["mc_mean"] = float(np.mean(valid_x))
                mc_res["mc_std"] = float(np.std(valid_x, ddof=1)) if len(valid_x) > 1 else 0.0
                mc_res["mc_q16"] = float(np.percentile(valid_x, 16))
                mc_res["mc_q84"] = float(np.percentile(valid_x, 84))
                mc_res["mc_valid_fraction"] = len(valid_x) / mc_samples
        except Exception:
            pass

    return CrossingResult(name=name, table=res_table, groups=groups_res, mc_results=mc_res)


# ------------------------------------------------------------------ 2. Line Fit
def fit_line(
    table: pl.DataFrame,
    x_expr: str,
    y_expr: str,
    u_x: str | None = None,
    u_y: str | None = None,
    where: str | None = None,
    method: str = "effective_variance",
    derived: Mapping[str, str] | None = None,
    scope: str = "tag",
    constants: Mapping[str, Constant] | None = None,
    mc_samples: int | None = None,
    name: str = "fit_line",
) -> FitLineResult:
    """Weighted line fit y = m x + b with effective variance and GLS covariance.

    Parameters
    ----------
    table : pl.DataFrame
        Step or results table.
    x_expr : str
        Formula or column for x (e.g. '1000 / (T_bett + 273.15)').
    y_expr : str
        Formula or column for y (e.g. 'log(k)').
    u_x : str | None
        Standard uncertainty of x (formula or column).
    u_y : str | None
        Standard uncertainty of y (formula or column).
    where : str | None
        Filter condition expression.
    method : str
        Fit method ('effective_variance').
    derived : Mapping[str, str] | None
        Formulas for derived quantities (e.g. {'Ea': '-m * 8.314462'}).
    scope : str
        Scope for between-row correlation of systematic uncertainties (default 'tag').
    constants : Mapping[str, Constant] | None
        Campaign constants.
    mc_samples : int | None
        Optional Monte Carlo samples count.
    name : str
        Name of the analysis.
    """
    sub = _filter_table(table, where)
    n = len(sub)
    if n < 2:
        raise ValueError(f"At least 2 points required for line fit; got {n}.")

    env, text_names = _table_env(sub)
    const_dict = dict(constants) if constants else {}
    for k, c in const_dict.items():
        if k not in env:
            env[k] = c.to_uarray()

    # 1. Evaluate x and u_x
    x_comp = compile_expr(x_expr, names=list(env.keys()), text_names=text_names)
    x_res = x_comp.evaluate(env)
    x_vals = np.broadcast_to(np.asarray(x_res.value, dtype=float), (n,)).copy()

    if u_x is not None:
        ux_comp = compile_expr(u_x, names=list(env.keys()), text_names=text_names)
        ux_res = ux_comp.evaluate(env)
        ux_vals = np.broadcast_to(np.asarray(ux_res.value, dtype=float), (n,)).copy()
    elif isinstance(x_res, UArray):
        ux_vals = np.broadcast_to(np.asarray(x_res.u(), dtype=float), (n,)).copy()
    elif f"u_{x_expr}" in sub.columns:
        ux_vals = sub[f"u_{x_expr}"].to_numpy().astype(float)
    else:
        ux_vals = np.zeros(n, dtype=float)

    # 2. Evaluate y, u_y_rand and covariance matrix V_y
    y_vals, uy_rand, V_y, budget = build_row_covariance(
        sub, y_expr=y_expr, constants=constants, u_y=u_y, scope_default=scope
    )

    # Filter non-finite points
    finite = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(ux_vals) & np.isfinite(uy_rand)
    x = x_vals[finite]
    ux = ux_vals[finite]
    y = y_vals[finite]
    uy_r = uy_rand[finite]
    V_y_filt = V_y[np.ix_(finite, finite)]

    N = len(x)
    if N < 2:
        raise ValueError(f"At least 2 finite points required for line fit; got {N}.")

    X_mat = np.column_stack([x, np.ones(N)])

    # Initial slope estimate
    m_init = float(np.polyfit(x, y, 1)[0])

    # 3. Fit A: Random-only fit (effective variance on diagonal)
    m_rand = m_init
    b_rand = 0.0
    cov_beta_rand = np.eye(2)
    sig2_rand = np.ones(N)

    for _ in range(5):
        sig2_rand = uy_r**2 + (m_rand * ux) ** 2
        sig2_rand = np.maximum(sig2_rand, 1e-15)
        W_rand = np.diag(1.0 / sig2_rand)
        cov_beta_rand = np.linalg.inv(X_mat.T @ W_rand @ X_mat)
        beta_rand = cov_beta_rand @ (X_mat.T @ W_rand @ y)
        m_rand, b_rand = float(beta_rand[0]), float(beta_rand[1])

    dof = max(1, N - 2)
    chi2_red_rand = float(np.sum((y - (m_rand * x + b_rand)) ** 2 / sig2_rand) / dof)
    birge_rand = max(1.0, math.sqrt(chi2_red_rand))
    u_m_zufall = float(math.sqrt(cov_beta_rand[0, 0]) * birge_rand)

    # 4. Fit B: Generalized Least Squares (GLS) with full covariance V_y
    m_gls = m_init
    b_gls = 0.0
    cov_beta_gls = np.eye(2)
    inv_V_gls = np.eye(N)

    for _ in range(5):
        V_tot = V_y_filt + np.diag((m_gls * ux) ** 2)
        # Regularize if needed for stability
        inv_V_gls = np.linalg.pinv(V_tot)
        cov_beta_gls = np.linalg.pinv(X_mat.T @ inv_V_gls @ X_mat)
        beta_gls = cov_beta_gls @ (X_mat.T @ inv_V_gls @ y)
        m_gls, b_gls = float(beta_gls[0]), float(beta_gls[1])

    res = y - (m_gls * x + b_gls)
    chi2_red = float((res.T @ inv_V_gls @ res) / dof)
    birge = max(1.0, math.sqrt(chi2_red))
    u_m = float(math.sqrt(cov_beta_gls[0, 0]) * birge)
    u_b = float(math.sqrt(cov_beta_gls[1, 1]) * birge)

    # 5. Derived quantities (e.g. Ea = -m * 8.314)
    derived_res: dict[str, tuple[float, float, float]] = {}  # name -> (val, u_corr, u_rand)
    derived_rand: dict[str, tuple[float, float]] = {}  # name -> (val, u) of the random-only fit
    if derived:
        for d_name, d_expr in derived.items():
            # Build env with m and b as UArray
            d_env: dict[str, Any] = dict(const_dict)
            d_env["m"] = UArray.input("m", m_gls, u_m, kind="zufall")
            d_env["b"] = UArray.input("b", b_gls, u_b, kind="zufall")
            comp = compile_expr(d_expr, names=list(d_env.keys()))
            eval_corr = comp.evaluate(d_env)
            val_d = float(eval_corr.value)
            u_d = float(eval_corr.u())

            # Evaluate with u_m_zufall
            d_env_rand: dict[str, Any] = dict(const_dict)
            d_env_rand["m"] = UArray.input("m", m_rand, u_m_zufall, kind="zufall")
            d_env_rand["b"] = UArray.input("b", b_rand, float(math.sqrt(cov_beta_rand[1, 1]) * birge_rand), kind="zufall")
            eval_rand = comp.evaluate(d_env_rand)
            u_d_rand = float(eval_rand.u())

            derived_res[d_name] = (val_d, u_d, u_d_rand)
            derived_rand[d_name] = (float(eval_rand.value), u_d_rand)

    # 6. Monte Carlo (optional)
    mc_res: dict[str, float] = {}
    if mc_samples is not None and mc_samples > 0:
        rng = np.random.default_rng(42)
        try:
            V_tot_final = V_y_filt + np.diag((m_gls * ux) ** 2)
            dy_sim = rng.multivariate_normal(np.zeros(N), V_tot_final, size=mc_samples)
            dx_sim = rng.normal(0, ux, size=(mc_samples, N))
            m_sims = []
            b_sims = []
            for s_idx in range(mc_samples):
                xs = x + dx_sim[s_idx]
                ys = y + dy_sim[s_idx]
                Xs = np.column_stack([xs, np.ones(N)])
                bs = np.linalg.pinv(Xs.T @ inv_V_gls @ Xs) @ (Xs.T @ inv_V_gls @ ys)
                m_sims.append(bs[0])
                b_sims.append(bs[1])
            m_arr = np.array(m_sims)
            b_arr = np.array(b_sims)
            mc_res["mc_m_mean"] = float(np.mean(m_arr))
            mc_res["mc_m_std"] = float(np.std(m_arr, ddof=1))
            mc_res["mc_b_mean"] = float(np.mean(b_arr))
            mc_res["mc_b_std"] = float(np.std(b_arr, ddof=1))
            for d_name, d_expr in (derived or {}).items():
                if d_name.lower() == "ea":
                    # common Arrhenius: Ea = -m * R
                    # If formula contains m
                    comp = compile_expr(d_expr, names=["m", "b"])
                    ea_sims = [float(comp.evaluate({"m": ms, "b": bs}).value) for ms, bs in zip(m_arr, b_arr)]
                    mc_res[f"mc_{d_name}_mean"] = float(np.mean(ea_sims))
                    mc_res[f"mc_{d_name}_std"] = float(np.std(ea_sims, ddof=1))
        except Exception:
            pass

    # 7. Summary table representation
    row_data: dict[str, Any] = {
        "n": [N],
        "m": [m_gls],
        "u_m": [u_m],
        "u_m_zufall": [u_m_zufall],
        "b": [b_gls],
        "u_b": [u_b],
        "chi2_red": [chi2_red],
        "birge": [birge],
    }
    for d_name, (d_val, d_u, d_u_zuf) in derived_res.items():
        row_data[d_name] = [d_val]
        row_data[f"u_{d_name}"] = [d_u]
        row_data[f"u_{d_name}_zufall"] = [d_u_zuf]

    res_df = pl.DataFrame(row_data)

    return FitLineResult(
        name=name,
        m=m_gls,
        u_m=u_m,
        u_m_zufall=u_m_zufall,
        b=b_gls,
        u_b=u_b,
        chi2_red=chi2_red,
        birge=birge,
        n=N,
        derived=derived_res,
        table=res_df,
        mc_results=mc_res,
        m_zufall=m_rand,
        b_zufall=b_rand,
        chi2_red_zufall=chi2_red_rand,
        derived_zufall=derived_rand,
    )


# ------------------------------------------------------------------ 3. Aggregate
def aggregate(
    table: pl.DataFrame,
    rows: str,
    cols: str | None = None,
    value: str = "",
    where: str | None = None,
    how: str = "mean",
) -> pl.DataFrame:
    """Aggregate or pivot table (mean, median, n, min, max, sum, std).

    Parameters
    ----------
    table : pl.DataFrame
        Step or results table.
    rows : str
        Row grouping column.
    cols : str | None
        Optional column for 2D pivot matrix.
    value : str
        Value column to aggregate.
    where : str | None
        Filter condition expression.
    how : str
        Aggregation method ('mean', 'median', 'n', 'min', 'max', 'sum', 'std').
    """
    sub = _filter_table(table, where)
    if len(sub) == 0:
        return pl.DataFrame()

    if rows not in sub.columns:
        raise ValueError(f"Row grouping column {rows!r} not in table; available: {sub.columns}.")
    if cols is not None and cols not in sub.columns:
        raise ValueError(f"Column grouping column {cols!r} not in table; available: {sub.columns}.")
    if value not in sub.columns:
        raise ValueError(f"Value column {value!r} not in table; available: {sub.columns}.")

    # Map `how` to polars aggregation function name
    how_lower = how.lower()
    if how_lower in ("n", "count"):
        agg_fn = "count"
    elif how_lower in ("mean", "median", "min", "max", "sum", "std"):
        agg_fn = how_lower
    else:
        raise ValueError(f"Unsupported aggregation method {how!r}; choose from mean, median, n, min, max, sum, std.")

    if cols is None:
        # 1D aggregation
        col_expr = pl.col(value)
        if agg_fn == "count":
            res_expr = col_expr.count().alias(f"{value}_n")
        elif agg_fn == "mean":
            res_expr = col_expr.mean().alias(f"{value}_mean")
        elif agg_fn == "median":
            res_expr = col_expr.median().alias(f"{value}_median")
        elif agg_fn == "min":
            res_expr = col_expr.min().alias(f"{value}_min")
        elif agg_fn == "max":
            res_expr = col_expr.max().alias(f"{value}_max")
        elif agg_fn == "sum":
            res_expr = col_expr.sum().alias(f"{value}_sum")
        elif agg_fn == "std":
            res_expr = col_expr.std().alias(f"{value}_std")

        return sub.group_by(rows).agg(res_expr).sort(rows)
    else:
        # 2D pivot matrix
        piv = sub.pivot(index=rows, on=cols, values=value, aggregate_function=agg_fn).sort(rows)
        # Sort columns nicely: index first, then column keys sorted
        other_cols = [c for c in piv.columns if c != rows]
        try:
            sorted_cols = sorted(other_cols, key=lambda c: float(c))
        except (ValueError, TypeError):
            sorted_cols = sorted(other_cols)
        return piv.select([rows] + sorted_cols)


__all__ = [
    "AggregateAnalysis",
    "Analysis",
    "CrossingAnalysis",
    "CrossingGroup",
    "CrossingResult",
    "FitLineAnalysis",
    "FitLineResult",
    "aggregate",
    "build_row_covariance",
    "crossing",
    "fit_line",
]
