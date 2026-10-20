"""View specifications for campaign evaluation results (Etappe 2b, Schritt 8, PR 7).

Defines serializable ViewSpec models:
- TableViewSpec (columns, filter expression, sort)
- XYViewSpec (scatter and line plots with error bars, styling, faceting, ref lines, secondary axis)
- MatrixViewSpec (rows, cols, value, status expression, subtext template, sequential color scale)
- TimelineViewSpec (time course of channels with step bands and per-step marks, one page per
  value of `page_by`, e.g. one page per measuring day)

Texts with fields (`MatrixViewSpec.subtext`, `TimelineViewSpec.band_label`, `page_title`) are
templates: `{formula:format}` takes a column or formula of the result table, e.g.
``"{c_in:.0f} mg/Nm³"`` (`render_template`).

These models are referenced in Evaluation.views in campaign.json and rendered in datualizer_gui/results_window.py.
Import rule: No GUI dependencies here (principle from AGENTS.md).
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Sequence, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator

from datualizer_core.calc.expr import FUNCTIONS, compile_expr


def _name(value: str, what: str) -> str:
    if not value.isidentifier() or value in FUNCTIONS:
        raise ValueError(
            f"{what} {value!r} must be a name usable in formulas (letters, digits, '_', no function name)."
        )
    return value


_FIELD = re.compile(r"\{([^{}]+)\}")
_OPEN, _CLOSE = "\x00", "\x01"  # stand-ins for literal braces '{{' and '}}'


def template_fields(template: str) -> list[tuple[str, str]]:
    """(formula, format spec) of every `{formula:format}` field; `{{` and `}}` are literal braces."""
    out = []
    for m in _FIELD.finditer(template.replace("{{", _OPEN).replace("}}", _CLOSE)):
        expr, _, fmt = m.group(1).partition(":")
        out.append((expr.strip(), fmt))
    return out


def check_template(template: str | None) -> str | None:
    """Raise ValueError if a field is no valid formula or format (names are checked on use)."""
    if template:
        for expr, fmt in template_fields(template):
            if not expr:
                raise ValueError(f"Empty field in template {template!r}.")
            compile_expr(expr)
            try:
                format(1.0, fmt)
            except ValueError:
                raise ValueError(f"Format {fmt!r} in template {template!r} is not a number format.") from None
    return template


def _format_value(value: Any, fmt: str) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "–"
    if isinstance(value, (bool, str)):
        return str(value)
    try:
        return format(float(value), fmt or "g")
    except (TypeError, ValueError):
        return str(value)


def render_template(table: Any, template: str, reduce: bool = False) -> list[str] | str:
    """The template filled from a result table: one text per row, or one for all rows (`reduce`).

    `reduce`: numbers are averaged over the rows (NaN left out), differing texts joined with ", ".
    A field without a value gives "–". Unknown names raise ExprError (with suggestions).
    """
    import numpy as np

    fields = template_fields(template)
    n = len(table)
    columns: list[np.ndarray] = []
    for expr, _ in fields:
        vals = eval_table_expr(table, expr) if n else None
        columns.append(np.broadcast_to(np.asarray(vals), (n,)) if n else np.array([]))

    def value(j: int, i: int | None) -> Any:
        arr = columns[j]
        if i is not None:
            v = arr[i]
            return v.item() if hasattr(v, "item") else v
        if not len(arr):
            return None
        if arr.dtype == object:
            texts = sorted({str(v) for v in arr if v is not None})
            return ", ".join(texts) if texts else None
        f = arr.astype(float)
        return float(np.nanmean(f)) if np.isfinite(f).any() else None

    def fill(i: int | None) -> str:
        k = iter(range(len(fields)))
        text = template.replace("{{", _OPEN).replace("}}", _CLOSE)
        out = _FIELD.sub(lambda m: (lambda j: _format_value(value(j, i), fields[j][1]))(next(k)), text)
        return out.replace(_OPEN, "{").replace(_CLOSE, "}")

    if reduce:
        return fill(None)
    return [fill(i) for i in range(n)]


class RefLine(BaseModel):
    """Reference line on an XY plot (e.g. y = 0.5 for T50 or room air baseline)."""

    model_config = ConfigDict(extra="forbid")

    axis: Literal["x", "y"] = "y"
    value: float
    label: str = ""
    color: str = "#c3c2b7"
    linestyle: Literal["-", "--", ":", "-."] = "-"
    width: float = 1.0


class SecondaryAxis(BaseModel):
    """Secondary axis on an XY plot related to the primary axis by mathematical transformations."""

    model_config = ConfigDict(extra="forbid")

    label: str = ""
    forward_expr: str = Field(
        description="Formula converting primary x to secondary value, e.g. '1000 / x - 273.15'"
    )
    inverse_expr: str | None = Field(
        default=None,
        description="Formula converting secondary value back to primary x, e.g. '1000 / (x + 273.15)'",
    )

    @field_validator("forward_expr")
    @classmethod
    def _forward_valid(cls, v: str) -> str:
        compile_expr(v, names=["x"])
        return v

    @field_validator("inverse_expr")
    @classmethod
    def _inverse_valid(cls, v: str | None) -> str | None:
        if v is not None and v.strip():
            compile_expr(v, names=["x"])
        return v


class TableViewSpec(BaseModel):
    """Tabular view specification with optional column selection, filter condition and sorting."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["table"] = "table"
    name: str = Field(min_length=1)
    title: str = ""
    subtitle: str | None = None
    columns: list[str] = Field(default_factory=list, description="Columns to display; empty = all")
    filter: str | None = Field(default=None, description="Filter condition expression")
    sort_by: list[str] = Field(default_factory=list, description="Columns to sort by")
    descending: bool = False

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "View name")

    @field_validator("filter")
    @classmethod
    def _filter_valid(cls, v: str | None) -> str | None:
        if v is not None and v.strip():
            compile_expr(v)
        return v


class XYViewSpec(BaseModel):
    """XY scatter and line plot view specification with error bars, styling, faceting and secondary axis."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["xy"] = "xy"
    name: str = Field(min_length=1)
    title: str = ""
    subtitle: str | None = None
    x: str | None = Field(default=None, description="x column name")
    y: str | None = Field(default=None, description="y column name")
    x_expr: str | None = Field(default=None, description="x formula expression, e.g. '1000 / (T_bett + 273.15)'")
    y_expr: str | None = Field(default=None, description="y formula expression, e.g. 'log(k_1_s)'")
    u_x: str | None = Field(default=None, description="Standard uncertainty for x (column or expression)")
    u_y: str | None = Field(default=None, description="Standard uncertainty for y (column or expression)")
    color_by: str | None = Field(default=None, description="Column to color points by (ordinal or categorical)")
    color_mode: Literal["auto", "ordinal", "categorical"] = "auto"
    marker_by: str | None = Field(default=None, description="Column mapping distinct values to marker symbols")
    filled: str | None = Field(
        default=None, description="Boolean expression for filled (True) vs unfilled/open (False) markers"
    )
    connect_by: str | None = Field(
        default=None, description="Column to group points by, sorted by x and connected with lines"
    )
    facet_by: str | None = Field(
        default=None, description="Column to split plot into small multiples (facets) with shared axes"
    )
    facet_cols: int = Field(default=3, ge=1, le=10, description="Number of columns in faceting grid")
    ref_lines: list[RefLine] = Field(default_factory=list, description="Horizontal or vertical reference lines")
    secondary_x: SecondaryAxis | None = Field(
        default=None, description="Optional secondary (top) x-axis with mathematical coordinate transformation"
    )
    filter: str | None = Field(default=None, description="Filter condition expression")
    x_label: str = Field(default="", description="Axis label for x")
    y_label: str = Field(default="", description="Axis label for y")
    x_lim: tuple[float, float] | None = Field(default=None, description="Optional (min, max) limits for x")
    y_lim: tuple[float, float] | None = Field(default=None, description="Optional (min, max) limits for y")

    @property
    def effective_x_expr(self) -> str:
        """The formula or column used for the x coordinate."""
        return self.x_expr or self.x or ""

    @property
    def effective_y_expr(self) -> str:
        """The formula or column used for the y coordinate."""
        return self.y_expr or self.y or ""

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "View name")

    @field_validator("filter", "filled")
    @classmethod
    def _expr_valid(cls, v: str | None) -> str | None:
        if v is not None and v.strip():
            compile_expr(v)
        return v

    @field_validator("x_expr")
    @classmethod
    def _x_expr_valid(cls, v: str | None) -> str | None:
        if v is not None and v.strip():
            compile_expr(v)
        return v

    @field_validator("y_expr")
    @classmethod
    def _y_expr_valid(cls, v: str | None) -> str | None:
        if v is not None and v.strip():
            compile_expr(v)
        return v


class MatrixViewSpec(BaseModel):
    """Matrix / heatmap view specification."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["matrix"] = "matrix"
    name: str = Field(min_length=1)
    title: str = ""
    subtitle: str | None = None
    rows: str = Field(min_length=1, description="Row grouping column, e.g. 'T_klasse'")
    cols: str = Field(min_length=1, description="Column grouping column, e.g. 'v_mfc'")
    value: str = Field(min_length=1, description="Value column or expression to average in each cell, e.g. 'X'")
    status: str | None = Field(
        default=None,
        description="Status boolean expression (e.g. 'stationaer'); if false, cell is drawn grey + '*'",
    )
    subtext: str | None = Field(
        default=None,
        description="Template of the second line in each cell, averaged over its stationary steps, e.g. "
        "'{c_in:.0f} mg/Nm³'; ' · n Stufen' is added for more than one step. None = 'n Stufe(n)'.",
    )
    cmap: str = Field(default="seq", description="Colormap name ('seq' = sequential blue ramp)")
    vmin: float | None = Field(default=0.0, description="Minimum value for color scaling")
    vmax: float | None = Field(default=1.0, description="Maximum value for color scaling")
    filter: str | None = Field(default=None, description="Filter condition expression")
    x_label: str = Field(default="", description="Axis label for columns (bottom)")
    y_label: str = Field(default="", description="Axis label for rows (left)")
    footer: str | None = Field(default=None, description="Optional explanation footer below the matrix")
    rows_descending: bool = Field(default=True, description="Largest row value at the top (e.g. hottest first)")

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "View name")

    @field_validator("subtext")
    @classmethod
    def _subtext_valid(cls, v: str | None) -> str | None:
        return check_template(v)

    @field_validator("filter", "status")
    @classmethod
    def _expr_valid(cls, v: str | None) -> str | None:
        if v is not None and v.strip():
            compile_expr(v)
        return v


class AlignedViewSpec(BaseModel):
    """Aligned time-series view specification (e.g. breakthrough curve X(t) over relative time)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["aligned"] = "aligned"
    name: str = Field(min_length=1)
    title: str = ""
    subtitle: str | None = None
    expr: str = Field(
        min_length=1,
        description="Formula over time series channel and step constants, e.g. '1 - (fid_mgC - B) * f / c_in'",
    )
    t0: Literal["step_start", "sequence_start"] = Field(
        default="sequence_start",
        description="Time origin: start of step, or start of cumulative loading sequence",
    )
    sequence_col: str | None = Field(
        default=None,
        description="Column identifying cumulative sequence (default: auto-detected from 'sequenz', 'sequence', 'beladung')",
    )
    rolling_s: float | None = Field(
        default=None,
        description="Optional moving average window in seconds (e.g. 10.0 s)",
    )
    facet_by: str | None = Field(
        default=None,
        description="Column to facet small multiples by, e.g. 'T_klasse'",
    )
    facet_cols: int = Field(default=3, ge=1, le=10)
    color_by: str | None = Field(
        default=None,
        description="Column to color curves by, e.g. 'V_MFC' (dosing level)",
    )
    color_mode: Literal["auto", "ordinal", "categorical"] = "auto"
    marker_by: str | None = Field(
        default=None,
        description="Column mapping values to step endpoint marker symbols, e.g. 'tag'",
    )
    filled: str | None = Field(
        default=None,
        description="Boolean condition for filled vs open endpoint marker, e.g. 'stationaer' or 'stat_ok'",
    )
    filter: str | None = Field(
        default=None,
        description="Filter condition on steps to include",
    )
    x_label: str = Field(default="Minuten seit Beginn der Beladung")
    y_label: str = Field(default="Umsatz X(t)")
    x_lim: tuple[float, float] | None = None
    y_lim: tuple[float, float] | None = None
    ref_lines: list[RefLine] = Field(default_factory=list)
    footer: str | None = Field(
        default="Fällt X während der Dosierung, speichert der Katalysator zuerst Ethanol und bricht dann durch; "
        "erst das Plateau ist der stationäre Umsatz."
    )

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "View name")

    @field_validator("filter", "filled")
    @classmethod
    def _cond_valid(cls, v: str | None) -> str | None:
        if v is not None and v.strip():
            compile_expr(v)
        return v

    @field_validator("expr")
    @classmethod
    def _expr_valid(cls, v: str) -> str:
        if v is not None and v.strip():
            compile_expr(v)
        return v


class BarsViewSpec(BaseModel):
    """Bar plot view specification for uncertainty budgets (total uncertainty bar above stacked variance shares)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["bars"] = "bars"
    name: str = Field(min_length=1)
    title: str = ""
    subtitle: str | None = None
    quantity: str = Field(default="X", description="Target quantity whose uncertainty budget is displayed")
    u_total: str | None = Field(
        default=None,
        description="Column for total standard uncertainty, e.g. 'u_X' (default: 'u_' + quantity)",
    )
    parts: list[Any] | None = Field(
        default=None,
        description="Uncertainty budget component columns, or (col, label) pairs. If None, auto-detected.",
    )
    sort_by: list[str] = Field(
        default_factory=lambda: ["T_bett"],
        description="Columns to sort steps by along the horizontal axis",
    )
    filter: str | None = Field(
        default=None,
        description="Filter condition on steps to include, e.g. 'stationaer' or 'stat_ok'",
    )
    x_label: str = Field(default="Messschritte")
    y_label: str = Field(default="Anteil an u²(X)")
    footer: str | None = Field(
        default="Systematische Anteile (syst.) wirken auf alle Stufen eines Tages gleich."
    )

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "View name")

    @field_validator("filter")
    @classmethod
    def _filter_valid(cls, v: str | None) -> str | None:
        if v is not None and v.strip():
            compile_expr(v)
        return v


class TimelineTrace(BaseModel):
    """A channel drawn over time: from every source that has it, or only from `sources`."""

    model_config = ConfigDict(extra="forbid")

    channel: str = Field(min_length=1, description="Channel name, e.g. 'fid_mgc' (calculated channels too)")
    sources: list[str] = Field(default_factory=list, description="Sources to take it from; empty = all that have it")
    label: str = ""
    color: str | None = Field(default=None, description="Line colour, e.g. '#2a78d6'; None = next of the palette")


class TimelineMarks(BaseModel):
    """One mark per step from the result table: a point at the step's start or end (with error bar),
    or a horizontal segment over a named window (e.g. the blank value of each zero phase)."""

    model_config = ConfigDict(extra="forbid")

    y: str = Field(min_length=1, description="Column or formula of the result table, e.g. 'X' or 'S_b'")
    u_y: str | None = Field(default=None, description="Standard uncertainty (column or formula) as error bar")
    at: Literal["end", "start", "window"] = "end"
    window: str = Field(default="default", description="With at='window': the named window the segment spans")
    filter: str | None = Field(default=None, description="Steps that get a mark, e.g. klasse == 'dosierung'")
    filled: str | None = Field(default=None, description="Condition for filled (true) vs open or muted (false) marks")
    color_by_band: bool = Field(default=True, description="Colour of the step's band (else the ink colour)")
    label: str = ""
    label_open: str = Field(default="", description="Legend text of open or muted marks (with `filled`)")

    @field_validator("y", "u_y", "filter", "filled")
    @classmethod
    def _expr_valid(cls, v: str | None) -> str | None:
        if v is not None and v.strip():
            compile_expr(v)
        return v


class TimelinePanel(BaseModel):
    """One plot of a timeline: channels and marks sharing a y axis."""

    model_config = ConfigDict(extra="forbid")

    traces: list[TimelineTrace] = Field(default_factory=list)
    marks: list[TimelineMarks] = Field(default_factory=list)
    y_label: str = ""
    y_lim: tuple[float, float] | None = None
    y_robust: float | None = Field(
        default=None, gt=0.0, le=100.0,
        description="Upper y limit from this percentile of the traces (x 1.12), lower limit 0; e.g. 99.7 against spikes",
    )
    height: float = Field(default=1.0, gt=0.0, description="Relative height of the panel")
    ref_lines: list[RefLine] = Field(default_factory=list)


class TimelineViewSpec(BaseModel):
    """Time course on master time: channels in stacked panels, steps as coloured bands, marks per step.

    `page_by`: a column of the result table (e.g. 'tag'); each value is a page in the report and a
    choice in the view, its time range from the first start to the last end of its steps.
    `bands`: which steps are drawn as bands, coloured by `band_color_by` (ordinal ramp as in the
    other views) and labelled with the template `band_label` where the band is wide enough.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["timeline"] = "timeline"
    name: str = Field(min_length=1)
    title: str = ""
    subtitle: str | None = None
    panels: list[TimelinePanel] = Field(min_length=1)
    page_by: str | None = Field(default=None, description="Column with one page per value, e.g. 'tag'")
    filter: str | None = Field(default=None, description="Steps of the view (time range, bands, marks)")
    bands: str | None = Field(default=None, description="Condition for steps drawn as bands; None = no bands")
    band_color_by: str | None = Field(default=None, description="Column colouring the bands, e.g. 'v_mfc'")
    band_label: str | None = Field(default=None, description="Template on the band, e.g. '{v_mfc:g}'")
    band_legend: str = Field(default="", description="Text after each band value in the legend, e.g. 'Nl/min'")
    page_title: str | None = Field(default=None, description="Template of the page title, e.g. 'Messtag {tag}'")
    footer: str | None = None

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "View name")

    @field_validator("filter", "bands")
    @classmethod
    def _cond_valid(cls, v: str | None) -> str | None:
        if v is not None and v.strip():
            compile_expr(v)
        return v

    @field_validator("band_label", "page_title")
    @classmethod
    def _template_valid(cls, v: str | None) -> str | None:
        return check_template(v)


class ReportViewSpec(BaseModel):
    """Multi-page report view specification combining views and a title/summary page."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["report"] = "report"
    name: str = Field(min_length=1)
    title: str = ""
    subtitle: str | None = None
    views: list[str] = Field(
        default_factory=list,
        description="Ordered list of view names to include in the report; empty = all non-report views",
    )
    cover: bool = Field(
        default=True,
        description="Include cover / summary page with campaign metadata, dates, and KPI tiles from analyses",
    )
    orientation: Literal["landscape", "portrait"] = "landscape"
    page_size: Literal["A4", "Letter"] = "A4"

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "View name")


ViewSpec = Annotated[
    Union[
        TableViewSpec, XYViewSpec, MatrixViewSpec, AlignedViewSpec, BarsViewSpec, TimelineViewSpec, ReportViewSpec
    ],
    Field(discriminator="kind"),
]

def table_env(table: Any) -> tuple[dict[str, Any], list[str]]:
    """Build evaluation environment from Polars DataFrame columns for compile_expr."""
    import polars as pl
    import numpy as np

    env: dict[str, Any] = {}
    text_names: list[str] = []
    for c in table.columns:
        if table[c].dtype in (pl.String, pl.Categorical):
            text_names.append(c)
            env[c] = np.array([None if v is None else str(v) for v in table[c].to_list()], dtype=object)
        elif table[c].dtype == pl.Boolean:
            env[c] = table[c].to_numpy().astype(bool)
        else:
            try:
                vals = table[c].to_numpy().astype(float)
                env[c] = vals
            except (ValueError, TypeError):
                text_names.append(c)
                env[c] = np.array([None if v is None else str(v) for v in table[c].to_list()], dtype=object)
    return env, text_names


def filter_table(table: Any, where: str | None) -> Any:
    """Filter Polars DataFrame by expression string."""
    if where is None or not str(where).strip():
        return table
    import numpy as np

    env, text_names = table_env(table)
    expr = compile_expr(where, names=list(env.keys()), text_names=text_names)
    res = expr.evaluate(env)
    val = res.value if hasattr(res, "value") else res
    if hasattr(val, "dtype") and val.dtype == bool:
        mask = val
    else:
        v = np.asarray(val, dtype=float)
        mask = (v != 0) & ~np.isnan(v)
    return table.filter(mask)


def eval_table_expr(table: Any, expr_str: str | None) -> Any:
    """Evaluate expression or column name on Polars DataFrame."""
    if expr_str is None or not str(expr_str).strip():
        return None
    import polars as pl
    import numpy as np

    if expr_str in table.columns:
        s = table[expr_str]
        if s.dtype == pl.Boolean:
            return s.to_numpy().astype(bool)
        try:
            return s.to_numpy().astype(float)
        except (ValueError, TypeError):
            return np.array(s.to_list(), dtype=object)
    env, text_names = table_env(table)
    expr = compile_expr(expr_str, names=list(env.keys()), text_names=text_names)
    res = expr.evaluate(env)
    val = res.value if hasattr(res, "value") else res
    return val


__all__ = [
    "AlignedViewSpec",
    "BarsViewSpec",
    "MatrixViewSpec",
    "RefLine",
    "ReportViewSpec",
    "SecondaryAxis",
    "TableViewSpec",
    "TimelineMarks",
    "TimelinePanel",
    "TimelineTrace",
    "TimelineViewSpec",
    "ViewSpec",
    "XYViewSpec",
    "check_template",
    "eval_table_expr",
    "filter_table",
    "render_template",
    "table_env",
    "template_fields",
]
