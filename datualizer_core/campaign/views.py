"""View specifications for campaign evaluation results (Etappe 2b, Schritt 8, PR 7).

Defines serializable ViewSpec models:
- TableViewSpec (columns, filter expression, sort)
- XYViewSpec (scatter and line plots with error bars, styling, faceting, ref lines, secondary axis)
- MatrixViewSpec (rows, cols, value, status expression, subtext, sequential color scale)

These models are referenced in Evaluation.views in campaign.json and rendered in datualizer_gui/results_window.py.
Import rule: No GUI dependencies here (principle from AGENTS.md).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Sequence, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator

from datualizer_core.calc.expr import FUNCTIONS, compile_expr


def _name(value: str, what: str) -> str:
    if not value.isidentifier() or value in FUNCTIONS:
        raise ValueError(
            f"{what} {value!r} must be a name usable in formulas (letters, digits, '_', no function name)."
        )
    return value


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
    cols: str = Field(min_length=1, description="Column grouping column, e.g. 'V_MFC'")
    value: str = Field(min_length=1, description="Value column or expression to average in each cell, e.g. 'X'")
    status: str | None = Field(
        default=None,
        description="Status boolean expression (e.g. 'stationaer'); if false, cell is drawn grey + '*'",
    )
    subtext: str | None = Field(
        default=None,
        description="Subtext expression or template in each cell, e.g. '{c_in_mg_m3:.0f} mg/Nm³'",
    )
    cmap: str = Field(default="seq", description="Colormap name ('seq' = sequential blue ramp)")
    vmin: float | None = Field(default=0.0, description="Minimum value for color scaling")
    vmax: float | None = Field(default=1.0, description="Maximum value for color scaling")
    filter: str | None = Field(default=None, description="Filter condition expression")
    x_label: str = Field(default="", description="Axis label for columns (bottom)")
    y_label: str = Field(default="", description="Axis label for rows (left)")
    footer: str | None = Field(default=None, description="Optional explanation footer below the matrix")

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        return _name(v, "View name")

    @field_validator("filter", "status")
    @classmethod
    def _expr_valid(cls, v: str | None) -> str | None:
        if v is not None and v.strip():
            compile_expr(v)
        return v


ViewSpec = Annotated[
    Union[TableViewSpec, XYViewSpec, MatrixViewSpec],
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
    "MatrixViewSpec",
    "RefLine",
    "SecondaryAxis",
    "TableViewSpec",
    "ViewSpec",
    "XYViewSpec",
    "eval_table_expr",
    "filter_table",
    "table_env",
]
