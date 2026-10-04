"""Long-format detection and long -> wide pivoting (Datenkatalog P1, P19).

A long export stores one measurement per row: (time, variable, value, ...).
The file name is never trusted; detection relies on the header and the data shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from datualizer_core.dataset import AuditLog
from datualizer_core.ingestion.type_inference import name_tokens
from datualizer_core.pipeline.operators import clean_column_name
from datualizer_core.schema import ColumnKind

_VARIABLE_TOKENS = frozenset({
    "roi", "variable", "channel", "kanal", "signal", "sensor",
    "messgroesse", "quantity", "metric", "measurement",
})
_VALUE_TOKENS = frozenset({"value", "wert", "messwert", "reading"})
# Columns describing a variable rather than a timestamp, even when they happen to be constant everywhere.
_CHANNEL_ATTR_TOKENS = frozenset({"unit", "einheit", "calculated", "computed", "derived", "berechnet"})


@dataclass(frozen=True)
class LongFormatSpec:
    """Columns that make up a long-format table."""

    variable_col: str
    value_col: str


@dataclass
class PivotResult:
    """Wide table produced from a long table, plus everything learned on the way."""

    df: pl.DataFrame
    column_kinds: dict[str, ColumnKind]
    channel_attrs: dict[str, dict[str, Any]]


def detect_long_format(
    df: pl.DataFrame, column_kinds: dict[str, ColumnKind], time_col: str
) -> LongFormatSpec | None:
    """Return the long-format spec if the table is a (time, variable, value) table, else None."""
    variable_cols = [
        c for c, k in column_kinds.items()
        if k is ColumnKind.CATEGORICAL and name_tokens(c) & _VARIABLE_TOKENS
    ]
    value_cols = [
        c for c, k in column_kinds.items()
        if k is ColumnKind.NUMERIC and name_tokens(c) & _VALUE_TOKENS
    ]
    if len(variable_cols) != 1 or len(value_cols) != 1 or time_col not in df.columns:
        return None

    # Several variables share one timestamp, otherwise it is just a wide table with a label column.
    if df[variable_cols[0]].n_unique() < 2 or df[time_col].n_unique() == df.height:
        return None
    return LongFormatSpec(variable_col=variable_cols[0], value_col=value_cols[0])


def pivot_long_to_wide(
    df: pl.DataFrame,
    spec: LongFormatSpec,
    column_kinds: dict[str, ColumnKind],
    time_col: str,
    audit_log: AuditLog,
) -> PivotResult:
    """Pivot a long table to wide: one row per timestamp, one numeric column per variable.

    - Columns constant within each timestamp (run_id, phase_status, ...) become row metadata.
    - Columns constant per variable (unit, is_calculated, ...) become channel attributes.
    - Duplicate (time, variable) keys keep the first value and are recorded in the audit log.
    """
    var, val = spec.variable_col, spec.value_col
    df = df.filter(pl.col(var).is_not_null()).with_columns(
        pl.col(var).map_elements(clean_column_name, return_dtype=pl.String)
    )
    channels = df[var].unique(maintain_order=True).to_list()

    other_cols = [c for c in df.columns if c not in (time_col, var, val)]
    row_meta = [
        c for c in other_cols
        if not name_tokens(c) & _CHANNEL_ATTR_TOKENS and _is_constant_within(df, time_col, c)
    ]
    channel_cols = [c for c in other_cols if c not in row_meta]

    channel_attrs: dict[str, dict[str, Any]] = {ch: {} for ch in channels}
    for col in channel_cols:
        if not _is_constant_within(df, var, col):
            continue
        for ch, attr in df.group_by(var, maintain_order=True).agg(pl.col(col).drop_nulls().first()).iter_rows():
            channel_attrs[ch][col] = attr

    df = df.with_row_index("_long_row")
    dupes = df.filter(pl.struct(time_col, var).is_duplicated() & ~pl.struct(time_col, var).is_first_distinct())
    for row_idx, ch, raw in dupes.select("_long_row", var, val).iter_rows():
        audit_log.add(row_idx, ch, raw, "duplicate_long_key", raw_column=var)

    index = [time_col, *row_meta]
    wide = (
        df.pivot(on=var, index=index, values=val, aggregate_function="first", maintain_order=True)
        .select(*index, *channels)
    )

    kinds = {c: column_kinds[c] for c in index}
    kinds.update({ch: ColumnKind.NUMERIC for ch in channels})
    return PivotResult(df=wide, column_kinds=kinds, channel_attrs=channel_attrs)


def _is_constant_within(df: pl.DataFrame, group_col: str, col: str) -> bool:
    """True if `col` has at most one distinct non-null value inside every `group_col` group."""
    counts = df.group_by(group_col).agg(pl.col(col).drop_nulls().n_unique().alias("n"))
    return counts["n"].max() <= 1
