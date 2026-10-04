"""Long-format detection and long -> wide pivoting (Datenkatalog P1, P19).

A long export stores one measurement per row: (time, variable, value, ...).
The file name is never trusted. Every role (variable/value column, per-channel attributes,
per-row metadata) can be set explicitly in `LongFormatConfig`; only unset roles are inferred
from the data shape and the configurable `Vocabulary`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from datualizer_core.dataset import AuditLog
from datualizer_core.ingestion.config import LongFormatConfig, LongFormatMode, Vocabulary
from datualizer_core.ingestion.type_inference import has_token
from datualizer_core.pipeline.operators import clean_column_name
from datualizer_core.schema import ColumnKind


class LongFormatError(ValueError):
    """Raised when a forced long-format configuration cannot be applied to the table."""


@dataclass(frozen=True)
class LongFormatSpec:
    """Resolved roles of a long-format table (all column names are loaded/cleaned names)."""

    variable_col: str
    value_col: str
    channel_attr_cols: tuple[str, ...]
    row_meta_cols: tuple[str, ...]
    dropped_cols: tuple[str, ...] = ()


@dataclass
class PivotResult:
    """Wide table produced from a long table, plus everything learned on the way."""

    df: pl.DataFrame
    column_kinds: dict[str, ColumnKind]
    channel_attrs: dict[str, dict[str, Any]]


def detect_long_format(
    df: pl.DataFrame,
    column_kinds: dict[str, ColumnKind],
    time_col: str,
    config: LongFormatConfig | None = None,
    vocabulary: Vocabulary | None = None,
) -> LongFormatSpec | None:
    """Resolve the long-format roles of a table, or return None if it is not (to be treated as) long."""
    config = config or LongFormatConfig()
    vocab = vocabulary or Vocabulary()
    if config.mode is LongFormatMode.OFF or time_col not in df.columns:
        return None
    forced = config.mode is LongFormatMode.FORCE

    variable_col = config.variable_col or _single(
        c for c, k in column_kinds.items()
        if k is ColumnKind.CATEGORICAL and has_token(c, vocab.long_variable_tokens)
    )
    value_col = config.value_col or _single(
        c for c, k in column_kinds.items()
        if k is ColumnKind.NUMERIC and has_token(c, vocab.long_value_tokens)
    )
    if variable_col is None or value_col is None:
        if forced:
            raise LongFormatError(
                "Long format forced, but variable/value column is ambiguous or missing; "
                "set long_format.variable_col and long_format.value_col."
            )
        return None
    for col in (variable_col, value_col):
        if col not in df.columns:
            raise LongFormatError(f"Configured long-format column '{col}' not found in {df.columns}.")

    # Auto mode: several variables must share one timestamp, otherwise it is a wide table with a label column.
    if not forced and (
        df[variable_col].n_unique() < 2 or df[time_col].n_unique() == df.height
    ):
        return None

    other_cols = [c for c in df.columns if c not in (time_col, variable_col, value_col)]
    attr_cols = config.channel_attr_cols
    meta_cols = config.row_meta_cols
    if attr_cols is None and meta_cols is None:
        meta_cols = [
            c for c in other_cols
            if not has_token(c, vocab.channel_attr_tokens) and _is_constant_within(df, time_col, c)
        ]
    if meta_cols is None:
        meta_cols = [c for c in other_cols if c not in attr_cols]
    dropped: list[str] = []
    if attr_cols is None:
        rest = [c for c in other_cols if c not in meta_cols]
        # Auto: a column varying per timestamp *and* per channel fits neither role.
        attr_cols = [c for c in rest if _is_constant_within(df, variable_col, c)]
        dropped = [c for c in rest if c not in attr_cols]
    else:
        dropped = [c for c in other_cols if c not in attr_cols and c not in meta_cols]

    return LongFormatSpec(
        variable_col=variable_col,
        value_col=value_col,
        channel_attr_cols=tuple(attr_cols),
        row_meta_cols=tuple(meta_cols),
        dropped_cols=tuple(dropped),
    )


def pivot_long_to_wide(
    df: pl.DataFrame,
    spec: LongFormatSpec,
    column_kinds: dict[str, ColumnKind],
    time_col: str,
    audit_log: AuditLog,
) -> PivotResult:
    """Pivot a long table to wide: one row per timestamp, one numeric column per variable.

    - `row_meta_cols` stay as row columns (run_id, phase_status, ...).
    - `channel_attr_cols` become per-channel attributes (unit, is_calculated, ...);
      an attribute that is not unique within a channel is recorded in the audit log.
    - Duplicate (time, variable) keys keep the first value and are recorded in the audit log.
    """
    var, val = spec.variable_col, spec.value_col
    df = df.filter(pl.col(var).is_not_null()).with_columns(
        pl.col(var).map_elements(clean_column_name, return_dtype=pl.String)
    )
    channels = df[var].unique(maintain_order=True).to_list()

    channel_attrs: dict[str, dict[str, Any]] = {ch: {} for ch in channels}
    for col in spec.dropped_cols:
        audit_log.add(-1, col, None, "dropped_long_column", raw_column=col)
    for col in spec.channel_attr_cols:
        per_channel = df.group_by(var, maintain_order=True).agg(
            pl.col(col).drop_nulls().first().alias("first"),
            pl.col(col).drop_nulls().n_unique().alias("n"),
        )
        for ch, first, n in per_channel.iter_rows():
            channel_attrs[ch][col] = first
            if n > 1:
                audit_log.add(-1, ch, col, "ambiguous_channel_attribute", raw_column=col)

    df = df.with_row_index("_long_row")
    key = pl.struct(time_col, var)
    dupes = df.filter(key.is_duplicated() & ~key.is_first_distinct())
    for row_idx, ch, raw in dupes.select("_long_row", var, val).iter_rows():
        audit_log.add(row_idx, ch, raw, "duplicate_long_key", raw_column=var)

    index = [time_col, *spec.row_meta_cols]
    wide = (
        df.pivot(on=var, index=index, values=val, aggregate_function="first", maintain_order=True)
        .select(*index, *channels)
    )

    kinds = {c: column_kinds[c] for c in index}
    kinds.update({ch: ColumnKind.NUMERIC for ch in channels})
    return PivotResult(df=wide, column_kinds=kinds, channel_attrs=channel_attrs)


def _single(candidates) -> str | None:
    """Return the only candidate, or None if there are zero or several."""
    found = list(candidates)
    return found[0] if len(found) == 1 else None


def _is_constant_within(df: pl.DataFrame, group_col: str, col: str) -> bool:
    """True if `col` has at most one distinct non-null value inside every `group_col` group."""
    counts = df.group_by(group_col).agg(pl.col(col).drop_nulls().n_unique().alias("n"))
    return counts["n"].max() <= 1
