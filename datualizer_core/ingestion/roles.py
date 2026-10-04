"""Measurement vs. parameter roles (Datenkatalog P7).

A setpoint such as `target_temperature`, `Rotameter` or `param_Temperatur` is stored like a
measurement, but it is constant inside a run. Plotting it as a channel is noise. The heuristic
"numeric and constant inside every run -> PARAMETER" only fills columns the user did not set
explicitly. Runs are defined by `RoleConfig.run_columns` or, if unset, by identifier/categorical
columns whose name contains one of `Vocabulary.run_tokens`.

Limit: a sensor that happens to read a constant value (e.g. `Temperatur = 25.0` in Run 1) cannot
be told apart from a setpoint by its values. It is proposed as PARAMETER and must be corrected
via `IngestionConfig.column_kinds`.
"""

from __future__ import annotations

from typing import Collection

import polars as pl

from datualizer_core.dataset import AuditLog
from datualizer_core.ingestion.config import RoleConfig, Vocabulary
from datualizer_core.ingestion.type_inference import has_token
from datualizer_core.schema import ColumnKind


class RoleConfigError(ValueError):
    """Raised when an explicit role configuration cannot be applied to the table."""


def resolve_run_columns(
    df: pl.DataFrame,
    column_kinds: dict[str, ColumnKind],
    config: RoleConfig,
    vocabulary: Vocabulary,
    raw_to_clean: dict[str, str] | None = None,
) -> list[str]:
    """Return the loaded column names that identify a run (empty if there is no run structure)."""
    if config.run_columns is not None:
        names = [(raw_to_clean or {}).get(c, c) for c in config.run_columns]
        missing = [c for c in names if c not in df.columns]
        if missing:
            raise RoleConfigError(f"Configured run column(s) {missing} not found in {df.columns}.")
        return names
    return [
        c for c, k in column_kinds.items()
        if k in (ColumnKind.IDENTIFIER, ColumnKind.CATEGORICAL) and has_token(c, vocabulary.run_tokens)
    ]


def assign_parameter_roles(
    df: pl.DataFrame,
    column_kinds: dict[str, ColumnKind],
    run_columns: list[str],
    explicit: Collection[str],
    config: RoleConfig,
    audit_log: AuditLog,
) -> dict[str, ColumnKind]:
    """Return column kinds with unset numeric columns that are constant per run marked as PARAMETER.

    Explicit PARAMETER columns that vary inside a run are kept as parameters, but recorded in
    the audit log (`non_constant_parameter`).
    """
    kinds = dict(column_kinds)
    candidates = (
        [c for c, k in kinds.items() if k is ColumnKind.NUMERIC and c not in explicit]
        if config.detect_parameters and run_columns
        else []
    )
    forced = [c for c, k in kinds.items() if k is ColumnKind.PARAMETER and c in explicit]
    if not candidates and not forced:
        return kinds

    keys = run_columns or [pl.lit(0).alias("_all")]
    cols = list(dict.fromkeys(candidates + forced))
    stats = df.group_by(keys).agg(
        *[pl.col(c).drop_nulls().n_unique().alias(f"{c}\0n") for c in cols],
        *[pl.col(c).count().alias(f"{c}\0count") for c in cols],
    )
    max_unique = {c: stats[f"{c}\0n"].max() for c in cols}
    max_count = {c: stats[f"{c}\0count"].max() for c in cols}

    for c in candidates:
        if max_unique[c] <= 1 and max_count[c] >= config.min_samples_per_run:
            kinds[c] = ColumnKind.PARAMETER
    for c in forced:
        if max_unique[c] > 1:
            audit_log.add(-1, c, None, "non_constant_parameter", raw_column=c)
    return kinds
