"""Units of numeric columns (Datenkatalog P20, rest of P9).

Headers such as `Massenstrom Waage (g/s)` carry the unit in the name; `clean_names` drops it from
the column name, so it is kept here as metadata instead of being lost. Long tables carry it as a
channel attribute. Explicit user units win over both.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import polars as pl

from datualizer_core.ingestion.config import UnitConfig, Vocabulary


class UnitConfigError(ValueError):
    """A unit setting names a column that is not numeric, or a header pattern is unusable."""


def unit_from_header(header: str, patterns: Sequence[str]) -> str | None:
    """Return the unit written in a header (first matching pattern), or None."""
    return split_header_unit(header, patterns)[1]


def split_header_unit(header: str, patterns: Sequence[str]) -> tuple[str, str | None]:
    """Split a header into (name without the unit part, unit); unit is None if no pattern matches."""
    for pattern in patterns:
        try:
            match = re.search(pattern, header)
        except re.error as exc:
            raise UnitConfigError(f"Invalid unit pattern {pattern!r}: {exc}") from exc
        if match is None:
            continue
        if "unit" not in match.groupdict():
            raise UnitConfigError(f"Unit pattern {pattern!r} has no named group 'unit'.")
        unit = (match.group("unit") or "").strip()
        if unit:
            return (header[: match.start()] + header[match.end():]).strip(), unit
    return header, None


def resolve_units(
    cfg: UnitConfig,
    numeric_columns: Sequence[str],
    raw_to_clean: Mapping[str, str],
    names: Mapping[str, str],
    channel_attrs: Mapping[str, Mapping[str, Any]],
    vocabulary: Vocabulary,
) -> dict[str, str]:
    """Return the unit of every numeric column that has one, keyed by loaded name.

    Explicit entries (also '' for "no unit") are kept as given, so the result replays exactly.
    """
    numeric = set(numeric_columns)
    units: dict[str, str] = {}
    if cfg.from_headers:
        for raw, clean in raw_to_clean.items():
            if clean in numeric:
                unit = unit_from_header(raw, cfg.header_patterns)
                if unit is not None:
                    units[clean] = unit
    tokens = [t.lower() for t in vocabulary.unit_tokens]
    for ch, attrs in channel_attrs.items():
        if ch not in numeric:
            continue
        for key, value in attrs.items():
            if any(t in key.lower() for t in tokens) and isinstance(value, str) and value.strip():
                units[ch] = value.strip()
                break
    for name, unit in cfg.units.items():
        loaded = names.get(name, name)
        if loaded not in numeric:
            raise UnitConfigError(
                f"units.units names '{name}', which is not a numeric column {sorted(numeric)}."
            )
        units[loaded] = unit.strip()
    return {c: units[c] for c in numeric_columns if c in units}


def compile_cell_pattern(cfg: UnitConfig) -> re.Pattern[str] | None:
    """Compile `cell_pattern` (None if cell units are off); fail loudly if it is unusable."""
    if not cfg.split_cell_units:
        return None
    try:
        pattern = re.compile(cfg.cell_pattern)
    except re.error as exc:
        raise UnitConfigError(f"Invalid cell unit pattern {cfg.cell_pattern!r}: {exc}") from exc
    if not {"value", "unit"} <= set(pattern.groupindex):
        raise UnitConfigError(f"Cell unit pattern {cfg.cell_pattern!r} needs named groups 'value' and 'unit'.")
    return pattern


CONFLICTS_SCHEMA = {"column": pl.String, "cell_unit": pl.String, "column_unit": pl.String, "n_rows": pl.Int64}


@dataclass
class CellUnitResult:
    units: dict[str, str]
    conflicts: pl.DataFrame
    audit: list[tuple[int, str, str, str, str]]  # row, column, raw value, reason, raw column


def reconcile_cell_units(
    cell_units: Mapping[str, Mapping[int, tuple[str, str]]],
    units: Mapping[str, str],
    numeric_columns: Sequence[str],
    explicit: set[str],
    non_null: Mapping[str, int],
    raw_names: Mapping[str, str],
) -> CellUnitResult:
    """Compare units found in cells (P6) with the column unit.

    - A column without unit whose every value carries the same cell unit takes that unit.
    - A cell unit that differs from the column unit, or appears in a column where other values
      have no unit, is a `unit_conflict`; the value is kept and marked, never dropped.
    - Otherwise the split is recorded as `value_with_unit`.
    """
    resolved = dict(units)
    rows: list[dict] = []
    audit: list[tuple[int, str, str, str, str]] = []
    for col, cells in cell_units.items():
        found = {u for _, u in cells.values()}
        column_unit = resolved.get(col) or None
        if (
            col in numeric_columns
            and column_unit is None
            and col not in explicit
            and len(found) == 1
            and len(cells) == non_null.get(col, -1)
        ):
            column_unit = resolved[col] = next(iter(found))
        consistent = col not in numeric_columns or (column_unit is not None and found == {column_unit})
        counts: dict[str, int] = {}
        for row, (raw, unit) in sorted(cells.items()):
            conflict = not consistent and unit != column_unit
            reason = "unit_conflict" if conflict else "value_with_unit"
            audit.append((row, col, raw, reason, raw_names.get(col, col)))
            if conflict:
                counts[unit] = counts.get(unit, 0) + 1
        rows += [
            {"column": col, "cell_unit": u, "column_unit": column_unit, "n_rows": n} for u, n in counts.items()
        ]
    conflicts = pl.DataFrame(rows, schema=CONFLICTS_SCHEMA) if rows else pl.DataFrame(schema=CONFLICTS_SCHEMA)
    ordered = {c: resolved[c] for c in numeric_columns if c in resolved}
    return CellUnitResult(ordered, conflicts, audit)
