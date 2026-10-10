"""Units of numeric columns (Datenkatalog P20, rest of P9).

Headers such as `Massenstrom Waage (g/s)` carry the unit in the name; `clean_names` drops it from
the column name, so it is kept here as metadata instead of being lost. Long tables carry it as a
channel attribute. Explicit user units win over both.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

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
