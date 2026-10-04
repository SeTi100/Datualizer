"""Column type inference: decides per column whether it is numeric, categorical or an identifier."""

from __future__ import annotations

import re
from typing import Iterable

from datualizer_core.schema import ColumnKind

# Name tokens that mark a column as metadata/identifier, regardless of how its values look.
# Prevents e.g. experiment_name = "250" from becoming the float 250.0.
_METADATA_TOKENS = frozenset({
    "id", "name", "status", "phase", "unit", "einheit", "type", "typ",
    "label", "comment", "kommentar", "bemerkung", "note", "notiz",
})

_INT_RE = re.compile(r"^[+-]?\d+$")

# Minimum share of parseable values (among non-empty, non-sentinel cells) for a column to count as numeric.
NUMERIC_RATIO_THRESHOLD = 0.5


def name_tokens(raw_name: str) -> set[str]:
    """Split a raw header into lowercase alphanumeric tokens."""
    return {t for t in re.split(r"[^0-9a-zäöüß]+", raw_name.lower()) if t}


def is_metadata_name(raw_name: str) -> bool:
    """Return True if the header name marks the column as metadata/identifier."""
    return bool(name_tokens(raw_name) & _METADATA_TOKENS)


def parse_number(v_str: str, decimal_sep: str) -> float | None:
    """Parse a stripped cell value as float, honouring decimal and thousand separators."""
    norm = v_str
    if decimal_sep == ",":
        if re.match(r"^-?\d{1,3}(?:\.\d{3})+(?:,\d+)?$", norm):
            norm = norm.replace(".", "").replace(",", ".")
        else:
            norm = norm.replace(",", ".")
    elif re.match(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$", norm):
        norm = norm.replace(",", "")
    try:
        return float(norm)
    except ValueError:
        return None


def infer_column_kind(
    raw_name: str,
    values: Iterable[object],
    decimal_sep: str,
    upper_sentinels: set[str],
) -> ColumnKind:
    """Infer the ColumnKind of a non-time column from its header name and raw string values."""
    candidates = []
    for raw in values:
        if raw is None:
            continue
        v_str = str(raw).strip()
        if v_str and v_str.upper() not in upper_sentinels:
            candidates.append(v_str)

    tokens = name_tokens(raw_name)
    if tokens & _METADATA_TOKENS:
        # Only explicit *_id columns become integer keys; names like "250" stay text.
        if "id" in tokens and candidates and all(_INT_RE.match(v) for v in candidates):
            return ColumnKind.IDENTIFIER
        return ColumnKind.CATEGORICAL

    # An entirely empty column stays a (empty) numeric channel, as before.
    if not candidates:
        return ColumnKind.NUMERIC

    n_numeric = sum(parse_number(v, decimal_sep) is not None for v in candidates)
    if n_numeric / len(candidates) >= NUMERIC_RATIO_THRESHOLD:
        return ColumnKind.NUMERIC
    return ColumnKind.CATEGORICAL
