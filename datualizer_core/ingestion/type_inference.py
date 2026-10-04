"""Column type inference: decides per column whether it is numeric, categorical or an identifier.

All name hints come from a configurable `Vocabulary`; explicit per-column kinds in
`IngestionConfig.column_kinds` are applied by the loader before this heuristic runs.
"""

from __future__ import annotations

import re
from typing import Iterable

from datualizer_core.ingestion.config import Vocabulary
from datualizer_core.schema import ColumnKind

_INT_RE = re.compile(r"^[+-]?\d+$")
_DEFAULT_VOCABULARY = Vocabulary()


def name_tokens(raw_name: str) -> set[str]:
    """Split a raw header into lowercase alphanumeric tokens."""
    return {t for t in re.split(r"[^0-9a-zäöüß]+", raw_name.lower()) if t}


def has_token(raw_name: str, tokens: Iterable[str]) -> bool:
    """True if the header contains one of the given tokens (case-insensitive)."""
    return bool(name_tokens(raw_name) & {t.lower() for t in tokens})


def is_metadata_name(raw_name: str, vocabulary: Vocabulary | None = None) -> bool:
    """Return True if the header name marks the column as metadata/identifier."""
    return has_token(raw_name, (vocabulary or _DEFAULT_VOCABULARY).metadata_tokens)


def is_integer_text(v_str: str) -> bool:
    """True if a stripped cell holds a plain integer."""
    return bool(_INT_RE.match(v_str))


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
    vocabulary: Vocabulary | None = None,
    numeric_ratio_threshold: float = 0.5,
) -> ColumnKind:
    """Infer the ColumnKind of a non-time column from its header name and raw string values."""
    vocab = vocabulary or _DEFAULT_VOCABULARY
    candidates = []
    for raw in values:
        if raw is None:
            continue
        v_str = str(raw).strip()
        if v_str and v_str.upper() not in upper_sentinels:
            candidates.append(v_str)

    if has_token(raw_name, vocab.metadata_tokens):
        # Only identifier-token columns become integer keys; names like "250" stay text.
        if (
            has_token(raw_name, vocab.identifier_tokens)
            and candidates
            and all(is_integer_text(v) for v in candidates)
        ):
            return ColumnKind.IDENTIFIER
        return ColumnKind.CATEGORICAL

    # An entirely empty column stays a (empty) numeric channel, as before.
    if not candidates:
        return ColumnKind.NUMERIC

    n_numeric = sum(parse_number(v, decimal_sep) is not None for v in candidates)
    if n_numeric / len(candidates) >= numeric_ratio_threshold:
        return ColumnKind.NUMERIC
    return ColumnKind.CATEGORICAL
