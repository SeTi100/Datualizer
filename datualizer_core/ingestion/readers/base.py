"""Common result type of all readers and the conversion of typed cells to raw text."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Sequence

import polars as pl


class ReaderError(ValueError):
    """A source cannot be read with the given options (e.g. sheet or table missing).

    If the user has to pick a part of the source, `choice` names the profile field to set
    (`"excel.sheet"` or `"sqlite.table"`) and `options` lists the valid values, so a GUI can
    offer them instead of only showing the message.
    """

    def __init__(self, message: str, *, choice: str | None = None, options: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.choice = choice
        self.options = list(options)


@dataclass(frozen=True)
class RawTable:
    """All-String table (one column per header) plus what the reader learned about the source."""

    df: pl.DataFrame
    decimal_separator: str = "."
    meta: dict[str, Any] = field(default_factory=dict)


def cell_to_text(value: Any) -> str | None:
    """Typed cell (Excel, SQLite) to the text the loader parses; None stays None.

    Numbers use a decimal point, date-times ISO 8601, times `HH:MM:SS[.ffffff]`.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, timedelta):
        return repr(value.total_seconds())
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def unique_headers(names: Sequence[str | None], fallback: Sequence[str]) -> list[str]:
    """Header names for a raw table: empty ones get `fallback[i]`, duplicates a `_2`, `_3` suffix."""
    out: list[str] = []
    seen: dict[str, int] = {}
    for name, fb in zip(names, fallback):
        base = (str(name).strip() if name is not None else "") or fb
        n = seen.get(base, 0) + 1
        seen[base] = n
        out.append(base if n == 1 else f"{base}_{n}")
    return out


def string_frame(headers: list[str], rows: Sequence[Sequence[Any]]) -> pl.DataFrame:
    """Build an all-String DataFrame from typed rows (column-wise, rows may be shorter)."""
    columns = {
        h: [cell_to_text(r[i]) if i < len(r) else None for r in rows] for i, h in enumerate(headers)
    }
    return pl.DataFrame(columns, schema={h: pl.String for h in headers})
