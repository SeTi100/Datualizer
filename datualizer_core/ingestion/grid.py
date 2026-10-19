"""The raw cells of a table as a grid: find headers by text and cells by the label next to them.

A lab journal changes while it is written: columns are inserted, a block of another day is
pasted below, the headers gain units. Anything read from such a table is therefore found by
text, never by position. This module holds the plain helpers (no campaign imports):

- `Grid`: the cells of a sheet with all its rows (also those above the header), or of a loaded
  table (header in row 1). Row numbers are 1-based as in Excel, columns have letters.
- `match_header`: a configured column name against the header texts (case, spaces and a unit
  suffix do not matter); an ambiguous name raises `GridError`.
- `find_header_row`: the first row that has all the names; else `GridError` with a suggestion.
- `locate`: a cell given by a label next to it (`CellRef`) or by address.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
import difflib
import re
from typing import Any, Iterable, Sequence

from datualizer_core.ingestion.readers.base import unique_headers
from datualizer_core.ingestion.steps import CellRef
from datualizer_core.pipeline.operators import clean_column_name

HEADER_SEARCH_ROWS = 200
LOCATE_MAX_STEPS = 50  # how far `locate` looks for the n-th non-empty cell

_UNIT = re.compile(r"\s*(\[[^\]]*\]|\([^)]*\)|\{[^}]*\})\s*$")
_CELL = re.compile(r"^\s*([A-Za-z]{1,3})\s*(\d+)\s*$")
_EXCEL_TIME_ONLY = (date(1899, 12, 29), date(1899, 12, 30), date(1899, 12, 31), date(1900, 1, 1))


class GridError(ValueError):
    """A header or labelled cell is missing or ambiguous; the message suggests what was meant."""


def column_letter(idx: int) -> str:
    """'A' for column 0, 'AA' for column 26."""
    letters = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def a1(row: int, col: int) -> str:
    """A1 address of a 1-based row and 0-based column."""
    return f"{column_letter(col)}{row}"


def parse_a1(ref: str) -> tuple[int, int]:
    """1-based row and 0-based column of an A1 address; ValueError otherwise."""
    m = _CELL.match(ref)
    if m is None or int(m.group(2)) < 1:
        raise ValueError(f"Not a cell address: {ref!r} (expected e.g. 'C3').")
    col = 0
    for ch in m.group(1).upper():
        col = col * 26 + ord(ch) - 64
    return int(m.group(2)), col - 1


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return isinstance(value, str) and value.strip() == ""


def excel_time_only(value: datetime) -> bool:
    """A date-time Excel produced from a time without date (day 0 of its calendar)."""
    return value.date() in _EXCEL_TIME_ONLY


def display_text(value: Any) -> str:
    """Short text of a cell as a person reads it (grid view, messages, header names)."""
    if is_empty(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else f"{value:.6g}"
    if isinstance(value, datetime):
        if excel_time_only(value):
            return value.strftime("%H:%M:%S")
        if value.time() == time(0) and value.tzinfo is None:
            return value.strftime("%d.%m.%Y")
        return value.strftime("%d.%m.%Y %H:%M:%S")
    if isinstance(value, date):
        return value.strftime("%d.%m.%Y")
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    return str(value).strip()


def header_key(text: Any) -> str:
    """Comparison key of a header: case and all spaces do not matter."""
    return "".join(str(text).split()).casefold()


def _base_key(text: Any) -> str:
    return header_key(_UNIT.sub("", str(text)))


@dataclass(frozen=True)
class Grid:
    """Raw cells of a table. `rows[0]` is row 1; values as stored (number, text, date-time, time)."""

    rows: list[list[Any]]
    kind: str = "sheet"  # "sheet" (a workbook sheet with all rows) or "table" (header in row 1)
    sheet: str | None = None
    sheets: tuple[str, ...] = ()
    reader_header_row: int | None = None  # the row the reader takes as header (preferred on a tie)
    row_days: dict[int, date] = field(default_factory=dict)  # date of a row's clock times (tables)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        w = 0
        for r in self.rows:
            for i in range(len(r) - 1, -1, -1):
                if not is_empty(r[i]):
                    w = max(w, i + 1)
                    break
        return w

    def value(self, row: int, col: int) -> Any:
        """Cell of a 1-based row and 0-based column (None outside the table)."""
        if 1 <= row <= len(self.rows) and 0 <= col < len(self.rows[row - 1]):
            return self.rows[row - 1][col]
        return None

    def row_empty(self, row: int) -> bool:
        return all(is_empty(v) for v in self.rows[row - 1]) if 1 <= row <= len(self.rows) else True

    def last_used_row(self) -> int:
        for r in range(len(self.rows), 0, -1):
            if not self.row_empty(r):
                return r
        return 0

    def headers(self, row: int) -> list[str]:
        """Header names of a row: the cell texts, empty ones `col_<letter>` (as the Excel reader names them)."""
        width = self.width
        texts = [None if is_empty(v) else display_text(v) for v in (self.value(row, c) for c in range(width))]
        return unique_headers(texts, [f"col_{column_letter(i)}" for i in range(width)])


def _levels(name: str):
    key, base = header_key(name), _base_key(name)
    clean, clean_base = clean_column_name(str(name)), clean_column_name(_UNIT.sub("", str(name)))
    return [
        lambda h: header_key(h) == key,
        lambda h: base != "" and _base_key(h) == base,
        lambda h: clean_column_name(h) == clean,
        lambda h: clean_column_name(_UNIT.sub("", h)) == clean_base,
    ]


def match_header(name: str, headers: Sequence[str]) -> int | None:
    """Index of the header `name` stands for, or None. Raises GridError if it is ambiguous.

    Tried in order, the first test that finds exactly one header wins: the same text (ignoring
    case and spaces), the same text without a unit suffix (`V_MFC` = `V_MFC [Nl/min]`), the same
    cleaned name (`v_mfc`), the same cleaned name without unit.
    """
    for test in _levels(name):
        hits = [i for i, h in enumerate(headers) if test(h)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            cols = ", ".join(f"{column_letter(i)} '{headers[i]}'" for i in hits)
            raise GridError(f"Column '{name}' is ambiguous: {cols}. Name the header exactly.")
    return None


def suggest_header(name: str, headers: Iterable[str]) -> str | None:
    """The header closest to `name` (difflib), or None."""
    named = [h for h in headers if not h.startswith("col_")]
    for key in (header_key, _base_key):
        keys = {key(h): h for h in named}
        hit = difflib.get_close_matches(key(name), list(keys), n=1, cutoff=0.5)
        if hit:
            return keys[hit[0]]
    return None


def missing_text(name: str, headers: Sequence[str]) -> str:
    s = suggest_header(name, headers)
    return f"no column '{name}'" + (f" (did you mean '{s}'?)" if s else "")


@dataclass(frozen=True)
class HeaderMatch:
    row: int
    headers: list[str]
    columns: dict[str, int]  # configured name -> 0-based column
    missing: list[str]  # optional names not found


def match_row(grid: Grid, row: int, required: Sequence[str], optional: Sequence[str] = ()) -> HeaderMatch:
    """Columns of all names in header row `row`; GridError naming every missing one (with suggestion)."""
    headers = grid.headers(row)
    columns, missing, errors = {}, [], []
    names = list(dict.fromkeys(required)) + [n for n in dict.fromkeys(optional) if n not in required]
    for name in names:
        idx = match_header(name, headers)
        if idx is not None:
            columns[name] = idx
        elif name in required:
            errors.append(missing_text(name, headers))
        else:
            missing.append(name)
    if errors:
        raise GridError(f"Header row {row}: " + "; ".join(errors) + ".")
    return HeaderMatch(row, headers, columns, missing)


def find_header_row(
    grid: Grid, required: Sequence[str], optional: Sequence[str] = (), row: int | None = None
) -> HeaderMatch:
    """The header row: `row` if given (checked), else the first row that has all `required` names.

    Nothing found: GridError for the row with the most of them (the reader's header row on a tie),
    naming what is missing there with a suggestion. A table (not a sheet) has its header in row 1.
    """
    if row is not None or grid.kind == "table":
        return match_row(grid, row or 1, required, optional)
    names = list(dict.fromkeys(required))
    best, best_n = None, -1
    for r in range(1, min(grid.n_rows, HEADER_SEARCH_ROWS) + 1):
        if grid.row_empty(r):
            continue
        headers = grid.headers(r)
        n = 0
        for name in names:
            try:
                n += match_header(name, headers) is not None
            except GridError:
                n += 1  # present but ambiguous: reported when this row is used
        if n == len(names):
            return match_row(grid, r, required, optional)
        if n > best_n or (n == best_n and r == grid.reader_header_row):
            best, best_n = r, n
    if best is None:
        raise GridError("The table is empty.")
    try:
        match_row(grid, best, required, optional)
    except GridError as exc:
        raise GridError(f"No row has all headers of the mapping. {exc}") from None
    raise GridError("No row has all headers of the mapping.")  # pragma: no cover (best row would have matched)


@dataclass(frozen=True)
class Located:
    """A cell found for a `CellRef`: its position and value, and where its label is."""

    row: int
    col: int
    value: Any
    label_cell: str | None = None

    @property
    def address(self) -> str:
        return a1(self.row, self.col)


def find_label(grid: Grid, label: str) -> list[tuple[int, int]]:
    """All cells (1-based row, 0-based column) whose text equals `label`, ignoring case and spaces."""
    key = header_key(label)
    return [
        (r + 1, c)
        for r, row in enumerate(grid.rows)
        for c, v in enumerate(row)
        if isinstance(v, str) and header_key(v) == key
    ]


def _label_suggestion(grid: Grid, label: str) -> str | None:
    texts = {header_key(v): v.strip() for row in grid.rows for v in row if isinstance(v, str) and v.strip()}
    hit = difflib.get_close_matches(header_key(label), list(texts), n=1, cutoff=0.6)
    return texts[hit[0]] if hit else None


def locate(grid: Grid, ref: CellRef) -> Located:
    """Cell of a `CellRef`. A missing or ambiguous label, or no value next to it, raises GridError."""
    where = f"sheet '{grid.sheet}'" if grid.sheet else "the table"
    if ref.cell is not None:
        row, col = parse_a1(ref.cell)
        return Located(row, col, grid.value(row, col))
    hits = find_label(grid, ref.label or "")
    if not hits:
        s = _label_suggestion(grid, ref.label or "")
        raise GridError(f"No cell '{ref.label}' in {where}" + (f" (did you mean '{s}'?)" if s else "") + ".")
    if len(hits) > 1:
        raise GridError(f"'{ref.label}' is in several cells of {where}: {', '.join(a1(r, c) for r, c in hits)}.")
    r, c = hits[0]
    dr, dc = (0, 1) if ref.direction == "right" else (1, 0)
    found = 0
    for k in range(1, LOCATE_MAX_STEPS + 1):
        v = grid.value(r + dr * k, c + dc * k)
        if not is_empty(v):
            found += 1
            if found == ref.nth:
                return Located(r + dr * k, c + dc * k, v, a1(r, c))
    raise GridError(f"No value {ref.describe()} ({a1(r, c)}) in {where}.")
