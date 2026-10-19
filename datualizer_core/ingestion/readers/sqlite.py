"""SQLite databases: a whole table or a read-only query.

The database is opened read-only. Besides the table, the reader reports which tables exist and
which AUTOINCREMENT ids are missing, so deleted experiments or runs become visible.

Columns holding JSON objects as text (`SqliteOptions.json_columns`, e.g. run parameters
`{"Temperatur": "150", "V_MFC": "1"}`) are split into one column per key; the JSON column stays.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datualizer_core.ingestion.readers.base import RawTable, ReaderError, cell_to_text, string_frame, unique_headers


class SqliteOptions(BaseModel):
    """Which data to read from a database: a table or a query (not both)."""

    model_config = ConfigDict(extra="forbid")

    table: str | None = Field(None, description="Table to read completely.")
    query: str | None = Field(None, description="Read-only SELECT, e.g. a join of readings with their runs.")
    json_columns: list[str] = Field(
        default_factory=list,
        description="Columns of the result holding JSON objects as text; each key becomes a column (missing key = empty).",
    )

    @model_validator(mode="after")
    def _one_source(self) -> "SqliteOptions":
        if self.table is not None and self.query is not None:
            raise ValueError("Give either 'table' or 'query', not both.")
        return self


def _ranges(ids: list[int]) -> list[list[int]]:
    """[1, 2, 3, 7] -> [[1, 3], [7, 7]]"""
    out: list[list[int]] = []
    for i in ids:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def _missing_ids(con: sqlite3.Connection, tables: list[str]) -> dict[str, list[list[int]]]:
    """Ids between 1 and the AUTOINCREMENT counter that no longer exist, as ranges."""
    if "sqlite_sequence" not in tables:
        return {}
    out: dict[str, list[list[int]]] = {}
    for name, seq in con.execute("SELECT name, seq FROM sqlite_sequence"):
        if name not in tables:
            continue
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
        if "id" not in cols:
            continue
        present = {r[0] for r in con.execute(f'SELECT id FROM "{name}"')}
        missing = [i for i in range(1, int(seq) + 1) if i not in present]
        if missing:
            out[name] = _ranges(missing)
    return out


def read_sqlite(path: str | Path, options: SqliteOptions | None = None) -> RawTable:
    """Read a table or query result of a SQLite file as an all-String table."""
    path = Path(path)
    opts = options or SqliteOptions()
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        user_tables = [t for t in tables if t != "sqlite_sequence"]
        if opts.query is not None:
            sql = opts.query
        elif opts.table is not None:
            if opts.table not in tables:
                raise ReaderError(
                    f"{path.name} has no table {opts.table!r}; tables: {user_tables}",
                    choice="sqlite.table",
                    options=user_tables,
                )
            sql = f'SELECT * FROM "{opts.table}"'
        elif len(user_tables) == 1:
            sql = f'SELECT * FROM "{user_tables[0]}"'
        else:
            raise ReaderError(
                f"{path.name} has several tables, choose one or give a query: {user_tables}",
                choice="sqlite.table",
                options=user_tables,
            )
        try:
            cur = con.execute(sql)
        except sqlite3.Error as exc:
            raise ReaderError(f"Query on {path.name} failed: {exc}") from exc
        names = [d[0] for d in cur.description]
        rows = cur.fetchall()
        meta = {
            "tables": {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in user_tables},
            "missing_ids": _missing_ids(con, tables),
            "query": sql,
        }
    finally:
        con.close()
    headers = unique_headers(names, [f"col_{i}" for i in range(len(names))])
    if opts.json_columns:
        headers, rows, meta["json_columns"] = _split_json(path, headers, rows, opts.json_columns)
    return RawTable(string_frame(headers, rows), ".", meta)


def _json_text(value: Any) -> Any:
    """A JSON value as the cell the loader reads: scalars as they are, objects and lists as JSON text."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _split_json(
    path: Path, headers: list[str], rows: list[tuple], columns: list[str]
) -> tuple[list[str], list[list[Any]], dict[str, dict[str, Any]]]:
    """Add one column per key of each JSON column, right after it.

    New columns are named after the key, or `<column>_<key>` if that name is taken. A row without
    the key, with an empty cell or with text that is not a JSON object gets empty cells; the
    latter are counted in the metadata (`invalid_rows`, first few in `invalid_examples`).
    """
    unknown = [c for c in columns if c not in headers]
    if unknown:
        raise ReaderError(f"{path.name}: json_columns {unknown} are not in the result; columns: {headers}")
    out_rows = [list(r) for r in rows]
    out_headers = list(headers)
    report: dict[str, dict[str, Any]] = {}
    for col in columns:
        idx = out_headers.index(col)
        parsed: list[dict] = []
        invalid: list[int] = []
        for i, r in enumerate(out_rows):
            cell = r[idx] if idx < len(r) else None
            obj: Any = {}
            if cell is not None and str(cell).strip():
                try:
                    obj = json.loads(cell)
                except (TypeError, ValueError):
                    obj = None
                if not isinstance(obj, dict):
                    invalid.append(i)
                    obj = {}
            parsed.append(obj)
        keys: list[str] = []
        for obj in parsed:
            keys += [k for k in obj if k not in keys]
        taken = set(out_headers)
        names = []
        for k in keys:
            name = k if k not in taken else f"{col}_{k}"
            n = 2
            while name in taken:
                name, n = f"{col}_{k}_{n}", n + 1
            taken.add(name)
            names.append(name)
        out_headers[idx + 1:idx + 1] = names
        for r, obj in zip(out_rows, parsed):
            r += [None] * (idx + 1 - len(r))
            r[idx + 1:idx + 1] = [cell_to_text(_json_text(obj.get(k))) for k in keys]
        report[col] = {"keys": keys, "columns": names, "invalid_rows": len(invalid), "invalid_examples": invalid[:5]}
    return out_headers, out_rows, report
