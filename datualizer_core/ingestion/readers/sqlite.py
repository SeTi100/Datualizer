"""SQLite databases: a whole table or a read-only query.

The database is opened read-only. Besides the table, the reader reports which tables exist and
which AUTOINCREMENT ids are missing, so deleted experiments or runs become visible.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datualizer_core.ingestion.readers.base import RawTable, ReaderError, string_frame, unique_headers


class SqliteOptions(BaseModel):
    """Which data to read from a database: a table or a query (not both)."""

    model_config = ConfigDict(extra="forbid")

    table: str | None = Field(None, description="Table to read completely.")
    query: str | None = Field(None, description="Read-only SELECT, e.g. a join of readings with their runs.")

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
    return RawTable(string_frame(headers, rows), ".", meta)
