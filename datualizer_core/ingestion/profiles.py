"""Source profiles: how a device export is read (reader, options, IngestionConfig), as JSON data.

A profile is data, not code (AGENTS.md, principle 1): device-specific names and settings live in
`datualizer_core/profiles/*.json` or in user files, never in `if` branches. `detect_profile`
picks a profile by file pattern and hints (header text, sheet or table names); the user can
always pass a profile explicitly.
"""

from __future__ import annotations

from enum import Enum
import fnmatch
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from datualizer_core.ingestion.config import IngestionConfig
from datualizer_core.ingestion.readers.excel import ExcelOptions, _detect_header, _rows_xls, _rows_xlsx
from datualizer_core.ingestion.readers.sqlite import SqliteOptions
from datualizer_core.ingestion.steps import StepColumns

PROFILE_DIR = Path(__file__).resolve().parents[1] / "profiles"
SQLITE_MAGIC = b"SQLite format 3\x00"
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}
HINT_SCAN_BYTES = 16384


class ReaderKind(str, Enum):
    DELIMITED = "delimited"
    EXCEL = "excel"
    SQLITE = "sqlite"


class OriginRule(str, Enum):
    NONE = "none"
    FILE_MTIME_END = "file_mtime_end"  # logger saved the file right after the last sample


class SourceProfile(BaseModel):
    """Everything needed to read one kind of device export reproducibly."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    reader: ReaderKind = ReaderKind.DELIMITED
    file_patterns: list[str] = Field(
        default_factory=list, description="File name patterns (case-insensitive), e.g. '*.dat'. Empty = any."
    )
    header_hints: list[str] = Field(
        default_factory=list,
        description="Texts that must all occur: in the first lines (text), header or sheet names (Excel), table names (SQLite).",
    )
    excel: ExcelOptions = Field(default_factory=ExcelOptions)
    sqlite: SqliteOptions = Field(default_factory=SqliteOptions)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    origin_rule: OriginRule = Field(
        OriginRule.NONE,
        description="How to find the absolute start of sources with only relative time (if TimeConfig.origin is unset).",
    )
    intervals: StepColumns | None = Field(
        None, description="Suggested column roles if the source lists measurement steps (lab journal)."
    )


def reader_for(path: str | Path) -> ReaderKind:
    """Reader kind from the file content (SQLite magic bytes) or suffix."""
    path = Path(path)
    with open(path, "rb") as f:
        if f.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC:
            return ReaderKind.SQLITE
    if path.suffix.lower() in EXCEL_SUFFIXES:
        return ReaderKind.EXCEL
    return ReaderKind.DELIMITED


def load_profile(path: str | Path) -> SourceProfile:
    """Read a profile JSON file."""
    return SourceProfile.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def builtin_profiles() -> list[SourceProfile]:
    """Profiles shipped in `datualizer_core/profiles/`, sorted by name."""
    return sorted((load_profile(p) for p in PROFILE_DIR.glob("*.json")), key=lambda p: p.name)


def get_profile(name: str) -> SourceProfile:
    """A shipped profile by name."""
    for p in builtin_profiles():
        if p.name == name:
            return p
    raise KeyError(f"No profile {name!r}; available: {[p.name for p in builtin_profiles()]}")


def _hint_pool(path: Path, kind: ReaderKind) -> str:
    """Text the hints are searched in, cheap enough to build for every candidate profile."""
    if kind is ReaderKind.SQLITE:
        import sqlite3

        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            return "\n".join(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'"))
        finally:
            con.close()
    if kind is ReaderKind.EXCEL:
        names, sheets = _rows_xls(path) if path.suffix.lower() == ".xls" else _rows_xlsx(path)
        parts = list(names)
        for rows in sheets.values():
            if rows:
                parts += [str(v) for v in rows[_detect_header(rows)] if v is not None]
        return "\n".join(parts)
    head = path.read_bytes()[:HINT_SCAN_BYTES]
    try:
        return head.decode("utf-8")
    except UnicodeDecodeError:
        return head.decode("latin-1")


def matching_profiles(path: str | Path, profiles: list[SourceProfile] | None = None) -> list[SourceProfile]:
    """All profiles that fit the file, most specific (most hints) first."""
    path = Path(path)
    kind = reader_for(path)
    candidates = [
        p for p in (profiles if profiles is not None else builtin_profiles())
        if p.reader is kind
        and (not p.file_patterns or any(fnmatch.fnmatch(path.name.lower(), pat.lower()) for pat in p.file_patterns))
    ]
    if not candidates:
        return []
    pool = _hint_pool(path, kind) if any(p.header_hints for p in candidates) else ""
    hits = [p for p in candidates if all(h in pool for h in p.header_hints)]
    return sorted(hits, key=lambda p: -len(p.header_hints))


def detect_profile(path: str | Path, profiles: list[SourceProfile] | None = None) -> SourceProfile | None:
    """The most specific matching profile, or None (then the plain default config is used)."""
    hits = matching_profiles(path, profiles)
    return hits[0] if hits else None
