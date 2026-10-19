"""`load_source`: load any supported file (text, Excel, SQLite, XML settings) with a profile into a dataset."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
from pathlib import Path

import polars as pl

from datualizer_core.dataset import SOURCES_SCHEMA, DualModeDataset
from datualizer_core.ingestion.config import IngestionConfig, TimeFormat
from datualizer_core.ingestion.loader import CSVLoader
from datualizer_core.ingestion.profiles import (
    OriginRule,
    ReaderKind,
    SourceProfile,
    detect_profile,
    get_profile,
    reader_for,
)
from datualizer_core.ingestion.readers import RawTable, read_delimited, read_excel, read_sqlite, read_xml_meta


def read_raw(path: Path, profile: SourceProfile | None) -> tuple[RawTable, ReaderKind]:
    """Read a file with the profile's reader (or the reader its content suggests)."""
    kind = profile.reader if profile is not None else reader_for(path)
    if kind is ReaderKind.EXCEL:
        return read_excel(path, profile.excel if profile else None), kind
    if kind is ReaderKind.SQLITE:
        return read_sqlite(path, profile.sqlite if profile else None), kind
    if kind is ReaderKind.XML:
        return read_xml_meta(path, profile.xml if profile else None), kind
    return read_delimited(path), kind


def with_choices(
    path: str | Path,
    profile: SourceProfile | None,
    *,
    sheet: str | None = None,
    table: str | None = None,
    clock_date: date | None = None,
) -> SourceProfile:
    """Copy of `profile` with the user's choices: Excel sheet, SQLite table, date of clock times.

    Without a profile, a minimal one for the file's reader is created, so the choice is stored in
    the same place (`excel.sheet`, `sqlite.table`, `ingestion.time.clock_date`) and replays.
    The given profile is never changed.
    """
    path = Path(path)
    out = (
        profile.model_copy(deep=True)
        if profile is not None
        else SourceProfile(name="manual", description="Choices made when opening the file.", reader=reader_for(path))
    )
    if sheet is not None:
        out.excel.sheet = sheet
    if table is not None:
        out.sqlite.table = table
        out.sqlite.query = None
    if clock_date is not None:
        out.ingestion.time.clock_date = clock_date
        out.ingestion.time.origin = None
    return out


def needs_clock_date(ds: DualModeDataset) -> bool:
    """True if the time column holds bare times of day and no date or origin places them."""
    spec = ds.ingestion_spec
    return ds.time_origin is None and spec.time.format is TimeFormat.CLOCK


def load_source(
    source: str | Path,
    profile: SourceProfile | str | None = None,
    config: IngestionConfig | None = None,
) -> DualModeDataset:
    """Load a text, Excel, SQLite or XML settings file into a DualModeDataset.

    `profile`: a SourceProfile, the name of a shipped profile, or None to detect one (falls back
    to the reader the file suggests and the default config). `config` replaces the profile's
    IngestionConfig, e.g. a dataset's `ingestion_spec` for an exact replay.
    `ds.source_meta` tells which profile and reader were used and what the reader found.
    An XML settings file gives a dataset without rows; its content is in `source_meta["xml"]`.
    """
    path = Path(source)
    if isinstance(profile, str):
        profile = get_profile(profile)
    elif profile is None:
        profile = detect_profile(path)
    cfg = config if config is not None else (profile.ingestion if profile is not None else IngestionConfig())

    raw, kind = read_raw(path, profile)
    sources = pl.DataFrame(
        [{
            "source": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_rows": len(raw.df),
            "n_kept": len(raw.df),
            "n_replaced": 0,
            "n_conflicts": 0,
            "duplicate_of": None,
        }],
        schema=SOURCES_SCHEMA,
    )
    meta = {"profile": profile.name if profile else None, "reader": kind.value, **raw.meta}
    if profile is not None and profile.calibration is not None:
        meta["calibration_fields"] = profile.calibration.model_dump()
    if kind is ReaderKind.XML:  # settings file: metadata only, no table, no time
        return DualModeDataset(pl.DataFrame(), sources=sources, source_meta=meta)
    ds = CSVLoader(cfg).load_frame(raw.df, raw.decimal_separator, sources=sources, source_meta=meta)

    rule = profile.origin_rule if profile is not None else OriginRule.NONE
    spec = ds.ingestion_spec
    if (
        rule is OriginRule.FILE_MTIME_END
        and ds.time_origin is None
        and spec.time.format in (TimeFormat.CLOCK, TimeFormat.NUMERIC)
        and len(ds) > 0
    ):
        # The file was saved right after the last sample: start = save time - duration.
        mtime = path.stat().st_mtime
        end = datetime.fromtimestamp(mtime, tz=timezone.utc) if spec.time.timezone else datetime.fromtimestamp(mtime)
        last = ds.df[ds.time_col].max() or 0.0
        spec.time.origin = end - timedelta(seconds=last)
        meta["time_origin_rule"] = {
            "rule": rule.value,
            "file_mtime": end.isoformat(),
            "note": "Assumes the logger saved the file right after the last sample.",
        }
        if raw.meta.get("footer_lines"):
            meta["time_origin_rule"]["warning"] = (
                f"{raw.meta['footer_lines']} trailing lines were dropped as footer; if they carried "
                "time stamps, the computed start is too late by their duration."
            )
        ds = CSVLoader(spec).load_frame(raw.df, raw.decimal_separator, sources=sources, source_meta=meta)
    return ds
