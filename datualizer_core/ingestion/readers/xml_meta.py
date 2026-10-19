"""Settings files in XML: selected blocks as source metadata, no table.

Device settings (Testa iFID `Settings*.xml`) hold a calibration next to things that must not
be copied anywhere: password hashes, PINs, system ids. The reader therefore keeps only the
blocks the profile lists (`XmlOptions.blocks`) and of those only the leaf values; every other
element is parsed by the XML library but never stored, logged or returned.

Values become numbers where they read as one, date-times in one of `datetime_formats` become
ISO 8601 text (still on the device clock), everything else stays text.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from datualizer_core.ingestion.readers.base import RawTable, ReaderError


class XmlOptions(BaseModel):
    """Which parts of an XML file are kept."""

    model_config = ConfigDict(extra="forbid")

    blocks: list[str] = Field(
        default_factory=list,
        description="Element paths below the root (ElementTree syntax, e.g. 'calib_data'); only their leaf values are kept.",
    )
    datetime_formats: list[str] = Field(
        default_factory=list, description="strptime formats of date-time values, e.g. '%d.%m.%Y %H:%M:%S'."
    )


def _value(text: str | None, formats: list[str]) -> int | float | str | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).isoformat(sep=" ")
        except ValueError:
            pass
    return text


def read_xml_meta(path: str | Path, options: XmlOptions | None = None) -> RawTable:
    """Read the listed blocks of an XML file into `meta["xml"]`; the table is empty.

    `meta["xml"]` maps each found block to `{field: value}` of its leaf children (nested
    elements are skipped); `meta["xml_missing_blocks"]` lists blocks that were not found.
    Without blocks nothing is kept. A file that is not XML raises ReaderError.
    """
    path = Path(path)
    opts = options or XmlOptions()
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ReaderError(f"{path.name} is not a readable XML file: {exc}") from exc
    blocks: dict[str, dict[str, int | float | str | None]] = {}
    missing: list[str] = []
    for name in opts.blocks:
        el = root.find(name)
        if el is None:
            missing.append(name)
            continue
        blocks[name] = {child.tag: _value(child.text, opts.datetime_formats) for child in el if len(child) == 0}
    return RawTable(pl.DataFrame(), ".", {"xml": blocks, "xml_missing_blocks": missing})
