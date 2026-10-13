"""Delimited text (CSV, TSV, `.dat`, `.txt`): the existing pre-scanner path."""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from datualizer_core.ingestion.pre_scanner import pre_scan
from datualizer_core.ingestion.readers.base import RawTable


def read_delimited(source: str | Path | TextIO) -> RawTable:
    """Pre-scan and read a delimited text source as an all-String table."""
    # Local import: loader imports the readers package for `load_source`
    from datualizer_core.ingestion.loader import read_raw_table

    res = pre_scan(source)
    return RawTable(
        read_raw_table(res),
        res.decimal_separator,
        {"delimiter": res.delimiter, "decimal_separator": res.decimal_separator,
         "skipped_lines": res.skip_rows, "footer_lines": res.footer_lines_count},
    )
