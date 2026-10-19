"""Readers turn a source file into an all-String raw table plus source metadata (Etappe 2b).

The XML reader is the exception: settings files have no table, only metadata.

Every reader hands its table to the same `CSVLoader.load_frame`, so type inference, audit,
long-format pivot, quality flags and spec replay work identically for every file format.
"""

from datualizer_core.ingestion.readers.base import RawTable, ReaderError, cell_to_text
from datualizer_core.ingestion.readers.delimited import read_delimited
from datualizer_core.ingestion.readers.excel import ExcelOptions, parse_cell, read_cells, read_excel, sheet_names
from datualizer_core.ingestion.readers.sqlite import SqliteOptions, read_sqlite
from datualizer_core.ingestion.readers.xml_meta import XmlOptions, read_xml_meta

__all__ = [
    "ExcelOptions",
    "RawTable",
    "ReaderError",
    "SqliteOptions",
    "XmlOptions",
    "cell_to_text",
    "parse_cell",
    "read_cells",
    "read_delimited",
    "read_excel",
    "read_sqlite",
    "read_xml_meta",
    "sheet_names",
]
