"""Readers turn a source file into an all-String raw table plus source metadata (Etappe 2b).

Every reader hands its table to the same `CSVLoader.load_frame`, so type inference, audit,
long-format pivot, quality flags and spec replay work identically for every file format.
"""

from datualizer_core.ingestion.readers.base import RawTable, ReaderError, cell_to_text
from datualizer_core.ingestion.readers.delimited import read_delimited
from datualizer_core.ingestion.readers.excel import ExcelOptions, read_excel
from datualizer_core.ingestion.readers.sqlite import SqliteOptions, read_sqlite

__all__ = [
    "ExcelOptions",
    "RawTable",
    "ReaderError",
    "SqliteOptions",
    "cell_to_text",
    "read_delimited",
    "read_excel",
    "read_sqlite",
]
