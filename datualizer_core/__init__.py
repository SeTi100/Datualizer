"""Datualizer Core: High-Performance Tidy Data Wrangling & Visualization Framework for Sensor & Engineering Data."""

from datualizer_core.dataset import AuditEntry, AuditLog, DualModeDataset
from datualizer_core.ingestion.config import (
    IngestionConfig,
    LongFormatConfig,
    LongFormatMode,
    RoleConfig,
    Vocabulary,
)
from datualizer_core.ingestion.loader import CSVLoader, load_csv
from datualizer_core.ingestion.pre_scanner import PreScanResult, PreScanner, pre_scan
from datualizer_core.schema import ColumnKind
from datualizer_core.pipeline.operators import clean_column_name, clean_names, drop_footer, unpivot

__all__ = [
    "AuditEntry",
    "ColumnKind",
    "AuditLog",
    "DualModeDataset",
    "CSVLoader",
    "IngestionConfig",
    "LongFormatConfig",
    "LongFormatMode",
    "RoleConfig",
    "Vocabulary",
    "load_csv",
    "PreScanResult",
    "PreScanner",
    "pre_scan",
    "clean_column_name",
    "clean_names",
    "drop_footer",
    "unpivot",
]
