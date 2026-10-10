"""Datualizer Core: High-Performance Tidy Data Wrangling & Visualization Framework for Sensor & Engineering Data."""

from datualizer_core.dataset import AuditEntry, AuditLog, DualModeDataset
from datualizer_core.ingestion.config import (
    IngestionConfig,
    LongFormatConfig,
    LongFormatMode,
    MergeConfig,
    MergeConflictMode,
    QualityConfig,
    RoleConfig,
    RunConfig,
    Vocabulary,
)
from datualizer_core.ingestion.loader import CSVLoader, load_csv
from datualizer_core.ingestion.merge import MergeError, load_csvs
from datualizer_core.ingestion.quality_rules import QualityConfigError
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
    "MergeConfig",
    "MergeConflictMode",
    "MergeError",
    "QualityConfig",
    "QualityConfigError",
    "RoleConfig",
    "RunConfig",
    "Vocabulary",
    "load_csv",
    "load_csvs",
    "PreScanResult",
    "PreScanner",
    "pre_scan",
    "clean_column_name",
    "clean_names",
    "drop_footer",
    "unpivot",
]
