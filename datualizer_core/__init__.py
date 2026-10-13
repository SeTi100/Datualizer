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
    TimeConfig,
    TimeFormat,
    UnitConfig,
    Vocabulary,
)
from datualizer_core.ingestion.loader import CSVLoader, load_csv
from datualizer_core.ingestion.merge import MergeError, load_csvs
from datualizer_core.ingestion.profiles import (
    OriginRule,
    ReaderKind,
    SourceProfile,
    builtin_profiles,
    detect_profile,
    get_profile,
    load_profile,
)
from datualizer_core.ingestion.readers import ExcelOptions, RawTable, ReaderError, SqliteOptions
from datualizer_core.ingestion.source_loader import load_source, needs_clock_date, with_choices
from datualizer_core.ingestion.quality_rules import QualityConfigError
from datualizer_core.ingestion.units import UnitConfigError
from datualizer_core.ingestion.pre_scanner import PreScanResult, PreScanner, pre_scan
from datualizer_core.schema import ColumnKind
from datualizer_core.timebase import TimeConfigError, date_in_text
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
    "TimeConfig",
    "TimeConfigError",
    "TimeFormat",
    "UnitConfig",
    "UnitConfigError",
    "Vocabulary",
    "load_csv",
    "load_csvs",
    "load_source",
    "needs_clock_date",
    "with_choices",
    "date_in_text",
    "load_profile",
    "get_profile",
    "builtin_profiles",
    "detect_profile",
    "SourceProfile",
    "ReaderKind",
    "OriginRule",
    "ExcelOptions",
    "SqliteOptions",
    "RawTable",
    "ReaderError",
    "PreScanResult",
    "PreScanner",
    "pre_scan",
    "clean_column_name",
    "clean_names",
    "drop_footer",
    "unpivot",
]
