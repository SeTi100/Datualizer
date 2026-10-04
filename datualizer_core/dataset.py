"""DualModeDataset: Encapsulates high-performance wide-format DataFrame with on-demand tidy long-format reshaping."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import polars as pl

from datualizer_core.pipeline.operators import unpivot
from datualizer_core.schema import ColumnKind


@dataclass(frozen=True)
class AuditEntry:
    """Single audit log entry for unconvertible values, sentinels, or data errors."""

    row_index: int
    column: str
    raw_value: Any
    reason: str
    raw_column: str = ""


@dataclass
class AuditLog:
    """Audit log container recording non-strict conversion errors and anomalies."""

    entries: list[AuditEntry] = field(default_factory=list)

    def add(
        self,
        row_index: int,
        column: str,
        raw_value: Any,
        reason: str,
        raw_column: str = "",
    ) -> None:
        """Record an audit entry."""
        self.entries.append(AuditEntry(row_index, column, raw_value, reason, raw_column))

    def has_errors(self) -> bool:
        """Check if any audit entries are present."""
        return len(self.entries) > 0

    def for_column(self, col: str) -> list[AuditEntry]:
        """Return all audit entries for a specific column name (matching cleaned or raw name)."""
        return [e for e in self.entries if e.column == col or e.raw_column == col]

    def to_dataframe(self) -> pl.DataFrame:
        """Convert audit log entries to a Polars DataFrame."""
        schema = {
            "row_index": pl.Int64,
            "column": pl.String,
            "raw_value": pl.String,
            "reason": pl.String,
        }
        if not self.entries:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(
            [
                {
                    "row_index": e.row_index,
                    "column": e.column,
                    "raw_value": str(e.raw_value),
                    "reason": e.reason,
                }
                for e in self.entries
            ],
            schema=schema,
        )

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, idx: int | slice):
        return self.entries[idx]

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"AuditLog(total_entries={len(self.entries)})"


class DualModeDataset:
    """Dual-Mode Dataset encapsulating wide-format DataFrame with on-demand long-format.

    - Primary Mode: Wide-format Polars DataFrame optimized for zero-copy NumPy/PyQtGraph plotting.
    - Secondary Mode: On-demand tidy long-format Polars DataFrame via `.to_long()`.
    - Audit Trail: Accessible via `.audit_log` detailing non-strict float conversions.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        audit_log: AuditLog | None = None,
        time_col: str = "time_seconds",
        column_kinds: dict[str, ColumnKind] | None = None,
        channel_attrs: dict[str, dict[str, Any]] | None = None,
        source_format: str = "wide",
    ) -> None:
        self._df = df
        self._audit_log = audit_log if audit_log is not None else AuditLog()
        self._channel_attrs = channel_attrs or {}
        self._source_format = source_format
        if time_col in df.columns:
            self._time_col = time_col
        elif "time_seconds" in df.columns:
            self._time_col = "time_seconds"
        elif "time" in df.columns:
            self._time_col = "time"
        elif len(df.columns) > 0:
            self._time_col = df.columns[0]
        else:
            self._time_col = time_col
        self._column_kinds = self._derive_column_kinds(column_kinds or {})
        self._cached_long_df: pl.DataFrame | None = None

    def _derive_column_kinds(self, given: dict[str, ColumnKind]) -> dict[str, ColumnKind]:
        """Complete column kinds from dtypes for columns the caller did not classify."""
        kinds: dict[str, ColumnKind] = {}
        for col, dtype in self._df.schema.items():
            if col in given:
                kinds[col] = given[col]
            elif col == self._time_col:
                kinds[col] = ColumnKind.TIME
            elif dtype.is_numeric():
                kinds[col] = ColumnKind.NUMERIC
            else:
                kinds[col] = ColumnKind.CATEGORICAL
        return kinds

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "DualModeDataset":
        """Load a DualModeDataset directly from a CSV/text file using the two-stage pre-scanner and loader."""
        from datualizer_core.ingestion.loader import load_csv
        return load_csv(path, **kwargs)

    @property
    def df(self) -> pl.DataFrame:
        """Return the underlying wide-format Polars DataFrame."""
        return self._df

    @property
    def wide(self) -> pl.DataFrame:
        """Alias for `.df`."""
        return self._df

    @property
    def audit_log(self) -> AuditLog:
        """Return the audit log."""
        return self._audit_log

    @property
    def time_col(self) -> str:
        """Return the primary time column name."""
        return self._time_col

    @property
    def columns(self) -> list[str]:
        """Return column names of the wide DataFrame."""
        return self._df.columns

    @property
    def column_kinds(self) -> dict[str, ColumnKind]:
        """Return the inferred ColumnKind for every column."""
        return dict(self._column_kinds)

    @property
    def channels(self) -> list[str]:
        """Return the numeric measurement channels (plottable columns, excluding time and metadata)."""
        return [c for c, k in self._column_kinds.items() if k is ColumnKind.NUMERIC]

    @property
    def channel_attrs(self) -> dict[str, dict[str, Any]]:
        """Return per-channel attributes such as unit or is_calculated (filled for long-format sources)."""
        return {ch: dict(attrs) for ch, attrs in self._channel_attrs.items()}

    @property
    def source_format(self) -> str:
        """Return the layout of the source file: 'wide' or 'long'."""
        return self._source_format

    @property
    def metadata_columns(self) -> list[str]:
        """Return identifier and categorical columns (e.g. run_id, phase_status)."""
        return [
            c for c, k in self._column_kinds.items()
            if k in (ColumnKind.IDENTIFIER, ColumnKind.CATEGORICAL)
        ]

    @property
    def schema(self) -> pl.Schema:
        """Return schema of the wide DataFrame."""
        return self._df.schema

    @property
    def shape(self) -> tuple[int, int]:
        """Return (n_rows, n_cols) of the wide DataFrame."""
        return self._df.shape

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self):
        """Iterate over column names of the wide DataFrame."""
        return iter(self._df.columns)

    def __contains__(self, key: str) -> bool:
        """Check if a column exists in the dataset."""
        if not isinstance(key, str):
            return False
        if key in self._df.columns:
            return True
        if key == "time" and "time_seconds" in self._df.columns:
            return True
        return False

    def __getitem__(self, key: str | Any) -> pl.Series:
        """Allow column access directly on dataset, e.g. dataset['kanal_1']."""
        if isinstance(key, str):
            if key in self._df.columns:
                return self._df[key]
            if key == "time" and "time_seconds" in self._df.columns:
                return self._df["time_seconds"]
        return self._df[key]

    def to_numpy(self, col: str, allow_copy: bool = True) -> np.ndarray:
        """Extract a column as a NumPy ndarray for PyQtGraph plotting.

        Note: When data contains nulls/NaNs or non-contiguous layouts, allow_copy
        enables converting nulls to np.nan.
        """
        series = self[col]
        try:
            return series.to_numpy(allow_copy=allow_copy)
        except TypeError:
            # Fallback for polars versions with different parameters
            return series.to_numpy()

    def to_long(
        self,
        id_vars: str | Sequence[str] | None = None,
        value_vars: str | Sequence[str] | None = None,
        variable_name: str = "channel",
        value_name: str = "value",
        use_cache: bool = True,
    ) -> pl.DataFrame:
        """Transform wide-format data into tidy long-format DataFrame on-demand.

        Results with default arguments are cached for subsequent calls.
        """
        is_default = (
            id_vars is None
            and value_vars is None
            and variable_name == "channel"
            and value_name == "value"
        )

        if is_default and use_cache and self._cached_long_df is not None:
            return self._cached_long_df

        if id_vars is None:
            # Metadata columns stay as identifiers; only numeric channels are melted.
            if self._time_col in self._df.columns:
                effective_id = [self._time_col]
            elif len(self._df.columns) > 0:
                effective_id = [self._df.columns[0]]
            else:
                effective_id = []
            effective_id += [c for c in self.metadata_columns if c not in effective_id]
            if value_vars is None:
                value_vars = [c for c in self.channels if c not in effective_id]
        elif isinstance(id_vars, str):
            effective_id = [id_vars]
        else:
            effective_id = list(id_vars)

        long_df = unpivot(
            self._df,
            id_vars=effective_id,
            value_vars=value_vars,
            variable_name=variable_name,
            value_name=value_name,
        )

        if is_default and use_cache:
            self._cached_long_df = long_df

        return long_df

    def to_arrow(self) -> Any:
        """Return underlying data as a PyArrow Table."""
        return self._df.to_arrow()

    def head(self, n: int = 5) -> pl.DataFrame:
        """Return first n rows of the wide DataFrame."""
        return self._df.head(n)

    def tail(self, n: int = 5) -> pl.DataFrame:
        """Return last n rows of the wide DataFrame."""
        return self._df.tail(n)

    def __repr__(self) -> str:
        return (
            f"DualModeDataset(rows={len(self._df)}, columns={len(self._df.columns)}, "
            f"time_col='{self._time_col}', source_format='{self._source_format}', "
            f"audit_errors={len(self._audit_log)})"
        )
