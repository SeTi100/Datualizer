"""DualModeDataset: Encapsulates high-performance wide-format DataFrame with on-demand tidy long-format reshaping."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
import numpy as np
import polars as pl

from datualizer_core import quality as _quality
from datualizer_core import runs as _runs
from datualizer_core.pipeline.operators import unpivot
from datualizer_core.schema import ColumnKind

if TYPE_CHECKING:
    from datualizer_core.ingestion.config import IngestionConfig


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


SOURCES_SCHEMA = {
    "source": pl.String,
    "sha256": pl.String,
    "n_rows": pl.Int64,
    "n_kept": pl.Int64,
    "n_replaced": pl.Int64,
    "n_conflicts": pl.Int64,
    "duplicate_of": pl.String,
}


def empty_sources() -> pl.DataFrame:
    """Return an empty source table (see `DualModeDataset.sources`)."""
    return pl.DataFrame(schema=SOURCES_SCHEMA)


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
        ingestion_spec: IngestionConfig | None = None,
        run_columns: Sequence[str] = (),
        aborted_run_fraction: float = 0.1,
        sources: pl.DataFrame | None = None,
        quality: _quality.QualitySettings | None = None,
        units: dict[str, str] | None = None,
    ) -> None:
        self._df = df
        self._units = {c: u for c, u in (units or {}).items() if u}
        self._quality = quality
        self._quality_flags: pl.DataFrame | None = None
        self._sources = sources if sources is not None else empty_sources()
        self._audit_log = audit_log if audit_log is not None else AuditLog()
        self._channel_attrs = channel_attrs or {}
        self._source_format = source_format
        self._ingestion_spec = ingestion_spec
        self._run_columns = list(run_columns)
        self._aborted_run_fraction = aborted_run_fraction
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
        self._fill_ratio: dict[str, float] | None = None
        self._runs: pl.DataFrame | None = None
        self._channel_availability: pl.DataFrame | None = None

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
    def parameters(self) -> list[str]:
        """Return numeric parameter columns (setpoints constant per run, not plotted)."""
        return [c for c, k in self._column_kinds.items() if k is ColumnKind.PARAMETER]

    @property
    def fill_ratio(self) -> dict[str, float]:
        """Return the share of non-null values per column (time column excluded), 0.0 to 1.0."""
        if self._fill_ratio is None:
            n = len(self._df)
            nulls = self._df.null_count().row(0, named=True) if self._df.width else {}
            self._fill_ratio = {
                c: (1.0 - nulls[c] / n) if n else 0.0
                for c in self._df.columns
                if c != self._time_col
            }
        return dict(self._fill_ratio)

    @property
    def empty_columns(self) -> list[str]:
        """Return columns without a single value (e.g. a channel that was never recorded).

        They are marked, not dropped: the column stays in `df` and keeps its role.
        """
        return [c for c, r in self.fill_ratio.items() if r == 0.0]

    @property
    def run_columns(self) -> list[str]:
        """Return the columns that together identify a run (empty if the data has no runs)."""
        return list(self._run_columns)

    @property
    def runs(self) -> pl.DataFrame:
        """Return one row per run: size, timing, sampling interval, parameter values, `aborted`.

        See `datualizer_core.runs.summarize_runs`. Empty if the data has no runs.
        """
        if self._runs is None:
            self._runs = _runs.summarize_runs(
                self._df, self._time_col, self._run_columns, self.parameters, self._aborted_run_fraction
            )
        return self._runs

    @property
    def channel_availability(self) -> pl.DataFrame:
        """Return the fill ratio of every channel inside every run (empty if the data has no runs)."""
        if self._channel_availability is None:
            self._channel_availability = _runs.channel_availability(
                self._df, self._run_columns, self.channels
            )
        return self._channel_availability

    def run_time(self) -> pl.Series:
        """Return seconds since the start of each row's run; the global time stays in `time_col`."""
        return _runs.run_time(self._df, self._time_col, self._run_columns)

    def select_run(self, key: Sequence[Any]) -> "DualModeDataset":
        """Return a new dataset with only the rows of one run and a run-relative time column.

        `key` holds one value per run column, e.g. `(2,)`. The original data is not changed.
        Raises KeyError if the dataset has no runs or the run does not exist.
        """
        key = tuple(key)
        if not self._run_columns or len(key) != len(self._run_columns):
            raise KeyError(f"Run {key!r} does not match run columns {self._run_columns}.")
        mask = pl.all_horizontal(
            pl.col(c).is_null() if v is None else pl.col(c) == v
            for c, v in zip(self._run_columns, key)
        )
        df = self._df.with_columns(self.run_time()).filter(mask)
        if df.is_empty():
            raise KeyError(f"Run {key!r} not found.")
        return DualModeDataset(
            df=df,
            audit_log=self._audit_log,
            time_col=self._time_col,
            column_kinds=self._column_kinds,
            channel_attrs=self._channel_attrs,
            source_format=self._source_format,
            ingestion_spec=self._ingestion_spec,
            run_columns=self._run_columns,
            aborted_run_fraction=self._aborted_run_fraction,
            sources=self._sources,
            quality=self._quality,
            units=self._units,
        )

    @property
    def units(self) -> dict[str, str]:
        """Return the unit of every numeric column that has one (from header, long table or config)."""
        return dict(self._units)

    @property
    def unitless_columns(self) -> list[str]:
        """Return channels and parameters without a unit (P9: e.g. a setpoint `snad2 = 180`).

        They are only marked; set a unit via `IngestionConfig(units=UnitConfig(units={...}))`.
        """
        return [c for c in (*self.channels, *self.parameters) if c not in self._units]

    @property
    def quality_flags(self) -> pl.DataFrame:
        """Return every quality flag as an event table `row, channel, flag` (P12–P17).

        `row` indexes `df`, `channel` is null for row-level flags (`gap`). Flags are `gap`,
        `missing`, `missing_calculated`, `stuck`, `jump`, `dropout` and `out_of_range`; see
        `datualizer_core.quality`. The data itself is never changed. Empty if detection is off
        or the dataset was built in memory without quality settings.
        """
        if self._quality_flags is None:
            if self._quality is None:
                self._quality_flags = _quality.empty_flags()
            else:
                self._quality_flags = _quality.detect_flags(
                    self._df, self._time_col, self._run_columns, self.channels, self._quality
                )
        return self._quality_flags

    @property
    def quality_summary(self) -> pl.DataFrame:
        """Return the number of flagged samples per channel and flag."""
        return (
            self.quality_flags.group_by("channel", "flag")
            .agg(pl.len().cast(pl.Int64).alias("n"))
            .sort("channel", "flag", nulls_last=False)
        )

    def flag_mask(self, flags: Sequence[str] | None = None, channel: str | None = None) -> pl.Series:
        """Return a Boolean Series over `df` rows: True where one of `flags` (default: any) is set.

        With `channel`, only that channel's flags count; row-level flags have no channel.
        """
        events = self.quality_flags
        if flags is not None:
            events = events.filter(pl.col("flag").is_in(list(flags)))
        if channel is not None:
            events = events.filter(pl.col("channel") == channel)
        mask = pl.Series("flagged", [False] * len(self._df), dtype=pl.Boolean)
        rows = events["row"].unique()
        return mask.scatter(rows, True) if len(rows) else mask

    def with_flag_columns(self, suffix: str = "_flags") -> pl.DataFrame:
        """Return a new wide frame with one flag column per channel (comma-joined flags or null).

        Row-level flags go into the column named after the time column. `df` is not changed.
        """
        columns = [(self._time_col, None), *[(ch, ch) for ch in self.channels]]
        return _quality.flag_columns(self._df, self.quality_flags, columns, suffix)

    @property
    def sources(self) -> pl.DataFrame:
        """Return one row per source file: path, SHA-256 and what the merge did with its rows.

        Columns: `source`, `sha256`, `n_rows` (data rows in the file), `n_kept` (rows in this
        dataset), `n_replaced` (rows superseded by a later file with the same key), `n_conflicts`
        (values of this file that differed from the winning file) and `duplicate_of` (set for
        byte-identical files, which are skipped). Empty for datasets built in memory.
        """
        return self._sources

    @property
    def channel_attrs(self) -> dict[str, dict[str, Any]]:
        """Return per-channel attributes such as unit or is_calculated (filled for long-format sources)."""
        return {ch: dict(attrs) for ch, attrs in self._channel_attrs.items()}

    @property
    def ingestion_spec(self) -> IngestionConfig | None:
        """Return the fully resolved IngestionConfig used to load this dataset (None if built in memory).

        Every automatic decision is written out explicitly, so the spec can be edited and passed
        back to `load_csv(..., config=spec)` to reproduce or correct the result.
        """
        return self._ingestion_spec.model_copy(deep=True) if self._ingestion_spec else None

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
            if self._time_col in self._df.columns:
                effective_id = [self._time_col]
            elif len(self._df.columns) > 0:
                effective_id = [self._df.columns[0]]
            else:
                effective_id = []
            # Metadata and parameter columns stay as identifiers; only numeric channels are melted.
            effective_id += [
                c for c in (*self.metadata_columns, *self.parameters) if c not in effective_id
            ]
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
