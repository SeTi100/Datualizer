"""User-configurable ingestion settings.

Principle: explicit user settings always win; heuristics only fill what is left open.
After loading, `DualModeDataset.ingestion_spec` holds a fully resolved copy of this config
(every heuristic decision written out), so it can be inspected, corrected and replayed.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from datualizer_core.schema import ColumnKind

DEFAULT_SENTINELS = [
    "", "NA", "N/A", "null", "NULL", "NaN", "None", "-999", "ERR", "#N/A", "#VALUE!",
    "nan", "NAN", "error", "ERROR", "undef", "UNDEF", "overflow", "OVERFLOW",
]


class Vocabulary(BaseModel):
    """Name tokens the heuristics look for. Defaults are a starting point, not a convention."""

    model_config = ConfigDict(extra="forbid")

    time_names: list[str] = Field(
        default_factory=lambda: ["time", "time_seconds", "zeit", "timestamp", "datetime", "date"],
        description="Exact header names (case-insensitive) tried as time column.",
    )
    metadata_tokens: list[str] = Field(
        default_factory=lambda: [
            "id", "name", "status", "phase", "unit", "einheit", "type", "typ",
            "label", "comment", "kommentar", "bemerkung", "note", "notiz",
        ],
        description="A header containing one of these tokens is metadata, never a numeric channel.",
    )
    identifier_tokens: list[str] = Field(
        default_factory=lambda: ["id"],
        description="Metadata headers with these tokens and only integer values become Int64 keys.",
    )
    long_variable_tokens: list[str] = Field(
        default_factory=lambda: [
            "roi", "variable", "channel", "kanal", "signal", "sensor",
            "messgroesse", "quantity", "metric", "measurement",
        ],
        description="Tokens of the column holding the variable name in long tables.",
    )
    long_value_tokens: list[str] = Field(
        default_factory=lambda: ["value", "wert", "messwert", "reading"],
        description="Tokens of the column holding the value in long tables.",
    )
    channel_attr_tokens: list[str] = Field(
        default_factory=lambda: ["unit", "einheit", "calculated", "computed", "derived", "berechnet"],
        description="Long-table columns that describe a variable even if they are constant everywhere.",
    )
    run_tokens: list[str] = Field(
        default_factory=lambda: ["run", "lauf", "batch"],
        description="Identifier/categorical columns with these tokens define a run (used for parameter roles).",
    )


class LongFormatMode(str, Enum):
    AUTO = "auto"    # detect from header and data shape
    FORCE = "force"  # always treat as long (variable_col/value_col must be resolvable)
    OFF = "off"      # never pivot, keep the table as it is


class LongFormatConfig(BaseModel):
    """How to treat (time, variable, value) tables. Any field left as None is decided automatically."""

    model_config = ConfigDict(extra="forbid")

    mode: LongFormatMode = LongFormatMode.AUTO
    pivot: bool = Field(True, description="Pivot detected long tables to wide (False keeps the raw long table).")
    variable_col: str | None = None
    value_col: str | None = None
    channel_attr_cols: list[str] | None = Field(
        None, description="Columns stored per channel (e.g. unit). None = decide from data shape and vocabulary."
    )
    row_meta_cols: list[str] | None = Field(
        None, description="Columns kept per timestamp (e.g. run_id). None = decide from data shape."
    )


class RoleConfig(BaseModel):
    """Measurement vs. parameter roles (Datenkatalog P7).

    A numeric column that is constant inside every run is a parameter (setpoint, setting) and not
    a plotted channel. Explicit kinds in `IngestionConfig.column_kinds` always win over this heuristic.
    """

    model_config = ConfigDict(extra="forbid")

    detect_parameters: bool = Field(True, description="Classify numeric columns constant per run as PARAMETER.")
    run_columns: list[str] | None = Field(
        None, description="Columns that together identify a run. None = auto via vocabulary.run_tokens."
    )
    min_samples_per_run: int = Field(
        2, ge=2,
        description="A column counts as constant only if at least one run has this many values of it.",
    )


class RunConfig(BaseModel):
    """Run segmentation settings (Datenkatalog P10, P11, P18). Runs are defined by `RoleConfig.run_columns`."""

    model_config = ConfigDict(extra="forbid")

    aborted_fraction: float = Field(
        0.1, ge=0.0, le=1.0,
        description="A run with fewer samples than this share of the median run is marked aborted. 0 = off.",
    )


class MergeConflictMode(str, Enum):
    NEWEST = "newest"  # the later file wins, every replaced differing value is audited
    ERROR = "error"    # any differing value for the same key raises MergeError


class MergeConfig(BaseModel):
    """Merging several files into one dataset (Datenkatalog P2, P3), see `load_csvs`.

    Byte-identical files are always skipped (and audited). Rows of different files with the same
    key are one sample: the file given later wins. Single-file loads ignore these settings.
    """

    model_config = ConfigDict(extra="forbid")

    key_columns: list[str] | None = Field(
        None,
        description=(
            "Raw headers that identify a sample across files. None = time column, plus the "
            "variable column for long tables."
        ),
    )
    conflict: MergeConflictMode = Field(
        MergeConflictMode.NEWEST,
        description="What to do if files disagree on a value for the same key.",
    )


class IngestionConfig(BaseModel):
    """Every decision the loader makes, as editable and JSON-serializable settings."""

    model_config = ConfigDict(extra="forbid")

    time_column: str | None = Field(None, description="Raw header of the time column. None = auto.")
    clean_column_names: bool = True
    sentinels: list[str] = Field(default_factory=lambda: list(DEFAULT_SENTINELS))
    numeric_ratio_threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Share of parseable values needed for an unclassified column to count as numeric.",
    )
    column_kinds: dict[str, ColumnKind] = Field(
        default_factory=dict,
        description=(
            "Explicit kind per column, keyed by raw or cleaned name (for long sources also by variable "
            "name). Overrides all heuristics."
        ),
    )
    long_format: LongFormatConfig = Field(default_factory=LongFormatConfig)
    roles: RoleConfig = Field(default_factory=RoleConfig)
    runs: RunConfig = Field(default_factory=RunConfig)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    vocabulary: Vocabulary = Field(default_factory=Vocabulary)

    def kind_override(self, raw_name: str, clean_name: str) -> ColumnKind | None:
        """Return the user-set kind for a column, matching raw name first, then cleaned name."""
        return self.column_kinds.get(raw_name) or self.column_kinds.get(clean_name)
