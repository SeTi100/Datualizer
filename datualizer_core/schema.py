"""Shared schema vocabulary for Datualizer datasets."""

from __future__ import annotations

from enum import Enum


class ColumnKind(str, Enum):
    """Semantic storage kind of a loaded column."""

    TIME = "time"
    NUMERIC = "numeric"          # Float64 measurement channel, plotted
    IDENTIFIER = "identifier"    # Int64 key such as run_id, not plotted
    CATEGORICAL = "categorical"  # String metadata such as phase_status, not plotted
