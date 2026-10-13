"""Shared schema vocabulary for Datualizer datasets."""

from __future__ import annotations

from enum import Enum


class ColumnKind(str, Enum):
    """Semantic storage kind of a loaded column."""

    TIME = "time"
    NUMERIC = "numeric"          # Float64 measurement channel, plotted
    PARAMETER = "parameter"      # Float64 setpoint/setting, constant per run, not plotted
    IDENTIFIER = "identifier"    # Int64 key such as run_id, not plotted
    CATEGORICAL = "categorical"  # String metadata such as phase_status, not plotted


class PlotStyle(str, Enum):
    """How a channel is drawn (display only). Shared by source profiles and campaign display settings."""

    LINE = "line"                # samples joined by straight lines
    POINTS = "points"            # samples only, no line
    STEP = "step"                # a value holds until the next sample (measurement steps)
    LINE_POINTS = "line_points"  # line with a marker on every sample
