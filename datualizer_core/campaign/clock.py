"""Clock models: map a source's own absolute time onto the campaign's master time (Etappe 2b).

Devices of one campaign run on their own clocks (an analyzer 16 min behind the PC, a scale
display 69 min behind, a video without frame time stamps). A ClockModel states how to correct
one source, explicitly and serializable, so the correction is visible and replayable.

Times are handled as epoch seconds (float). Date-times without zone are read in the campaign zone.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from datualizer_core.dataset import DualModeDataset
from datualizer_core.timebase import origin_epoch_us


def to_epoch_s(value: datetime, zone: str) -> float:
    """Epoch seconds of a date-time; without offset it is read in `zone`."""
    return origin_epoch_us(value, zone)[0] / 1e6


def epoch_seconds(ds: DualModeDataset, zone: str) -> np.ndarray | None:
    """Absolute time of every row as epoch seconds (NaN where missing), or None without origin.

    Wall-clock datasets (no zone) are read in the campaign zone `zone`.
    """
    abs_time = ds.time_abs
    if abs_time is None:
        return None
    if abs_time.dtype.time_zone is None:  # type: ignore[union-attr]
        abs_time = abs_time.dt.replace_time_zone(zone, ambiguous="earliest", non_existent="null")
    return (abs_time.dt.epoch("us").cast(pl.Float64) / 1e6).to_numpy().astype(float)


class ClockAnchor(BaseModel):
    """One event seen on both clocks, e.g. a scale display minute change in a video frame."""

    model_config = ConfigDict(extra="forbid")

    source_time: datetime = Field(description="Time of the event on the source's clock.")
    master_time: datetime = Field(description="Time of the same event on the master clock.")
    note: str = ""


class CellPair(BaseModel):
    """Two cells holding the same event on two clocks, e.g. journal 'Start FID' C3 and 'Start Real' C4."""

    model_config = ConfigDict(extra="forbid")

    source_cell: str = Field(description="A1 reference of the time on the source's clock, e.g. 'C3'.")
    master_cell: str = Field(description="A1 reference of the same event on the master clock, e.g. 'C4'.")
    note: str = ""


class ClockCells(BaseModel):
    """Where clock pairs are written down in a workbook of the campaign (e.g. the lab journal).

    Which cells hold which clock differs from journal to journal, so all of it is configured,
    nothing is looked up by label.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="Campaign source whose file holds the cells.")
    sheet: str | None = Field(None, description="Sheet name; None = the only sheet of the file.")
    day: date | None = Field(
        None, description="Date of times written without date (read in the campaign zone)."
    )
    pairs: list[CellPair] = Field(min_length=1)
    resolution_s: float = Field(
        1.0, gt=0.0, description="Reading step of the written times; half of it is the smallest uncertainty."
    )


class ClockModel(BaseModel):
    """Correction from source time to master time.

    - No anchors: master = source + offset_s + drift_ppm * 1e-6 * (source - reference_time).
    - One anchor: a pure offset (`offset_s` and drift are ignored).
    - Two or more anchors: piecewise linear between anchors, linear beyond the outer ones
      (absorbs dropped video frames or a drifting clock).
    """

    model_config = ConfigDict(extra="forbid")

    offset_s: float = Field(0.0, description="Seconds added to the source time.")
    drift_ppm: float = Field(0.0, description="Clock rate error in parts per million.")
    reference_time: datetime | None = Field(
        None, description="Time at which the drift term is zero. None = first sample of the source."
    )
    anchors: list[ClockAnchor] = Field(default_factory=list)
    method: str = Field(
        "none", description="How the model was found: manual, cross_correlation, journal_pairs, anchors, …"
    )
    uncertainty_s: float | None = Field(None, ge=0.0, description="Estimated uncertainty of the mapping.")
    note: str = Field("", description="Assumption behind the model, shown next to results.")
    cells: ClockCells | None = Field(
        None, description="Cell pairs the offset was read from (method 'journal_pairs'), kept for replay."
    )

    @model_validator(mode="after")
    def _distinct_anchors(self) -> "ClockModel":
        times = [a.source_time for a in self.anchors]
        if len(set(times)) != len(times):
            raise ValueError("Clock anchors need distinct source times.")
        return self

    def is_identity(self) -> bool:
        return not self.anchors and self.offset_s == 0.0 and self.drift_ppm == 0.0

    def shifted(self, lag_s: float, **changes) -> "ClockModel":
        """The same mapping moved by `lag_s` seconds on master time (anchors move with it)."""
        update: dict = dict(changes)
        if self.anchors:
            update["anchors"] = [
                a.model_copy(update={"master_time": a.master_time + timedelta(seconds=lag_s)}) for a in self.anchors
            ]
        else:
            update["offset_s"] = self.offset_s + lag_s
        return self.model_copy(update=update)

    def apply(self, source_s: np.ndarray, zone: str) -> np.ndarray:
        """Master epoch seconds for source epoch seconds (NaN stays NaN)."""
        t = np.asarray(source_s, dtype=float)
        if len(self.anchors) == 1:
            a = self.anchors[0]
            return t + (to_epoch_s(a.master_time, zone) - to_epoch_s(a.source_time, zone))
        if len(self.anchors) >= 2:
            pts = sorted((to_epoch_s(a.source_time, zone), to_epoch_s(a.master_time, zone)) for a in self.anchors)
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            out = np.interp(t, xs, ys)
            lo, hi = t < xs[0], t > xs[-1]
            out[lo] = ys[0] + (t[lo] - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
            out[hi] = ys[-1] + (t[hi] - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            out[np.isnan(t)] = np.nan
            return out
        out = t + self.offset_s
        if self.drift_ppm:
            if self.reference_time is not None:
                ref = to_epoch_s(self.reference_time, zone)
            else:
                finite = t[np.isfinite(t)]
                ref = float(finite[0]) if len(finite) else 0.0
            out = out + self.drift_ppm * 1e-6 * (t - ref)
        return out
