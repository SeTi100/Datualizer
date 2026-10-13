"""How the channels of a campaign are drawn: plot style, y scale and display name (alias).

Stored per source and channel in `campaign.json`. Display only: nothing here changes, cuts or
filters data (AGENTS.md, principle 4). Values outside a y range stay in the track and are
marked at the plot edge.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from datualizer_core.schema import PlotStyle

CONSTANT_PAD = 0.5  # half height of the range shown for a constant channel (in its unit, or 5 % of the value)


class YScaleMode(str, Enum):
    AUTO = "auto"      # the plot fits all visible values
    ROBUST = "robust"  # percentile range, so rare outliers (OCR misreadings) do not flatten the rest
    FIXED = "fixed"    # min and max as given


class YScale(BaseModel):
    """Y range of one channel."""

    model_config = ConfigDict(extra="forbid")

    mode: YScaleMode = YScaleMode.AUTO
    low_percentile: float = Field(1.0, ge=0.0, le=100.0, description="Lower bound of the robust range.")
    high_percentile: float = Field(99.0, ge=0.0, le=100.0, description="Upper bound of the robust range.")
    min: float | None = Field(None, description="Lower bound of the fixed range.")
    max: float | None = Field(None, description="Upper bound of the fixed range.")

    @model_validator(mode="after")
    def _check(self) -> "YScale":
        if self.mode is YScaleMode.FIXED:
            if self.min is None or self.max is None or not self.min < self.max:
                raise ValueError(f"Fixed y range needs min < max, got min={self.min}, max={self.max}.")
        if self.mode is YScaleMode.ROBUST and not self.low_percentile < self.high_percentile:
            raise ValueError(
                f"Robust y range needs low < high percentile, got {self.low_percentile}, {self.high_percentile}."
            )
        return self

    def limits(self, y: np.ndarray) -> tuple[float, float] | None:
        """(low, high) to show for these values; None = auto (the plot fits what is visible)."""
        if self.mode is YScaleMode.FIXED:
            return float(self.min), float(self.max)  # type: ignore[arg-type]
        if self.mode is YScaleMode.AUTO:
            return None
        finite = np.asarray(y, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return None
        lo, hi = (float(v) for v in np.percentile(finite, [self.low_percentile, self.high_percentile]))
        if hi <= lo:  # constant channel: show a band around the value instead of an empty range
            pad = abs(lo) * 0.05 or CONSTANT_PAD
            lo, hi = lo - pad, hi + pad
        return lo, hi


class ChannelDisplay(BaseModel):
    """Display settings of one channel of one source; defaults are not stored in campaign.json."""

    model_config = ConfigDict(extra="forbid")

    alias: str = Field("", description="Display name for axis, tree and readout; the channel name in the data stays.")
    style: PlotStyle | None = Field(None, description="Plot style; None = default of the source profile.")
    y_scale: YScale = Field(default_factory=YScale)
    color: str | None = Field(
        None, pattern=r"^#[0-9a-fA-F]{6}$", description="Line colour '#rrggbb' for the manual colour scheme."
    )

    def is_default(self) -> bool:
        return self == ChannelDisplay()
