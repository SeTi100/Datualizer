"""Time axes of the campaign plot: clock time, date + time, or relative to a reference t0.

The x values are always master epoch seconds; only the tick labels change. `TimeViewBox`
optionally restricts the mouse wheel to the time axis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import numpy as np
import pyqtgraph as pg


class TimeAxisMode(str, Enum):
    AUTO = "auto"          # pyqtgraph's date axis: format follows the zoom level
    CLOCK = "clock"        # time of day only
    DATETIME = "datetime"  # date and time on every tick
    RELATIVE = "relative"  # t − t0


NICE_SPACINGS_S = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
    1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800,
    3600, 2 * 3600, 3 * 3600, 6 * 3600, 12 * 3600, 86400, 2 * 86400, 7 * 86400,
]
LABEL_PADDING_PX = 16


def local_datetime(epoch_s: float, utc_offset_s: int) -> datetime:
    """Wall time for an epoch second with a fixed UTC offset (east positive)."""
    return datetime.fromtimestamp(epoch_s + utc_offset_s, tz=timezone.utc).replace(tzinfo=None)


def absolute_format(mode: TimeAxisMode, spacing_s: float) -> str:
    """strftime format of a tick label; finer spacing shows seconds and milliseconds."""
    clock = "%H:%M:%S.%f" if spacing_s < 1 else "%H:%M:%S" if spacing_s < 60 else "%H:%M"
    return clock if mode is TimeAxisMode.CLOCK else f"%d.%m.%Y {clock}"


def format_absolute(epoch_s: float, utc_offset_s: int, mode: TimeAxisMode, spacing_s: float) -> str:
    fmt = absolute_format(mode, spacing_s)
    text = local_datetime(epoch_s, utc_offset_s).strftime(fmt)
    return text[:-3] if "%f" in fmt else text  # milliseconds


def format_relative(delta_s: float, spacing_s: float = 1.0) -> str:
    """Signed duration: '12.5 s' below a minute, else '[-]H:MM:SS' (with milliseconds for fine ticks)."""
    if not np.isfinite(delta_s):
        return ""
    sign = "-" if delta_s < 0 else ""
    d = abs(delta_s)
    if d < 60 and spacing_s < 60:
        decimals = 0 if spacing_s >= 1 else min(3, int(np.ceil(-np.log10(spacing_s))))
        return f"{sign}{d:.{decimals}f} s"
    whole = int(d)
    h, rest = divmod(whole, 3600)
    m, s = divmod(rest, 60)
    text = f"{sign}{h}:{m:02d}:{s:02d}"
    if spacing_s < 1:
        text += f".{int(round((d - whole) * 1000)):03d}"
    return text


def nice_spacing(span_s: float, max_ticks: int) -> float:
    """Smallest spacing from a 1-2-5 / minute / hour series that gives at most `max_ticks` ticks."""
    target = span_s / max(max_ticks, 1)
    for spacing in NICE_SPACINGS_S:
        if spacing >= target:
            return float(spacing)
    return float(10 ** np.ceil(np.log10(target)))


class AbsoluteTimeAxis(pg.DateAxisItem):
    """Date axis with fixed label formats: clock time or date + time (AUTO = pyqtgraph's own)."""

    def __init__(self, mode: TimeAxisMode, utc_offset_s: int, **kwargs) -> None:
        super().__init__(orientation="bottom", utcOffset=-utc_offset_s, **kwargs)  # pyqtgraph: seconds west
        self.mode = TimeAxisMode(mode)
        self.utc_offset_s = utc_offset_s

    def setZoomLevelForDensity(self, density: float) -> None:
        super().setZoomLevelForDensity(density)
        if self.mode is TimeAxisMode.AUTO:
            return
        # longer labels than pyqtgraph's own: keep them apart
        example = format_absolute(0.0, 0, self.mode, 0.5 if density < 0.01 else 30 if density < 1 else 600)
        self.minSpacing = max(self.minSpacing, density * (self.fontMetrics.boundingRect(example).width() + LABEL_PADDING_PX))

    def tickStrings(self, values, scale, spacing):
        if self.mode is TimeAxisMode.AUTO:
            return super().tickStrings(values, scale, spacing)
        return [format_absolute(v, self.utc_offset_s, self.mode, spacing) for v in values]


class RelativeTimeAxis(pg.AxisItem):
    """Time since a reference t0 (epoch seconds); ticks sit on t0 + k · spacing."""

    def __init__(self, t0: float, **kwargs) -> None:
        super().__init__(orientation="bottom", **kwargs)
        self.mode = TimeAxisMode.RELATIVE
        self.t0 = float(t0)
        self.enableAutoSIPrefix(False)

    def tickSpacing(self, minVal, maxVal, size):
        if maxVal <= minVal or size <= 0:
            return []
        example = format_relative(-(maxVal - minVal) - 3600, 1)
        label_px = self.fontMetrics.boundingRect(example).width() + LABEL_PADDING_PX if hasattr(self, "fontMetrics") else 80
        major = nice_spacing(maxVal - minVal, int(size // max(label_px, 1)))
        return [(major, self.t0 % major)]

    def tickStrings(self, values, scale, spacing):
        return [format_relative(v - self.t0, spacing) for v in values]


def make_time_axis(mode: TimeAxisMode, utc_offset_s: int, t0: float | None) -> pg.AxisItem:
    """Bottom axis for master time in the given mode (relative needs t0)."""
    mode = TimeAxisMode(mode)
    if mode is TimeAxisMode.RELATIVE:
        return RelativeTimeAxis(t0 if t0 is not None else 0.0)
    return AbsoluteTimeAxis(mode, utc_offset_s)


class TimeViewBox(pg.ViewBox):
    """ViewBox whose mouse wheel can be limited to the time (x) axis."""

    def __init__(self, wheel_time_only: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.wheel_time_only = wheel_time_only

    def wheelEvent(self, ev, axis=None):
        if self.wheel_time_only and axis is None:
            axis = 0
        super().wheelEvent(ev, axis)
