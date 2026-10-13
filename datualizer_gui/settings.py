"""User preferences of the GUI (Settings menu), stored per user in an INI file via QSettings.

Only personal preferences live here. What belongs to a campaign (how a channel is drawn, its
time zone and t0) is saved in campaign.json instead.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings

COVERAGE_INFO = "tools/coverage_info"  # show the coverage strip in the campaign window
WHEEL_TIME_ONLY = "view/wheel_time_only"  # mouse wheel zooms the time axis only
Y_FIT_VISIBLE = "view/y_fit_visible"  # auto y range fits the visible time span
TIME_AXIS = "time/axis_mode"  # TimeAxisMode value: auto, clock, datetime, relative
MIN_SUBPLOT_HEIGHT = "layout/min_subplot_height"  # px; below that the plot area scrolls (0 = off)
GROUP_BY = "layout/group_by"  # GroupBy value: channel, unit, source
COLOR_BY = "layout/color_by"  # ColorBy value: channel, source, manual
READOUT_MODE = "readout/mode"  # ReadoutMode value: bar, tooltip, table
READOUT_DECIMALS = "readout/decimals"  # decimals of values; -1 = 4 significant digits

DEFAULTS: dict[str, object] = {
    COVERAGE_INFO: False,
    WHEEL_TIME_ONLY: True,
    Y_FIT_VISIBLE: True,
    TIME_AXIS: "auto",
    MIN_SUBPLOT_HEIGHT: 120,
    GROUP_BY: "channel",
    COLOR_BY: "channel",
    READOUT_MODE: "bar",
    READOUT_DECIMALS: -1,
}


def app_settings() -> QSettings:
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "Datualizer", "Datualizer")


def get_bool(key: str) -> bool:
    value = app_settings().value(key, DEFAULTS.get(key, False))
    return value in (True, "true", "1", 1)  # INI files return strings


def set_bool(key: str, value: bool) -> None:
    set_value(key, bool(value))


def get_int(key: str) -> int:
    default = int(DEFAULTS.get(key, 0))  # type: ignore[arg-type]
    try:
        return int(app_settings().value(key, default))
    except (TypeError, ValueError):  # edited by hand
        return default


def get_str(key: str) -> str:
    return str(app_settings().value(key, DEFAULTS.get(key, "")))


def set_value(key: str, value: object) -> None:
    s = app_settings()
    s.setValue(key, value)
    s.sync()
