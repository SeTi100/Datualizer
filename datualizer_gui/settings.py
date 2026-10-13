"""User preferences of the GUI (Settings menu), stored per user in an INI file via QSettings."""

from __future__ import annotations

from PySide6.QtCore import QSettings

COVERAGE_INFO = "tools/coverage_info"  # show the coverage strip in the campaign window
INTERVAL_BANDS = "tools/interval_bands"  # draw measurement steps as bands in the campaign plots

DEFAULTS = {COVERAGE_INFO: False, INTERVAL_BANDS: True}


def app_settings() -> QSettings:
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "Datualizer", "Datualizer")


def get_bool(key: str) -> bool:
    value = app_settings().value(key, DEFAULTS.get(key, False))
    return value in (True, "true", "1", 1)  # INI files return strings


def set_bool(key: str, value: bool) -> None:
    s = app_settings()
    s.setValue(key, bool(value))
    s.sync()
