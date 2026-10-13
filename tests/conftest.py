"""Shared test setup."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_gui_settings(tmp_path_factory: pytest.TempPathFactory) -> None:
    """GUI preferences go to a fresh folder per test, never to the user's real settings."""
    from PySide6.QtCore import QSettings

    folder: Path = tmp_path_factory.mktemp("settings")
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(folder))
