"""Application entry point for Datualizer Desktop GUI."""

from __future__ import annotations

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from datualizer_gui.main_window import DatualizerMainWindow, apply_modern_theme


def create_app() -> tuple[QApplication, DatualizerMainWindow]:
    """Create and configure QApplication and main window instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("Datualizer")
    app.setOrganizationName("Datualizer")
    app.setStyle("Fusion")
    apply_modern_theme(app)

    main_win = DatualizerMainWindow(auto_load=True)
    return app, main_win


def main() -> None:
    """Launch the Datualizer desktop GUI application."""
    app, main_win = create_app()
    main_win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
