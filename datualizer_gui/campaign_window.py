"""Campaign window: all sources of a measurement campaign on one master time axis (Etappe 2b).

Optional (Settings → Tools → Coverage Info): a coverage strip on top shows when each source has
values; gaps are hatched (step 4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QDockWidget, QFileDialog, QMainWindow, QMessageBox, QStatusBar, QWidget

from datualizer_core.campaign import Campaign, CampaignSession
from datualizer_gui.components.campaign_canvas import CampaignPlotCanvas
from datualizer_gui.components.source_dialog import load_source_interactive
from datualizer_gui.components.sources_panel import SourcesPanel
from datualizer_gui.settings import COVERAGE_INFO, get_bool, set_bool

SOURCE_FILTER = (
    "Measurement Files (*.csv *.tsv *.txt *.dat *.xls *.xlsx *.xlsm *.db *.sqlite *.sqlite3);;All Files (*)"
)


class CampaignWindow(QMainWindow):
    """Sources dock on the left, linked subplots on master time in the center."""

    def __init__(self, session: CampaignSession | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Datualizer – Campaign")
        self.resize(1400, 850)
        self._session = session if session is not None else CampaignSession()
        self._path: Path | None = None

        self.canvas = CampaignPlotCanvas()
        self.setCentralWidget(self.canvas)
        self.sources_panel = SourcesPanel()
        self.dock_sources = QDockWidget("Sources", self)
        self.dock_sources.setWidget(self.sources_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_sources)

        self.sources_panel.selection_changed.connect(self.refresh_plot)
        self.sources_panel.clock_changed.connect(self._on_clock_changed)
        self.sources_panel.add_requested.connect(self._on_add)
        self.sources_panel.remove_requested.connect(self.remove_source)

        menu = self.menuBar().addMenu("&Campaign")
        for text, key, slot in [
            ("&Add Source...", "Ctrl+Shift+A", self._on_add),
            ("&Open Campaign...", "Ctrl+Shift+K", self._on_open),
            ("&Save Campaign...", "Ctrl+Shift+S", self._on_save),
            ("&Reset Zoom", "Ctrl+R", self.canvas.reset_zoom),
        ]:
            action = QAction(text, self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(slot)
            menu.addAction(action)
        settings_menu = self.menuBar().addMenu("S&ettings")
        tools_menu = settings_menu.addMenu("&Tools")
        self.action_coverage = QAction("&Coverage Info", self)
        self.action_coverage.setCheckable(True)
        self.action_coverage.setChecked(get_bool(COVERAGE_INFO))
        self.action_coverage.setToolTip("Strip above the plots: one lane per source, gaps hatched")
        self.action_coverage.toggled.connect(self.set_coverage_visible)
        tools_menu.addAction(self.action_coverage)
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self._refresh_all()

    @property
    def session(self) -> CampaignSession:
        return self._session

    # ------------------------------------------------------------- operations
    def add_sources(self, paths: Sequence[str | Path]) -> None:
        for p in paths:
            try:
                loaded = load_source_interactive(self, p, ask_clock_date=True)
                if loaded is None:  # user cancelled the sheet/table choice
                    continue
                ds, profile = loaded
                src = self._session.add_source(p, profile, dataset=ds)
            except Exception as exc:  # show, do not crash the window
                QMessageBox.critical(self, "Cannot add source", f"{Path(p).name}:\n\n{exc}")
                continue
            self.status_bar.showMessage(
                f"Added {src.label} [profile: {src.profile.name if src.profile else 'none'}]"
            )
        self._refresh_all()

    def remove_source(self, source_id: str) -> None:
        self._session.remove_source(source_id)
        self._refresh_all()

    def open_campaign(self, path: str | Path) -> None:
        self._session = CampaignSession.open(path)
        self._path = Path(path)
        self._refresh_all()
        self.status_bar.showMessage(f"Opened {self._path.name}")

    def save_campaign(self, path: str | Path) -> None:
        self._session.save(path)
        self._path = Path(path)
        self.status_bar.showMessage(f"Saved {self._path.name}")

    @property
    def coverage_visible(self) -> bool:
        return self.action_coverage.isChecked()

    def set_coverage_visible(self, on: bool) -> None:
        """Show or hide the coverage strip; the choice is remembered (Settings → Tools)."""
        if self.action_coverage.isChecked() != on:
            self.action_coverage.setChecked(on)  # emits toggled, which calls this again
            return
        set_bool(COVERAGE_INFO, on)
        self.refresh_plot()

    def refresh_plot(self) -> None:
        tracks = []
        for sid, ch in self.sources_panel.checked_channels():
            if self._session.master_time(sid) is not None:
                tracks.append(self._session.track(sid, ch))
        coverage = (
            [self._session.coverage(src.id) for src in self._session.campaign.sources]
            if self.coverage_visible else []
        )
        self.canvas.set_tracks(tracks, self._session.campaign.timezone, coverage)

    # ----------------------------------------------------------------- internals
    def _refresh_all(self) -> None:
        self.sources_panel.set_session(self._session)
        self.refresh_plot()
        name = self._session.campaign.name
        self.setWindowTitle(f"Datualizer – Campaign: {name}" + (f" ({self._path.name})" if self._path else ""))

    def _on_clock_changed(self, source_id: str) -> None:
        self.sources_panel.set_session(self._session)
        self.sources_panel.select_source(source_id)
        self.refresh_plot()

    def _on_add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Source(s) to Campaign", "", SOURCE_FILTER)
        if paths:
            self.add_sources(paths)

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Campaign", "", "Campaign (*.json);;All Files (*)")
        if path:
            try:
                self.open_campaign(path)
            except Exception as exc:
                QMessageBox.critical(self, "Cannot open campaign", str(exc))

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Campaign", "campaign.json", "Campaign (*.json)")
        if path:
            self.save_campaign(path)


def new_campaign_window(parent: QWidget | None = None) -> CampaignWindow:
    return CampaignWindow(CampaignSession(Campaign()), parent)
