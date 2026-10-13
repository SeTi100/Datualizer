"""Campaign window: all sources of a measurement campaign on one master time axis (Etappe 2b).

Optional (Settings → Tools → Coverage Info): a coverage strip on top shows when each source has
values; gaps are hatched (step 4).

Settings come in two levels: how a channel is drawn (style, y scale, alias) belongs to the
campaign and is saved in campaign.json; personal preferences are kept in QSettings
(`datualizer_gui.settings`). Both are display only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)

from datualizer_core.campaign import Campaign, CampaignSession
from datualizer_gui.components.campaign_canvas import CampaignPlotCanvas
from datualizer_gui.components.source_dialog import load_source_interactive
from datualizer_gui.components.sources_panel import SourcesPanel
from datualizer_gui.components.time_axis import TimeAxisMode
from datualizer_gui.settings import (
    COVERAGE_INFO,
    MIN_SUBPLOT_HEIGHT,
    TIME_AXIS,
    WHEEL_TIME_ONLY,
    Y_FIT_VISIBLE,
    get_bool,
    get_int,
    get_str,
    set_bool,
    set_value,
)

TIME_AXIS_NAMES = {
    TimeAxisMode.AUTO: "&Automatic",
    TimeAxisMode.CLOCK: "&Clock Time",
    TimeAxisMode.DATETIME: "&Date + Time",
    TimeAxisMode.RELATIVE: "&Relative (t \u2212 t0)",
}

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
        self.sources_panel.display_changed.connect(self._on_display_changed)
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
        self.action_channel_display = QAction("Channel &Display...", self)
        self.action_channel_display.setShortcut(QKeySequence("Ctrl+D"))
        self.action_channel_display.setToolTip("Display name, plot style and y scale of the selected channel")
        self.action_channel_display.triggered.connect(self._on_channel_display)
        settings_menu.addAction(self.action_channel_display)
        settings_menu.addSeparator()

        time_menu = settings_menu.addMenu("&Time Axis")
        group = QActionGroup(time_menu)
        self.time_axis_actions: dict[TimeAxisMode, QAction] = {}
        for mode, text in TIME_AXIS_NAMES.items():
            action = QAction(text, time_menu)
            action.setCheckable(True)
            action.setData(mode.value)  # Qt returns str enums as plain str
            group.addAction(action)
            time_menu.addAction(action)
            self.time_axis_actions[mode] = action
        group.triggered.connect(lambda a: self.set_time_axis_mode(TimeAxisMode(a.data())))
        time_menu.addSeparator()
        self.action_t0_here = time_menu.addAction("Set t0 at &Crosshair", self._on_t0_at_crosshair)
        self.action_t0_here.setToolTip("Reference of the relative axis; saved with the campaign")
        time_menu.addAction("&Reset t0 (Start of Data)", lambda: self.set_time_reference(None))
        settings_menu.addAction("Campaign Time &Zone...", self._on_time_zone)
        settings_menu.addSeparator()

        self.action_wheel_time = QAction("Mouse &Wheel Zooms Time Only", self)
        self.action_wheel_time.setToolTip("Over a y axis the wheel still zooms that axis")
        self.action_y_fit = QAction("&Fit Y to Visible Time Span", self)
        self.action_y_fit.setToolTip("Auto-scaled subplots fit the values in view (not fixed or robust ranges)")
        for action, key in ((self.action_wheel_time, WHEEL_TIME_ONLY), (self.action_y_fit, Y_FIT_VISIBLE)):
            action.setCheckable(True)
            action.setChecked(get_bool(key))
            action.toggled.connect(lambda on, key=key: self._on_view_option(key, on))
            settings_menu.addAction(action)
        settings_menu.addAction("Minimum Subplot &Height...", self._on_min_height)
        settings_menu.addSeparator()
        tools_menu = settings_menu.addMenu("&Tools")
        self.action_coverage = QAction("&Coverage Info", self)
        self.action_coverage.setCheckable(True)
        self.action_coverage.setChecked(get_bool(COVERAGE_INFO))
        self.action_coverage.setToolTip("Strip above the plots: one lane per source, gaps hatched")
        self.action_coverage.toggled.connect(self.set_coverage_visible)
        tools_menu.addAction(self.action_coverage)
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self._apply_view_settings()
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
        self.canvas.set_time_axis(self.canvas.time_axis_mode, self._session.time_reference_s)
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

    # ----------------------------------------------------------- time and view
    def set_time_axis_mode(self, mode: TimeAxisMode) -> None:
        """Clock time, date + time or relative to t0; remembered as a personal preference."""
        mode = TimeAxisMode(mode)
        set_value(TIME_AXIS, mode.value)
        self.time_axis_actions[mode].setChecked(True)
        self.canvas.set_time_axis(mode, self._session.time_reference_s)

    def set_time_reference(self, epoch_s: float | None) -> None:
        """t0 of the relative axis (None = start of the data shown); saved with the campaign."""
        self._session.set_time_reference(epoch_s)
        if self.canvas.time_axis_mode is not TimeAxisMode.RELATIVE:
            self.set_time_axis_mode(TimeAxisMode.RELATIVE)
        else:
            self.canvas.set_time_axis(TimeAxisMode.RELATIVE, self._session.time_reference_s)
        self.status_bar.showMessage(
            "t0 = start of the data shown" if epoch_s is None else f"t0 = {self.canvas.format_time(epoch_s)}"
        )

    def set_timezone(self, zone: str) -> None:
        """Master zone of the campaign (raises TimeConfigError for an unknown zone)."""
        self._session.set_timezone(zone)
        self._refresh_all()
        self.status_bar.showMessage(f"Campaign time zone: {zone}")

    def _apply_view_settings(self) -> None:
        self.canvas.set_view_options(get_bool(WHEEL_TIME_ONLY), get_bool(Y_FIT_VISIBLE))
        self.canvas.set_min_subplot_height(get_int(MIN_SUBPLOT_HEIGHT))
        try:
            mode = TimeAxisMode(get_str(TIME_AXIS))
        except ValueError:  # edited by hand
            mode = TimeAxisMode.AUTO
        self.time_axis_actions[mode].setChecked(True)
        self.canvas.set_time_axis(mode, self._session.time_reference_s)

    def _on_view_option(self, key: str, on: bool) -> None:
        set_bool(key, on)
        self.canvas.set_view_options(self.action_wheel_time.isChecked(), self.action_y_fit.isChecked())

    def _on_t0_at_crosshair(self) -> None:
        x = self.canvas.crosshair_x
        if x is None:
            QMessageBox.information(self, "Set t0", "Move the mouse over the plot first.")
            return
        self.set_time_reference(x)

    def _on_time_zone(self) -> None:
        from zoneinfo import available_timezones

        try:
            zones = sorted(available_timezones())
        except Exception:  # no zone database
            zones = []
        current = self._session.campaign.timezone
        if current not in zones:
            zones.insert(0, current)
        zone, ok = QInputDialog.getItem(
            self, "Campaign Time Zone",
            "Master zone of the axis and readout.\nSources without their own zone are read in it.",
            zones, zones.index(current), True,
        )
        if ok and zone and zone != current:
            try:
                self.set_timezone(zone)
            except Exception as exc:
                QMessageBox.critical(self, "Campaign Time Zone", str(exc))

    def _on_min_height(self) -> None:
        px, ok = QInputDialog.getInt(
            self, "Minimum Subplot Height", "Height in pixels (0 = fit all subplots, no scrolling):",
            get_int(MIN_SUBPLOT_HEIGHT), 0, 2000, 10,
        )
        if ok:
            set_value(MIN_SUBPLOT_HEIGHT, px)
            self.canvas.set_min_subplot_height(px)

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

    def _on_display_changed(self, source_id: str, channel: str) -> None:
        self.sources_panel.set_session(self._session)  # aliases in the tree
        self.refresh_plot()
        self.status_bar.showMessage(f"Display of {channel} changed (saved with the campaign)")

    def _on_channel_display(self) -> None:
        current = self.sources_panel.current_channel()
        if current is None:
            QMessageBox.information(self, "Channel Display", "Select a channel in the source list first.")
            return
        self.sources_panel.edit_display(*current)

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
