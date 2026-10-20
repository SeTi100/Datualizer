"""Campaign window: all sources of a measurement campaign on one master time axis (Etappe 2b).

Optional (Settings → Tools → Coverage Info): a coverage strip on top shows when each source has
values; gaps are hatched (step 4). Measurement steps (step 5) come from a step table (lab journal)
or a right click → Marker ○ ○, are listed in the Steps dock and drawn as bands
(Settings → Tools → Interval Bands, off by default).

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

from datualizer_core.campaign import (
    DEFAULT_WINDOW,
    Campaign,
    CampaignSession,
    IntervalSet,
    TableViewSpec,
    master_datetime,
)
from datualizer_gui.components.campaign_canvas import CampaignPlotCanvas
from datualizer_gui.components.intervals_panel import IntervalsPanel
from datualizer_gui.components.source_dialog import load_source_interactive
from datualizer_gui.components.sources_panel import SourcesPanel
from datualizer_gui.components.export_dialog import ExportDialog, options_from_json, options_to_json
from datualizer_gui.components.plot_export import ExportFormat, ExportOptions, export_plot
from datualizer_gui.components.plot_groups import ColorBy, GroupBy
from datualizer_gui.components.readout import AUTO_DECIMALS, ReadoutMode
from datualizer_gui.components.results_panel import ResultsPanel
from datualizer_gui.components.time_axis import TimeAxisMode
from datualizer_gui.components.window_inspector import WindowInspector
from datualizer_gui.settings import (
    COLOR_BY,
    COVERAGE_INFO,
    EXPORT_DIR,
    EXPORT_OPTIONS,
    GROUP_BY,
    INTERVAL_BANDS,
    MIN_SUBPLOT_HEIGHT,
    READOUT_DECIMALS,
    READOUT_MODE,
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

GROUP_NAMES = {
    GroupBy.CHANNEL: "One Subplot per &Channel",
    GroupBy.UNIT: "Group by &Unit",
    GroupBy.SOURCE: "Group by &Source",
}
COLOR_NAMES = {
    ColorBy.CHANNEL: "Colours by &Channel",
    ColorBy.SOURCE: "Colours by &Source",
    ColorBy.MANUAL: "&Manual Colours (Channel Display)",
}

READOUT_NAMES = {
    ReadoutMode.BAR: "&Bar Above the Plots",
    ReadoutMode.TOOLTIP: "&Tooltip at the Cursor",
    ReadoutMode.TABLE: "T&able in a Dock",
}

SOURCE_FILTER = (
    "Measurement Files (*.csv *.tsv *.txt *.dat *.xls *.xlsx *.xlsm *.db *.sqlite *.sqlite3);;All Files (*)"
)


class CampaignWindow(QMainWindow):
    """Sources and Steps docks on the left, linked subplots on master time in the center."""

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
        self.dock_values = QDockWidget("Values", self)
        self.dock_values.setWidget(self.canvas.readout_table)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_values)
        self.dock_values.hide()
        self.intervals_panel = IntervalsPanel()
        self.dock_steps = QDockWidget("Steps", self)
        self.dock_steps.setWidget(self.intervals_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_steps)
        self.tabifyDockWidget(self.dock_sources, self.dock_steps)
        self.dock_sources.raise_()
        self.window_inspector = WindowInspector()
        self.dock_window = QDockWidget("Window", self)
        self.dock_window.setObjectName("dock_window")
        self.dock_window.setWidget(self.window_inspector)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_window)
        self.dock_window.hide()

        self.results_panel = ResultsPanel()
        self.dock_results = QDockWidget("Results", self)
        self.dock_results.setObjectName("dock_results")
        self.dock_results.setWidget(self.results_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_results)
        self.tabifyDockWidget(self.dock_window, self.dock_results)
        self.dock_results.hide()

        self.sources_panel.selection_changed.connect(self.refresh_plot)
        self.sources_panel.clock_changed.connect(self._on_clock_changed)
        self.sources_panel.view_range = self.canvas.view_x_range
        self.sources_panel.display_changed.connect(self._on_display_changed)
        self.sources_panel.data_changed.connect(lambda sid: self.refresh_plot())
        self.sources_panel.add_requested.connect(self._on_add)
        self.sources_panel.remove_requested.connect(self.remove_source)
        self.intervals_panel.changed.connect(self.refresh_steps)
        self.intervals_panel.zoom_requested.connect(self.zoom_to_step)
        self.intervals_panel.inspect_requested.connect(self.inspect_step)
        self.canvas.interval_inspect_requested.connect(self.inspect_step)
        self.window_inspector.override_changed.connect(self.set_step_window)
        self.window_inspector.override_cleared.connect(self.reset_step_window)
        self.window_inspector.step_changed.connect(self.intervals_panel.select_step)
        self.canvas.interval_marked.connect(self.add_step)
        self.canvas.interval_moved.connect(self.move_step)
        self.canvas.interval_rename_requested.connect(self._on_rename_step)
        self.canvas.interval_delete_requested.connect(self.delete_step)
        self.canvas.mask_requested.connect(
            lambda lo, hi, chans: self.sources_panel.new_mask(start=lo, end=hi, channels=chans)
        )
        self.canvas.mask_edit_requested.connect(self.sources_panel.edit_mask)
        self.canvas.mask_remove_requested.connect(self.sources_panel.remove_mask)
        self._results_window = None
        self.results_panel.step_double_clicked.connect(self.inspect_step)
        self.results_panel.evaluation_changed.connect(self._on_evaluation_changed)
        self.results_panel.open_views_requested.connect(self.open_results_window)

        menu = self.menuBar().addMenu("&Campaign")
        for text, key, slot in [
            ("&Add Source...", "Ctrl+Shift+A", self._on_add),
            ("&Open Campaign...", "Ctrl+Shift+K", self._on_open),
            ("&Save Campaign...", "Ctrl+Shift+S", self._on_save),
            ("&Reset Zoom", "Ctrl+R", self.canvas.reset_zoom),
            ("&Export Plot...", "Ctrl+E", self._on_export),
            ("&Export Report...", "Ctrl+Shift+E", self._on_export_report),
            ("Evaluation from &Template...", "", self._on_apply_recipe),
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

        readout_menu = settings_menu.addMenu("&Values under Cursor")
        self.readout_actions = self._radio_actions(readout_menu, READOUT_NAMES, lambda v: self.set_readout(mode=ReadoutMode(v)))
        readout_menu.addSeparator()
        readout_menu.addAction("&Decimals...", self._on_decimals)

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
        layout_menu = settings_menu.addMenu("&Subplots and Colours")
        self.group_actions = self._radio_actions(layout_menu, GROUP_NAMES, lambda v: self.set_grouping(group_by=GroupBy(v)))
        layout_menu.addSeparator()
        self.color_actions = self._radio_actions(layout_menu, COLOR_NAMES, lambda v: self.set_grouping(color_by=ColorBy(v)))
        settings_menu.addSeparator()
        tools_menu = settings_menu.addMenu("&Tools")
        self.action_coverage = QAction("&Coverage Info", self)
        self.action_coverage.setCheckable(True)
        self.action_coverage.setChecked(get_bool(COVERAGE_INFO))
        self.action_coverage.setToolTip("Strip above the plots: one lane per source, gaps hatched")
        self.action_coverage.toggled.connect(self.set_coverage_visible)
        tools_menu.addAction(self.action_coverage)
        self.action_bands = QAction("&Interval Bands", self)
        self.action_bands.setCheckable(True)
        self.action_bands.setChecked(get_bool(INTERVAL_BANDS))
        self.action_bands.setToolTip("Measurement steps as bands over all plots")
        self.action_bands.toggled.connect(self.set_bands_visible)
        tools_menu.addAction(self.action_bands)
        self.action_results = self.dock_results.toggleViewAction()
        self.action_results.setText("&Results Dock")
        self.action_results.setToolTip("Results table, formulas, criteria and scenarios")
        tools_menu.addAction(self.action_results)
        self.action_results_views = QAction("Results &Views...", self)
        self.action_results_views.setToolTip("Open separate Results Views window with XY plots, matrices and tables")
        self.action_results_views.triggered.connect(self.open_results_window)
        tools_menu.addAction(self.action_results_views)
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
            msg = f"Added {src.label} [profile: {src.profile.name if src.profile else 'none'}]"
            if src.profile is not None and src.profile.intervals is not None:
                # the profile lists steps: use its suggestion, the user can change it (Steps from Source...)
                try:
                    self._session.set_mapping(self._session.suggest_mapping(src.id))
                    msg += f", {len(self._session.intervals().intervals)} steps from it"
                except ValueError as exc:
                    msg += f", steps not read: {exc}"
            self.status_bar.showMessage(msg)
        self._refresh_all()

    def remove_source(self, source_id: str) -> None:
        self._session.remove_source(source_id)
        self._refresh_all()

    def open_campaign(self, path: str | Path) -> None:
        self._session = CampaignSession.open(path)
        self._path = Path(path)
        self.canvas.set_time_axis(self.canvas.time_axis_mode, self._session.time_reference_s)
        self._refresh_all()
        notes = len(self._session.warnings)
        # changed files and step tables are listed in the Sources dock (session.warnings)
        self.status_bar.showMessage(
            f"Opened {self._path.name}" + (f" · {notes} notes on changed files, see Sources" if notes else "")
        )

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
    def _radio_actions(self, menu, names: dict, slot) -> dict:
        """Exclusive checkable actions for an enum; `slot` gets the enum value string."""
        group = QActionGroup(menu)
        actions = {}
        for value, text in names.items():
            action = QAction(text, menu)
            action.setCheckable(True)
            action.setData(value.value)  # Qt returns str enums as plain str
            group.addAction(action)
            menu.addAction(action)
            actions[value] = action
        group.triggered.connect(lambda a: slot(a.data()))
        return actions

    def set_grouping(self, group_by: GroupBy | None = None, color_by: ColorBy | None = None) -> None:
        """Subplots per channel, unit or source and the colour scheme; remembered as preferences."""
        group_by = GroupBy(group_by) if group_by is not None else self._setting_enum(GROUP_BY, GroupBy)
        color_by = ColorBy(color_by) if color_by is not None else self._setting_enum(COLOR_BY, ColorBy)
        set_value(GROUP_BY, group_by.value)
        set_value(COLOR_BY, color_by.value)
        self.group_actions[group_by].setChecked(True)
        self.color_actions[color_by].setChecked(True)
        self.canvas.set_grouping(group_by, color_by)

    def set_readout(self, mode: ReadoutMode | None = None, decimals: int | None = None) -> None:
        """Values under the cursor as bar, tooltip or table; decimals (AUTO_DECIMALS = 4 significant
        digits). Remembered as preferences."""
        mode = ReadoutMode(mode) if mode is not None else self._setting_enum(READOUT_MODE, ReadoutMode)
        decimals = decimals if decimals is not None else get_int(READOUT_DECIMALS)
        set_value(READOUT_MODE, mode.value)
        set_value(READOUT_DECIMALS, decimals)
        self.readout_actions[mode].setChecked(True)
        self.dock_values.setVisible(mode is ReadoutMode.TABLE)
        self.canvas.set_readout(mode, decimals)

    def _on_decimals(self) -> None:
        value, ok = QInputDialog.getInt(
            self, "Decimals", "Decimals of the values (-1 = automatic, 4 significant digits):",
            get_int(READOUT_DECIMALS), AUTO_DECIMALS, 12, 1,
        )
        if ok:
            self.set_readout(decimals=value)

    @staticmethod
    def _setting_enum(key: str, enum):
        try:
            return enum(get_str(key))
        except ValueError:  # edited by hand
            return next(iter(enum))

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
        mode = self._setting_enum(TIME_AXIS, TimeAxisMode)
        self.time_axis_actions[mode].setChecked(True)
        self.canvas.set_time_axis(mode, self._session.time_reference_s)
        self.set_grouping()
        self.set_readout()

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

    @property
    def bands_visible(self) -> bool:
        return self.action_bands.isChecked()

    def set_bands_visible(self, on: bool) -> None:
        """Show or hide the step bands; the choice is remembered (Settings → Tools)."""
        if self.action_bands.isChecked() != on:
            self.action_bands.setChecked(on)
            return
        set_bool(INTERVAL_BANDS, on)
        self.refresh_steps()

    def refresh_plot(self) -> None:
        tracks = []
        for sid, ch in self.sources_panel.checked_channels():
            if self._session.master_time(sid) is not None:
                tracks.append(self._session.track(sid, ch))
        coverage = (
            [self._session.coverage(src.id) for src in self._session.campaign.sources]
            if self.coverage_visible else []
        )
        sources = self._session.campaign.sources
        self.canvas.set_tracks(
            tracks, self._session.campaign.timezone, coverage,
            source_order=[s.id for s in sources], source_labels={s.id: s.label or s.id for s in sources},
        )
        self.canvas.set_masks(self._session.mask_spans())
        self.window_inspector.set_channels(self.sources_panel.checked_channels())
        self.refresh_steps()

    def refresh_steps(self) -> None:
        """Steps changed (mapping, window rule, manual step, clock): update the table and the bands."""
        self.intervals_panel.refresh()
        self.canvas.set_intervals(self._session.intervals() if self.bands_visible else IntervalSet())
        if self.window_inspector.step_id is not None:
            self.window_inspector.refresh()
        self.results_panel.refresh()

    # ------------------------------------------------------------------ steps
    def add_step(self, start: float, end: float) -> None:
        """A step set with the marker (master epoch seconds)."""
        zone = self._session.campaign.timezone
        iv = self._session.add_interval(master_datetime(start, zone), master_datetime(end, zone))
        self.refresh_steps()
        self.status_bar.showMessage(f"Added step '{iv.label}' ({end - start:.0f} s)")

    def move_step(self, interval_id: str, start: float, end: float) -> None:
        zone = self._session.campaign.timezone
        try:
            self._session.update_interval(
                interval_id, start=master_datetime(start, zone), end=master_datetime(end, zone)
            )
        except ValueError as exc:
            self.status_bar.showMessage(f"Step not moved: {exc}")
        self.refresh_steps()

    def rename_step(self, interval_id: str, label: str) -> None:
        self._session.update_interval(interval_id, label=label)
        self.refresh_steps()

    def delete_step(self, interval_id: str) -> None:
        self._session.remove_interval(interval_id)
        self.refresh_steps()

    def inspect_step(self, interval_id: str) -> None:
        """Open the Window dock on a step: statistics of the selected channels in its window."""
        self.window_inspector.set_channels(self.sources_panel.checked_channels())
        self.dock_window.show()
        self.dock_window.raise_()
        self.window_inspector.show_step(interval_id)
        self.window_inspector.setFocus()

    def open_results_window(self) -> Any:
        """Open or raise the Results Views window (tabs for Table, XY and Matrix views)."""
        from datualizer_gui.results_window import ResultsWindow

        if self._results_window is None:
            self._results_window = ResultsWindow(self._session, parent=self)
            self._results_window.step_inspected.connect(self.inspect_step)
        else:
            self._results_window.set_session(self._session)
        self._results_window.show()
        self._results_window.raise_()
        self._results_window.activateWindow()
        return self._results_window

    def set_step_window(self, interval_id: str, start: float, end: float, window: str = DEFAULT_WINDOW) -> None:
        """Window of one step set by hand (master epoch seconds; `window`: which named window); saved with the campaign."""
        try:
            self._session.set_window_override(interval_id, start, end, window)
        except (KeyError, ValueError) as exc:
            self.status_bar.showMessage(f"Window not changed: {exc}")
            return
        self.refresh_steps()
        self.status_bar.showMessage(f"Window of the step set by hand ({end - start:.0f} s, saved with the campaign)")

    def reset_step_window(self, interval_id: str, window: str = DEFAULT_WINDOW) -> None:
        self._session.clear_window_override(interval_id, window)
        self.refresh_steps()
        self.status_bar.showMessage("Window of the step follows the window rule again")

    def zoom_to_step(self, interval_id: str) -> None:
        try:
            self.canvas.zoom_to_step(self._session.intervals().get(interval_id))
        except KeyError:
            pass

    # ----------------------------------------------------------------- internals
    def _refresh_all(self) -> None:
        self.sources_panel.set_session(self._session)
        self.intervals_panel.set_session(self._session)
        self.window_inspector.set_session(self._session)
        self.results_panel.set_session(self._session)
        if self._results_window is not None:
            self._results_window.set_session(self._session)
        self.refresh_plot()
        name = self._session.campaign.name
        self.setWindowTitle(f"Datualizer – Campaign: {name}" + (f" ({self._path.name})" if self._path else ""))

    def apply_recipe(self, path: str | Path) -> list[str]:
        """Take the evaluation of a recipe (examples/rezept_katalyse_voc.json); returns what does not fit yet."""
        notes = self._session.apply_recipe(path)
        self._refresh_all()
        self.status_bar.showMessage(f"Evaluation from {Path(path).name}" + (f" · {len(notes)} notes" if notes else ""))
        return notes

    def _on_apply_recipe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Evaluation from Template", "", "Recipe or campaign (*.json)")
        if not path:
            return
        if self._session.campaign.evaluation.measures and QMessageBox.question(
            self, "Evaluation from Template", "Replace the evaluation of this campaign (sources and steps stay)?"
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            notes = self.apply_recipe(path)
        except Exception as exc:  # show, do not crash the window
            QMessageBox.critical(self, "Evaluation from Template", str(exc))
            return
        if notes:
            QMessageBox.information(self, "Evaluation from Template", "\n".join(notes))

    def _on_evaluation_changed(self) -> None:
        self.refresh_steps()
        self.status_bar.showMessage("Evaluation settings updated")

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
    def _on_rename_step(self, interval_id: str) -> None:
        current = next((iv.label for iv in self._session.campaign.intervals if iv.id == interval_id), "")
        label, ok = QInputDialog.getText(self, "Rename Step", "Name:", text=current)
        if ok and label.strip():
            self.rename_step(interval_id, label.strip())

    def export_plot(self, path: str | Path, options: ExportOptions | None = None) -> Path:
        """Write the plot as shown (time span, y ranges) to PNG, SVG or PDF; the options are remembered."""
        options = options if options is not None else options_from_json(get_str(EXPORT_OPTIONS))
        out = export_plot(self.canvas, path, options)
        set_value(EXPORT_OPTIONS, options_to_json(options))
        set_value(EXPORT_DIR, str(out.resolve().parent))
        self.status_bar.showMessage(f"Exported {out.name}")
        return out

    def _on_export(self) -> None:
        if not self.canvas.all_plots():
            QMessageBox.information(self, "Export Plot", "Select channels or show the coverage strip first.")
            return
        start = str(Path(get_str(EXPORT_DIR) or ".") / f"{self._session.campaign.name}.png")
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export Plot", start, "PNG image (*.png);;SVG vector graphic (*.svg);;PDF document (*.pdf)"
        )
        if not path:
            return
        if Path(path).suffix.lower() not in (".png", ".svg", ".pdf"):
            path += "." + chosen.split("*.")[-1].rstrip(")")
        dialog = ExportDialog(options_from_json(get_str(EXPORT_OPTIONS)), ExportFormat.from_path(path), self)
        if dialog.exec() != ExportDialog.DialogCode.Accepted:
            return
        try:
            self.export_plot(path, dialog.options())
        except Exception as exc:  # show, do not crash the window
            QMessageBox.critical(self, "Export Plot", str(exc))

    def _on_export_report(self) -> None:
        start = str(Path(get_str(EXPORT_DIR) or ".") / f"{self._session.campaign.name}_report.pdf")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", start, "PDF document (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            out = self.export_report(path)
            QMessageBox.information(self, "Export Report", f"Report saved to:\n{out}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Report", str(exc))

    def export_report(self, path: str | Path) -> Path:
        """Export the campaign report as multi-page PDF.

        The first report view (`kind: "report"`) sets which views appear, in which order, and the
        title and subtitle; without one, every view is a page.
        """
        from datualizer_gui.results_window import ResultsWindow, report_views
        from datualizer_gui.components.plot_export import export_report

        if self._results_window is None:
            self._results_window = ResultsWindow(session=self._session, parent=self)
        self._results_window.refresh()

        ev = self._session.campaign.evaluation
        views = ev.views if ev and ev.views else []
        report = next((v for v in views if v.kind == "report"), None)
        title, subtitle = self._session.campaign.name, None
        if report is not None:
            views = report_views(views, report)
            title, subtitle = report.title or title, report.subtitle
        if not views:
            views = [TableViewSpec(name="results", title="Results Table")]

        out = export_report(
            session=self._session,
            views=views,
            view_widgets=self._results_window._view_widgets,
            path=path,
            options=ExportOptions(width_mm=297, height_mm=210, dpi=300),
            cover=True if report is None else report.cover,
            title=title,
            subtitle=subtitle,
            tiles=report.tiles if report is not None else (),
        )
        self.status_bar.showMessage(f"Exported report {Path(out).name}")
        return Path(out)

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
