"""Plot canvas for a campaign: tracks in subplots (per channel, unit or source), each with its own x array on master time."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
import weakref

import numpy as np
import polars as pl
import pyqtgraph as pg
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QScrollArea, QToolTip, QVBoxLayout, QWidget

from datualizer_core.campaign import IntervalSet, MaskSpan, ResolvedInterval, SourceCoverage, Track
from datualizer_gui.components import curves
from datualizer_gui.components.curves import (
    FastCurve,
    OutOfRangeMarkers,
    add_curve,
    add_edge_markers,
    curve_pen,
    ensure_pixmap_cache,
    light_grid,
    rate_limited_mouse,
)
from datualizer_gui.components.coverage_strip import (
    LEFT_AXIS_MIN_PX,
    CoverageItem,
    build_coverage_plot,
    describe_gap,
    lane_label,
    left_axis_width,
)
from datualizer_gui.components.interval_bands import PENDING_PEN, StepBand, StepsItem, marker_icon, marker_text
from datualizer_gui.components.mask_bands import MasksItem, mask_text, masks_for
from datualizer_gui.components.plot_canvas import CHANNEL_COLORS
from datualizer_gui.components.plot_groups import ColorBy, GroupBy, PlotGroup, group_tracks, track_colors, track_key
from datualizer_gui.components.readout import VALUE_WIDTH, ReadoutLabel, ReadoutMode, ReadoutTable, format_value
from datualizer_gui.components.time_axis import TimeAxisMode, TimeViewBox, format_relative, make_time_axis


def utc_offset_s(epoch_s: float, zone: str) -> int:
    """Offset of `zone` from UTC at an instant, in seconds (east positive)."""
    s = pl.Series([int(epoch_s * 1e6)]).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    local = s.dt.convert_time_zone(zone).dt.replace_time_zone(None)
    return int((local.dt.epoch("us")[0] - int(epoch_s * 1e6)) / 1e6)


def group_limits(tracks: Sequence[Track]) -> tuple[float, float] | None:
    """Y range of a subplot: the union of its tracks' fixed or robust ranges; auto (None) as
    soon as one track is auto-scaled."""
    limits = [tr.y_scale.limits(tr.y) for tr in tracks]
    if not limits or any(lim is None for lim in limits):
        return None
    return min(lo for lo, _ in limits), max(hi for _, hi in limits)  # type: ignore[misc]


def connect_mask(values: np.ndarray, gap_rows: np.ndarray) -> np.ndarray:
    """Connect i to i+1 only if both are finite and no time gap starts at i+1."""
    return curves.connect_mask(values, gap_rows)


class CampaignPlotCanvas(QWidget):
    """Coverage strip on top, stacked subplots below, all with linked x axes on master time.

    Unlike the dataset canvas, every track keeps its own x array, so sources with different
    sampling and clocks share one time axis without resampling.

    Measurement steps are drawn as bands over all subplots. Right click → "Marker ○ ○" sets a
    step by hand: the first click its start (first circle green), the second its end.
    Masked spans are hatched in the subplots of the channels they concern; right click →
    "Mask Range..." asks to mask the started marker's span or the visible time span.
    """

    interval_marked = Signal(float, float)  # master start, end of a step set with the marker
    interval_moved = Signal(str, float, float)  # manual step dragged to a new start, end
    interval_rename_requested = Signal(str)
    interval_delete_requested = Signal(str)
    interval_inspect_requested = Signal(str)  # open the window inspector on this step
    mask_requested = Signal(float, float, object)  # master start, end, [(source, channel)] of the subplot
    mask_edit_requested = Signal(str, int)  # source id, index of the mask in the source
    mask_remove_requested = Signal(str, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tracks: list[Track] = []
        self._plot_items: dict[str, pg.PlotItem] = {}
        self._curves: dict[str, FastCurve] = {}
        self._crosshair_lines: dict[str, pg.InfiniteLine] = {}
        self._edge_markers: dict[str, OutOfRangeMarkers] = {}  # tracks with a fixed or robust y range
        self._y_limits: dict[str, tuple[float, float]] = {}  # per subplot
        self._groups: list[PlotGroup] = []
        self._group_tracks: dict[str, tuple[Track, ...]] = {}  # tracks each kept subplot was drawn from
        self._colors: dict[str, str] = {}  # per track key
        self._group_by = GroupBy.CHANNEL
        self._color_by = ColorBy.CHANNEL
        self._source_order: list[str] = []
        self._source_labels: dict[str, str] = {}
        self._coverage: list[SourceCoverage] = []
        self.coverage_plot: pg.PlotItem | None = None
        self.coverage_item: CoverageItem | None = None
        self._coverage_line: pg.InfiniteLine | None = None
        self._zone = "UTC"
        self._utc_offset = 0
        self._user_range: tuple[float, float] | None = None  # set when the user zooms or pans
        self._axis_mode = TimeAxisMode.AUTO
        self._t0_setting: float | None = None  # reference for relative time; None = first sample
        self._t0: float | None = None
        self._wheel_time_only = False
        self._y_fit_visible = False
        self._min_subplot_height = 0
        self.crosshair_x: float | None = None  # master time of the last crosshair position
        self._intervals = IntervalSet()
        self._steps_items: list[tuple[pg.PlotItem, StepsItem]] = []  # one per subplot
        self._bands: list[StepBand] = []  # draggable regions of manual steps only
        self._marker_start: float | None = None  # master time of a started, unfinished marker
        self._pending_lines: list[tuple[pg.PlotItem, pg.InfiniteLine]] = []
        self._menu_x: float | None = None  # master time of the last right click
        self._menu_step: ResolvedInterval | None = None  # step under the last right click
        self._masks: list[MaskSpan] = []
        self._masks_items: list[tuple[pg.PlotItem, MasksItem]] = []  # subplots with masks
        self._menu_key: str | None = None  # subplot of the last right click (None: coverage strip)
        self._menu_mask: MaskSpan | None = None  # mask under the last right click

        self.action_marker = QAction(marker_icon(0), marker_text(0), self)
        self.action_marker.setToolTip("First click: start of a step, second click: its end")
        self.action_marker.triggered.connect(self._on_marker)
        self.action_marker_cancel = QAction("Discard marker", self)
        self.action_marker_cancel.triggered.connect(self.cancel_marker)
        self.action_inspect_step = QAction("Inspect step", self)
        self.action_inspect_step.setToolTip("Statistics of the step's window in the Window dock")
        self.action_inspect_step.triggered.connect(
            lambda: self._menu_step and self.interval_inspect_requested.emit(self._menu_step.id)
        )
        self.action_zoom_step = QAction("Zoom to step", self)
        self.action_zoom_step.triggered.connect(lambda: self._menu_step and self.zoom_to_step(self._menu_step))
        self.action_rename_step = QAction("Rename step...", self)
        self.action_rename_step.triggered.connect(
            lambda: self._menu_step and self.interval_rename_requested.emit(self._menu_step.id)
        )
        self.action_delete_step = QAction("Delete step", self)
        self.action_delete_step.triggered.connect(
            lambda: self._menu_step and self.interval_delete_requested.emit(self._menu_step.id)
        )
        self.action_mask_range = QAction("Mask Range...", self)
        self.action_mask_range.setToolTip(
            "Flag a time span as masked (left out of the statistics): from the started marker to here, "
            "else the visible time span"
        )
        self.action_mask_range.triggered.connect(self._on_mask_range)
        self.action_edit_mask = QAction("Edit Mask...", self)
        self.action_edit_mask.triggered.connect(
            lambda: self._menu_mask and self.mask_edit_requested.emit(self._menu_mask.source_id, self._menu_mask.index)
        )
        self.action_remove_mask = QAction("Remove Mask", self)
        self.action_remove_mask.triggered.connect(
            lambda: self._menu_mask and self.mask_remove_requested.emit(self._menu_mask.source_id, self._menu_mask.index)
        )
        self._step_actions = [
            self.action_marker, self.action_marker_cancel, self.action_inspect_step, self.action_zoom_step,
            self.action_rename_step, self.action_delete_step,
            self.action_mask_range, self.action_edit_mask, self.action_remove_mask,
        ]
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        esc.activated.connect(self.cancel_marker)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.crosshair_label = ReadoutLabel("Add sources and select channels to plot them on one time axis.")
        layout.addWidget(self.crosshair_label)
        self.readout_table = ReadoutTable()  # shown by the window in a dock (readout mode TABLE)
        self._readout_mode = ReadoutMode.BAR
        self._decimals: int | None = None  # None: 4 significant digits
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground("#18181c")
        ensure_pixmap_cache()
        self._mouse_proxy = rate_limited_mouse(self.graphics_layout.scene(), self._on_mouse_moved)
        self.scroll_area = QScrollArea()  # scrolls once the subplots reach their minimum height
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.setWidget(self.graphics_layout)
        layout.addWidget(self.scroll_area)

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    @property
    def plot_items(self) -> dict[str, pg.PlotItem]:
        return self._plot_items

    @property
    def curves(self) -> dict[str, FastCurve]:
        return self._curves

    @property
    def edge_markers(self) -> dict[str, OutOfRangeMarkers]:
        """Edge markers per track key (tracks in a subplot with a fixed or robust y range)."""
        return self._edge_markers

    @property
    def groups(self) -> list[PlotGroup]:
        return list(self._groups)

    @property
    def colors(self) -> dict[str, str]:
        """Colour per track key."""
        return dict(self._colors)

    def set_grouping(self, group_by: GroupBy, color_by: ColorBy) -> None:
        """Subplot per channel, per unit or per source; colours by channel, by source or manual."""
        group_by, color_by = GroupBy(group_by), ColorBy(color_by)
        if (group_by, color_by) == (self._group_by, self._color_by):
            return
        self._group_by, self._color_by = group_by, color_by
        self.set_tracks(self._tracks, self._zone, self._coverage)

    @property
    def coverage(self) -> list[SourceCoverage]:
        return list(self._coverage)

    def set_tracks(
        self,
        tracks: Sequence[Track],
        zone: str = "UTC",
        coverage: Sequence[SourceCoverage] = (),
        source_order: Sequence[str] | None = None,
        source_labels: dict[str, str] | None = None,
    ) -> None:
        """Show the coverage of these sources and these tracks on one time axis.

        A range the user zoomed to is kept, otherwise everything is shown. Subplots whose tracks
        are still the same Track objects are kept with their drawn curves, so ticking one more
        channel does not redraw all the others. `source_order` (the campaign's sources) gives
        each source its colour; `source_labels` names source subplots.
        """
        self._tracks = list(tracks)
        self._coverage = list(coverage)
        if source_order is not None:
            self._source_order = list(source_order)
        if source_labels is not None:
            self._source_labels = dict(source_labels)
        starts = [float(np.nanmin(t.x)) for t in self._tracks if np.isfinite(t.x).any()]
        starts += [c.span[0] for c in self._coverage if c.span is not None]
        offset = utc_offset_s(min(starts), zone) if starts else 0
        t0 = self._t0_setting if self._t0_setting is not None else (min(starts) if starts else None)
        if self._axis_mode is not TimeAxisMode.RELATIVE:
            t0 = None  # only the relative axis depends on it: no redraw when the data start moves
        old_plots, old_curves, old_lines, old_groups = self._plot_items, self._curves, self._crosshair_lines, self._group_tracks
        old_markers = self._edge_markers
        if (offset, zone, t0) != (self._utc_offset, self._zone, self._t0):
            old_groups = {}  # axis labels depend on offset and t0: build all subplots anew
        self._zone, self._utc_offset, self._t0 = zone, offset, t0
        for p in old_plots.values():  # unlink first: the old coverage strip is deleted below
            p.setXLink(None)
        self.graphics_layout.clear()  # detaches the subplots, kept ones are added back below
        self._plot_items, self._curves, self._crosshair_lines, self._group_tracks = {}, {}, {}, {}
        self._edge_markers, self._y_limits = {}, {}
        self.coverage_plot = self.coverage_item = self._coverage_line = None
        self._groups = group_tracks(self._tracks, self._group_by, self._source_labels)
        self._colors = track_colors(self._tracks, self._color_by, CHANNEL_COLORS, self._source_order)

        first: pg.PlotItem | None = None
        if self._coverage:  # rebuilt every time: cheap, and the lanes follow the source list
            first, self.coverage_item = build_coverage_plot(
                self.graphics_layout, self._coverage, self._lane_colors(), self._utc_offset, axis=self._make_axis()
            )
            first.getAxis("bottom").setStyle(showValues=not self._tracks)
            first.getViewBox().sigRangeChangedManually.connect(self._on_manual_range)
            self._coverage_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen("#888888", style=pg.QtCore.Qt.DashLine)
            )
            first.addItem(self._coverage_line, ignoreBounds=True)
            self.coverage_plot = first
        for i, group in enumerate(self._groups):
            gkey = group.key
            row = i + 1  # row 0 belongs to the coverage strip
            limits = group_limits(group.tracks)
            if limits is not None:
                self._y_limits[gkey] = limits
            kept = old_groups.get(gkey)
            if kept is not None and len(kept) == len(group.tracks) and all(a is b for a, b in zip(kept, group.tracks)):
                p, line = old_plots[gkey], old_lines[gkey]
                self.graphics_layout.addItem(p, row=row, col=0)
                for tr in group.tracks:
                    key = track_key(tr)
                    pen = curve_pen(self._colors[key])
                    self._curves[key] = old_curves[key]
                    self._curves[key].set_pen(pen)
                    if key in old_markers:
                        old_markers[key].setPen(pen.color())
                        old_markers[key].setBrush(pen.color())
                        self._edge_markers[key] = old_markers[key]
            else:
                p, line = self._new_plot(row, group, limits is not None)
            p.setXLink(first)  # None for the first subplot
            if first is None:
                first = p
            self._plot_items[gkey], self._crosshair_lines[gkey] = p, line
            self._group_tracks[gkey] = tuple(group.tracks)
        width = LEFT_AXIS_MIN_PX
        if self.coverage_plot is not None:
            width = left_axis_width(self.coverage_plot, [lane_label(c) for c in self._coverage])
        for p in self._all_plots():
            p.getAxis("left").setWidth(width)
        self._apply_view_options()
        self._apply_min_height()
        if first is not None:
            self._apply_range()
        self._apply_y_ranges()
        for p in self._all_plots():
            self._install_step_menu(p)
        self._draw_bands()
        self._draw_masks()
        self._draw_pending()
        if not self._tracks and not self._coverage:
            self.crosshair_label.setText("Add sources and select channels to plot them on one time axis.")

    # ------------------------------------------------------------ view options
    @property
    def time_axis_mode(self) -> TimeAxisMode:
        return self._axis_mode

    @property
    def t0(self) -> float | None:
        """Reference of the relative time axis (the one set, else the first sample shown)."""
        return self._t0

    def set_time_axis(self, mode: TimeAxisMode, t0: float | None = None) -> None:
        """Label the time axis as clock time, date + time or t - t0 (t0 None = first sample)."""
        mode = TimeAxisMode(mode)
        if (mode, t0) == (self._axis_mode, self._t0_setting):
            return
        self._axis_mode, self._t0_setting = mode, t0
        self._rebuild()

    def set_view_options(self, wheel_time_only: bool, y_fit_visible: bool) -> None:
        """Mouse wheel on the time axis only; y of auto-scaled subplots fits the visible part."""
        self._wheel_time_only, self._y_fit_visible = wheel_time_only, y_fit_visible
        self._apply_view_options()

    def set_min_subplot_height(self, px: int) -> None:
        """Subplots never get lower than this (0 = fit all); the plot area scrolls instead."""
        self._min_subplot_height = max(int(px), 0)
        self._apply_min_height()

    def _rebuild(self) -> None:
        self._group_tracks = {}  # nothing to keep: the axes change
        self.set_tracks(self._tracks, self._zone, self._coverage)

    def _make_axis(self) -> pg.AxisItem:
        return make_time_axis(self._axis_mode, self._utc_offset, self._t0)

    def _apply_view_options(self) -> None:
        for p in self._all_plots():
            vb = p.getViewBox()
            if isinstance(vb, TimeViewBox):
                vb.wheel_time_only = self._wheel_time_only
            if p is not self.coverage_plot:
                p.setAutoVisible(y=self._y_fit_visible)

    def _apply_min_height(self) -> None:
        rows = len(self._plot_items)
        height = rows * self._min_subplot_height
        if self.coverage_plot is not None and rows:
            height += int(self.coverage_plot.minimumHeight())
        self.graphics_layout.setMinimumHeight(height if self._min_subplot_height else 0)
        self._update_time_labels()

    @property
    def scrolls(self) -> bool:
        """True if the subplots are taller than the visible area (then each shows its time labels)."""
        return self.graphics_layout.minimumHeight() > self.scroll_area.viewport().height()

    def _update_time_labels(self) -> None:
        """Time labels under the last subplot only, or under every subplot while scrolling."""
        plots = list(self._plot_items.values())
        every = self.scrolls
        for i, p in enumerate(plots):
            p.getAxis("bottom").setStyle(showValues=every or i == len(plots) - 1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_time_labels()

    # ----------------------------------------------------------------- export
    def view_x_range(self) -> tuple[float, float] | None:
        """Master time range currently shown."""
        first = self._master_plot()
        return tuple(first.getViewBox().viewRange()[0]) if first is not None else None  # type: ignore[return-value]

    def export_copy(self) -> "CampaignPlotCanvas":
        """A new canvas with the same tracks, subplots, colours, time axis and x/y ranges,
        with the step bands but without crosshair, for rendering into a file (see `plot_export`). Not shown."""
        c = CampaignPlotCanvas()
        c._axis_mode, c._t0_setting = self._axis_mode, self._t0_setting
        c._group_by, c._color_by = self._group_by, self._color_by
        c.set_tracks(self._tracks, self._zone, self._coverage, self._source_order, self._source_labels)
        rng = self.view_x_range()
        if rng is not None:
            c.set_view_range(*rng)
        for key, p in c._plot_items.items():
            shown = self._plot_items.get(key)
            if shown is not None:
                p.enableAutoRange(axis="y", enable=False)
                p.setYRange(*shown.getViewBox().viewRange()[1], padding=0)
        c.set_intervals(self._intervals)  # step bands as shown (empty if switched off)
        c.set_masks(self._masks)
        for line in [*c._crosshair_lines.values(), c._coverage_line]:
            if line is not None:
                line.hide()
        return c

    def all_plots(self) -> list[pg.PlotItem]:
        """Coverage strip (if shown) and all subplots, top to bottom."""
        return self._all_plots()

    # ------------------------------------------------------------------ steps
    @property
    def intervals(self) -> IntervalSet:
        return self._intervals

    @property
    def bands(self) -> list[StepBand]:
        """Draggable regions of the steps set by hand (one per step and subplot)."""
        return list(self._bands)

    @property
    def steps_items(self) -> list[StepsItem]:
        """The items painting all steps, one per subplot."""
        return [item for _, item in self._steps_items]

    @property
    def marker_state(self) -> int:
        """0: no marker started, 1: start set and waiting for the end."""
        return 0 if self._marker_start is None else 1

    def set_intervals(self, intervals: IntervalSet) -> None:
        """Draw these steps as bands over all subplots."""
        self._intervals = intervals
        self._draw_bands()

    # ------------------------------------------------------------------ masks
    @property
    def masks(self) -> list[MaskSpan]:
        return list(self._masks)

    @property
    def masks_items(self) -> dict[str, MasksItem]:
        """The items painting the masks, per subplot key (only subplots with masks)."""
        keys = {id(p): k for k, p in self._plot_items.items()}
        return {keys[id(p)]: item for p, item in self._masks_items if id(p) in keys}

    def set_masks(self, masks: Sequence[MaskSpan]) -> None:
        """Hatch these masks in the subplots of the channels they concern."""
        self._masks = list(masks)
        self._draw_masks()

    def masks_at(self, x: float, key: str | None = None) -> list[MaskSpan]:
        """Masks at master time `x` that concern the subplot `key` (default: any track shown)."""
        tracks = self._group_tracks.get(key, ()) if key is not None else self._tracks
        return [m for m in masks_for(self._masks, tracks) if m.start <= x <= m.end]

    def request_mask(self, x: float, key: str | None = None) -> None:
        """What "Mask Range..." does after a right click at master time `x` in subplot `key`."""
        if self._marker_start is not None:
            start, end = sorted((self._marker_start, x))
            self.cancel_marker()
        else:
            rng = self.view_x_range()
            if rng is None:
                return
            start, end = rng
        tracks = self._group_tracks.get(key, ()) if key is not None else self._tracks
        self.mask_requested.emit(start, end, [(tr.source_id, tr.channel) for tr in tracks])

    def _on_mask_range(self) -> None:
        if self._menu_x is not None:
            self.request_mask(self._menu_x, self._menu_key)

    def _draw_masks(self) -> None:
        for p, item in self._masks_items:
            p.removeItem(item)
        self._masks_items = []
        if not self._masks:
            return
        for key, p in self._plot_items.items():
            hits = masks_for(self._masks, self._group_tracks.get(key, ()))
            if hits:
                item = MasksItem(hits, self._source_labels)
                p.addItem(item, ignoreBounds=True)
                self._masks_items.append((p, item))

    def place_marker(self, x: float) -> None:
        """The "Marker ○ ○" action at master time `x` (what a right click does)."""
        self._menu_x = x
        self._on_marker()

    def cancel_marker(self) -> None:
        self._marker_start = None
        self._update_marker_action()
        self._draw_pending()

    def zoom_to_step(self, iv: ResolvedInterval) -> None:
        margin = 0.05 * (iv.end - iv.start)
        self.set_view_range(iv.start - margin, iv.end + margin)

    def step_at(self, x: float) -> ResolvedInterval | None:
        """The step under master time `x`: a manual one before a derived one, then the shortest."""
        hits = [iv for iv in self._intervals.intervals if iv.start <= x <= iv.end]
        hits.sort(key=lambda iv: (not iv.manual, iv.end - iv.start))
        return hits[0] if hits else None

    def _on_marker(self) -> None:
        x = self._menu_x
        if x is None:
            return
        if self._marker_start is None:
            self._marker_start = x
        else:
            start, end = sorted((self._marker_start, x))
            self._marker_start = None
            if end > start:
                self.interval_marked.emit(start, end)
        self._update_marker_action()
        self._draw_pending()

    def _update_marker_action(self) -> None:
        state = self.marker_state
        self.action_marker.setIcon(marker_icon(state))
        self.action_marker.setText(marker_text(state))

    def _prepare_step_menu(self, x: float, key: str | None = None) -> None:
        """Set up the step and mask entries for a right click at master time `x` in subplot `key`."""
        self._menu_x = x
        self._menu_key = key
        self._menu_step = self.step_at(x)
        hits = self.masks_at(x, key) if key is not None else []
        self._menu_mask = hits[0] if hits else None
        self.action_edit_mask.setVisible(self._menu_mask is not None)
        self.action_remove_mask.setVisible(self._menu_mask is not None)
        if self._menu_mask is not None:
            reason = self._menu_mask.reason or "no reason"
            self.action_edit_mask.setText(f"Edit Mask '{reason}'...")
            self.action_remove_mask.setText(f"Remove Mask '{reason}'")
        self.action_marker_cancel.setVisible(self._marker_start is not None)
        on_step = self._menu_step is not None
        self.action_zoom_step.setVisible(on_step)
        self.action_inspect_step.setVisible(on_step)
        manual = on_step and self._menu_step.manual
        self.action_rename_step.setVisible(manual)
        self.action_delete_step.setVisible(manual)
        if on_step:
            self.action_zoom_step.setText(f"Zoom to step '{self._menu_step.label}'")
            self.action_inspect_step.setText(f"Inspect step '{self._menu_step.label}'")

    def _install_step_menu(self, plot: pg.PlotItem) -> None:
        vb = plot.getViewBox()
        if getattr(vb, "_has_step_menu", False):
            return
        vb._has_step_menu = True
        vb.setMenuEnabled(True)
        menu = vb.menu
        before = menu.actions()[0] if menu.actions() else None
        for action in self._step_actions:
            menu.insertAction(before, action)
        if before is not None:
            menu.insertSeparator(before)
        # Weak references only: a closure holding the view box would keep its Python wrapper alive
        # after Qt deleted the subplot, and linked views would then call into a deleted ViewBox
        # (RuntimeError on Windows, segfault on Linux CI).
        canvas_ref, vb_ref = weakref.ref(self), weakref.ref(vb)

        def raise_menu(ev) -> None:
            canvas, view = canvas_ref(), vb_ref()
            if canvas is None or view is None:
                return
            key = next((k for k, p in canvas._plot_items.items() if p.getViewBox() is view), None)
            canvas._prepare_step_menu(view.mapSceneToView(ev.scenePos()).x(), key)
            type(view).raiseContextMenu(view, ev)

        vb.raiseContextMenu = raise_menu

    def _draw_bands(self) -> None:
        for band in self._bands:
            band.remove()
        for p, item in self._steps_items:
            p.removeItem(item)
        self._bands, self._steps_items = [], []
        if not self._intervals.intervals:
            return
        master = self._master_plot()
        manual = [iv for iv in self._intervals.intervals if iv.manual]
        for p in self._all_plots():
            item = StepsItem(self._intervals.intervals, with_labels=p is master)
            p.addItem(item, ignoreBounds=True)
            self._steps_items.append((p, item))
            self._bands += [StepBand(p, iv, self.interval_moved.emit) for iv in manual]

    def _draw_pending(self) -> None:
        for p, line in self._pending_lines:
            p.removeItem(line)
        self._pending_lines = []
        if self._marker_start is None:
            return
        for p in self._all_plots():
            line = pg.InfiniteLine(pos=self._marker_start, angle=90, movable=False, pen=PENDING_PEN)
            p.addItem(line, ignoreBounds=True)
            self._pending_lines.append((p, line))

    def _master_plot(self) -> pg.PlotItem | None:
        """The subplot all others are x-linked to (the coverage strip if shown)."""
        if self.coverage_plot is not None:
            return self.coverage_plot
        return next(iter(self._plot_items.values()), None)

    def _all_plots(self) -> list[pg.PlotItem]:
        return ([self.coverage_plot] if self.coverage_plot is not None else []) + list(self._plot_items.values())

    def _new_plot(self, row: int, group: PlotGroup, edge_markers: bool) -> tuple[pg.PlotItem, pg.InfiniteLine]:
        """Subplot with all curves of a group (legend if more than one), edge markers if the
        group has a fixed or robust y range."""
        p = self.graphics_layout.addPlot(
            row=row, col=0, axisItems={"bottom": self._make_axis()}, viewBox=TimeViewBox(self._wheel_time_only)
        )
        light_grid(p, alpha=0.25)
        p.setLabel("left", group.axis_label, units=group.unit or None)
        p.setTitle(group.title, size="9pt", color="#aaaaaa")
        legend = None
        if group.legend:
            legend = p.addLegend(offset=(10, 5), labelTextSize="8pt", brush=pg.mkBrush(24, 24, 28, 200))
        for tr in group.tracks:
            key = track_key(tr)
            pen = curve_pen(self._colors[key])
            curve = add_curve(p, tr.x, tr.y, tr.gap_rows, pen, tr.style)
            if legend is not None:
                legend.addItem(curve, tr.label if self._group_by is not GroupBy.SOURCE else tr.name)
            self._curves[key] = curve
            if edge_markers:
                self._edge_markers[key] = add_edge_markers(p, tr.x, tr.y, pen.color())
        line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#888888", style=pg.QtCore.Qt.DashLine))
        p.addItem(line, ignoreBounds=True)
        p.getViewBox().sigRangeChangedManually.connect(self._on_manual_range)
        return p, line

    def _lane_colors(self) -> list[str]:
        """Coverage lane colours: with colours by source, the same as the source's curves."""
        if self._color_by is not ColorBy.SOURCE:
            return list(CHANNEL_COLORS)
        order = self._source_order or [c.source_id for c in self._coverage]
        return [
            CHANNEL_COLORS[(order.index(c.source_id) if c.source_id in order else i) % len(CHANNEL_COLORS)]
            for i, c in enumerate(self._coverage)
        ]

    def full_range(self) -> tuple[float, float] | None:
        """Union of the spans of all tracks and sources (linked axes otherwise follow the first plot only)."""
        spans = [(np.nanmin(t.x), np.nanmax(t.x)) for t in self._tracks if np.isfinite(t.x).any()]
        spans += [c.span for c in self._coverage if c.span is not None]
        if not spans:
            return None
        return float(min(a for a, _ in spans)), float(max(b for _, b in spans))

    def set_view_range(self, start: float, end: float) -> None:
        """Show master time `start`..`end`; kept across rebuilds until `reset_zoom`."""
        self._user_range = (start, end)
        self._apply_range()

    def reset_zoom(self) -> None:
        self._user_range = None
        self._apply_range()
        self._apply_y_ranges(force=True)

    def y_limits(self, key: str) -> tuple[float, float] | None:
        """Fixed or robust y range of a subplot (None = auto)."""
        return self._y_limits.get(key)

    def _apply_y_ranges(self, force: bool = False) -> None:
        """Set fixed or robust y ranges (values outside are marked at the edge); auto for the rest.

        A y range the user zoomed to is kept across rebuilds unless `force`.
        """
        for key, p in self._plot_items.items():
            limits = self._y_limits.get(key)
            if limits is None:
                if force:
                    p.enableAutoRange(axis="y")
            elif force or p.getViewBox().autoRangeEnabled()[1]:
                p.enableAutoRange(axis="y", enable=False)
                p.setYRange(*limits, padding=0.02)

    def _apply_range(self) -> None:
        first = self._master_plot()
        if first is None:
            return
        rng = self._user_range or self.full_range()
        if rng is not None:
            # x auto-range of a newly added subplot would pull all linked subplots to its own span
            for p in self._all_plots():
                p.enableAutoRange(axis="x", enable=False)
            first.setXRange(*rng, padding=0 if self._user_range else 0.01)

    def _on_manual_range(self, *_args) -> None:
        first = self._master_plot()
        if first is None:
            return
        lo, hi = first.getViewBox().viewRange()[0]
        self._user_range = (lo, hi)

    def format_time(self, epoch_s: float) -> str:
        t = datetime.fromtimestamp(epoch_s + self._utc_offset, tz=timezone.utc).replace(tzinfo=None)
        text = t.strftime("%d.%m.%Y %H:%M:%S")
        if self._axis_mode is TimeAxisMode.RELATIVE and self._t0 is not None:
            text += f" (t \u2212 t0 = {format_relative(epoch_s - self._t0)})"
        return text

    @property
    def readout_mode(self) -> ReadoutMode:
        return self._readout_mode

    def set_readout(self, mode: ReadoutMode, decimals: int | None = None) -> None:
        """Values under the crosshair as a bar, a tooltip at the cursor or a table (dock).

        `decimals`: fixed number of decimals, None (or negative) = 4 significant digits.
        """
        self._readout_mode = ReadoutMode(mode)
        self._decimals = decimals if decimals is not None and decimals >= 0 else None
        self.crosshair_label.setVisible(self._readout_mode is ReadoutMode.BAR)
        if self._readout_mode is not ReadoutMode.TOOLTIP:
            QToolTip.hideText()
        if self.crosshair_x is not None:
            self.update_crosshair(self.crosshair_x)

    def readout_rows(self, x: float) -> tuple[str, list[tuple[str, str, str]], list[str]]:
        """(time text, rows of (track label, value, unit), notes: gaps, masks and the current step) at master time `x`."""
        rows = []
        for tr in self._tracks:
            curve = self._curves.get(track_key(tr))
            if not len(tr.x) or curve is None:
                continue
            # what the curve draws there: steps hold their value, nothing in a break ("—")
            rows.append((tr.label, format_value(curve.value_at(x), self._decimals).strip(), tr.unit))
        notes = [g for g in (describe_gap(c, x) for c in self._coverage) if g]
        notes += [mask_text(m, self._source_labels.get(m.source_id, "")) for m in self.masks_at(x)]
        step = self.step_at(x)
        if step is not None:
            notes.append(f"step: {step.label}")
        return self.format_time(x), rows, notes

    def update_crosshair(self, x: float, global_pos: QPoint | None = None) -> str:
        """Move the crosshair to master time `x`, show the values and return them as one line.

        `global_pos`: cursor position on screen, for the tooltip readout.
        """
        self.crosshair_x = x
        for line in self._crosshair_lines.values():
            line.setPos(x)
        if self._coverage_line is not None:
            self._coverage_line.setPos(x)
        time_text, rows, notes = self.readout_rows(x)
        values = [f"{name} = {value.rjust(VALUE_WIDTH)} {unit}".rstrip() for name, value, unit in rows]
        text = " | ".join([time_text, *values, *notes])
        if self._readout_mode is ReadoutMode.BAR:
            self.crosshair_label.setText(text)
        elif self._readout_mode is ReadoutMode.TABLE:
            self.readout_table.show_values(time_text, rows, notes)
        elif global_pos is not None:
            lines = [time_text, *[f"{name} = {value} {unit}".rstrip() for name, value, unit in rows], *notes]
            QToolTip.showText(global_pos + QPoint(16, 16), "\n".join(lines), self.graphics_layout)
        return text

    def _on_mouse_moved(self, pos) -> None:
        for p in self._all_plots():
            if p.sceneBoundingRect().contains(pos):
                view = self.graphics_layout
                self.update_crosshair(p.getViewBox().mapSceneToView(pos).x(), view.mapToGlobal(view.mapFromScene(pos)))
                return
