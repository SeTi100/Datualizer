"""Plot canvas for a campaign: one subplot per track, each with its own x array on master time."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
import weakref

import numpy as np
import polars as pl
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QVBoxLayout, QWidget

from datualizer_core.campaign import IntervalSet, ResolvedInterval, SourceCoverage, Track
from datualizer_gui.components import curves
from datualizer_gui.components.curves import (
    FastCurve,
    add_curve,
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
from datualizer_gui.components.interval_bands import PENDING_PEN, StepBand, marker_icon, marker_text
from datualizer_gui.components.plot_canvas import CHANNEL_COLORS
from datualizer_gui.components.readout import ReadoutLabel, format_value


def utc_offset_s(epoch_s: float, zone: str) -> int:
    """Offset of `zone` from UTC at an instant, in seconds (east positive)."""
    s = pl.Series([int(epoch_s * 1e6)]).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    local = s.dt.convert_time_zone(zone).dt.replace_time_zone(None)
    return int((local.dt.epoch("us")[0] - int(epoch_s * 1e6)) / 1e6)


def connect_mask(values: np.ndarray, gap_rows: np.ndarray) -> np.ndarray:
    """Connect i to i+1 only if both are finite and no time gap starts at i+1."""
    return curves.connect_mask(values, gap_rows)


class CampaignPlotCanvas(QWidget):
    """Coverage strip on top, stacked subplots below, all with linked x axes on master time.

    Unlike the dataset canvas, every track keeps its own x array, so sources with different
    sampling and clocks share one time axis without resampling.

    Measurement steps are drawn as bands over all subplots. Right click → "Marker ○ ○" sets a
    step by hand: the first click its start (first circle green), the second its end.
    """

    interval_marked = Signal(float, float)  # master start, end of a step set with the marker
    interval_moved = Signal(str, float, float)  # manual step dragged to a new start, end
    interval_rename_requested = Signal(str)
    interval_delete_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tracks: list[Track] = []
        self._plot_items: dict[str, pg.PlotItem] = {}
        self._curves: dict[str, FastCurve] = {}
        self._crosshair_lines: dict[str, pg.InfiniteLine] = {}
        self._track_of: dict[str, Track] = {}  # track each kept subplot was drawn from
        self._coverage: list[SourceCoverage] = []
        self.coverage_plot: pg.PlotItem | None = None
        self.coverage_item: CoverageItem | None = None
        self._coverage_line: pg.InfiniteLine | None = None
        self._zone = "UTC"
        self._utc_offset = 0
        self._user_range: tuple[float, float] | None = None  # set when the user zooms or pans
        self._intervals = IntervalSet()
        self._bands: list[StepBand] = []
        self._marker_start: float | None = None  # master time of a started, unfinished marker
        self._pending_lines: list[tuple[pg.PlotItem, pg.InfiniteLine]] = []
        self._menu_x: float | None = None  # master time of the last right click
        self._menu_step: ResolvedInterval | None = None  # step under the last right click

        self.action_marker = QAction(marker_icon(0), marker_text(0), self)
        self.action_marker.setToolTip("First click: start of a step, second click: its end")
        self.action_marker.triggered.connect(self._on_marker)
        self.action_marker_cancel = QAction("Discard marker", self)
        self.action_marker_cancel.triggered.connect(self.cancel_marker)
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
        self._step_actions = [
            self.action_marker, self.action_marker_cancel, self.action_zoom_step,
            self.action_rename_step, self.action_delete_step,
        ]
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        esc.activated.connect(self.cancel_marker)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.crosshair_label = ReadoutLabel("Add sources and select channels to plot them on one time axis.")
        layout.addWidget(self.crosshair_label)
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground("#18181c")
        ensure_pixmap_cache()
        self._mouse_proxy = rate_limited_mouse(self.graphics_layout.scene(), self._on_mouse_moved)
        layout.addWidget(self.graphics_layout)

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
    def coverage(self) -> list[SourceCoverage]:
        return list(self._coverage)

    def set_tracks(
        self, tracks: Sequence[Track], zone: str = "UTC", coverage: Sequence[SourceCoverage] = ()
    ) -> None:
        """Show the coverage of these sources and these tracks on one time axis.

        A range the user zoomed to is kept, otherwise everything is shown. Subplots of tracks
        that are still shown (the same Track object) are kept with their drawn curve, so ticking
        one more channel does not redraw all the others.
        """
        self._tracks = list(tracks)
        self._coverage = list(coverage)
        starts = [float(np.nanmin(t.x)) for t in self._tracks if np.isfinite(t.x).any()]
        starts += [c.span[0] for c in self._coverage if c.span is not None]
        offset = utc_offset_s(min(starts), zone) if starts else 0
        old_plots, old_curves, old_lines, old_tracks = self._plot_items, self._curves, self._crosshair_lines, self._track_of
        if (offset, zone) != (self._utc_offset, self._zone):
            old_tracks = {}  # date axis labels depend on the offset: build all subplots anew
        self._zone, self._utc_offset = zone, offset
        for p in old_plots.values():  # unlink first: the old coverage strip is deleted below
            p.setXLink(None)
        self.graphics_layout.clear()  # detaches the subplots, kept ones are added back below
        self._plot_items, self._curves, self._crosshair_lines, self._track_of = {}, {}, {}, {}
        self.coverage_plot = self.coverage_item = self._coverage_line = None

        first: pg.PlotItem | None = None
        if self._coverage:  # rebuilt every time: cheap, and the lanes follow the source list
            first, self.coverage_item = build_coverage_plot(
                self.graphics_layout, self._coverage, CHANNEL_COLORS, self._utc_offset
            )
            first.getAxis("bottom").setStyle(showValues=not self._tracks)
            first.getViewBox().sigRangeChangedManually.connect(self._on_manual_range)
            self._coverage_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen("#888888", style=pg.QtCore.Qt.DashLine)
            )
            first.addItem(self._coverage_line, ignoreBounds=True)
            self.coverage_plot = first
        last = len(self._tracks) - 1
        for i, tr in enumerate(self._tracks):
            key = f"{tr.source_id}/{tr.channel}"
            pen = curve_pen(CHANNEL_COLORS[i % len(CHANNEL_COLORS)])
            row = i + 1  # row 0 belongs to the coverage strip
            if old_tracks.get(key) is tr:
                p, curve, line = old_plots[key], old_curves[key], old_lines[key]
                self.graphics_layout.addItem(p, row=row, col=0)
                curve.setPen(pen)
            else:
                p, curve, line = self._new_plot(row, tr, pen)
            p.setXLink(first)  # None for the first subplot
            p.getAxis("bottom").setStyle(showValues=i == last)
            if first is None:
                first = p
            self._plot_items[key], self._curves[key], self._crosshair_lines[key] = p, curve, line
            self._track_of[key] = tr
        width = LEFT_AXIS_MIN_PX
        if self.coverage_plot is not None:
            width = left_axis_width(self.coverage_plot, [lane_label(c) for c in self._coverage])
        for p in self._all_plots():
            p.getAxis("left").setWidth(width)
        if first is not None:
            self._apply_range()
        for p in self._all_plots():
            self._install_step_menu(p)
        self._draw_bands()
        self._draw_pending()
        if not self._tracks and not self._coverage:
            self.crosshair_label.setText("Add sources and select channels to plot them on one time axis.")

    # ------------------------------------------------------------------ steps
    @property
    def intervals(self) -> IntervalSet:
        return self._intervals

    @property
    def bands(self) -> list[StepBand]:
        return list(self._bands)

    @property
    def marker_state(self) -> int:
        """0: no marker started, 1: start set and waiting for the end."""
        return 0 if self._marker_start is None else 1

    def set_intervals(self, intervals: IntervalSet) -> None:
        """Draw these steps as bands over all subplots."""
        self._intervals = intervals
        self._draw_bands()

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

    def _prepare_step_menu(self, x: float) -> None:
        """Set up the step entries for a right click at master time `x`."""
        self._menu_x = x
        self._menu_step = self.step_at(x)
        self.action_marker_cancel.setVisible(self._marker_start is not None)
        on_step = self._menu_step is not None
        self.action_zoom_step.setVisible(on_step)
        manual = on_step and self._menu_step.manual
        self.action_rename_step.setVisible(manual)
        self.action_delete_step.setVisible(manual)
        if on_step:
            self.action_zoom_step.setText(f"Zoom to step '{self._menu_step.label}'")

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
            canvas._prepare_step_menu(view.mapSceneToView(ev.scenePos()).x())
            type(view).raiseContextMenu(view, ev)

        vb.raiseContextMenu = raise_menu

    def _draw_bands(self) -> None:
        for band in self._bands:
            band.remove()
        self._bands = []
        master = self._master_plot()
        for p in self._all_plots():
            for iv in self._intervals.intervals:
                self._bands.append(StepBand(p, iv, self.interval_moved.emit, with_label=p is master))

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

    def _new_plot(self, row: int, tr: Track, pen) -> tuple[pg.PlotItem, FastCurve, pg.InfiniteLine]:
        # pyqtgraph's utcOffset is seconds west of UTC
        axis = pg.DateAxisItem(orientation="bottom", utcOffset=-self._utc_offset)
        p = self.graphics_layout.addPlot(row=row, col=0, axisItems={"bottom": axis})
        light_grid(p, alpha=0.25)
        p.setLabel("left", tr.channel, units=tr.unit or None)
        p.setTitle(tr.label, size="9pt", color="#aaaaaa")
        curve = add_curve(p, tr.x, tr.y, tr.gap_rows, pen)
        line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#888888", style=pg.QtCore.Qt.DashLine))
        p.addItem(line, ignoreBounds=True)
        p.getViewBox().sigRangeChangedManually.connect(self._on_manual_range)
        return p, curve, line

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
        for p in self._plot_items.values():
            p.enableAutoRange(axis="y")

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
        return t.strftime("%d.%m.%Y %H:%M:%S")

    def update_crosshair(self, x: float) -> str:
        """Move the crosshair to master time `x` and return the readout text."""
        for line in self._crosshair_lines.values():
            line.setPos(x)
        if self._coverage_line is not None:
            self._coverage_line.setPos(x)
        parts = [self.format_time(x)]
        for tr in self._tracks:
            if not len(tr.x):
                continue
            idx = int(np.clip(np.searchsorted(tr.x, x), 1, len(tr.x) - 1))
            j = idx - 1 if abs(tr.x[idx - 1] - x) <= abs(tr.x[idx] - x) else idx
            val = tr.y[j]
            parts.append(f"{tr.label} = {format_value(val)} {tr.unit}".rstrip())
        parts += [g for g in (describe_gap(c, x) for c in self._coverage) if g]
        step = self.step_at(x)
        if step is not None:
            parts.append(f"step: {step.label}")
        text = " | ".join(parts)
        self.crosshair_label.setText(text)
        return text

    def _on_mouse_moved(self, pos) -> None:
        for p in self._all_plots():
            if p.sceneBoundingRect().contains(pos):
                self.update_crosshair(p.getViewBox().mapSceneToView(pos).x())
                return
