"""Window inspector: one measurement step up close, with its statistics (Etappe 2b, step 6, package D).

Top: the selected channels around the step (time relative to the step start), the evaluated
window highlighted and draggable, the OLS line (solid) and the Theil-Sen line (dashed), samples
left out by their flags as crosses, the two support points of the two-point value as circles.
Bottom: the statistics with uncertainties, the traffic light per channel and the assumptions.

Dragging the window edges sets a window for this step only (`WindowOverride`, saved in
campaign.json); "Reset Window" goes back to the window rule. Page Down / Page Up step through
the steps, so a whole journal is reviewed quickly. The box next to the title chooses which window
is shown and dragged: `default` (the window rule) or a named window of `Evaluation.windows`
(PR 3); a named window that is not for the step's class says so instead of showing statistics.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGraphicsScene,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.campaign import DEFAULT_WINDOW, CampaignSession, ResolvedInterval, StepWindow
from datualizer_core.stats import WindowResult
from datualizer_gui.components.curves import add_curve, curve_pen, ensure_pixmap_cache, light_grid
from datualizer_gui.components.plot_canvas import CHANNEL_COLORS
from datualizer_gui.components.readout import ReadoutLabel

MAX_CHANNELS = 8  # subplots in the inspector; more are listed in the table only
PAD_FRACTION = 0.15  # shown on each side of the step, as a fraction of its length
MIN_PAD_S = 10.0
STEP_BRUSH = QColor(120, 160, 255, 22)
WINDOW_BRUSH = QColor(255, 255, 255, 36)
WINDOW_HOVER = QColor(255, 255, 255, 60)
OVERRIDE_PEN = QColor("#ffb86c")  # edges of a window set by hand
OLS_COLOR = QColor("#f1fa8c")
THEIL_SEN_COLOR = QColor("#8be9fd")
EXCLUDED_COLOR = QColor("#ff5555")
SUPPORT_COLOR = QColor("#50fa7b")
LEVEL_COLORS = {"ok": QColor("#2ecc71"), "warn": QColor("#f1c40f"), "fail": QColor("#e74c3c")}
TABLE_COLUMNS = ["", "Channel", "Quantity", "Value", "±", "Unit"]


def level_icon(level: str, size: int = 12) -> QIcon:
    """Filled circle in the traffic-light colour of `level`."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(LEVEL_COLORS.get(level, QColor("#888888"))))
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pix)


def _num(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.5g}"


def _scaled(value: float | None, factor: float) -> float | None:
    return None if value is None else value * factor


def value_and_u(value: float | None, u: float | None) -> tuple[str, str]:
    """Value rounded to the uncertainty (two significant digits of u), as in a measurement report."""
    if value is None or not math.isfinite(value):
        return "—", _num(u)
    if u is None or not math.isfinite(u) or u <= 0:
        return f"{value:.8g}", _num(u)
    decimals = 1 - math.floor(math.log10(u))
    if decimals <= 0:
        return f"{value:.0f}", f"{u:.2g}"
    return f"{value:.{decimals}f}", f"{u:.{decimals}f}"


def stats_rows(r: WindowResult, unit: str) -> list[tuple[str, str, str, str]]:
    """(quantity, value, ±, unit) rows of one result, in reading order."""
    per_s, per_h = f"{unit}/s", f"{unit}/h"  # "/s" without a unit
    rows = [
        ("Samples (left out / empty)", f"{r.n} ({r.n_excluded} / {r.n_missing})", "", ""),
        ("Coverage (not covered)", f"{r.coverage:.1%}", f"{r.integral_gap_s:.1f} s", ""),
        ("Mean ± SEM_eff", *value_and_u(r.mean, r.sem_eff), unit),
        ("SEM classic (independent samples)", _num(r.sem), "", unit),
        ("Median", "—" if r.median is None else f"{r.median:.8g}", "", unit),
        ("SD", _num(r.sd), "", unit),
        ("Min / Max", "—" if r.min is None else f"{r.min:.8g} / {r.max:.8g}", "", unit),
        ("Quantiles 5 % / 95 %", "—" if r.quantiles is None else f"{r.quantiles[0]:.8g} / {r.quantiles[1]:.8g}", "", unit),
        ("ρ₁ / n_eff", f"{_num(r.rho1)} / {_num(r.n_eff)}", "", ""),
        ("Slope (OLS) ± SE", *value_and_u(r.slope, r.slope_se_eff), per_s),
        ("Slope (OLS) per hour", *value_and_u(_scaled(r.slope, 3600), _scaled(r.slope_se_eff, 3600)), per_h),
        ("SE classic / Newey-West / AR(1)", f"{_num(r.slope_se)} / {_num(r.slope_se_hac)} / {_num(r.slope_se_ar1)}", "", per_s),
    ]
    if r.slope_theil_sen is not None:
        lo, hi = r.slope_ts_ci if r.slope_ts_ci is not None else (None, None)
        rows.append(("Slope (Theil-Sen) [95 %]", _num(r.slope_theil_sen), f"[{_num(lo)}, {_num(hi)}]", per_s))
    rows += [
        ("R² / Durbin-Watson", f"{_num(r.r2)} / {_num(r.durbin_watson)}", "", ""),
        ("Two-point Δy/Δt", *value_and_u(r.two_point, r.two_point_u), per_s),
        ("Integral", *value_and_u(r.integral, r.integral_u), f"{unit}·s" if unit else "s"),
        ("2nd half − 1st half", _num(r.halves_diff), "", unit),
    ]
    return rows


class _ChannelView:
    """One subplot: curve, step band, draggable window, fit lines, excluded and support points."""

    def __init__(self, plot: pg.PlotItem, color: str) -> None:
        self.plot = plot
        self.color = color
        self.region: pg.LinearRegionItem | None = None
        self.curve = None
        self.ols = None
        self.theil_sen = None
        self.excluded: pg.ScatterPlotItem | None = None
        self.support: pg.ScatterPlotItem | None = None


class WindowInspector(QWidget):
    """Dock contents "Window": plots and statistics of one step."""

    override_changed = Signal(str, float, float, str)  # step id, master start, end of the window set by hand, window
    override_cleared = Signal(str, str)  # step id, window: back to the window's rule
    step_changed = Signal(str)  # step id now shown

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session: CampaignSession | None = None
        self._channels: list[tuple[str, str]] = []
        self._step_id: str | None = None
        self._views: list[_ChannelView] = []
        self._results: dict[tuple[str, str], WindowResult] = {}
        self._window = DEFAULT_WINDOW
        self._step_window: StepWindow | None = None
        self._syncing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        head = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setToolTip("Previous step (Page Up)")
        self.btn_next = QPushButton("▶")
        self.btn_next.setToolTip("Next step (Page Down)")
        for b in (self.btn_prev, self.btn_next):
            b.setFixedWidth(32)
        self.title = ReadoutLabel("Double-click a step in the Steps dock or right click → Inspect step.")
        self.window_box = QComboBox()
        self.window_box.setToolTip("Window shown and dragged: the window rule (default) or a named window")
        self.window_box.addItem(DEFAULT_WINDOW)
        self.btn_reset = QPushButton("Reset Window")
        self.btn_reset.setToolTip("Use the window rule for this step again")
        head.addWidget(self.btn_prev)
        head.addWidget(self.title, stretch=1)
        head.addWidget(self.btn_next)
        head.addWidget(self.window_box)
        head.addWidget(self.btn_reset)
        layout.addLayout(head)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("#18181c")
        # the plots are rebuilt on every step; Qt's BSP index kept stale entries of removed items
        # and crashed while painting (segfault, Linux offscreen). Few items: no index needed.
        self.graphics.scene().setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        ensure_pixmap_cache()
        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(TABLE_COLUMNS)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.checks = QListWidget()
        self.assumptions = QTextBrowser()
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(QLabel("Checks"))
        side_layout.addWidget(self.checks, stretch=1)
        side_layout.addWidget(QLabel("Assumptions"))
        side_layout.addWidget(self.assumptions, stretch=1)
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.addWidget(self.table)
        bottom.addWidget(side)
        bottom.setStretchFactor(0, 3)
        bottom.setStretchFactor(1, 2)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.graphics)
        split.addWidget(bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, stretch=1)

        self.btn_prev.clicked.connect(self.previous_step)
        self.btn_next.clicked.connect(self.next_step)
        self.btn_reset.clicked.connect(self._on_reset)
        self.window_box.currentTextChanged.connect(self._on_window_chosen)
        for key, slot in ((Qt.Key.Key_PageDown, self.next_step), (Qt.Key.Key_PageUp, self.previous_step)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
        self._update_buttons()

    # ------------------------------------------------------------------ state
    @property
    def step_id(self) -> str | None:
        return self._step_id

    @property
    def channels(self) -> list[tuple[str, str]]:
        return list(self._channels)

    @property
    def results(self) -> dict[tuple[str, str], WindowResult]:
        """Statistics of the step shown, per (source, channel)."""
        return dict(self._results)

    @property
    def views(self) -> list[_ChannelView]:
        return list(self._views)

    @property
    def window_name(self) -> str:
        """Window shown: `default` (the window rule) or a named window."""
        return self._window

    def set_window_name(self, name: str) -> None:
        if self.window_box.findText(name) < 0:
            raise KeyError(f"No window {name!r}.")
        self.window_box.setCurrentText(name)

    def _on_window_chosen(self, name: str) -> None:
        if name and name != self._window:
            self._window = name
            self.refresh()

    def _fill_windows(self) -> None:
        names = self._session.campaign.evaluation.window_names() if self._session is not None else [DEFAULT_WINDOW]
        if self._window not in names:
            self._window = DEFAULT_WINDOW
        self.window_box.blockSignals(True)
        self.window_box.clear()
        self.window_box.addItems(names)
        self.window_box.setCurrentText(self._window)
        self.window_box.blockSignals(False)
        self.window_box.setVisible(len(names) > 1)

    def set_session(self, session: CampaignSession | None) -> None:
        self._session = session
        if session is None or self._step_id not in {iv.id for iv in session.intervals().intervals}:
            self._step_id = None
        self.refresh()

    def set_channels(self, channels: Sequence[tuple[str, str]]) -> None:
        """(source id, channel) pairs to show; sources without absolute time are left out."""
        channels = list(channels)
        if channels != self._channels:
            self._channels = channels
            self.refresh()

    def show_step(self, step_id: str) -> None:
        self._step_id = step_id
        self.refresh()
        self.step_changed.emit(step_id)

    def _steps(self) -> list[ResolvedInterval]:
        return self._session.intervals().intervals if self._session is not None else []

    def _index(self) -> int:
        ids = [iv.id for iv in self._steps()]
        return ids.index(self._step_id) if self._step_id in ids else -1

    def next_step(self) -> None:
        self._go(+1)

    def previous_step(self) -> None:
        self._go(-1)

    def _go(self, delta: int) -> None:
        steps = self._steps()
        if not steps:
            return
        i = self._index()
        j = 0 if i < 0 else min(max(i + delta, 0), len(steps) - 1)
        if j != i:
            self.show_step(steps[j].id)

    def _update_buttons(self) -> None:
        n, i = len(self._steps()), self._index()
        self.btn_prev.setEnabled(i > 0)
        self.btn_next.setEnabled(n > 0 and i < n - 1)

    # ---------------------------------------------------------------- render
    def refresh(self) -> None:
        """Recompute and redraw the step shown (data, clocks, window or channels changed)."""
        for v in self._views:  # unlink first, as in the campaign canvas: linked views of deleted plots crash
            v.plot.setXLink(None)
        self.graphics.clear()
        self._views = []
        self._results = {}
        self.table.setRowCount(0)
        self.checks.clear()
        self.assumptions.clear()
        self._step_window = None
        self._update_buttons()
        self._fill_windows()
        session = self._session
        iv = None
        if session is not None and self._step_id is not None:
            try:
                iv = session.intervals().get(self._step_id)
            except KeyError:
                self._step_id = None
        if iv is None:
            self.title.setText("Double-click a step in the Steps dock or right click → Inspect step.")
            self.btn_reset.setEnabled(False)
            return
        i, n = self._index(), len(self._steps())
        params = ", ".join(f"{k} = {v}" for k, v in iv.params.items() if v is not None)
        head = f"Step {i + 1}/{n}: {iv.label}" + (f"  ({params})" if params else "")
        sw = self._step_window = session.step_window(iv, self._window)
        if not sw.applies:
            self.title.setText(f"{head}  ·  {sw.note}")
            self.btn_reset.setEnabled(False)
            self.checks.addItem(QListWidgetItem(level_icon("warn"), f"No statistics: {sw.note}."))
            return
        win = sw.window if sw.window is not None else (iv.start, iv.end)
        rule = "window rule" if sw.name == DEFAULT_WINDOW else f"window '{sw.name}'"
        how = f"{rule}, set by hand" if sw.override and sw.name != DEFAULT_WINDOW else ("set by hand" if sw.override else rule)
        self.title.setText(
            f"{head}  ·  window {win[0] - iv.start:.0f}–{win[1] - iv.start:.0f} s of {iv.duration_s:.0f} s ({how})"
        )
        self.btn_reset.setEnabled(sw.override)
        channels = [(s, c) for s, c in self._channels if session.master_time(s) is not None]
        if not channels:
            item = QListWidgetItem("Select channels in the Sources dock to evaluate them.")
            self.checks.addItem(item)
            return
        assumptions: list[str] = []
        for k, (sid, ch) in enumerate(channels):
            res = session.window_stats(iv, sid, ch, window=self._window)
            self._results[(sid, ch)] = res
            for a in res.assumptions:
                if a not in assumptions:
                    assumptions.append(a)
            if k < MAX_CHANNELS:
                self._draw_channel(k, iv, sid, ch, res)
        self._fill_table(channels)
        self.assumptions.setPlainText("\n".join(f"• {a}" for a in assumptions))
        if len(channels) > MAX_CHANNELS:
            self.checks.addItem(f"{len(channels) - MAX_CHANNELS} more channels in the table only (no plot).")

    def _draw_channel(self, row: int, iv: ResolvedInterval, sid: str, ch: str, res: WindowResult) -> None:
        session = self._session
        tr = session.track(sid, ch)
        pad = max(PAD_FRACTION * iv.duration_s, MIN_PAD_S)
        lo, hi = iv.start - pad, iv.end + pad
        i0, i1 = np.searchsorted(tr.x, [lo, hi], side="left")
        i1 = min(i1 + 1, len(tr.x))
        x = tr.x[i0:i1] - iv.start
        y = tr.y[i0:i1]
        gaps = tr.gap_rows[(tr.gap_rows > i0) & (tr.gap_rows < i1)] - i0
        color = tr.color or CHANNEL_COLORS[row % len(CHANNEL_COLORS)]

        p = self.graphics.addPlot(row=row, col=0)
        light_grid(p, alpha=0.2)
        p.setLabel("left", tr.name, units=tr.unit or None)
        p.setTitle(tr.label, size="9pt", color="#aaaaaa")
        if self._views:
            p.setXLink(self._views[0].plot)
        p.setLabel("bottom", "t − step start", units="s")
        view = _ChannelView(p, color)
        step_band = pg.LinearRegionItem((0, iv.duration_s), movable=False, brush=STEP_BRUSH, pen=pg.mkPen(None))
        p.addItem(step_band, ignoreBounds=True)
        view.curve = add_curve(p, x, y, gaps, curve_pen(color), tr.style)

        sw = self._step_window
        win = sw.window if sw.window is not None else (iv.start, iv.end)  # as session.window_stats
        w0, w1 = win[0] - iv.start, win[1] - iv.start
        edge_pen = pg.mkPen(OVERRIDE_PEN if sw.override else QColor(220, 220, 220), width=1)
        region = pg.LinearRegionItem(
            (w0, w1), movable=True, brush=WINDOW_BRUSH, hoverBrush=WINDOW_HOVER, pen=edge_pen,
            bounds=(0, iv.duration_s),
        )
        region.setZValue(-5)
        p.addItem(region, ignoreBounds=True)
        region.sigRegionChanged.connect(self._sync_regions)
        region.sigRegionChangeFinished.connect(self._on_region_finished)
        view.region = region

        tw = np.array([w0, w1])
        # both lines are given at the window start (t counted from there)
        if res.slope is not None and res.intercept is not None:
            view.ols = add_curve(p, tw, res.intercept + res.slope * (tw - w0), np.zeros(0, int), curve_pen(OLS_COLOR))
        if res.slope_theil_sen is not None and res.intercept_theil_sen is not None:
            pen = curve_pen(THEIL_SEN_COLOR)
            pen.setStyle(Qt.PenStyle.DashLine)
            yts = res.intercept_theil_sen + res.slope_theil_sen * (tw - w0)
            view.theil_sen = add_curve(p, tw, yts, np.zeros(0, int), pen)
        if len(res.excluded_rows):
            view.excluded = pg.ScatterPlotItem(
                tr.x[res.excluded_rows] - iv.start, tr.y[res.excluded_rows], symbol="x", size=9,
                pen=pg.mkPen(EXCLUDED_COLOR, width=1.5), brush=None,
            )
            p.addItem(view.excluded)
        if res.two_point_points is not None:
            (t1, y1), (t2, y2) = res.two_point_points
            view.support = pg.ScatterPlotItem(
                [t1 - iv.start, t2 - iv.start], [y1, y2], symbol="o", size=11,
                pen=pg.mkPen(SUPPORT_COLOR, width=1.5), brush=None,
            )
            p.addItem(view.support)
        p.setXRange(-pad, iv.duration_s + pad, padding=0)
        self._views.append(view)

    def _fill_table(self, channels: list[tuple[str, str]]) -> None:
        rows: list[tuple[str, str, str, str, str, str]] = []
        for sid, ch in channels:
            res = self._results[(sid, ch)]
            tr = self._session.track(sid, ch)
            for k, (quantity, value, u, unit) in enumerate(stats_rows(res, tr.unit)):
                rows.append((res.level if k == 0 else "", tr.name if k == 0 else "", quantity, value, u, unit))
            findings = [(getattr(c, "level", "warn"), str(c)) for c in res.checks] or [("ok", "all checks passed")]
            for level, text in findings:
                item = QListWidgetItem(level_icon(level), f"{tr.name}: {text}")
                item.setData(Qt.ItemDataRole.UserRole, level)
                self.checks.addItem(item)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, text in enumerate(row):
                if j == 0:
                    item = QTableWidgetItem(level_icon(text), "") if text else QTableWidgetItem("")
                    item.setToolTip(text)
                else:
                    item = QTableWidgetItem(text)
                    if j in (3, 4):
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, item)

    def table_rows(self) -> list[list[str]]:
        return [
            [self.table.item(i, j).text() for j in range(1, self.table.columnCount())]
            for i in range(self.table.rowCount())
        ]

    # ---------------------------------------------------------------- window
    def window_region(self) -> tuple[float, float] | None:
        """Window as shown, relative to the step start (s)."""
        if not self._views or self._views[0].region is None:
            return None
        lo, hi = self._views[0].region.getRegion()
        return float(lo), float(hi)

    def _sync_regions(self, region: pg.LinearRegionItem) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            for v in self._views:
                if v.region is not None and v.region is not region:
                    v.region.setRegion(region.getRegion())
        finally:
            self._syncing = False

    def set_window(self, start_rel: float, end_rel: float) -> None:
        """Set the window as if dragged (relative to the step start, s)."""
        if not self._views:
            return
        region = self._views[0].region
        region.setRegion((start_rel, end_rel))
        region.sigRegionChangeFinished.emit(region)

    def _on_region_finished(self, region: pg.LinearRegionItem) -> None:
        if self._syncing or self._session is None or self._step_id is None:
            return
        iv = self._session.intervals().get(self._step_id)
        lo, hi = region.getRegion()
        if hi - lo <= 0:
            return
        sw = self._session.step_window(iv, self._window)
        current = sw.window if sw.window is not None else (iv.start, iv.end)
        new = (iv.start + float(lo), iv.start + float(hi))
        if abs(new[0] - current[0]) < 1e-6 and abs(new[1] - current[1]) < 1e-6:
            return
        # later: redrawing deletes this region, which must not happen inside its own signal
        step_id, window = self._step_id, self._window
        QTimer.singleShot(0, lambda: self.override_changed.emit(step_id, *new, window))

    def _on_reset(self) -> None:
        if self._step_id is not None:
            self.override_cleared.emit(self._step_id, self._window)
