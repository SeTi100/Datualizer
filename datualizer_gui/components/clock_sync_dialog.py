"""Align the clock of one source (Etappe 2b, step 7): cross-correlation or clock pairs from a journal.

Tab "Cross-correlation": a reference track of another source and a channel of this source,
overlaid (normalized, display only) inside the chosen span; the slider shifts this source, the
correlation curve r(shift) shows how well each shift fits and where the best one is. Drag the
shaded region to narrow the span that is compared.

Tab "Journal pairs": cells of a workbook in the campaign that hold the same event on both clocks
(e.g. "Start FID" / "Start Real"). The anchors are listed with their offsets; the suggestion is
their median, the spread its uncertainty.

"Apply" returns a ClockModel with `method` and `note`; nothing changes before that.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsScene,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.campaign import (
    CROSS_CORRELATION,
    CampaignSession,
    CellPair,
    ClockAnchor,
    ClockCells,
    ClockModel,
    XCorrResult,
    pair_offsets,
)
from datualizer_core.ingestion.readers import ReaderError, sheet_names
from datualizer_core.timebase import date_in_text
from datualizer_gui.components.curves import add_curve, curve_pen, light_grid

EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")
SLIDER_STEPS_PER_S = 10  # slider resolution 0.1 s; the spin box takes finer values
REF_COLOR = "#8be9fd"
MOV_COLOR = "#ffb86c"
BEST_COLOR = "#50fa7b"
SPAN_BRUSH = QColor(255, 255, 255, 28)
SPAN_HOVER = QColor(255, 255, 255, 50)
LEVEL_TEXT_COLORS = {"ok": "#2ecc71", "warn": "#f1c40f", "fail": "#e74c3c"}
PAIR_COLUMNS = ["Source cell", "Master cell", "Note"]
ANCHOR_COLUMNS = ["Pair", "Source time", "Master time", "Offset [s]"]


def _normalized(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(y − mean) / std of the values inside the span; only for the overlay."""
    inside = y[mask & np.isfinite(y)]
    if len(inside) < 2 or np.std(inside) == 0:
        return y - (np.mean(inside) if len(inside) else 0.0)
    return (y - np.mean(inside)) / np.std(inside)


def _plot_widget() -> pg.PlotWidget:
    w = pg.PlotWidget()
    w.scene().setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)  # rebuilt often
    light_grid(w.getPlotItem(), 0.25)
    return w


class ClockSyncDialog(QDialog):
    """Suggest a clock model for `target_id`; `clock_model()` holds it after Apply."""

    def __init__(
        self,
        session: CampaignSession,
        target_id: str,
        view_range: tuple[float, float] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._target = target_id
        self._old = session.campaign.source(target_id).clock
        src = session.campaign.source(target_id)
        self.setWindowTitle(f"Align Clock – {src.label or src.id}")
        self.resize(900, 760)
        span = session.span(target_id)
        if view_range is None:
            view_range = span if span is not None else (0.0, 1.0)
        self._view = (float(view_range[0]), float(view_range[1]))
        self._result: XCorrResult | None = None
        self._pairs_model: ClockModel | None = None
        self._anchors: list[ClockAnchor] = []
        self._mov_curve = None
        self._corr_curve = None
        self._auto_note = ""

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Current clock: {self._describe(self._old)}"))
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_xcorr_tab(), "Cross-correlation")
        self.tabs.addTab(self._build_pairs_tab(), "Journal pairs")
        self.tabs.currentChanged.connect(lambda _i: (self._refresh_note(), self._update_apply()))
        layout.addWidget(self.tabs, stretch=1)

        note_row = QFormLayout()
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Assumption behind the clock model (saved with it)")
        note_row.addRow("Note", self.note_edit)
        layout.addLayout(note_row)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._rebuild_overlay()
        self._update_apply()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _describe(clock: ClockModel) -> str:
        if clock.is_identity():
            return "none (source time = master time)"
        if clock.anchors:
            return f"{len(clock.anchors)} anchors ({clock.method})"
        u = f" ± {clock.uncertainty_s:g}" if clock.uncertainty_s is not None else ""
        return f"offset {clock.offset_s:+.3f}{u} s, drift {clock.drift_ppm:g} ppm ({clock.method})"

    # --------------------------------------------------- cross-correlation tab
    def _build_xcorr_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        form = QFormLayout()
        self.ref_combo = QComboBox()
        self.mov_combo = QComboBox()
        for s in self._session.campaign.sources:
            ds = self._session.datasets.get(s.id)
            if ds is None or self._session.master_time(s.id) is None:
                continue
            for ch in ds.channels:
                text = f"{s.label or s.id}: {s.channel_display(ch).alias or ch}"
                if s.id == self._target:
                    self.mov_combo.addItem(text, (s.id, ch))
                else:
                    self.ref_combo.addItem(text, (s.id, ch))
        self.derivative_check = QCheckBox("Compare derivatives (ignores offsets and slow drifts)")
        half = max((self._view[1] - self._view[0]) / 2.0, 60.0)
        self.lag_min_spin = self._seconds_spin(-half)
        self.lag_max_spin = self._seconds_spin(half)
        lag_row = QHBoxLayout()
        lag_row.addWidget(self.lag_min_spin)
        lag_row.addWidget(QLabel("to"))
        lag_row.addWidget(self.lag_max_spin)
        self.correlate_button = QPushButton("Correlate")
        lag_row.addWidget(self.correlate_button)
        form.addRow("Reference", self.ref_combo)
        form.addRow("This source", self.mov_combo)
        form.addRow("", self.derivative_check)
        form.addRow("Search shifts", lag_row)
        v.addLayout(form)

        self.overlay = _plot_widget()
        self.overlay.setLabel("bottom", "t − span start [s]")
        self.overlay.setLabel("left", "normalized (display only)")
        self.overlay.addLegend(offset=(10, 10))
        v.addWidget(self.overlay, stretch=3)
        self.span_region = pg.LinearRegionItem(
            values=(0.0, self._view[1] - self._view[0]), movable=True, brush=SPAN_BRUSH, hoverBrush=SPAN_HOVER
        )
        self.span_region.setToolTip("Compared span: drag the region or its edges")
        self.span_region.setZValue(-10)

        shift_row = QHBoxLayout()
        shift_row.addWidget(QLabel("Shift this source"))
        self.shift_slider = QSlider(Qt.Orientation.Horizontal)
        self.shift_spin = self._seconds_spin(0.0)
        self.shift_spin.setDecimals(3)
        shift_row.addWidget(self.shift_slider, stretch=1)
        shift_row.addWidget(self.shift_spin)
        v.addLayout(shift_row)

        self.corr_plot = _plot_widget()
        self.corr_plot.setLabel("bottom", "shift [s]")
        self.corr_plot.setLabel("left", "r")
        self.shift_line = pg.InfiniteLine(0.0, angle=90, pen=pg.mkPen(MOV_COLOR, width=1))
        self.corr_plot.addItem(self.shift_line)
        self.best_line = pg.InfiniteLine(0.0, angle=90, pen=pg.mkPen(BEST_COLOR, width=1, style=Qt.PenStyle.DashLine))
        self.best_line.hide()
        self.corr_plot.addItem(self.best_line)
        v.addWidget(self.corr_plot, stretch=2)

        self.xcorr_label = QLabel("Choose the tracks and a span with a clear feature, then Correlate.")
        self.xcorr_label.setWordWrap(True)
        self.xcorr_label.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.xcorr_label)

        self._update_slider_range()
        self.ref_combo.currentIndexChanged.connect(lambda _i: self._rebuild_overlay())
        self.mov_combo.currentIndexChanged.connect(lambda _i: self._rebuild_overlay())
        self.lag_min_spin.editingFinished.connect(self._update_slider_range)
        self.lag_max_spin.editingFinished.connect(self._update_slider_range)
        self.shift_slider.valueChanged.connect(self._on_slider)
        self.shift_spin.valueChanged.connect(self._on_shift_spin)
        self.correlate_button.clicked.connect(self.correlate)
        return tab

    @staticmethod
    def _seconds_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1e7, 1e7)
        spin.setDecimals(1)
        spin.setSuffix(" s")
        spin.setValue(value)
        return spin

    def _update_slider_range(self) -> None:
        lo, hi = self.lag_min_spin.value(), self.lag_max_spin.value()
        self.shift_slider.blockSignals(True)
        self.shift_slider.setRange(int(round(lo * SLIDER_STEPS_PER_S)), int(round(hi * SLIDER_STEPS_PER_S)))
        self.shift_slider.setValue(int(round(self.shift_spin.value() * SLIDER_STEPS_PER_S)))
        self.shift_slider.blockSignals(False)

    @staticmethod
    def _select_track(combo: QComboBox, source_id: str, channel: str) -> None:
        for i in range(combo.count()):  # Qt hands tuples back as lists, so findData does not match
            if tuple(combo.itemData(i)) == (source_id, channel):
                combo.setCurrentIndex(i)
                return
        raise KeyError(f"No track {source_id}/{channel} to choose.")

    def set_reference(self, source_id: str, channel: str) -> None:
        self._select_track(self.ref_combo, source_id, channel)

    def set_moving(self, channel: str) -> None:
        self._select_track(self.mov_combo, self._target, channel)

    def span(self) -> tuple[float, float]:
        """Compared span in master epoch seconds (the shaded region)."""
        a, b = self.span_region.getRegion()
        return self._view[0] + a, self._view[0] + b

    def set_span(self, start: float, end: float) -> None:
        self.span_region.setRegion((start - self._view[0], end - self._view[0]))

    def shift(self) -> float:
        return self.shift_spin.value()

    def set_shift(self, value: float) -> None:
        self.shift_spin.setValue(value)

    def _on_slider(self, value: int) -> None:
        self.shift_spin.blockSignals(True)
        self.shift_spin.setValue(value / SLIDER_STEPS_PER_S)
        self.shift_spin.blockSignals(False)
        self._shift_changed()

    def _on_shift_spin(self, value: float) -> None:
        self.shift_slider.blockSignals(True)
        self.shift_slider.setValue(int(round(value * SLIDER_STEPS_PER_S)))
        self.shift_slider.blockSignals(False)
        self._shift_changed()

    def _shift_changed(self) -> None:
        self.shift_line.setValue(self.shift())
        self._draw_moving()
        self._refresh_note()
        self._update_apply()

    def _rebuild_overlay(self) -> None:
        plot = self.overlay.getPlotItem()
        region = self.span_region.getRegion()
        plot.clear()
        if plot.legend is not None:
            plot.legend.clear()
        plot.addItem(self.span_region)
        self.span_region.setRegion(region)
        self._mov_curve = None
        ref = self.ref_combo.currentData()
        if ref is not None:
            tr = self._session.track(*ref)
            mask = (tr.x >= self._view[0]) & (tr.x <= self._view[1])
            curve = add_curve(plot, tr.x - self._view[0], _normalized(tr.y, mask), tr.gap_rows, curve_pen(REF_COLOR))
            plot.legend.addItem(curve, f"Reference – {tr.label}")
        self._draw_moving()
        plot.setXRange(0.0, self._view[1] - self._view[0], padding=0.02)

    def _draw_moving(self) -> None:
        plot = self.overlay.getPlotItem()
        if self._mov_curve is not None:
            plot.removeItem(self._mov_curve)
            if plot.legend is not None:
                plot.legend.removeItem(self._mov_curve)
            self._mov_curve = None
        mov = self.mov_combo.currentData()
        if mov is None:
            return
        tr = self._session.track(*mov)
        x = tr.x + self.shift() - self._view[0]
        mask = (x >= 0) & (x <= self._view[1] - self._view[0])
        self._mov_curve = add_curve(plot, x, _normalized(tr.y, mask), tr.gap_rows, curve_pen(MOV_COLOR))
        plot.legend.addItem(self._mov_curve, f"This source, shifted – {tr.label}")

    def correlate(self) -> XCorrResult | None:
        """Run the cross-correlation in the span and move the slider to the best shift."""
        ref, mov = self.ref_combo.currentData(), self.mov_combo.currentData()
        if ref is None or mov is None:
            self.xcorr_label.setText("Needs a channel of another source as reference and one of this source.")
            return None
        start, end = self.span()
        try:
            res = self._session.cross_correlate(
                ref, mov, start, end, self.lag_min_spin.value(), self.lag_max_spin.value(),
                derivative=self.derivative_check.isChecked(),
            )
        except ValueError as exc:
            self._result = None
            self.xcorr_label.setText(f"<span style='color:{LEVEL_TEXT_COLORS['fail']}'>{exc}</span>")
            self._update_apply()
            return None
        self._result = res
        plot = self.corr_plot.getPlotItem()
        if self._corr_curve is not None:
            plot.removeItem(self._corr_curve)
        self._corr_curve = add_curve(plot, res.lags_s, res.r_curve, np.array([], dtype=int), curve_pen(REF_COLOR))
        self.best_line.setValue(res.lag_s)
        self.best_line.show()
        self.corr_plot.getPlotItem().setXRange(res.lag_range[0], res.lag_range[1], padding=0.02)
        color = LEVEL_TEXT_COLORS[res.level]
        lines = [f"<b style='color:{color}'>{res.summary()}</b>"]
        lines += [f"<span style='color:{LEVEL_TEXT_COLORS[c.level]}'>• {c}</span>" for c in res.checks]
        lines += [f"• {n}" for n in res.notes]
        self.xcorr_label.setText("<br>".join(lines))
        self.set_shift(res.lag_s)
        self._refresh_note()
        return res

    def _by_correlation(self) -> bool:
        """The shift is still the correlation result (as far as the spin box shows it)."""
        tol = 0.51 * 10.0 ** -self.shift_spin.decimals()
        return self._result is not None and abs(self.shift() - self._result.lag_s) <= tol

    def _xcorr_note(self) -> str:
        if self._by_correlation():
            res = self._result
            ref = self.ref_combo.currentText()
            return f"{res.summary()} against {ref}; ± half the grid step, a wrong peak is not covered."
        return f"Shifted by hand by {self.shift():+.3f} s in the align dialog." if self.shift() else ""

    def _xcorr_model(self) -> ClockModel:
        res = self._result
        by_corr = self._by_correlation()
        return self._old.shifted(
            res.lag_s if by_corr else self.shift(),
            method=CROSS_CORRELATION if by_corr else "manual",
            uncertainty_s=res.uncertainty_s if by_corr else None,
            note=self.note_edit.text(),
            cells=None,
        )

    # -------------------------------------------------------- journal pairs tab
    def _build_pairs_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        form = QFormLayout()
        self.cells_source_combo = QComboBox()
        for s in self._session.campaign.sources:
            if self._session.resolve(s).suffix.lower() in EXCEL_SUFFIXES:
                self.cells_source_combo.addItem(s.label or s.id, s.id)
        self.sheet_combo = QComboBox()
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("dd.MM.yyyy")
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.001, 3600.0)
        self.resolution_spin.setDecimals(3)
        self.resolution_spin.setSuffix(" s")
        self.resolution_spin.setValue(1.0)
        self.resolution_spin.setToolTip("Step of the written times; half of it is the smallest uncertainty.")
        form.addRow("Workbook", self.cells_source_combo)
        form.addRow("Sheet", self.sheet_combo)
        form.addRow("Date of the times", self.date_edit)
        form.addRow("Reading step", self.resolution_spin)
        v.addLayout(form)

        v.addWidget(QLabel("Cell pairs: the same event on this source's clock and on the master clock."))
        self.pairs_table = QTableWidget(0, len(PAIR_COLUMNS))
        self.pairs_table.setHorizontalHeaderLabels(PAIR_COLUMNS)
        self.pairs_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.pairs_table, stretch=1)
        row = QHBoxLayout()
        add_btn = QPushButton("Add Pair")
        remove_btn = QPushButton("Remove Pair")
        self.read_button = QPushButton("Read Pairs")
        row.addWidget(add_btn)
        row.addWidget(remove_btn)
        row.addStretch(1)
        row.addWidget(self.read_button)
        v.addLayout(row)

        self.anchor_table = QTableWidget(0, len(ANCHOR_COLUMNS))
        self.anchor_table.setHorizontalHeaderLabels(ANCHOR_COLUMNS)
        self.anchor_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.anchor_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.anchor_table, stretch=1)
        self.pairs_label = QLabel("Enter the cells, then Read Pairs.")
        self.pairs_label.setWordWrap(True)
        v.addWidget(self.pairs_label)

        add_btn.clicked.connect(lambda: self.add_pair())
        remove_btn.clicked.connect(self._remove_pair)
        self.read_button.clicked.connect(self.read_pairs)
        self.cells_source_combo.currentIndexChanged.connect(lambda _i: self._fill_sheets())
        self.sheet_combo.currentTextChanged.connect(self._suggest_date)

        cells = self._old.cells
        if cells is not None and self.cells_source_combo.findData(cells.source_id) >= 0:
            self.cells_source_combo.setCurrentIndex(self.cells_source_combo.findData(cells.source_id))
        self._fill_sheets()
        if cells is not None:  # what was used last time
            if cells.sheet:
                self.sheet_combo.setCurrentText(cells.sheet)
            if cells.day is not None:
                self.date_edit.setDate(QDate(cells.day.year, cells.day.month, cells.day.day))
            self.resolution_spin.setValue(cells.resolution_s)
            for p in cells.pairs:
                self.add_pair(p.source_cell, p.master_cell, p.note)
        else:
            self.add_pair()
        return tab

    def _fill_sheets(self) -> None:
        self.sheet_combo.clear()
        sid = self.cells_source_combo.currentData()
        if sid is None:
            return
        try:
            names = sheet_names(self._session.resolve(self._session.campaign.source(sid)))
        except Exception as exc:  # unreadable workbook: say so, keep the dialog usable
            self.pairs_label.setText(f"Cannot list the sheets: {exc}")
            return
        self.sheet_combo.addItems(names)

    def _suggest_date(self, sheet: str) -> None:
        day = date_in_text(sheet)
        if day is not None:
            self.date_edit.setDate(QDate(day.year, day.month, day.day))

    def add_pair(self, source_cell: str = "", master_cell: str = "", note: str = "") -> None:
        r = self.pairs_table.rowCount()
        self.pairs_table.insertRow(r)
        for c, text in enumerate((source_cell, master_cell, note)):
            self.pairs_table.setItem(r, c, QTableWidgetItem(text))

    def _remove_pair(self) -> None:
        rows = sorted({i.row() for i in self.pairs_table.selectedIndexes()}, reverse=True)
        for r in rows or [self.pairs_table.rowCount() - 1]:
            if r >= 0:
                self.pairs_table.removeRow(r)

    def cell_config(self) -> ClockCells:
        pairs = []
        for r in range(self.pairs_table.rowCount()):
            texts = [(self.pairs_table.item(r, c).text().strip() if self.pairs_table.item(r, c) else "") for c in range(3)]
            if texts[0] or texts[1]:
                pairs.append(CellPair(source_cell=texts[0].upper(), master_cell=texts[1].upper(), note=texts[2]))
        d = self.date_edit.date()
        return ClockCells(
            source_id=self.cells_source_combo.currentData(),
            sheet=self.sheet_combo.currentText() or None,
            day=date(d.year(), d.month(), d.day()),
            pairs=pairs,
            resolution_s=self.resolution_spin.value(),
        )

    def read_pairs(self) -> ClockModel | None:
        """Read the configured cells; list the anchors and suggest median ± spread."""
        self._pairs_model, self._anchors = None, []
        self.anchor_table.setRowCount(0)
        try:
            if self.cells_source_combo.currentData() is None:
                raise ValueError("The campaign has no workbook (.xlsx/.xls) to read pairs from.")
            model, anchors = self._session.clock_from_cells(self.cell_config())
        except (ValueError, ReaderError) as exc:  # pydantic ValidationError is a ValueError
            self.pairs_label.setText(f"<span style='color:{LEVEL_TEXT_COLORS['fail']}'>{exc}</span>")
            self._update_apply()
            return None
        zone = self._session.campaign.timezone
        po = pair_offsets(anchors, zone, model.cells.resolution_s if model.cells else 1.0)
        for a, off in zip(anchors, po.offsets_s):
            r = self.anchor_table.rowCount()
            self.anchor_table.insertRow(r)
            for c, text in enumerate((
                a.note, a.source_time.strftime("%d.%m.%Y %H:%M:%S"), a.master_time.strftime("%d.%m.%Y %H:%M:%S"),
                f"{off:+.3f}",
            )):
                self.anchor_table.setItem(r, c, QTableWidgetItem(text))
        self._pairs_model, self._anchors = model, anchors
        self.pairs_label.setText(
            f"Offset {model.offset_s:+.3f} s ± {model.uncertainty_s:g} s "
            f"(median of {len(anchors)} pair(s), spread {po.spread_s:g} s)."
        )
        self._refresh_note()
        self._update_apply()
        return model

    @property
    def anchors(self) -> list[ClockAnchor]:
        return list(self._anchors)

    # ---------------------------------------------------------------- result
    def _refresh_note(self) -> None:
        """Fill in the note of the active result unless the user wrote their own."""
        if not hasattr(self, "note_edit") or self.note_edit.text() not in ("", self._auto_note):
            return
        if self.tabs.currentIndex() == 1:
            note = self._pairs_model.note if self._pairs_model is not None else ""
        else:
            note = self._xcorr_note()
        self._auto_note = note
        self.note_edit.setText(note)

    def _update_apply(self) -> None:
        on_pairs = self.tabs.currentIndex() == 1
        ok = self._pairs_model is not None if on_pairs else self.mov_combo.count() > 0
        if hasattr(self, "buttons"):
            self.buttons.button(QDialogButtonBox.StandardButton.Apply).setEnabled(ok)

    def clock_model(self) -> ClockModel | None:
        """The model "Apply" writes: journal pairs or the shift of the active tab."""
        if self.tabs.currentIndex() == 1:
            if self._pairs_model is None:
                return None
            return self._pairs_model.model_copy(update={"note": self.note_edit.text()})
        return self._xcorr_model()
