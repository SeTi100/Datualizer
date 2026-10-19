"""Measurement steps of a campaign: table, step mapping dialog and window rule (Etappe 2b, step 5).

The column "Class" shows the step class from `Evaluation.step_classes` (PR 3); a step without a
class or with several is marked and its hint is in the tooltip.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.calc import ExprError
from datualizer_core.campaign import (
    CampaignSession,
    IntervalConfigError,
    IntervalMapping,
    WindowRule,
    master_datetime,
)
from datualizer_core.campaign.intervals import resolve_column
from datualizer_core.ingestion.steps import EndKind, EndRule
from datualizer_core.schema import ColumnKind

ROLE_ID = Qt.ItemDataRole.UserRole
COLUMNS = ["Step", "Class", "Start", "End", "Duration", "Origin", "Parameters"]
HINT_COLOR = QColor("#ffb86c")  # a class cell with a hint (no class, or several)
TIME_COLUMN = "(time column of the source)"
OWN_CLOCK = "(clock of this source)"
END_KINDS = [
    (EndKind.COLUMN, "End time column"),
    (EndKind.DURATION, "Duration column (s)"),
    (EndKind.NEXT_START, "Until the next step starts"),
    (EndKind.FIXED, "Fixed length"),
]


def _duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class IntervalsPanel(QWidget):
    """Table of all steps, problems, and the window rule; opens the mapping dialog."""

    changed = Signal()  # a mapping, the window rule or a manual step changed
    zoom_requested = Signal(str)  # step id
    inspect_requested = Signal(str)  # step id: open the window inspector

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session: CampaignSession | None = None
        self._updating = False
        layout = QVBoxLayout(self)

        buttons = QHBoxLayout()
        self.btn_mapping = QPushButton("Steps from Source...")
        self.btn_mapping.setToolTip("Which columns of a source (e.g. the lab journal) give start, end and name")
        self.btn_delete = QPushButton("Delete Step")
        buttons.addWidget(self.btn_mapping)
        buttons.addWidget(self.btn_delete)
        layout.addLayout(buttons)

        hint = QLabel("Right click in a plot → Marker ○ ○ sets a step by hand.")
        hint.setStyleSheet("color: #999999;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.cellDoubleClicked.connect(self._on_double_click)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self.action_inspect = QAction("Inspect Step", self.table)
        self.action_inspect.setToolTip("Statistics of the step's window in the Window dock")
        self.action_inspect.triggered.connect(lambda: self._emit_selected(self.inspect_requested))
        self.action_zoom = QAction("Zoom to Step", self.table)
        self.action_zoom.triggered.connect(lambda: self._emit_selected(self.zoom_requested))
        self.table.addActions([self.action_inspect, self.action_zoom])
        layout.addWidget(self.table, stretch=1)

        self.problem_label = QLabel()
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet("color: #ffb86c;")
        self.problem_label.hide()
        layout.addWidget(self.problem_label)

        box = QGroupBox("Evaluated window of each step")
        form = QFormLayout(box)
        self.skip_start = self._seconds_spin()
        self.skip_end = self._seconds_spin()
        self.length = self._seconds_spin()
        self.length.setSpecialValueText("whole step")
        self.anchor = QComboBox()
        self.anchor.addItems(["end", "start"])
        form.addRow("Skip at start", self.skip_start)
        form.addRow("Skip at end", self.skip_end)
        form.addRow("Length", self.length)
        form.addRow("Stick to", self.anchor)
        layout.addWidget(box)

        self.btn_mapping.clicked.connect(self.open_mapping_dialog)
        self.btn_delete.clicked.connect(self._delete_selected)
        for spin in (self.skip_start, self.skip_end, self.length):
            spin.editingFinished.connect(self._apply_rule)
        self.anchor.currentIndexChanged.connect(self._apply_rule)

    @staticmethod
    def _seconds_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0, 1e6)
        spin.setDecimals(1)
        spin.setSuffix(" s")
        return spin

    # ------------------------------------------------------------------ state
    def set_session(self, session: CampaignSession | None) -> None:
        self._session = session
        self.refresh()

    def refresh(self) -> None:
        self._updating = True
        self.table.setRowCount(0)
        session = self._session
        if session is not None:
            zone = session.campaign.timezone
            steps = session.intervals()
            problems = list(steps.problems)
            try:
                classes = session.step_classes()
            except ExprError as exc:  # a class formula that does not fit the steps: say so, show no classes
                classes = None
                problems.append(str(exc).split("\n")[0])
            self.table.setRowCount(len(steps.intervals))
            for row, iv in enumerate(steps.intervals):
                params = ", ".join(f"{k} = {v}" for k, v in iv.params.items() if v is not None)
                klasse = classes.of(iv.id) if classes is not None else None
                hint = classes.notes.get(iv.id) if classes is not None else None
                cells = [
                    iv.label,
                    klasse or ("—" if hint else ""),
                    master_datetime(iv.start, zone).strftime("%d.%m. %H:%M:%S"),
                    master_datetime(iv.end, zone).strftime("%H:%M:%S"),
                    _duration(iv.duration_s),
                    ("manual" if iv.manual else iv.origin) + (" · window ✎" if iv.window_override else ""),
                    params,
                ]
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    item.setData(ROLE_ID, iv.id)
                    if col == 1 and hint:
                        item.setForeground(HINT_COLOR)
                        item.setToolTip(hint)
                    self.table.setItem(row, col, item)
            self.problem_label.setText("\n".join(problems))
            self.problem_label.setVisible(bool(problems))
            rule = session.campaign.window_rule
            self.skip_start.setValue(rule.skip_start_s)
            self.skip_end.setValue(rule.skip_end_s)
            self.length.setValue(rule.length_s or 0.0)
            self.anchor.setCurrentText(rule.anchor)
        else:
            self.problem_label.hide()
        self.table.resizeColumnsToContents()
        self._updating = False

    def selected_id(self) -> str | None:
        item = self.table.item(self.table.currentRow(), 0) if self.table.currentRow() >= 0 else None
        return item.data(ROLE_ID) if item is not None else None

    # -------------------------------------------------------------- handlers
    def _on_double_click(self, row: int, _col: int) -> None:
        """Zoom the plots to the step and open it in the window inspector."""
        item = self.table.item(row, 0)
        if item is not None:
            self.zoom_requested.emit(item.data(ROLE_ID))
            self.inspect_requested.emit(item.data(ROLE_ID))

    def _emit_selected(self, signal) -> None:
        sid = self.selected_id()
        if sid is not None:
            signal.emit(sid)

    def select_step(self, step_id: str) -> None:
        """Select the row of a step (e.g. when the inspector moves on)."""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.data(ROLE_ID) == step_id:
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.blockSignals(False)
                self.table.scrollToItem(item)
                return

    def _delete_selected(self) -> None:
        sid = self.selected_id()
        if self._session is None or sid is None:
            return
        if not any(iv.id == sid for iv in self._session.campaign.intervals):
            QMessageBox.information(
                self, "Step from a source",
                "This step comes from a source. Change its mapping (Steps from Source...) instead.",
            )
            return
        self._session.remove_interval(sid)
        self.changed.emit()

    def _apply_rule(self) -> None:
        if self._updating or self._session is None:
            return
        rule = WindowRule(
            skip_start_s=self.skip_start.value(),
            skip_end_s=self.skip_end.value(),
            length_s=self.length.value() or None,
            anchor=self.anchor.currentText(),
        )
        if rule != self._session.campaign.window_rule:
            self._session.set_window_rule(rule)
            self.changed.emit()

    def open_mapping_dialog(self) -> None:
        if self._session is None or not self._session.datasets:
            QMessageBox.information(self, "No sources", "Add a source that lists the steps (e.g. a lab journal) first.")
            return
        dialog = MappingDialog(self._session, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.changed.emit()


class MappingDialog(QDialog):
    """Choose which columns of a source give the steps; pre-filled with the suggestion, with a preview."""

    def __init__(self, session: CampaignSession, parent: QWidget | None = None, source_id: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Steps from Source")
        self.resize(560, 640)
        self._session = session
        self._filling = False
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.source = QComboBox()
        for src in session.campaign.sources:
            if src.id in session.datasets:
                self.source.addItem(src.label or src.id, src.id)
        self.start = QComboBox()
        self.end_kind = QComboBox()
        for kind, text in END_KINDS:
            self.end_kind.addItem(text, kind.value)  # Qt hands str enums back as plain str
        self.end_column = QComboBox()
        self.end_seconds = QDoubleSpinBox()
        self.end_seconds.setRange(0.1, 1e6)
        self.end_seconds.setValue(300)
        self.end_seconds.setSuffix(" s")
        self.clock = QComboBox()
        self.labels = QListWidget()
        self.fill_down = QListWidget()
        self.params = QListWidget()
        for w in (self.labels, self.fill_down, self.params):
            w.setMaximumHeight(110)
        form.addRow("Source", self.source)
        form.addRow("Start", self.start)
        form.addRow("End", self.end_kind)
        form.addRow("End column", self.end_column)
        form.addRow("Length", self.end_seconds)
        form.addRow("Times on clock of", self.clock)
        form.addRow("Name from", self.labels)
        form.addRow("Only in first row of a block", self.fill_down)
        form.addRow("Set points", self.params)
        layout.addLayout(form)
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.preview, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.btn_remove = buttons.addButton("Remove Mapping", QDialogButtonBox.ButtonRole.DestructiveRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.btn_remove.clicked.connect(self._remove)
        layout.addWidget(buttons)

        self.source.currentIndexChanged.connect(self._fill)
        for combo in (self.start, self.end_kind, self.end_column, self.clock):
            combo.currentIndexChanged.connect(self._update_preview)
        self.end_seconds.valueChanged.connect(self._update_preview)
        for lw in (self.labels, self.fill_down, self.params):
            lw.itemChanged.connect(self._update_preview)
        if source_id is not None:
            self.source.setCurrentIndex(max(0, self.source.findData(source_id)))
        self._fill()

    # ---------------------------------------------------------------- fill
    def _fill(self) -> None:
        sid = self.source.currentData()
        if sid is None:
            return
        self._filling = True
        ds = self._session.datasets[sid]
        existing = next((m for m in self._session.campaign.interval_mappings if m.source_id == sid), None)
        mapping = existing or self._session.suggest_mapping(sid)

        def loaded(name: str | None) -> str | None:
            try:
                return resolve_column(ds, name) if name else None
            except IntervalConfigError:
                return None

        cols = [c for c in ds.df.columns if ds.column_kinds.get(c) is not ColumnKind.TIME]
        self.start.clear()
        self.start.addItem(TIME_COLUMN, None)
        for c in cols:
            self.start.addItem(c, c)
        self.start.setCurrentIndex(max(0, self.start.findData(loaded(mapping.start_column))))
        self.end_kind.setCurrentIndex(self.end_kind.findData(mapping.end.kind.value))
        self.end_column.clear()
        for c in cols:
            self.end_column.addItem(c, c)
        if mapping.end.column:
            self.end_column.setCurrentIndex(max(0, self.end_column.findData(loaded(mapping.end.column))))
        if mapping.end.seconds:
            self.end_seconds.setValue(mapping.end.seconds)
        self.clock.clear()
        self.clock.addItem(OWN_CLOCK, None)
        for src in self._session.campaign.sources:
            if src.id != sid:
                self.clock.addItem(src.label or src.id, src.id)
        self.clock.setCurrentIndex(max(0, self.clock.findData(mapping.clock_source)))
        labels = {loaded(n) for n in mapping.label_columns}
        fill = {loaded(n) for n in mapping.fill_down}
        params = {loaded(n) for n in mapping.parameter_columns}
        self._fill_list(self.labels, cols, labels)
        self._fill_list(self.fill_down, cols, fill)
        self._fill_list(self.params, [c for c in cols if c in ds.channels], params)
        self._filling = False
        self._update_preview()

    @staticmethod
    def _fill_list(lw: QListWidget, names: list[str], checked: set) -> None:
        lw.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name in checked else Qt.CheckState.Unchecked)
            lw.addItem(item)

    @staticmethod
    def _checked(lw: QListWidget) -> list[str]:
        return [lw.item(i).text() for i in range(lw.count()) if lw.item(i).checkState() == Qt.CheckState.Checked]

    # ------------------------------------------------------------- mapping
    def mapping(self) -> IntervalMapping:
        """The mapping as currently chosen (raises ValueError if incomplete)."""
        kind = EndKind(self.end_kind.currentData())
        end = EndRule(
            kind=kind,
            column=self.end_column.currentData() if kind in (EndKind.COLUMN, EndKind.DURATION) else None,
            seconds=self.end_seconds.value() if kind is EndKind.FIXED else None,
        )
        labels = self._checked(self.labels)
        return IntervalMapping(
            source_id=self.source.currentData(),
            start_column=self.start.currentData(),
            end=end,
            label_columns=labels,
            fill_down=[c for c in self._checked(self.fill_down) if c in labels],
            parameter_columns=self._checked(self.params),
            clock_source=self.clock.currentData(),
        )

    def _update_preview(self, *_args) -> None:
        if self._filling:
            return
        kind = EndKind(self.end_kind.currentData())
        self.end_column.setEnabled(kind in (EndKind.COLUMN, EndKind.DURATION))
        self.end_seconds.setEnabled(kind is EndKind.FIXED)
        try:
            steps, problems = self._session.preview_steps(self.mapping())
        except (ValueError, KeyError) as exc:
            self.preview.setText(f"<span style='color:#ff6b6b'>{exc}</span>")
            return
        zone = self._session.campaign.timezone
        lines = [f"<b>{len(steps)} steps</b>"]
        for iv in steps[:6]:
            lines.append(
                f"{master_datetime(iv.start, zone):%H:%M:%S} – {master_datetime(iv.end, zone):%H:%M:%S}  {iv.label}"
            )
        if len(steps) > 6:
            lines.append("…")
        lines += [f"<span style='color:#ffb86c'>{p}</span>" for p in problems]
        self.preview.setText("<br>".join(lines))

    def accept(self) -> None:
        try:
            self._session.set_mapping(self.mapping())
        except (ValueError, KeyError) as exc:
            QMessageBox.critical(self, "Cannot use this mapping", str(exc))
            return
        super().accept()

    def _remove(self) -> None:
        self._session.remove_mapping(self.source.currentData())
        super().accept()
