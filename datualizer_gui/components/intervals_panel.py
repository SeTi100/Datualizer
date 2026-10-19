"""Measurement steps of a campaign: table, step mapping dialog and window rule (Etappe 2b, step 5).

The mapping dialog shows the sheet itself (`step_mapping_dialog.StepMappingDialog`, PR 9b).

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
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.calc import ExprError
from datualizer_core.campaign import CampaignSession, WindowRule, master_datetime
from datualizer_gui.components.step_mapping_dialog import StepMappingDialog

ROLE_ID = Qt.ItemDataRole.UserRole
COLUMNS = ["Step", "Class", "Start", "End", "Duration", "Origin", "Parameters"]
HINT_COLOR = QColor("#ffb86c")  # a class cell with a hint (no class, or several)


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
        dialog = StepMappingDialog(self._session, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.changed.emit()


# The mapping dialog works on the sheet itself (PR 9b); kept under its old name for callers.
MappingDialog = StepMappingDialog
