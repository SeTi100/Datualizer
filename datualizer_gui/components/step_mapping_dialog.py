"""Dialog "Steps from Source": set the step mapping directly on the sheet (PR 9b).

The raw sheet (or table) is shown as a grid with Excel letters and row numbers, including the
rows above the header. Everything is set by clicking it:

- a cell of the header row → menu: start, end, duration, name, name (block), parameter, id, ignore;
  assigned columns are coloured and tagged in the column header;
- a row number → "Header row here"; right click → table starts/ends here, leave out the row;
  dragging over row numbers selects a range ("Use rows … as table");
- right click on any other cell → keep it as cell value, or as one side of a clock pair.

Rows are coloured by what became of them: step (green), left out (grey, reason in the tooltip),
problem (orange). Stop rule, `where` filter (checked while typing), end fallback and parameter
defaults are fields below the grid. On the right: the steps as a table and a small timeline,
updated live. The result is an `IntervalMapping` in campaign.json.
"""

from __future__ import annotations

from datetime import date
import re
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QDate, QModelIndex, QPoint, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.calc import ExprError
from datualizer_core.calc.expr import compile_expr
from datualizer_core.campaign import (
    CampaignSession,
    CellPair,
    ClockCells,
    IntervalConfigError,
    IntervalMapping,
    master_datetime,
)
from datualizer_core.campaign.intervals import CellValue, StepDerivation, row_names
from datualizer_core.ingestion.grid import (
    Grid,
    GridError,
    a1,
    column_letter,
    display_text,
    find_header_row,
    is_empty,
    locate,
    match_header,
    parse_a1,
)
from datualizer_core.ingestion.steps import CellRef, EndKind, EndRule, RowSelection, StopRule

ROLE_COLORS = {
    "start": QColor(46, 204, 113, 110),
    "end": QColor(231, 76, 60, 110),
    "duration": QColor(231, 76, 60, 110),
    "name": QColor(52, 152, 219, 110),
    "parameter": QColor(155, 89, 182, 110),
    "id": QColor(241, 196, 15, 110),
}
ROW_COLORS = {
    "step": QColor(46, 204, 113, 45),
    "excluded": QColor(128, 128, 128, 70),
    "problem": QColor(255, 152, 0, 90),
    "header": QColor(52, 152, 219, 60),
}
CELL_COLOR = QColor(0, 188, 212, 120)  # cells kept as cell value or clock pair
TIME_COLUMN = "(time column of the source)"
OWN_CLOCK = "(clock of this source)"
END_KINDS = [
    (EndKind.COLUMN, "End time column"),
    (EndKind.DURATION, "Duration column (s)"),
    (EndKind.NEXT_START, "Until the next step starts"),
    (EndKind.FIXED, "Fixed length"),
]
FALLBACKS = [(None, "none (report the row)")] + END_KINDS
STOPS = [
    (StopRule.NONE, "none: to the last row"),
    (StopRule.TIME_JUMP, "where the start time jumps back"),
    (StopRule.FIRST_EMPTY_START, "at the first row without start"),
]
PREVIEW_COLUMNS = ["Step", "Start", "End", "Duration", "Row", "Parameters"]


def _duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_defaults(text: str) -> dict[str, float | str]:
    """'luft = 100; gas = N2' → {'luft': 100.0, 'gas': 'N2'}; ValueError for a part without '='."""
    out: dict[str, float | str] = {}
    for part in text.replace("\n", ";").split(";"):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"'{part.strip()}' is not name = value")
        name, value = (x.strip() for x in part.split("=", 1))
        if not name:
            raise ValueError(f"'{part.strip()}' has no name")
        try:
            out[name] = float(value.replace(",", "."))
        except ValueError:
            out[name] = value.strip("'\"")
    return out


def format_defaults(defaults: dict[str, float | str]) -> str:
    return "; ".join(f"{k} = {display_text(v) if isinstance(v, float) else v}" for k, v in defaults.items())


_VALUE_TEXT = re.compile(r"^\s*[-+]?(\d+([.,]\d*)?|\d{1,2}:\d{2}(:\d{2})?)\s*$")


def is_label(value: Any) -> bool:
    """A text that names something (not a number or time written as text)."""
    return isinstance(value, str) and value.strip() != "" and not _VALUE_TEXT.match(value)


def label_left(grid: Grid, row: int, col: int) -> tuple[str, int] | None:
    """The label left of a cell in its row and which value right of it the cell is (nth), or None."""
    nth = 0
    for c in range(col, -1, -1):
        v = grid.value(row, c)
        if is_empty(v):
            continue
        if c < col and is_label(v):
            return v.strip(), nth
        nth += 1
    return None


# ------------------------------------------------------------------ grid model
class SheetModel(QAbstractTableModel):
    """The raw cells, coloured by row state and column role."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.grid: Grid | None = None
        self.derivation: StepDerivation | None = None
        self.header_row: int | None = None
        self.roles: dict[int, str] = {}
        self.marked: dict[tuple[int, int], str] = {}  # (row, col) -> what the cell is kept as

    def set_grid(self, grid: Grid | None) -> None:
        self.beginResetModel()
        self.grid = grid
        self.derivation = None
        self.endResetModel()

    def set_state(self, derivation: StepDerivation | None, header_row: int | None, roles: dict[int, str]) -> None:
        self.derivation, self.header_row, self.roles = derivation, header_row, roles
        if self.rowCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, max(0, self.columnCount() - 1))

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        return 0 if self.grid is None or parent.isValid() else self.grid.n_rows

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        return 0 if self.grid is None or parent.isValid() else max(self.grid.width, 1)

    def state(self, row: int):
        d = self.derivation
        return d.rows.get(row) if d is not None else None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if self.grid is None or not index.isValid():
            return None
        row, col = index.row() + 1, index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return display_text(self.grid.value(row, col))
        if role == Qt.ItemDataRole.BackgroundRole:
            if (row, col) in self.marked:
                return CELL_COLOR
            if row == self.header_row:
                return ROLE_COLORS.get(self.roles.get(col, "").split(",")[0], ROW_COLORS["header"])
            st = self.state(row)
            return ROW_COLORS.get(st.status) if st is not None else None
        if role == Qt.ItemDataRole.ToolTipRole:
            parts = [a1(row, col)]
            if (row, col) in self.marked:
                parts.append(self.marked[(row, col)])
            st = self.state(row)
            if st is not None and st.status not in ("above",):
                parts.append({"step": "step", "excluded": "left out", "problem": "problem", "header": "header row"}[st.status]
                             + (f": {st.reason}" if st.reason else "") + (f" ({st.step_id})" if st.step_id else ""))
            if row == self.header_row and col in self.roles:
                parts.append(f"role: {self.roles[col]}")
            return "\n".join(parts)
        if role == Qt.ItemDataRole.FontRole and row == self.header_row:
            f = QFont()
            f.setBold(True)
            return f
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                tag = self.roles.get(section)
                return column_letter(section) + (f"\n{tag}" if tag else "")
            return str(section + 1)
        if role == Qt.ItemDataRole.BackgroundRole and orientation == Qt.Orientation.Horizontal:
            tag = self.roles.get(section)
            return ROLE_COLORS.get(tag.split(",")[0]) if tag else None
        if role == Qt.ItemDataRole.ToolTipRole and orientation == Qt.Orientation.Vertical:
            st = self.state(section + 1)
            return st.reason if st is not None and st.reason else None
        return None


# ------------------------------------------------------------------ timeline
class StepTimeline(QWidget):
    """The steps as bars on their time axis (alternating shades; a reported row is not drawn)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.steps: list = []
        self.zone = "UTC"
        self.selected: str | None = None
        self.setMinimumHeight(54)

    def set_steps(self, steps: list, zone: str) -> None:
        self.steps, self.zone = list(steps), zone
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.setPen(QPen(self.palette().mid().color()))
        p.drawRect(0, 0, w - 1, h - 1)
        if not self.steps:
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "no steps")
            return
        t0 = min(s.start for s in self.steps)
        t1 = max(s.end for s in self.steps)
        span = max(t1 - t0, 1.0)
        pad = 6
        y0, bar_h = 6, h - 26
        shades = [QColor(46, 204, 113, 200), QColor(39, 174, 96, 200)]
        for i, s in enumerate(sorted(self.steps, key=lambda s: s.start)):
            x0 = pad + (s.start - t0) / span * (w - 2 * pad)
            x1 = pad + (s.end - t0) / span * (w - 2 * pad)
            color = QColor(52, 152, 219, 230) if s.id == self.selected else shades[i % 2]
            p.fillRect(QRectF(x0, y0, max(x1 - x0, 1.0), bar_h), color)
        p.setPen(QPen(self.palette().text().color()))
        p.drawText(QRectF(pad, h - 18, w / 2, 16), Qt.AlignmentFlag.AlignLeft,
                   master_datetime(t0, self.zone).strftime("%d.%m. %H:%M"))
        p.drawText(QRectF(w / 2, h - 18, w / 2 - pad, 16), Qt.AlignmentFlag.AlignRight,
                   master_datetime(t1, self.zone).strftime("%H:%M"))


# ------------------------------------------------------------------ dialog
class StepMappingDialog(QDialog):
    """Set which cells of a source give the steps, on the sheet itself, with a live preview."""

    mapping_applied = Signal()

    def __init__(self, session: CampaignSession, parent: QWidget | None = None, source_id: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Steps from Source")
        self.resize(1400, 820)
        self._session = session
        self._filling = False
        self._grid: Grid | None = None
        self._derivation: StepDerivation | None = None
        self._header_row: int | None = None  # set by hand (None = found by the headers)
        self._found_header: int | None = None
        self._resolved = None
        # roles by header name
        self.start_column: str | None = None
        self.end_column_name: str | None = None
        self.label_columns: list[str] = []
        self.fill_down: list[str] = []
        self.parameter_columns: list[str] = []
        self.id_column: str | None = None
        self.skip_rows: list[int] = []
        self.first_row: int | None = None
        self.last_row: int | None = None
        self.cell_values: list[CellValue] = []
        self.pairs: list[CellPair] = []
        self._pending_pair: dict[str, Any] = {}
        self._fallback_column_name: str | None = None
        self._filled_header: int | None = None  # header row the column combos were filled from

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.source = QComboBox()
        for src in session.campaign.sources:
            if src.id in session.datasets:
                self.source.addItem(src.label or src.id, src.id)
        self.sheet = QComboBox()
        self.sheet_label = QLabel("Sheet")
        self.use_day = QCheckBox("Date of clock times")
        self.day = QDateEdit()
        self.day.setCalendarPopup(True)
        self.day.setDisplayFormat("dd.MM.yyyy")
        self.clock = QComboBox()
        for w in (QLabel("Source"), self.source, self.sheet_label, self.sheet, self.use_day, self.day,
                  QLabel("Times on clock of"), self.clock):
            top.addWidget(w)
        top.addStretch(1)
        layout.addLayout(top)

        hint = QLabel(
            "Click a cell of the header row to give its column a role · click a row number for the header row · "
            "right click a row number to start, end or leave out rows · right click a cell to keep it as value or clock pair"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #999999;")
        layout.addWidget(hint)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QSplitter(Qt.Orientation.Vertical)
        self.model = SheetModel(self)
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.horizontalHeader().setDefaultSectionSize(92)
        self.view.verticalHeader().setDefaultSectionSize(22)
        self.view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.view.verticalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        left.addWidget(self.view)

        tabs = QTabWidget()
        rows_tab = QWidget()
        rf = QFormLayout(rows_tab)
        self.header_info = QLabel()
        self.btn_header_auto = QPushButton("Find header by names")
        hb = QHBoxLayout()
        hb.addWidget(self.header_info, 1)
        hb.addWidget(self.btn_header_auto)
        rf.addRow("Header row", hb)
        self.range_info = QLabel()
        self.btn_range_clear = QPushButton("Whole table")
        rb = QHBoxLayout()
        rb.addWidget(self.range_info, 1)
        rb.addWidget(self.btn_range_clear)
        rf.addRow("Rows", rb)
        self.stop = QComboBox()
        for rule, text in STOPS:
            self.stop.addItem(text, rule.value)
        self.time_jump = QDoubleSpinBox()
        self.time_jump.setRange(0.1, 10000)
        self.time_jump.setValue(5)
        self.time_jump.setSuffix(" min")
        sb = QHBoxLayout()
        sb.addWidget(self.stop, 1)
        sb.addWidget(QLabel("jump back more than"))
        sb.addWidget(self.time_jump)
        rf.addRow("Stop", sb)
        self.where = QLineEdit()
        self.where.setPlaceholderText("formula over the row values, e.g. index <= 20 or datum == '17.10.2026'")
        self.where_status = QLabel()
        self.where_status.setWordWrap(True)
        rf.addRow("Only rows where", self.where)
        rf.addRow("", self.where_status)
        self.skip_info = QLabel()
        self.btn_skip_clear = QPushButton("Read them again")
        kb = QHBoxLayout()
        kb.addWidget(self.skip_info, 1)
        kb.addWidget(self.btn_skip_clear)
        rf.addRow("Left out", kb)
        tabs.addTab(rows_tab, "Rows")

        cols_tab = QWidget()
        cf = QFormLayout(cols_tab)
        self.start = QComboBox()
        self.end_kind = QComboBox()
        for kind, text in END_KINDS:
            self.end_kind.addItem(text, kind.value)  # Qt hands str enums back as plain str
        self.end_column = QComboBox()
        self.end_seconds = QDoubleSpinBox()
        self.end_seconds.setRange(0.1, 1e6)
        self.end_seconds.setValue(300)
        self.end_seconds.setSuffix(" s")
        self.fallback_kind = QComboBox()
        for kind, text in FALLBACKS:
            self.fallback_kind.addItem(text, kind.value if kind is not None else None)
        self.fallback_column = QComboBox()
        self.fallback_seconds = QDoubleSpinBox()
        self.fallback_seconds.setRange(0.1, 1e6)
        self.fallback_seconds.setValue(300)
        self.fallback_seconds.setSuffix(" s")
        self.roles_info = QLabel()
        self.roles_info.setWordWrap(True)
        self.defaults = QLineEdit()
        self.defaults.setPlaceholderText("luft = 100; gas = N2   (where the column is missing or a cell is empty)")
        self.defaults_status = QLabel()
        eb = QHBoxLayout()
        eb.addWidget(self.end_kind)
        eb.addWidget(self.end_column)
        eb.addWidget(self.end_seconds)
        fb = QHBoxLayout()
        fb.addWidget(self.fallback_kind)
        fb.addWidget(self.fallback_column)
        fb.addWidget(self.fallback_seconds)
        cf.addRow("Start", self.start)
        cf.addRow("End", eb)
        cf.addRow("If no end", fb)
        cf.addRow("Roles", self.roles_info)
        cf.addRow("Parameter defaults", self.defaults)
        cf.addRow("", self.defaults_status)
        tabs.addTab(cols_tab, "Columns")

        cells_tab = QWidget()
        xf = QVBoxLayout(cells_tab)
        xf.addWidget(QLabel("Cell values (kept with every step; found by the text left of them):"))
        self.cells_table = QTableWidget(0, 3)
        self.cells_table.setHorizontalHeaderLabels(["Name", "Cell", "Value"])
        self.cells_table.horizontalHeader().setStretchLastSection(True)
        self.cells_table.setMaximumHeight(110)
        xf.addWidget(self.cells_table)
        self.btn_cell_remove = QPushButton("Remove cell value")
        xf.addWidget(self.btn_cell_remove)
        xf.addWidget(QLabel("Clock pairs (same event on the source clock and on the master clock):"))
        self.pairs_table = QTableWidget(0, 2)
        self.pairs_table.setHorizontalHeaderLabels(["Source clock", "Master clock"])
        self.pairs_table.horizontalHeader().setStretchLastSection(True)
        self.pairs_table.setMaximumHeight(110)
        xf.addWidget(self.pairs_table)
        pb = QHBoxLayout()
        self.pair_target = QComboBox()
        self.btn_pairs_clear = QPushButton("Remove pairs")
        pb.addWidget(QLabel("Apply offset to clock of"))
        pb.addWidget(self.pair_target, 1)
        pb.addWidget(self.btn_pairs_clear)
        xf.addLayout(pb)
        self.pair_result = QLabel()
        self.pair_result.setWordWrap(True)
        xf.addWidget(self.pair_result)
        tabs.addTab(cells_tab, "Cells")
        self.tabs = tabs
        left.addWidget(tabs)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 1)
        left.setSizes([560, 230])
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.steps_table = QTableWidget(0, len(PREVIEW_COLUMNS))
        self.steps_table.setHorizontalHeaderLabels(PREVIEW_COLUMNS)
        self.steps_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.steps_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.steps_table.verticalHeader().hide()
        self.timeline = StepTimeline()
        rv.addWidget(self.steps_table, stretch=3)
        rv.addWidget(self.timeline)
        rv.addWidget(self.preview, stretch=1)
        split.addWidget(right)
        split.setSizes([860, 540])
        layout.addWidget(split, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.btn_remove = buttons.addButton("Remove Mapping", QDialogButtonBox.ButtonRole.DestructiveRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.btn_remove.clicked.connect(self._remove)
        layout.addWidget(buttons)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._update_preview)

        self.source.currentIndexChanged.connect(self._fill)
        self.sheet.currentIndexChanged.connect(self._sheet_changed)
        for combo in (self.start, self.end_kind, self.end_column, self.clock, self.stop,
                      self.fallback_kind, self.fallback_column, self.pair_target):
            combo.currentIndexChanged.connect(self._changed)
        for spin in (self.end_seconds, self.time_jump, self.fallback_seconds):
            spin.valueChanged.connect(self._changed)
        self.use_day.toggled.connect(self._changed)
        self.day.dateChanged.connect(self._changed)
        self.where.textChanged.connect(self._changed)
        self.defaults.textChanged.connect(self._changed)
        self.btn_header_auto.clicked.connect(lambda: self._set_header(None))
        self.btn_range_clear.clicked.connect(self._clear_range)
        self.btn_skip_clear.clicked.connect(self._clear_skips)
        self.btn_cell_remove.clicked.connect(self._remove_cell_value)
        self.btn_pairs_clear.clicked.connect(self._clear_pairs)
        self.view.clicked.connect(self._cell_clicked)
        self.view.customContextMenuRequested.connect(self._cell_menu)
        self.view.verticalHeader().sectionClicked.connect(self._row_clicked)
        self.view.verticalHeader().customContextMenuRequested.connect(self._row_menu)
        self.steps_table.itemSelectionChanged.connect(self._step_selected)
        if source_id is not None:
            self.source.setCurrentIndex(max(0, self.source.findData(source_id)))
        self._fill()

    # ---------------------------------------------------------------- fill
    def _existing(self, sid: str) -> IntervalMapping | None:
        return next((m for m in self._session.campaign.interval_mappings if m.source_id == sid), None)

    def _fill(self) -> None:
        sid = self.source.currentData()
        if sid is None:
            return
        self._filling = True
        existing = self._existing(sid)
        m = existing or self._session.suggest_mapping(sid)
        self._resolved = m.resolved
        self.start_column = m.start_column
        self.label_columns = list(m.label_columns)
        self.fill_down = list(m.fill_down)
        self.parameter_columns = list(m.parameter_columns)
        self.id_column = m.id_column
        self._header_row = m.rows.header_row
        self.first_row, self.last_row = m.rows.first_row, m.rows.last_row
        self.skip_rows = list(m.rows.skip_rows)
        self.cell_values = [cv.model_copy() for cv in m.cell_values]
        clock = self._session.campaign.source(sid).clock
        self.pairs = list(clock.cells.pairs) if clock.cells is not None else []

        src = self._session.campaign.source(sid)
        meta = self._session.datasets[sid].source_meta
        is_sheet = meta.get("reader") == "excel"
        self.sheet.blockSignals(True)
        self.sheet.clear()
        if is_sheet:
            for name in meta.get("sheets") or [meta.get("sheet")]:
                self.sheet.addItem(name, name)
            self.sheet.setCurrentIndex(max(0, self.sheet.findData(m.sheet or meta.get("sheet"))))
        self.sheet.blockSignals(False)
        self.sheet.setVisible(is_sheet)
        self.sheet_label.setVisible(is_sheet)
        cfg = src.spec or (src.profile.ingestion if src.profile else None)
        source_day = cfg.time.clock_date if cfg is not None else None
        self.use_day.setChecked(m.day is not None)
        d = m.day or source_day or date.today()
        self.day.setDate(QDate(d.year, d.month, d.day))
        self.use_day.setToolTip(f"Unchecked: the date of the source ({source_day or 'none'})")
        self.clock.clear()
        self.clock.addItem(OWN_CLOCK, None)
        self.pair_target.clear()
        self.pair_target.addItem("(do not apply)", None)
        for other in self._session.campaign.sources:
            if other.id != sid:
                self.clock.addItem(other.label or other.id, other.id)
                self.pair_target.addItem(other.label or other.id, other.id)
        self.clock.setCurrentIndex(max(0, self.clock.findData(m.clock_source)))
        target = next((s.id for s in self._session.campaign.sources
                       if s.clock.cells is not None and s.clock.cells.source_id == sid), None)
        self.pair_target.setCurrentIndex(max(0, self.pair_target.findData(target)))
        if target is not None:
            self.pairs = list(self._session.campaign.source(target).clock.cells.pairs)
        self.end_kind.setCurrentIndex(self.end_kind.findData(m.end.kind.value))
        self.end_column_name = m.end.column
        if m.end.seconds:
            self.end_seconds.setValue(m.end.seconds)
        fb = m.end.fallback
        self.fallback_kind.setCurrentIndex(self.fallback_kind.findData(fb.kind.value if fb else None))
        self._fallback_column_name = fb.column if fb else None
        if fb is not None and fb.seconds:
            self.fallback_seconds.setValue(fb.seconds)
        self.stop.setCurrentIndex(self.stop.findData(m.rows.stop.value))
        self.time_jump.setValue(m.rows.time_jump_min)
        self.where.setText(m.rows.where or "")
        self.defaults.setText(format_defaults(m.parameter_defaults))
        self._load_grid()
        self._filling = False
        self._update_preview()

    def _sheet_changed(self) -> None:
        if self._filling:
            return
        self._header_row = None
        self._load_grid()
        self._changed()

    def _load_grid(self) -> None:
        sid = self.source.currentData()
        try:
            self._grid = self._session.step_grid(sid, self.sheet.currentData() if self.sheet.isVisible() else None)
        except IntervalConfigError as exc:
            self._grid = None
            self.preview.setText(f"<span style='color:#ff6b6b'>{exc}</span>")
        self.model.set_grid(self._grid)
        self._fill_columns()

    def _header_names(self) -> list[str]:
        row = self._header_row or self._found_header
        if self._grid is None or row is None:
            return []
        return self._grid.headers(row)

    def _fill_columns(self) -> None:
        """Start and end combos: the header names of the header row (as the mapping names them)."""
        names = self._header_names()
        self._filled_header = self._header_row or self._found_header
        was = self._filling
        self._filling = True
        for combo, stored, first in (
            (self.start, self.start_column, TIME_COLUMN),
            (self.end_column, self.end_column_name, None),
            (self.fallback_column, self._fallback_column_name, None),
        ):
            current = combo.currentData() if combo.count() else stored
            combo.clear()
            if first is not None:
                combo.addItem(first, None)
            for n in names:
                combo.addItem(n, n)
            if current is not None and combo.findData(current) < 0:
                combo.addItem(current, current)  # keep a configured name the header row lacks (reported)
            combo.setCurrentIndex(max(0, combo.findData(current)))
        self._filling = was

    # ------------------------------------------------------------- mapping
    def _end_rule(self) -> EndRule:
        kind = EndKind(self.end_kind.currentData())
        fallback = None
        fk = self.fallback_kind.currentData()
        if fk is not None:
            fkind = EndKind(fk)
            fallback = EndRule(
                kind=fkind,
                column=self.fallback_column.currentData() if fkind in (EndKind.COLUMN, EndKind.DURATION) else None,
                seconds=self.fallback_seconds.value() if fkind is EndKind.FIXED else None,
            )
        return EndRule(
            kind=kind,
            column=self.end_column.currentData() if kind in (EndKind.COLUMN, EndKind.DURATION) else None,
            seconds=self.end_seconds.value() if kind is EndKind.FIXED else None,
            fallback=fallback,
        )

    def mapping(self) -> IntervalMapping:
        """The mapping as currently chosen (raises ValueError if incomplete)."""
        qd = self.day.date()
        sheet = self.sheet.currentData() if self.sheet.isVisible() else None
        meta_sheet = self._session.datasets[self.source.currentData()].source_meta.get("sheet")
        labels = list(self.label_columns)
        return IntervalMapping(
            source_id=self.source.currentData(),
            start_column=self.start.currentData(),
            end=self._end_rule(),
            label_columns=labels,
            fill_down=[c for c in self.fill_down if c in labels],
            parameter_columns=list(self.parameter_columns),
            parameter_defaults=parse_defaults(self.defaults.text()),
            id_column=self.id_column,
            rows=RowSelection(
                header_row=self._header_row, first_row=self.first_row, last_row=self.last_row,
                stop=StopRule(self.stop.currentData()), time_jump_min=self.time_jump.value(),
                where=self.where.text().strip() or None, skip_rows=sorted(set(self.skip_rows)),
            ),
            clock_source=self.clock.currentData(),
            sheet=sheet if sheet != meta_sheet else None,
            day=date(qd.year(), qd.month(), qd.day()) if self.use_day.isChecked() else None,
            cell_values=list(self.cell_values),
            resolved=self._resolved,
        )

    def clock_cells(self) -> ClockCells | None:
        """The clock pairs as configured (None without pairs)."""
        if not self.pairs:
            return None
        m_day = self.mapping().day
        src = self._session.campaign.source(self.source.currentData())
        cfg = src.spec or (src.profile.ingestion if src.profile else None)
        day = m_day or (cfg.time.clock_date if cfg is not None else None)
        sheet = self.sheet.currentData() if self.sheet.isVisible() else None
        return ClockCells(source_id=src.id, sheet=sheet, day=day, pairs=list(self.pairs))

    # ------------------------------------------------------------- preview
    def _changed(self, *_args) -> None:
        if not self._filling:
            self._timer.start()

    def refresh_now(self) -> None:
        """Update the preview at once (the dialog waits a moment while typing)."""
        self._timer.stop()
        self._update_preview()

    def _roles_by_column(self) -> dict[int, str]:
        names = self._header_names()
        out: dict[int, list[str]] = {}

        def add(name: str | None, role: str) -> None:
            if not name:
                return
            try:
                idx = match_header(name, names)
            except GridError:
                idx = None
            if idx is not None:
                out.setdefault(idx, []).append(role)

        src = self._session.campaign.source(self.source.currentData())
        add(self.start.currentData() or self._session._time_name(src), "start")
        rule = self._end_rule()
        while rule is not None:
            add(rule.column, "end" if rule.kind is EndKind.COLUMN else "duration")
            rule = rule.fallback
        for n in self.label_columns:
            add(n, "name")
        for n in self.parameter_columns:
            add(n, "parameter")
        add(self.id_column, "id")
        return {c: ", ".join(v) for c, v in out.items()}

    def _update_preview(self) -> None:
        if self._filling or self.source.currentData() is None:
            return
        kind = EndKind(self.end_kind.currentData())
        self.end_column.setEnabled(kind in (EndKind.COLUMN, EndKind.DURATION))
        self.end_seconds.setEnabled(kind is EndKind.FIXED)
        fk = self.fallback_kind.currentData()
        self.fallback_column.setEnabled(fk in (EndKind.COLUMN.value, EndKind.DURATION.value))
        self.fallback_seconds.setEnabled(fk == EndKind.FIXED.value)
        self.time_jump.setEnabled(self.stop.currentData() == StopRule.TIME_JUMP.value)
        try:
            self.defaults_status.setText("")
            m = self.mapping()
        except ValueError as exc:
            self._update_info()
            self._show_error(str(exc).split("\n")[0])
            return
        self._check_where(m)
        try:
            d = self._session.derive_mapping(m)
        except (IntervalConfigError, KeyError, ValueError) as exc:
            self._derivation = None
            self._found_header = self._guess_header(m)
            self._fill_columns()
            self.model.set_state(None, self._header_row or self._found_header, self._roles_by_column())
            self._update_info()
            self._show_error(str(exc))
            return
        self._derivation = d
        self._found_header = d.header.row if d.header else None
        if self._filled_header != self.header_row():
            self._fill_columns()
        self._update_info()
        self.model.set_state(d, self.header_row(), self._roles_by_column())
        self._show_steps(d)
        self._update_pairs()

    def _guess_header(self, m: IntervalMapping) -> int | None:
        """A header row to show while the mapping does not fit (the one set by hand, else the reader's)."""
        if m.rows.header_row is not None:
            return m.rows.header_row
        if self._grid is None:
            return None
        try:
            return find_header_row(self._grid, [n for n in [m.start_column, *m.end.columns()] if n]).row
        except GridError:
            return self._grid.reader_header_row or 1

    def _show_error(self, text: str) -> None:
        self.steps_table.setRowCount(0)
        self.timeline.set_steps([], self._session.campaign.timezone)
        self.preview.setText(f"<b>0 steps</b><br><span style='color:#ff6b6b'>{text}</span>")

    def _show_steps(self, d: StepDerivation) -> None:
        zone = self._session.campaign.timezone
        steps = d.intervals
        self.steps_table.setRowCount(len(steps))
        for row, iv in enumerate(steps):
            params = ", ".join(f"{k} = {display_text(v)}" for k, v in iv.params.items() if v is not None)
            cells = [
                iv.label,
                master_datetime(iv.start, zone).strftime("%d.%m. %H:%M:%S"),
                master_datetime(iv.end, zone).strftime("%H:%M:%S"),
                _duration(iv.duration_s),
                iv.origin.rsplit("row ", 1)[-1],
                params,
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, iv.id)
                if iv.note:
                    item.setToolTip(iv.note)
                    if col == 2:
                        item.setForeground(QColor("#ffb86c"))
                self.steps_table.setItem(row, col, item)
        self.steps_table.resizeColumnsToContents()
        self.timeline.set_steps(steps, zone)
        lines = [f"<b>{len(steps)} steps</b> · header row {d.header.row}"]
        lines += [f"<span style='color:#ffb86c'>{p}</span>" for p in d.problems]
        self.preview.setText("<br>".join(lines))

    def _check_where(self, m: IntervalMapping) -> None:
        text = m.rows.where
        if not text:
            self.where_status.setText("")
            return
        header = self._derivation.header if self._derivation is not None else None
        if self._grid is None or header is None:
            self.where_status.setText("")
            return
        names, text_names = row_names(self._grid, header)
        try:
            compile_expr(text, names=names, text_names=text_names)
            self.where_status.setText("<span style='color:#2ecc71'>✓ formula fits the table</span>")
        except ExprError as exc:
            self.where_status.setText(f"<span style='color:#ff6b6b'>{exc.message}</span><br>"
                                      f"<span style='color:#999999'>names: {', '.join(names)}</span>")

    def _update_info(self) -> None:
        header = self._header_row or self._found_header
        how = "set by hand" if self._header_row else "first row with all headers of the mapping"
        self.header_info.setText(f"{header} ({how})" if header else "—")
        if self.first_row or self.last_row:
            self.range_info.setText(f"{self.first_row or 'below header'} … {self.last_row or 'end'}")
        else:
            self.range_info.setText("below the header to the end of the table")
        self.skip_info.setText(", ".join(str(r) for r in sorted(set(self.skip_rows))) or "none")
        parts = []
        for role, names in (("start", [self.start.currentData() or TIME_COLUMN]),
                            ("name", self.label_columns), ("block name", self.fill_down),
                            ("parameters", self.parameter_columns), ("id", [self.id_column] if self.id_column else [])):
            if names:
                parts.append(f"{role}: {', '.join(names)}")
        self.roles_info.setText(" · ".join(parts))
        self.model.marked = self._marked_cells()
        self.cells_table.setRowCount(len(self.cell_values))
        for i, cv in enumerate(self.cell_values):
            try:
                hit = self._session.cell_value(cv, self.mapping_for_cells())
                value, where = display_text(hit.value), hit.address
            except (IntervalConfigError, ValueError) as exc:
                value, where = str(exc), "?"
            for c, text in enumerate([cv.name, f"{cv.describe()} → {where}", value]):
                self.cells_table.setItem(i, c, QTableWidgetItem(text))

    def mapping_for_cells(self) -> IntervalMapping:
        sheet = self.sheet.currentData() if self.sheet.isVisible() else None
        return IntervalMapping(source_id=self.source.currentData(), sheet=sheet)

    def _marked_cells(self) -> dict[tuple[int, int], str]:
        out: dict[tuple[int, int], str] = {}
        if self._grid is None:
            return out
        for cv in self.cell_values:
            try:
                hit = locate(self._grid, cv)
                out[(hit.row, hit.col)] = f"cell value '{cv.name}'"
            except (GridError, ValueError):
                pass
        for i, p in enumerate(self.pairs):
            for side, ref in zip(("source clock", "master clock"), p.refs()):
                try:
                    hit = locate(self._grid, ref)
                    out[(hit.row, hit.col)] = f"clock pair {i + 1}: {side}"
                except (GridError, ValueError):
                    pass
        return out

    def _update_pairs(self) -> None:
        self.pairs_table.setRowCount(len(self.pairs))
        for i, p in enumerate(self.pairs):
            src_ref, mst_ref = p.refs()
            self.pairs_table.setItem(i, 0, QTableWidgetItem(src_ref.describe()))
            self.pairs_table.setItem(i, 1, QTableWidgetItem(mst_ref.describe()))
        if not self.pairs:
            self.pair_result.setText("Right click the time of an event on the source clock, then the same event on the master clock.")
            return
        try:
            model, anchors = self._session.clock_from_cells(self.clock_cells())
            target = self.pair_target.currentData()
            applied = f" → applied to {target} on OK" if target else " (choose the clock to apply it to)"
            self.pair_result.setText(
                f"Offset {model.offset_s:g} s ± {model.uncertainty_s:g} s from "
                + ", ".join(a.note for a in anchors) + applied
            )
        except (ValueError, KeyError) as exc:
            self.pair_result.setText(f"<span style='color:#ff6b6b'>{exc}</span>")

    def _step_selected(self) -> None:
        items = self.steps_table.selectedItems()
        sid = items[0].data(Qt.ItemDataRole.UserRole) if items else None
        self.timeline.selected = sid
        self.timeline.update()
        if sid is None or self._derivation is None:
            return
        row = next((r for r, st in self._derivation.rows.items() if st.step_id == sid), None)
        if row is not None:
            self.view.scrollTo(self.model.index(row - 1, 0))

    # ------------------------------------------------------- clicking cells
    def header_row(self) -> int | None:
        return self._header_row or self._found_header

    def _cell_clicked(self, index: QModelIndex) -> None:
        if index.row() + 1 == self.header_row():
            self.role_menu(index.column()).popup(self.view.viewport().mapToGlobal(self.view.visualRect(index).bottomLeft()))

    def role_menu(self, col: int) -> QMenu:
        """Menu of roles for the column `col` of the header row."""
        names = self._header_names()
        name = names[col] if col < len(names) else f"col_{column_letter(col)}"
        menu = QMenu(self)
        menu.setTitle(name)
        current = self._roles_by_column().get(col, "")
        entries = [
            ("Start", "start"), ("End", "end"), ("Duration (s)", "duration"), ("Name", "name"),
            ("Name, block (filled down)", "block"), ("Parameter", "parameter"), ("ID", "id"),
        ]
        for text, role in entries:
            act = QAction(text, menu)
            act.setCheckable(True)
            act.setChecked(self._has_role(name, role, current))
            act.triggered.connect(lambda checked, r=role: self.set_role(name, r, checked))
            menu.addAction(act)
        menu.addSeparator()
        ignore = QAction("Ignore", menu)
        ignore.triggered.connect(lambda: self.set_role(name, "ignore", True))
        menu.addAction(ignore)
        return menu

    def _has_role(self, name: str, role: str, current: str) -> bool:
        if role == "block":
            return name in self.fill_down
        if role == "start":
            return self.start.currentData() == name
        return role in current.split(", ")

    def set_role(self, name: str, role: str, on: bool = True) -> None:
        """Give header `name` a role (or take it away); 'ignore' removes all its roles."""
        def drop(lst: list[str]) -> list[str]:
            return [n for n in lst if n != name]

        if role == "ignore" or not on:
            if role in ("ignore", "start") and self.start.currentData() == name:
                self.start.setCurrentIndex(0)
            if role in ("ignore", "end", "duration") and self.end_column.currentData() == name:
                self.end_kind.setCurrentIndex(self.end_kind.findData(EndKind.NEXT_START.value))
            if role in ("ignore", "name", "block"):
                self.label_columns = drop(self.label_columns) if role != "block" else self.label_columns
                self.fill_down = drop(self.fill_down)
            if role in ("ignore", "parameter"):
                self.parameter_columns = drop(self.parameter_columns)
            if role in ("ignore", "id") and self.id_column == name:
                self.id_column = None
        elif role == "start":
            self.start_column = name
            self.start.setCurrentIndex(max(0, self.start.findData(name)))
        elif role in ("end", "duration"):
            self.end_column_name = name
            self.end_kind.setCurrentIndex(self.end_kind.findData(
                EndKind.COLUMN.value if role == "end" else EndKind.DURATION.value))
            self.end_column.setCurrentIndex(max(0, self.end_column.findData(name)))
        elif role == "name":
            if name not in self.label_columns:
                self.label_columns.append(name)
        elif role == "block":
            if name not in self.label_columns:
                self.label_columns.append(name)
            if name not in self.fill_down:
                self.fill_down.append(name)
        elif role == "parameter":
            if name not in self.parameter_columns:
                self.parameter_columns.append(name)
        elif role == "id":
            self.id_column = name
        self.refresh_now()

    def _row_clicked(self, section: int) -> None:
        menu = QMenu(self)
        act = QAction(f"Header row here (row {section + 1})", menu)
        act.triggered.connect(lambda: self._set_header(section + 1))
        menu.addAction(act)
        menu.popup(self.view.verticalHeader().mapToGlobal(QPoint(0, self.view.verticalHeader().sectionViewportPosition(section))))

    def _selected_rows(self) -> list[int]:
        return sorted({i.row() + 1 for i in self.view.selectionModel().selectedRows()})

    def _row_menu_entries(self, row: int) -> list[tuple[str, Any]]:
        rows = self._selected_rows()
        entries = [
            (f"Header row here (row {row})", lambda: self._set_header(row)),
            (f"Table starts here (row {row})", lambda: self.set_range(first=row)),
            (f"Table ends here (row {row})", lambda: self.set_range(last=row)),
        ]
        if len(rows) > 1 and rows == list(range(rows[0], rows[-1] + 1)):
            entries.append((f"Use rows {rows[0]}–{rows[-1]} as table", lambda: self.set_range(rows[0], rows[-1])))
        if row in self.skip_rows:
            entries.append((f"Read row {row} again", lambda: self.toggle_skip(row)))
        else:
            entries.append((f"Leave out row {row}", lambda: self.toggle_skip(row)))
        return entries

    def row_menu_for(self, row: int) -> QMenu:
        """Context menu of row number `row` (also offers the selected range)."""
        menu = QMenu(self)
        for text, fn in self._row_menu_entries(row):
            act = QAction(text, menu)
            act.triggered.connect(fn)
            menu.addAction(act)
        return menu

    def _row_menu(self, pos: QPoint) -> None:
        section = self.view.verticalHeader().logicalIndexAt(pos)
        if section >= 0:
            self.row_menu_for(section + 1).popup(self.view.verticalHeader().mapToGlobal(pos))

    def _set_header(self, row: int | None) -> None:
        self._header_row = row
        if row is not None and self.first_row is not None and self.first_row <= row:
            self.first_row = None
        self._fill_columns()
        self.refresh_now()

    def set_range(self, first: int | None = None, last: int | None = None) -> None:
        if first is not None:
            self.first_row = first
            if self.last_row is not None and self.last_row < first:
                self.last_row = None
        if last is not None:
            self.last_row = last
            if self.first_row is not None and self.first_row > last:
                self.first_row = None
        self.refresh_now()

    def _clear_range(self) -> None:
        self.first_row = self.last_row = None
        self.refresh_now()

    def toggle_skip(self, row: int) -> None:
        self.skip_rows = [r for r in self.skip_rows if r != row] if row in self.skip_rows else self.skip_rows + [row]
        self.refresh_now()

    def _clear_skips(self) -> None:
        self.skip_rows = []
        self.refresh_now()

    def cell_menu_for(self, row: int, col: int) -> QMenu:
        menu = QMenu(self)
        if row == self.header_row():
            return self.role_menu(col)
        entries = [
            ("Keep as cell value…", lambda: self.add_cell_value(row, col)),
            ("Clock pair: time on the source clock", lambda: self.add_pair_side(row, col, "source")),
            ("Clock pair: same event on the master clock", lambda: self.add_pair_side(row, col, "master")),
        ]
        for text, fn in entries:
            act = QAction(text, menu)
            act.triggered.connect(fn)
            menu.addAction(act)
        menu.addSeparator()
        for text, fn in self._row_menu_entries(row):
            act = QAction(text, menu)
            act.triggered.connect(fn)
            menu.addAction(act)
        return menu

    def _cell_menu(self, pos: QPoint) -> None:
        index = self.view.indexAt(pos)
        if index.isValid():
            self.cell_menu_for(index.row() + 1, index.column()).popup(self.view.viewport().mapToGlobal(pos))

    def _ref_for(self, row: int, col: int) -> dict[str, Any]:
        """A cell as label + nth if there is a text left of it, else as address."""
        hit = label_left(self._grid, row, col) if self._grid is not None else None
        return {"label": hit[0], "nth": hit[1]} if hit is not None else {"cell": a1(row, col)}

    def add_cell_value(self, row: int, col: int, name: str | None = None) -> None:
        ref = self._ref_for(row, col)
        if name is None:
            suggestion = ref.get("label", a1(row, col)).split()[0].lower() if ref.get("label") else "wert"
            name, ok = QInputDialog.getText(self, "Cell value", "Parameter name in the steps:", text=suggestion)
            if not ok or not name.strip():
                return
        self.cell_values = [cv for cv in self.cell_values if cv.name != name.strip()]
        self.cell_values.append(CellValue(name=name.strip(), **ref))
        self.tabs.setCurrentIndex(2)
        self.refresh_now()

    def _remove_cell_value(self) -> None:
        row = self.cells_table.currentRow()
        if 0 <= row < len(self.cell_values):
            del self.cell_values[row]
            self.refresh_now()

    def add_pair_side(self, row: int, col: int, side: str) -> None:
        """One side of a clock pair; the pair is complete when both sides are set."""
        ref = self._ref_for(row, col)
        self._pending_pair[side] = ref
        if "source" in self._pending_pair and "master" in self._pending_pair:
            s, m = self._pending_pair["source"], self._pending_pair["master"]
            kw: dict[str, Any] = {}
            if "label" in s and "label" in m and s["nth"] == m["nth"]:
                kw = {"source_label": s["label"], "master_label": m["label"], "nth": s["nth"]}
            else:
                kw = {"source_cell": a1(*self._addr(s, row, col)), "master_cell": a1(*self._addr(m, row, col))}
            self.pairs.append(CellPair(**kw))
            self._pending_pair = {}
        self.tabs.setCurrentIndex(2)
        self.refresh_now()

    def _addr(self, ref: dict[str, Any], row: int, col: int) -> tuple[int, int]:
        if "cell" in ref:
            return parse_a1(ref["cell"])
        hit = locate(self._grid, CellRef(label=ref["label"], nth=ref["nth"]))
        return hit.row, hit.col

    def _clear_pairs(self) -> None:
        self.pairs = []
        self._pending_pair = {}
        self.refresh_now()

    # ---------------------------------------------------------------- done
    def accept(self) -> None:
        self._timer.stop()
        try:
            m = self.mapping()
            self._session.set_mapping(m)
            target = self.pair_target.currentData()
            if self.pairs and target is not None:
                model, _ = self._session.clock_from_cells(self.clock_cells())
                self._session.set_clock(target, model)
        except (ValueError, KeyError) as exc:
            QMessageBox.critical(self, "Cannot use this mapping", str(exc))
            return
        self.mapping_applied.emit()
        super().accept()

    def reject(self) -> None:
        self._timer.stop()
        super().reject()

    def _remove(self) -> None:
        self._timer.stop()
        self._session.remove_mapping(self.source.currentData())
        super().accept()

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)

