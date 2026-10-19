"""Results dock: calculated quantities, uncertainty budgets, criteria, scenarios and results table (Etappe 2b, step 8, PR 5).

Shows:
- Results Table: steps with measures, relations, calculated quantities with uncertainties and
  budget columns, evaluated criteria booleans, ratings and reasons. Filterable, sortable,
  with CSV export and column visibility toggles. Double-clicking a row inspects the step in
  the Window dock.
- Calculations: list of formulas evaluated over steps with live syntax checking during typing.
- Constants & Scenarios: constants table (value, u, kind, scope, unit, note) and scenario switcher.
- Criteria & Ratings: condition formulas with live syntax checking and rating rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.calc import ExprError, compile_expr
from datualizer_core.calc.uncertain import Constant, KINDS
from datualizer_core.campaign import CampaignSession, Evaluation
from datualizer_core.campaign.evaluation import Calculation, Criterion, Rating, Scenario
from datualizer_gui.components.readout import ReadoutLabel

HINT_COLOR = QColor("#ffb86c")
VALID_COLOR = "#50fa7b"
ERROR_COLOR = "#ff5555"


class NumericTableWidgetItem(QTableWidgetItem):
    """Table item that sorts numerically when a numeric sort value is provided."""

    def __init__(self, text: str, sort_value: Any = None) -> None:
        super().__init__(text)
        self.sort_value = sort_value

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if hasattr(other, "sort_value") and self.sort_value is not None and other.sort_value is not None:
            try:
                return self.sort_value < other.sort_value
            except TypeError:
                pass
        return super().__lt__(other)


def _fmt(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        if np.isnan(val):
            return ""
        return f"{val:.4g}"
    return str(val)


class ResultsPanel(QWidget):
    """Dock widget for evaluation results, constants, formulas, criteria and scenarios."""

    step_double_clicked = Signal(str)  # step_id
    evaluation_changed = Signal()  # emitted when configuration is updated

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session: CampaignSession | None = None
        self._results_df: pl.DataFrame | None = None
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs)

        # Tab 0: Results Table
        self._build_results_tab()

        # Tab 1: Calculations
        self._build_calculations_tab()

        # Tab 2: Constants & Scenarios
        self._build_constants_tab()

        # Tab 3: Criteria & Ratings
        self._build_criteria_tab()

    # ------------------------------------------------------------------ Tab 0: Results
    def _build_results_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(2, 2, 2, 2)

        # Toolbar
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("Scenario:"))
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItem("(Base)")
        self.scenario_combo.currentIndexChanged.connect(self._on_scenario_changed)
        toolbar.addWidget(self.scenario_combo)

        toolbar.addSpacing(10)
        toolbar.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter steps...")
        self.filter_edit.textChanged.connect(self._apply_filter)
        toolbar.addWidget(self.filter_edit, stretch=1)

        self.btn_columns = QPushButton("Columns...")
        self.btn_columns.setToolTip("Show or hide columns in the results table")
        self.btn_columns.clicked.connect(self._open_columns_menu)
        toolbar.addWidget(self.btn_columns)

        self.btn_export = QPushButton("Export CSV...")
        self.btn_export.setToolTip("Export the current results table to a CSV file")
        self.btn_export.clicked.connect(self.export_csv)
        toolbar.addWidget(self.btn_export)

        self.btn_recompute = QPushButton("Recompute")
        self.btn_recompute.setToolTip("Recalculate results from campaign data and formulas")
        self.btn_recompute.clicked.connect(self.refresh)
        toolbar.addWidget(self.btn_recompute)

        layout.addLayout(toolbar)

        # Table
        self.table = QTableWidget(0, 0)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.table.horizontalHeader().setSectionsMovable(True)
        layout.addWidget(self.table, stretch=1)

        # Status readout
        self.status_label = ReadoutLabel("No results loaded.")
        layout.addWidget(self.status_label)

        self.tabs.addTab(tab, "Results")

    # ------------------------------------------------------------------ Tab 1: Calculations
    def _build_calculations_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: List / Table of Calculations
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("<b>Calculated Quantities</b> (evaluated in list order):"))
        self.calc_table = QTableWidget(0, 4)
        self.calc_table.setHorizontalHeaderLabels(["Name", "Expression", "Unit", "Note"])
        self.calc_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.calc_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.calc_table.itemSelectionChanged.connect(self._on_calc_selected)
        left_layout.addWidget(self.calc_table, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_add_calc = QPushButton("New")
        self.btn_add_calc.clicked.connect(self._clear_calc_form)
        self.btn_del_calc = QPushButton("Delete")
        self.btn_del_calc.clicked.connect(self._delete_calc)
        btn_row.addWidget(self.btn_add_calc)
        btn_row.addWidget(self.btn_del_calc)
        left_layout.addLayout(btn_row)

        splitter.addWidget(left)

        # Right: Formula Editor with Live Syntax Check
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Calculation Editor")
        form = QFormLayout(box)

        self.calc_name_edit = QLineEdit()
        self.calc_name_edit.setPlaceholderText("e.g. X, c_in, V_N")
        form.addRow("Name:", self.calc_name_edit)

        self.calc_expr_edit = QLineEdit()
        self.calc_expr_edit.setPlaceholderText("e.g. 1 - (S - B) * f / c_in")
        self.calc_expr_edit.textChanged.connect(self._on_calc_expr_changed)
        form.addRow("Formula:", self.calc_expr_edit)

        self.calc_syntax_label = QLabel("Enter formula")
        self.calc_syntax_label.setStyleSheet("color: #999999;")
        self.calc_syntax_label.setWordWrap(True)
        form.addRow("Syntax:", self.calc_syntax_label)

        self.calc_unit_edit = QLineEdit()
        self.calc_unit_edit.setPlaceholderText("e.g. mg/Nm3, 1/s, °C")
        form.addRow("Unit:", self.calc_unit_edit)

        self.calc_note_edit = QLineEdit()
        self.calc_note_edit.setPlaceholderText("Optional description")
        form.addRow("Note:", self.calc_note_edit)

        self.btn_save_calc = QPushButton("Save Calculation")
        self.btn_save_calc.clicked.connect(self._save_calc)
        form.addRow("", self.btn_save_calc)

        right_layout.addWidget(box)
        right_layout.addStretch(1)

        splitter.addWidget(right)
        splitter.setSizes([350, 300])

        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Calculations")

    # ------------------------------------------------------------------ Tab 2: Constants
    def _build_constants_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Constants Table
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("<b>Constants</b> (uncertain inputs and physical values):"))
        self.const_table = QTableWidget(0, 7)
        self.const_table.setHorizontalHeaderLabels(["Name", "Value", "u", "Kind", "Scope", "Unit", "Note"])
        self.const_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.const_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.const_table.itemSelectionChanged.connect(self._on_const_selected)
        left_layout.addWidget(self.const_table, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_add_const = QPushButton("New")
        self.btn_add_const.clicked.connect(self._clear_const_form)
        self.btn_del_const = QPushButton("Delete")
        self.btn_del_const.clicked.connect(self._delete_const)
        btn_row.addWidget(self.btn_add_const)
        btn_row.addWidget(self.btn_del_const)
        left_layout.addLayout(btn_row)

        splitter.addWidget(left)

        # Right: Constant Editor & Scenarios
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Constant Editor")
        form = QFormLayout(box)

        self.const_name_edit = QLineEdit()
        self.const_name_edit.setPlaceholderText("e.g. f, luft_m3_h, T_ref")
        form.addRow("Name:", self.const_name_edit)

        self.const_value_edit = QLineEdit()
        self.const_value_edit.setPlaceholderText("Numeric value")
        form.addRow("Value:", self.const_value_edit)

        self.const_u_edit = QLineEdit()
        self.const_u_edit.setPlaceholderText("Standard uncertainty (0.0)")
        form.addRow("u (1σ):", self.const_u_edit)

        self.const_kind_combo = QComboBox()
        self.const_kind_combo.addItems(list(KINDS))
        form.addRow("Kind:", self.const_kind_combo)

        self.const_scope_edit = QLineEdit()
        self.const_scope_edit.setPlaceholderText("Optional scope (e.g. 'tag')")
        form.addRow("Scope:", self.const_scope_edit)

        self.const_unit_edit = QLineEdit()
        form.addRow("Unit:", self.const_unit_edit)

        self.const_note_edit = QLineEdit()
        form.addRow("Note:", self.const_note_edit)

        self.btn_save_const = QPushButton("Save Constant")
        self.btn_save_const.clicked.connect(self._save_const)
        form.addRow("", self.btn_save_const)

        right_layout.addWidget(box)

        # Scenarios Group
        scen_box = QGroupBox("Scenarios")
        scen_form = QFormLayout(scen_box)
        self.scen_table = QTableWidget(0, 2)
        self.scen_table.setHorizontalHeaderLabels(["Scenario", "Overrides"])
        self.scen_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.scen_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        scen_form.addRow(self.scen_table)

        right_layout.addWidget(scen_box)
        right_layout.addStretch(1)

        splitter.addWidget(right)
        splitter.setSizes([380, 280])

        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Constants")

    # ------------------------------------------------------------------ Tab 3: Criteria & Ratings
    def _build_criteria_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Criteria
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("<b>Criteria</b> (evaluated per step; text is recorded when violated):"))
        self.crit_table = QTableWidget(0, 3)
        self.crit_table.setHorizontalHeaderLabels(["Name", "Expression", "Reason (Text)"])
        self.crit_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.crit_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.crit_table.itemSelectionChanged.connect(self._on_crit_selected)
        left_layout.addWidget(self.crit_table, stretch=1)

        crit_box = QGroupBox("Criterion Editor")
        crit_form = QFormLayout(crit_box)
        self.crit_name_edit = QLineEdit()
        self.crit_name_edit.setPlaceholderText("e.g. c_ok, flach, t_ok")
        crit_form.addRow("Name:", self.crit_name_edit)

        self.crit_expr_edit = QLineEdit()
        self.crit_expr_edit.setPlaceholderText("e.g. c_in >= 60")
        self.crit_expr_edit.textChanged.connect(self._on_crit_expr_changed)
        crit_form.addRow("Formula:", self.crit_expr_edit)

        self.crit_syntax_label = QLabel("Enter formula")
        self.crit_syntax_label.setStyleSheet("color: #999999;")
        crit_form.addRow("Syntax:", self.crit_syntax_label)

        self.crit_text_edit = QLineEdit()
        self.crit_text_edit.setPlaceholderText("Reason when violated (e.g. c_in < 60)")
        crit_form.addRow("Text:", self.crit_text_edit)

        crit_btn_row = QHBoxLayout()
        self.btn_save_crit = QPushButton("Save Criterion")
        self.btn_save_crit.clicked.connect(self._save_crit)
        self.btn_del_crit = QPushButton("Delete")
        self.btn_del_crit.clicked.connect(self._delete_crit)
        crit_btn_row.addWidget(self.btn_save_crit)
        crit_btn_row.addWidget(self.btn_del_crit)
        crit_form.addRow("", crit_btn_row)

        left_layout.addWidget(crit_box)
        splitter.addWidget(left)

        # Right: Ratings
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("<b>Rating Rules</b> (first matching rule sets step 'bewertung'):"))
        self.rating_table = QTableWidget(0, 4)
        self.rating_table.setHorizontalHeaderLabels(["Label", "All of", "Any of", "None of"])
        self.rating_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.rating_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.rating_table, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([350, 300])

        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Criteria & Ratings")

    # ------------------------------------------------------------------ State & Refresh
    def set_session(self, session: CampaignSession | None) -> None:
        self._session = session
        self._update_scenario_combo()
        self.refresh()

    def _update_scenario_combo(self) -> None:
        curr = self.scenario_combo.currentText()
        self.scenario_combo.blockSignals(True)
        self.scenario_combo.clear()
        self.scenario_combo.addItem("(Base)")
        if self._session is not None and self._session.campaign.evaluation:
            for name in self._session.campaign.evaluation.scenario_names():
                self.scenario_combo.addItem(name)
        idx = self.scenario_combo.findText(curr)
        self.scenario_combo.setCurrentIndex(max(0, idx))
        self.scenario_combo.blockSignals(False)

    def current_scenario_name(self) -> str | None:
        text = self.scenario_combo.currentText()
        return None if text in ("", "(Base)") else text

    def refresh(self) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self._refresh_config_views()
            self._compute_and_populate_results()
        finally:
            self._updating = False

    def _refresh_config_views(self) -> None:
        ev = self._session.campaign.evaluation if self._session else Evaluation()

        # Calculations table
        self.calc_table.setRowCount(len(ev.calculations))
        for i, c in enumerate(ev.calculations):
            self.calc_table.setItem(i, 0, QTableWidgetItem(c.name))
            self.calc_table.setItem(i, 1, QTableWidgetItem(c.expr))
            self.calc_table.setItem(i, 2, QTableWidgetItem(c.unit))
            self.calc_table.setItem(i, 3, QTableWidgetItem(c.note))
        self.calc_table.resizeColumnsToContents()

        # Constants table
        self.const_table.setRowCount(len(ev.constants))
        for i, (name, c) in enumerate(ev.constants.items()):
            self.const_table.setItem(i, 0, QTableWidgetItem(name))
            self.const_table.setItem(i, 1, NumericTableWidgetItem(str(c.value), c.value))
            self.const_table.setItem(i, 2, NumericTableWidgetItem(str(c.u), c.u))
            self.const_table.setItem(i, 3, QTableWidgetItem(c.kind))
            self.const_table.setItem(i, 4, QTableWidgetItem(c.scope))
            self.const_table.setItem(i, 5, QTableWidgetItem(c.unit))
            self.const_table.setItem(i, 6, QTableWidgetItem(c.note))
        self.const_table.resizeColumnsToContents()

        # Scenarios table
        self.scen_table.setRowCount(len(ev.scenarios))
        for i, s in enumerate(ev.scenarios):
            overrides = ", ".join(f"{k}={v.value}" for k, v in s.constants.items())
            self.scen_table.setItem(i, 0, QTableWidgetItem(s.name))
            self.scen_table.setItem(i, 1, QTableWidgetItem(overrides))
        self.scen_table.resizeColumnsToContents()

        # Criteria table
        self.crit_table.setRowCount(len(ev.criteria))
        for i, cr in enumerate(ev.criteria):
            self.crit_table.setItem(i, 0, QTableWidgetItem(cr.name))
            self.crit_table.setItem(i, 1, QTableWidgetItem(cr.expr))
            self.crit_table.setItem(i, 2, QTableWidgetItem(cr.text))
        self.crit_table.resizeColumnsToContents()

        # Ratings table
        self.rating_table.setRowCount(len(ev.ratings))
        for i, r in enumerate(ev.ratings):
            self.rating_table.setItem(i, 0, QTableWidgetItem(r.label))
            self.rating_table.setItem(i, 1, QTableWidgetItem(", ".join(r.all_of)))
            self.rating_table.setItem(i, 2, QTableWidgetItem(", ".join(r.any_of)))
            self.rating_table.setItem(i, 3, QTableWidgetItem(", ".join(r.none_of)))
        self.rating_table.resizeColumnsToContents()

    def _compute_and_populate_results(self) -> None:
        session = self._session
        if session is None:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.status_label.setText("No campaign session.")
            self._results_df = None
            return

        scenario = self.current_scenario_name()
        try:
            df = session.results(scenario=scenario)
        except Exception as exc:
            self.table.setRowCount(0)
            self.status_label.setText(f"Error evaluating results: {exc}")
            self._results_df = None
            return

        self._results_df = df
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)

        has_sid = "step_id" in df.columns

        for r_idx in range(len(df)):
            sid = str(df["step_id"][r_idx]) if has_sid else ""
            for c_idx, col in enumerate(df.columns):
                raw_val = df[col][r_idx]
                item_text = _fmt(raw_val)
                item = NumericTableWidgetItem(item_text, raw_val)
                if sid:
                    item.setData(Qt.ItemDataRole.UserRole, sid)
                self.table.setItem(r_idx, c_idx, item)

        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self._apply_filter()

        scen_text = f"Scenario: {scenario}" if scenario else "Scenario: (Base)"
        self.status_label.setText(f"Showing {len(df)} steps. {scen_text}")

    # ------------------------------------------------------------------ Actions & Slots
    def _on_scenario_changed(self) -> None:
        self._compute_and_populate_results()

    def _apply_filter(self) -> None:
        text = self.filter_edit.text().strip().lower()
        visible_count = 0
        total_count = self.table.rowCount()
        for r in range(total_count):
            if not text:
                self.table.setRowHidden(r, False)
                visible_count += 1
            else:
                match = False
                for c in range(self.table.columnCount()):
                    item = self.table.item(r, c)
                    if item and text in item.text().lower():
                        match = True
                        break
                self.table.setRowHidden(r, not match)
                if match:
                    visible_count += 1
        scen = self.current_scenario_name() or "(Base)"
        self.status_label.setText(f"Showing {visible_count} of {total_count} steps. Scenario: {scen}")

    def _open_columns_menu(self) -> None:
        menu = QMenu(self)
        act_show_all = menu.addAction("Show All Columns")
        act_show_all.triggered.connect(self._show_all_columns)

        act_hide_budget = menu.addAction("Hide Budget Columns (b_*)")
        act_hide_budget.triggered.connect(self._hide_budget_columns)

        menu.addSeparator()
        for c in range(self.table.columnCount()):
            header = self.table.horizontalHeaderItem(c)
            name = header.text() if header else f"Col {c}"
            act = menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(not self.table.isColumnHidden(c))
            act.toggled.connect(lambda checked, col=c: self.table.setColumnHidden(col, not checked))

        menu.exec(self.btn_columns.mapToGlobal(self.btn_columns.rect().bottomLeft()))

    def _show_all_columns(self) -> None:
        for c in range(self.table.columnCount()):
            self.table.setColumnHidden(c, False)

    def _hide_budget_columns(self) -> None:
        for c in range(self.table.columnCount()):
            header = self.table.horizontalHeaderItem(c)
            if header and header.text().startswith("b_"):
                self.table.setColumnHidden(c, True)

    def export_csv(self) -> None:
        if self._results_df is None or len(self._results_df) == 0:
            QMessageBox.information(self, "Export CSV", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results CSV", "results.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            try:
                self._results_df.write_csv(path, separator=";")
            except Exception as exc:
                QMessageBox.critical(self, "Export CSV Error", f"Cannot write CSV:\n\n{exc}")

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        item = self.table.item(row, col)
        if item is not None:
            sid = item.data(Qt.ItemDataRole.UserRole)
            if sid:
                self.step_double_clicked.emit(str(sid))

    # ------------------------------------------------------------------ Calculations Editor
    def _on_calc_expr_changed(self, text: str) -> None:
        text = text.strip()
        if not text:
            self.calc_syntax_label.setText("Enter formula")
            self.calc_syntax_label.setStyleSheet("color: #999999;")
            return
        try:
            compile_expr(text)
            self.calc_syntax_label.setText("✓ Syntax valid")
            self.calc_syntax_label.setStyleSheet(f"color: {VALID_COLOR}; font-weight: bold;")
        except ExprError as exc:
            self.calc_syntax_label.setText(f"✗ {exc.message}")
            self.calc_syntax_label.setStyleSheet(f"color: {ERROR_COLOR};")
        except Exception as exc:
            self.calc_syntax_label.setText(f"✗ {exc}")
            self.calc_syntax_label.setStyleSheet(f"color: {ERROR_COLOR};")

    def _on_calc_selected(self) -> None:
        row = self.calc_table.currentRow()
        if row >= 0:
            self.calc_name_edit.setText(self.calc_table.item(row, 0).text())
            self.calc_expr_edit.setText(self.calc_table.item(row, 1).text())
            self.calc_unit_edit.setText(self.calc_table.item(row, 2).text())
            self.calc_note_edit.setText(self.calc_table.item(row, 3).text())

    def _clear_calc_form(self) -> None:
        self.calc_table.clearSelection()
        self.calc_name_edit.clear()
        self.calc_expr_edit.clear()
        self.calc_unit_edit.clear()
        self.calc_note_edit.clear()

    def _save_calc(self) -> None:
        if not self._session:
            return
        name = self.calc_name_edit.text().strip()
        expr = self.calc_expr_edit.text().strip()
        unit = self.calc_unit_edit.text().strip()
        note = self.calc_note_edit.text().strip()
        if not name or not expr:
            QMessageBox.warning(self, "Invalid Calculation", "Calculation requires a Name and an Expression.")
            return
        try:
            calc = Calculation(name=name, expr=expr, unit=unit, note=note)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Calculation", str(exc))
            return

        ev = self._session.campaign.evaluation
        idx = next((i for i, c in enumerate(ev.calculations) if c.name == name), None)
        new_calcs = list(ev.calculations)
        if idx is not None:
            new_calcs[idx] = calc
        else:
            new_calcs.append(calc)

        try:
            new_ev = ev.model_copy(update={"calculations": new_calcs})
            new_ev._consistent()
            self._session.campaign.evaluation = new_ev
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Save Calculation", str(exc))
            return

        self.refresh()
        self.evaluation_changed.emit()

    def _delete_calc(self) -> None:
        row = self.calc_table.currentRow()
        if row < 0 or not self._session:
            return
        name = self.calc_table.item(row, 0).text()
        ev = self._session.campaign.evaluation
        new_calcs = [c for c in ev.calculations if c.name != name]
        try:
            new_ev = ev.model_copy(update={"calculations": new_calcs})
            new_ev._consistent()
            self._session.campaign.evaluation = new_ev
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Delete Calculation", str(exc))
            return
        self._clear_calc_form()
        self.refresh()
        self.evaluation_changed.emit()

    # ------------------------------------------------------------------ Constants Editor
    def _on_const_selected(self) -> None:
        row = self.const_table.currentRow()
        if row >= 0:
            self.const_name_edit.setText(self.const_table.item(row, 0).text())
            self.const_value_edit.setText(self.const_table.item(row, 1).text())
            self.const_u_edit.setText(self.const_table.item(row, 2).text())
            idx = self.const_kind_combo.findText(self.const_table.item(row, 3).text())
            self.const_kind_combo.setCurrentIndex(max(0, idx))
            self.const_scope_edit.setText(self.const_table.item(row, 4).text())
            self.const_unit_edit.setText(self.const_table.item(row, 5).text())
            self.const_note_edit.setText(self.const_table.item(row, 6).text())

    def _clear_const_form(self) -> None:
        self.const_table.clearSelection()
        self.const_name_edit.clear()
        self.const_value_edit.clear()
        self.const_u_edit.clear()
        self.const_kind_combo.setCurrentIndex(0)
        self.const_scope_edit.clear()
        self.const_unit_edit.clear()
        self.const_note_edit.clear()

    def _save_const(self) -> None:
        if not self._session:
            return
        name = self.const_name_edit.text().strip()
        val_str = self.const_value_edit.text().strip()
        u_str = self.const_u_edit.text().strip() or "0.0"
        kind = self.const_kind_combo.currentText()
        scope = self.const_scope_edit.text().strip()
        unit = self.const_unit_edit.text().strip()
        note = self.const_note_edit.text().strip()
        if not name or not val_str:
            QMessageBox.warning(self, "Invalid Constant", "Constant requires a Name and a Value.")
            return
        try:
            val = float(val_str)
            u = float(u_str)
            c = Constant(name=name, value=val, u=u, kind=kind, scope=scope, unit=unit, note=note)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Constant", str(exc))
            return

        ev = self._session.campaign.evaluation
        new_consts = dict(ev.constants)
        new_consts[name] = c
        try:
            new_ev = ev.model_copy(update={"constants": new_consts})
            new_ev._consistent()
            self._session.campaign.evaluation = new_ev
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Save Constant", str(exc))
            return

        self.refresh()
        self.evaluation_changed.emit()

    def _delete_const(self) -> None:
        row = self.const_table.currentRow()
        if row < 0 or not self._session:
            return
        name = self.const_table.item(row, 0).text()
        ev = self._session.campaign.evaluation
        new_consts = {k: v for k, v in ev.constants.items() if k != name}
        try:
            new_ev = ev.model_copy(update={"constants": new_consts})
            new_ev._consistent()
            self._session.campaign.evaluation = new_ev
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Delete Constant", str(exc))
            return
        self._clear_const_form()
        self.refresh()
        self.evaluation_changed.emit()

    # ------------------------------------------------------------------ Criteria Editor
    def _on_crit_expr_changed(self, text: str) -> None:
        text = text.strip()
        if not text:
            self.crit_syntax_label.setText("Enter formula")
            self.crit_syntax_label.setStyleSheet("color: #999999;")
            return
        try:
            compile_expr(text)
            self.crit_syntax_label.setText("✓ Syntax valid")
            self.crit_syntax_label.setStyleSheet(f"color: {VALID_COLOR}; font-weight: bold;")
        except ExprError as exc:
            self.crit_syntax_label.setText(f"✗ {exc.message}")
            self.crit_syntax_label.setStyleSheet(f"color: {ERROR_COLOR};")
        except Exception as exc:
            self.crit_syntax_label.setText(f"✗ {exc}")
            self.crit_syntax_label.setStyleSheet(f"color: {ERROR_COLOR};")

    def _on_crit_selected(self) -> None:
        row = self.crit_table.currentRow()
        if row >= 0:
            self.crit_name_edit.setText(self.crit_table.item(row, 0).text())
            self.crit_expr_edit.setText(self.crit_table.item(row, 1).text())
            self.crit_text_edit.setText(self.crit_table.item(row, 2).text())

    def _save_crit(self) -> None:
        if not self._session:
            return
        name = self.crit_name_edit.text().strip()
        expr = self.crit_expr_edit.text().strip()
        text = self.crit_text_edit.text().strip()
        if not name or not expr or not text:
            QMessageBox.warning(self, "Invalid Criterion", "Criterion requires Name, Formula and Reason Text.")
            return
        try:
            cr = Criterion(name=name, expr=expr, text=text)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Criterion", str(exc))
            return

        ev = self._session.campaign.evaluation
        idx = next((i for i, c in enumerate(ev.criteria) if c.name == name), None)
        new_crits = list(ev.criteria)
        if idx is not None:
            new_crits[idx] = cr
        else:
            new_crits.append(cr)

        try:
            new_ev = ev.model_copy(update={"criteria": new_crits})
            new_ev._consistent()
            self._session.campaign.evaluation = new_ev
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Save Criterion", str(exc))
            return

        self.refresh()
        self.evaluation_changed.emit()

    def _delete_crit(self) -> None:
        row = self.crit_table.currentRow()
        if row < 0 or not self._session:
            return
        name = self.crit_table.item(row, 0).text()
        ev = self._session.campaign.evaluation
        new_crits = [c for c in ev.criteria if c.name != name]
        try:
            new_ev = ev.model_copy(update={"criteria": new_crits})
            new_ev._consistent()
            self._session.campaign.evaluation = new_ev
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Delete Criterion", str(exc))
            return
        self.crit_table.clearSelection()
        self.crit_name_edit.clear()
        self.crit_expr_edit.clear()
        self.crit_text_edit.clear()
        self.refresh()
        self.evaluation_changed.emit()
