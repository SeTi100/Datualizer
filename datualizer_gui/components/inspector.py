"""Inspector Panel: File loading, metadata inspection, channel checklist, and error audit."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.dataset import DualModeDataset
from datualizer_core.ingestion.pre_scanner import pre_scan


class InspectorPanel(QWidget):
    """Inspector panel providing file management, metadata readout, channel selection,

    and audit error inspection.
    """

    file_selected = Signal(str)
    files_selected = Signal(list)  # Several files to load and merge
    files_added = Signal(list)  # Files to merge into the current data
    channels_toggled = Signal(list)  # Emits list of selected channel names
    run_selected = Signal(object)  # Emits the run key tuple, or None for all runs

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: DualModeDataset | None = None
        self._file_path: str = ""
        self._all_channels: list[str] = []
        self._run_keys: list[tuple | None] = []
        self._is_updating_ui = False

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # Scroll area container for long content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        # 1. File Actions
        file_group = QGroupBox("File & Ingestion")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("Open CSV...")
        self.btn_open.setStyleSheet("font-weight: bold;")
        self.btn_open.clicked.connect(self._on_open_file_clicked)
        btn_row.addWidget(self.btn_open)

        self.btn_add = QPushButton("Add (merge)...")
        self.btn_add.setToolTip("Merge further files into the current data; later files win on overlaps")
        self.btn_add.clicked.connect(self._on_add_files_clicked)
        btn_row.addWidget(self.btn_add)

        self.btn_quick_load = QPushButton("Quick Load: SA_testmessung_1")
        self.btn_quick_load.clicked.connect(self._on_quick_load_clicked)
        btn_row.addWidget(self.btn_quick_load)

        file_layout.addLayout(btn_row)

        self.lbl_filepath = QLabel("No file loaded")
        self.lbl_filepath.setWordWrap(True)
        self.lbl_filepath.setStyleSheet("color: #888888; font-size: 10px;")
        file_layout.addWidget(self.lbl_filepath)

        layout.addWidget(file_group)

        # 2. Metadata Inspection Area
        meta_group = QGroupBox("Dataset Metadata")
        meta_layout = QVBoxLayout(meta_group)
        meta_layout.setSpacing(3)

        self.lbl_rows = QLabel("Rows: -")
        self.lbl_cols = QLabel("Columns: -")
        self.lbl_delimiter = QLabel("Delimiter: -")
        self.lbl_decimal = QLabel("Decimal Separator: -")
        self.lbl_timespan = QLabel("Time Span: -")

        for lbl in (
            self.lbl_rows,
            self.lbl_cols,
            self.lbl_delimiter,
            self.lbl_decimal,
            self.lbl_timespan,
        ):
            lbl.setStyleSheet("font-size: 11px;")
            meta_layout.addWidget(lbl)

        layout.addWidget(meta_group)

        # 2b. Runs (only shown when the data has run columns)
        self.run_group = QGroupBox("Runs")
        run_layout = QVBoxLayout(self.run_group)
        run_layout.setSpacing(4)
        self.run_table = QTableWidget(0, 6)
        self.run_table.setHorizontalHeaderLabels(
            ["Run", "Samples", "Duration", "Δt", "Parameters", "Status"]
        )
        self.run_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.run_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.run_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.run_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.run_table.verticalHeader().setVisible(False)
        self.run_table.setMaximumHeight(180)
        self.run_table.itemSelectionChanged.connect(self._on_run_selection_changed)
        run_layout.addWidget(self.run_table)
        self.run_group.setHidden(True)
        layout.addWidget(self.run_group)

        # 3. Channel Checklist
        chan_group = QGroupBox("Channels & Subplots")
        chan_layout = QVBoxLayout(chan_group)
        chan_layout.setSpacing(4)

        chan_btn_row = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self.select_all_channels)
        chan_btn_row.addWidget(self.btn_select_all)

        self.btn_deselect_all = QPushButton("Deselect All")
        self.btn_deselect_all.clicked.connect(self.deselect_all_channels)
        chan_btn_row.addWidget(self.btn_deselect_all)

        chan_layout.addLayout(chan_btn_row)

        self.channel_list_widget = QListWidget()
        self.channel_list_widget.setStyleSheet(
            "QListWidget::item { padding: 4px; border-bottom: 1px solid #2d2d38; }"
        )
        self.channel_list_widget.itemChanged.connect(self._on_item_changed)
        chan_layout.addWidget(self.channel_list_widget)

        layout.addWidget(chan_group)

        # 4. Error Audit Area
        audit_group = QGroupBox("Conversion Audit Log")
        audit_layout = QVBoxLayout(audit_group)
        audit_layout.setSpacing(4)

        self.lbl_audit_summary = QLabel("Audit Log: No dataset loaded")
        self.lbl_audit_summary.setStyleSheet("font-weight: bold; font-size: 11px;")
        audit_layout.addWidget(self.lbl_audit_summary)

        self.audit_table = QTableWidget(0, 4)
        self.audit_table.setHorizontalHeaderLabels(["Row", "Column", "Raw Value", "Reason"])
        self.audit_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.audit_table.setAlternatingRowColors(True)
        self.audit_table.setMaximumHeight(150)
        audit_layout.addWidget(self.audit_table)

        layout.addWidget(audit_group)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def set_dataset(
        self,
        dataset: DualModeDataset | None,
        file_path: str | Path = "",
        delimiter: str = ",",
        decimal_sep: str = ",",
    ) -> None:
        """Bind dataset, update metadata, channel list, and audit entries."""
        self._dataset = dataset
        self._file_path = str(file_path)
        self._is_updating_ui = True

        if self._dataset is None:
            self.lbl_filepath.setText("No file loaded")
            self.lbl_rows.setText("Rows: -")
            self.lbl_cols.setText("Columns: -")
            self.lbl_delimiter.setText("Delimiter: -")
            self.lbl_decimal.setText("Decimal Separator: -")
            self.lbl_timespan.setText("Time Span: -")
            self.channel_list_widget.clear()
            self._all_channels = []
            self._fill_run_table()
            self.lbl_audit_summary.setText("Audit Log: No dataset loaded")
            self.audit_table.setRowCount(0)
            self._is_updating_ui = False
            return

        # 1. File path & Metadata
        sources = self._dataset.sources
        if len(sources) > 1:
            self.lbl_filepath.setText(self._sources_text(sources))
        elif self._file_path:
            p = Path(self._file_path)
            self.lbl_filepath.setText(f"{p.name} ({p.resolve()})")
        else:
            self.lbl_filepath.setText("In-Memory Dataset")

        n_rows = len(self._dataset)
        n_cols = len(self._dataset.columns)
        self.lbl_rows.setText(f"Rows: {n_rows:,}")
        self.lbl_cols.setText(f"Columns: {n_cols}")
        self.lbl_delimiter.setText(f"Delimiter: {repr(delimiter)}")
        self.lbl_decimal.setText(f"Decimal Separator: {repr(decimal_sep)}")

        # Time span
        time_col = self._dataset.time_col
        if time_col in self._dataset.columns and len(self._dataset) > 0:
            time_series = self._dataset[time_col]
            t_min = time_series.min()
            t_max = time_series.max()
            if t_min is not None and t_max is not None:
                duration_s = t_max - t_min
                m, s = divmod(int(duration_s), 60)
                h, m = divmod(m, 60)
                span_str = (
                    f"{t_min:.1f}s – {t_max:.1f}s (Duration: {h:02d}:{m:02d}:{s:02d})"
                )
                self.lbl_timespan.setText(f"Time Span: {span_str}")
            else:
                self.lbl_timespan.setText("Time Span: Empty time range")
        else:
            self.lbl_timespan.setText("Time Span: N/A")

        # 2. Channels Checklist
        self.channel_list_widget.clear()
        self._all_channels = list(self._dataset.channels)
        for ch in self._all_channels:
            item = QListWidgetItem(ch)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self.channel_list_widget.addItem(item)
        self._apply_fill_ratios(self._dataset.fill_ratio)
        self._fill_run_table()

        # 3. Audit Log Entries
        audit_log = self._dataset.audit_log
        err_count = len(audit_log)
        if err_count == 0:
            self.lbl_audit_summary.setText("Audit Log: 0 Conversion Errors (Clean)")
            self.lbl_audit_summary.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 11px;")
            self.audit_table.setRowCount(0)
        else:
            self.lbl_audit_summary.setText(f"Audit Log: {err_count} Conversion Anomalies")
            self.lbl_audit_summary.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 11px;")
            self.audit_table.setRowCount(err_count)
            for row_idx, entry in enumerate(audit_log.entries):
                self.audit_table.setItem(row_idx, 0, QTableWidgetItem(str(entry.row_index)))
                self.audit_table.setItem(row_idx, 1, QTableWidgetItem(str(entry.column)))
                self.audit_table.setItem(row_idx, 2, QTableWidgetItem(str(entry.raw_value)))
                self.audit_table.setItem(row_idx, 3, QTableWidgetItem(str(entry.reason)))

        self._is_updating_ui = False
        self.channels_toggled.emit(self.get_selected_channels())

    def _apply_fill_ratios(self, fill_ratio: dict[str, float]) -> None:
        """Check channels with values; gray out and uncheck channels without any value."""
        for idx in range(self.channel_list_widget.count()):
            item = self.channel_list_widget.item(idx)
            ratio = fill_ratio.get(item.text(), 0.0)
            if ratio == 0.0:
                # Empty channels stay listed (marked, not dropped) but are not plotted by default
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setForeground(Qt.GlobalColor.gray)
                item.setToolTip("No values (fill ratio 0 %)")
            else:
                item.setCheckState(Qt.CheckState.Checked)
                item.setData(Qt.ItemDataRole.ForegroundRole, None)
                item.setToolTip(f"Fill ratio: {ratio * 100:.0f} %")

    def _fill_run_table(self) -> None:
        """List all runs with size, timing, parameters and an 'aborted' badge (P10, P11, P18)."""
        self.run_table.blockSignals(True)
        self.run_table.setRowCount(0)
        self._run_keys = []
        ds = self._dataset
        if ds is None or not ds.run_columns:
            self.run_group.setHidden(True)
            self.run_table.blockSignals(False)
            return

        runs = ds.runs
        self.run_table.setRowCount(len(runs) + 1)
        self.run_table.setItem(0, 0, QTableWidgetItem("All runs"))
        self.run_table.setItem(0, 1, QTableWidgetItem(f"{len(ds):,}"))
        for col in range(2, self.run_table.columnCount()):
            self.run_table.setItem(0, col, QTableWidgetItem(""))
        self._run_keys.append(None)

        for row, run in enumerate(runs.iter_rows(named=True), start=1):
            key = tuple(run[c] for c in ds.run_columns)
            self._run_keys.append(key)
            params = ", ".join(f"{p}={run[p]:g}" for p in ds.parameters if run.get(p) is not None)
            duration, dt = run["duration_s"], run["median_dt_s"]
            cells = [
                " / ".join(str(v) for v in key),
                f"{run['n_samples']:,}",
                f"{duration:.1f} s" if duration is not None else "-",
                f"{dt:.3g} s" if dt is not None else "-",
                params,
                "aborted" if run["aborted"] else "",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if run["aborted"]:
                    item.setForeground(Qt.GlobalColor.red)
                self.run_table.setItem(row, col, item)

        self.run_group.setHidden(False)
        self.run_table.selectRow(0)
        self.run_table.blockSignals(False)

    def select_run(self, key: tuple | None) -> None:
        """Select a run by key (None = all runs); emits `run_selected`."""
        if key not in self._run_keys:
            raise KeyError(f"Run {key!r} not listed.")
        self.run_table.selectRow(self._run_keys.index(key))

    def _on_run_selection_changed(self) -> None:
        rows = self.run_table.selectionModel().selectedRows()
        if not rows or self._dataset is None:
            return
        key = self._run_keys[rows[0].row()]
        if key is None:
            ratios = self._dataset.fill_ratio
        else:
            avail = self._dataset.channel_availability
            for c, v in zip(self._dataset.run_columns, key):
                avail = avail.filter(avail[c].is_null() if v is None else avail[c] == v)
            ratios = avail.row(0, named=True)
        self._is_updating_ui = True
        self._apply_fill_ratios(ratios)
        self._is_updating_ui = False
        self.run_selected.emit(key)

    def get_selected_channels(self) -> list[str]:
        """Return list of channel names currently checked."""
        selected: list[str] = []
        for idx in range(self.channel_list_widget.count()):
            item = self.channel_list_widget.item(idx)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected

    def set_selected_channels(self, channels: Sequence[str]) -> None:
        """Set check state for specific channels."""
        self._is_updating_ui = True
        target = set(channels)
        for idx in range(self.channel_list_widget.count()):
            item = self.channel_list_widget.item(idx)
            state = Qt.CheckState.Checked if item.text() in target else Qt.CheckState.Unchecked
            item.setCheckState(state)
        self._is_updating_ui = False
        self.channels_toggled.emit(self.get_selected_channels())

    def select_all_channels(self) -> None:
        """Select all available channels in the list."""
        self._is_updating_ui = True
        for idx in range(self.channel_list_widget.count()):
            self.channel_list_widget.item(idx).setCheckState(Qt.CheckState.Checked)
        self._is_updating_ui = False
        self.channels_toggled.emit(self.get_selected_channels())

    def deselect_all_channels(self) -> None:
        """Deselect all channels in the list."""
        self._is_updating_ui = True
        for idx in range(self.channel_list_widget.count()):
            self.channel_list_widget.item(idx).setCheckState(Qt.CheckState.Unchecked)
        self._is_updating_ui = False
        self.channels_toggled.emit(self.get_selected_channels())

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if not self._is_updating_ui:
            self.channels_toggled.emit(self.get_selected_channels())

    @staticmethod
    def _sources_text(sources) -> str:
        """One line per merged file: rows kept, replaced by a later file, or skipped as duplicate."""
        lines = [f"{len(sources)} files merged:"]
        for row in sources.iter_rows(named=True):
            name = Path(row["source"]).name
            if row["duplicate_of"] is not None:
                lines.append(f"• {name}: duplicate of {Path(row['duplicate_of']).name}, skipped")
            else:
                text = f"• {name}: {row['n_kept']:,} of {row['n_rows']:,} rows kept"
                if row["n_conflicts"]:
                    text += f", {row['n_conflicts']} conflicting values"
                lines.append(text)
        return "\n".join(lines)

    def _on_open_file_clicked(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Measurement CSV (select several to merge)",
            "",
            "CSV / Delimited Files (*.csv *.tsv *.txt);;All Files (*)",
        )
        if len(file_paths) == 1:
            self.file_selected.emit(file_paths[0])
        elif file_paths:
            self.files_selected.emit(file_paths)

    def _on_add_files_clicked(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Measurement CSV (merge into current data)",
            "",
            "CSV / Delimited Files (*.csv *.tsv *.txt);;All Files (*)",
        )
        if file_paths:
            self.files_added.emit(file_paths)

    def _on_quick_load_clicked(self) -> None:
        target_name = "SA_testmessung_1.csv"
        # Check current working directory or relative to project root
        candidates = [
            Path.cwd() / target_name,
            Path(__file__).resolve().parent.parent.parent / target_name,
        ]
        for c in candidates:
            if c.exists():
                self.file_selected.emit(str(c))
                return

        # Fallback emit target name directly
        self.file_selected.emit(target_name)
