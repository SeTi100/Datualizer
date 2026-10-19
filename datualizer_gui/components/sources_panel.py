"""Source list of a campaign: channels to plot, their display settings and the clock model of each source.

Calculated channels are listed like the others (in italics, formula and unit in the tooltip);
each source with masks gets a "Masks" node. Both are edited from the context menu.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.campaign import ChannelDisplay, CampaignSession, ClockModel
from datualizer_core.ingestion.calibration import calibration_fields
from datualizer_core.ingestion.source_loader import load_source
from datualizer_core.schema import PlotStyle
from datualizer_gui.components.channel_display_dialog import STYLE_NAMES, ChannelDisplayDialog
from datualizer_gui.components.clock_sync_dialog import ClockSyncDialog
from datualizer_gui.components.derived_dialog import TIME_FORMAT, CalibrationDialog, DerivedChannelDialog, MaskDialog

ROLE_SOURCE = Qt.ItemDataRole.UserRole
ROLE_CHANNEL = Qt.ItemDataRole.UserRole + 1
ROLE_MASK = Qt.ItemDataRole.UserRole + 2  # index of the mask in its source

SETTINGS_FILTER = "Device Settings (*.xml);;All Files (*)"


def channel_tooltip(session: CampaignSession, source_id: str, channel: str) -> str:
    """Name in the data, unit and, for a calculated channel, its formulas and origin."""
    src = session.campaign.source(source_id)
    ds = session.datasets.get(source_id)
    unit = ds.units.get(channel, "") if ds is not None else ""
    lines = [f"Channel in the data: {channel}", f"Unit: {unit or '(none)'}"]
    dc = src.derived_channel(channel)
    if dc is not None:
        lines[0] = f"Calculated channel: {channel}"
        lines.append(dc.describe())
        if dc.note:
            lines.append(f"Origin: {dc.note}")
    lines.append("Right-click or double-click for display settings.")
    return "\n".join(lines)


def calibration_tooltip(meta: dict) -> str:
    """The calibration values a settings file holds (empty if it has none)."""
    fields = calibration_fields(meta)
    block = (meta.get("xml") or {}).get(fields.block) if fields is not None else None
    if not block:
        return ""
    values = ", ".join(f"{k} = {v}" for k, v in block.items())
    return f"Calibration ({fields.block}): {values}\nRight-click: calibrated channel from it"


def _fmt(epoch_s: float, zone: str) -> str:
    import polars as pl

    s = pl.Series([int(epoch_s * 1e6)]).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    return s.dt.convert_time_zone(zone).dt.strftime("%d.%m. %H:%M:%S")[0]


class SourcesPanel(QWidget):
    """Tree of sources (with time span, profile, clock offset) and their channels (checkable)."""

    selection_changed = Signal()      # checked channels changed
    clock_changed = Signal(str)       # source id whose clock model was edited
    display_changed = Signal(str, str)  # source id, channel whose display settings were edited
    add_requested = Signal()
    remove_requested = Signal(str)
    data_changed = Signal(str)  # source id whose calculated channels or masks changed

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session: CampaignSession | None = None
        self._updating = False
        # Visible master time span of the plot (set by the campaign window), default span of the align dialog
        self.view_range: Callable[[], tuple[float, float] | None] = lambda: None

        layout = QVBoxLayout(self)
        buttons = QHBoxLayout()
        self.btn_add = QPushButton("Add Source...")
        self.btn_remove = QPushButton("Remove")
        buttons.addWidget(self.btn_add)
        buttons.addWidget(self.btn_remove)
        layout.addLayout(buttons)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Source / Channel", "Profile", "Time span (master)", "Clock"])
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.currentItemChanged.connect(lambda *_: self._show_clock())
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.tree, stretch=1)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #ffb86c;")
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        box = QGroupBox("Clock of selected source")
        form = QFormLayout(box)
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-1e7, 1e7)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSuffix(" s")
        self.drift_spin = QDoubleSpinBox()
        self.drift_spin.setRange(-1e6, 1e6)
        self.drift_spin.setDecimals(2)
        self.drift_spin.setSuffix(" ppm")
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Assumption, e.g. 'FID clock behind PC, incl. transport delay'")
        self.anchor_label = QLabel()
        self.method_label = QLabel()
        self.btn_align = QPushButton("Align Clock...")
        self.btn_align.setToolTip("Find the offset by cross-correlation with another source or from journal clock pairs")
        form.addRow("Offset", self.offset_spin)
        form.addRow("Drift", self.drift_spin)
        form.addRow("Note", self.note_edit)
        form.addRow("Anchors", self.anchor_label)
        form.addRow("Found by", self.method_label)
        form.addRow("", self.btn_align)
        layout.addWidget(box)
        self.clock_box = box

        self.btn_add.clicked.connect(self.add_requested)
        self.btn_remove.clicked.connect(self._on_remove)
        self.offset_spin.editingFinished.connect(self._apply_clock)
        self.drift_spin.editingFinished.connect(self._apply_clock)
        self.note_edit.editingFinished.connect(self._apply_clock)
        self.btn_align.clicked.connect(lambda: self.align_clock())

    # ------------------------------------------------------------------ state
    def set_session(self, session: CampaignSession | None) -> None:
        checked = set(self.checked_channels())
        self._session = session
        self._updating = True
        self.tree.clear()
        if session is not None:
            for src in session.campaign.sources:
                ds = session.datasets.get(src.id)
                span = session.span(src.id)
                span_txt = (
                    f"{_fmt(span[0], session.campaign.timezone)} – {_fmt(span[1], session.campaign.timezone)}"
                    if span else "no absolute time"
                )
                clock = src.clock
                u = f" ± {clock.uncertainty_s:.3g}" if clock.uncertainty_s is not None else ""
                clock_txt = "—" if clock.is_identity() else (
                    f"{len(clock.anchors)} anchors" if clock.anchors else f"{clock.offset_s:+.1f}{u} s"
                )
                item = QTreeWidgetItem([src.label or src.id, src.profile.name if src.profile else "-", span_txt, clock_txt])
                item.setData(0, ROLE_SOURCE, src.id)
                cal_tip = calibration_tooltip(ds.source_meta) if ds is not None else ""
                if cal_tip:
                    item.setToolTip(0, cal_tip)
                self.tree.addTopLevelItem(item)
                for ch in (ds.channels if ds is not None else []):
                    alias = src.channel_display(ch).alias
                    child = QTreeWidgetItem([f"{alias} ({ch})" if alias else ch, ds.units.get(ch, ""), "", ""])
                    child.setToolTip(0, channel_tooltip(session, src.id, ch))
                    if src.derived_channel(ch) is not None:
                        font = child.font(0)
                        font.setItalic(True)
                        child.setFont(0, font)
                    child.setData(0, ROLE_SOURCE, src.id)
                    child.setData(0, ROLE_CHANNEL, ch)
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    on = (src.id, ch) in checked or (not checked and ch == ds.channels[0])
                    child.setCheckState(0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                    item.addChild(child)
                if src.masks:
                    self._add_masks(item, src.id)
                item.setExpanded(True)
            warnings = list(session.warnings)
            for src in session.campaign.sources:
                warnings += [f"{src.id}: {p}" for p in session.derived_problems(src.id)]
            self.warning_label.setText("\n".join(warnings))
            self.warning_label.setVisible(bool(warnings))
        for col in range(4):
            self.tree.resizeColumnToContents(col)
        self._updating = False
        self._show_clock()

    def _add_masks(self, item: QTreeWidgetItem, source_id: str) -> None:
        src = self._session.campaign.source(source_id)
        node = QTreeWidgetItem([f"Masks ({len(src.masks)})", "", "", ""])
        node.setData(0, ROLE_SOURCE, source_id)
        node.setToolTip(0, "Time spans flagged 'masked' (left out of the statistics). Right-click to edit.")
        for i, m in enumerate(src.masks):
            what = ", ".join(m.channels) if m.channels else "all channels"
            span = f"{m.start.strftime(TIME_FORMAT)} – {m.end.strftime('%H:%M:%S')}"
            child = QTreeWidgetItem([span, what, m.reason, ""])
            child.setToolTip(0, f"{span}\n{what}\n{m.reason or 'no reason given'}")
            child.setData(0, ROLE_SOURCE, source_id)
            child.setData(0, ROLE_MASK, i)
            node.addChild(child)
        item.addChild(node)
        node.setExpanded(True)

    def checked_channels(self) -> list[tuple[str, str]]:
        out = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                c = top.child(j)
                if c.checkState(0) == Qt.CheckState.Checked:
                    out.append((c.data(0, ROLE_SOURCE), c.data(0, ROLE_CHANNEL)))
        return out

    def set_checked(self, source_id: str, channel: str, on: bool = True) -> None:
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                c = top.child(j)
                if c.data(0, ROLE_SOURCE) == source_id and c.data(0, ROLE_CHANNEL) == channel:
                    c.setCheckState(0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def select_source(self, source_id: str) -> None:
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if top.data(0, ROLE_SOURCE) == source_id:
                self.tree.setCurrentItem(top)

    def current_source(self) -> str | None:
        item = self.tree.currentItem()
        return item.data(0, ROLE_SOURCE) if item is not None else None

    def current_channel(self) -> tuple[str, str] | None:
        """(source id, channel) of the selected channel row, None for a source row."""
        item = self.tree.currentItem()
        if item is None or item.data(0, ROLE_CHANNEL) is None:
            return None
        return item.data(0, ROLE_SOURCE), item.data(0, ROLE_CHANNEL)

    def display_dialog(self, source_id: str, channel: str) -> ChannelDisplayDialog:
        """Dialog pre-filled with the channel's current display settings (not shown yet)."""
        assert self._session is not None
        src = self._session.campaign.source(source_id)
        ds = self._session.datasets.get(source_id)
        values = ds.df[channel].cast(float).to_numpy() if ds is not None else None
        default = src.profile.plot_style if src.profile is not None else PlotStyle.LINE
        return ChannelDisplayDialog(channel, src.channel_display(channel), default, values, self)

    def edit_display(self, source_id: str, channel: str) -> bool:
        """Open the display dialog for a channel; True if settings were applied."""
        dialog = self.display_dialog(source_id, channel)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        self.apply_display(source_id, channel, dialog.display())
        return True

    def apply_display(self, source_id: str, channel: str, display: ChannelDisplay) -> None:
        if self._session is None:
            return
        self._session.set_display(source_id, channel, display)
        self.display_changed.emit(source_id, channel)

    def channel_menu(self, source_id: str, channel: str) -> QMenu:
        """Context menu of a channel row: display settings, a quick plot style choice, masks and
        calculated channels."""
        assert self._session is not None
        src = self._session.campaign.source(source_id)
        menu = QMenu(self)
        menu.addAction("Display Settings...", lambda: self.edit_display(source_id, channel))
        styles = menu.addMenu("Plot Style")
        group = QActionGroup(styles)
        current = src.plot_style(channel)
        for style, name in STYLE_NAMES.items():
            action = QAction(name, styles)
            action.setCheckable(True)
            action.setChecked(style is current)
            action.setData(style.value)  # Qt returns str enums as plain str
            group.addAction(action)
            styles.addAction(action)
        group.triggered.connect(
            lambda action: self.apply_display(
                source_id, channel, src.channel_display(channel).model_copy(update={"style": PlotStyle(action.data())})
            )
        )
        menu.addSeparator()
        menu.addAction("Mask Range...", lambda: self.new_mask(source_id=source_id, channels=[(source_id, channel)]))
        menu.addAction("New Calculated Channel...", lambda: self.new_derived(source_id))
        if src.derived_channel(channel) is not None:
            menu.addAction("Edit Calculated Channel...", lambda: self.edit_derived(source_id, channel))
            menu.addAction("Delete Calculated Channel", lambda: self.remove_derived(source_id, channel))
        menu.addAction("Calibrated Channel from Settings File...", lambda: self.calibration_from_file((source_id, channel)))
        return menu

    def source_menu(self, source_id: str) -> QMenu:
        """Context menu of a source row: align clock, masks, calculated channels, a calibration it holds."""
        assert self._session is not None
        menu = QMenu(self)
        menu.addAction("Align Clock...", lambda: self.align_clock(source_id))
        ds = self._session.datasets.get(source_id)
        if ds is not None and ds.channels:
            menu.addAction("Mask Range...", lambda: self.new_mask(source_id=source_id))
            menu.addAction("New Calculated Channel...", lambda: self.new_derived(source_id))
        if ds is not None and calibration_tooltip(ds.source_meta):
            label = self._session.campaign.source(source_id).label or source_id
            menu.addAction(
                "Calibrated Channel from These Settings...",
                lambda: self.calibration_dialog(ds.source_meta, label),
            )
        return menu

    def mask_menu(self, source_id: str, index: int) -> QMenu:
        menu = QMenu(self)
        menu.addAction("Edit Mask...", lambda: self.edit_mask(source_id, index))
        menu.addAction("Remove Mask", lambda: self.remove_mask(source_id, index))
        return menu

    def align_dialog(self, source_id: str) -> ClockSyncDialog:
        """Align dialog for a source (not shown yet), starting on the visible time span."""
        assert self._session is not None
        return ClockSyncDialog(self._session, source_id, self.view_range(), self)

    def align_clock(self, source_id: str | None = None) -> bool:
        """Open the align dialog; True if a clock model was applied."""
        sid = source_id or self.current_source()
        if sid is None or self._session is None:
            return False
        dialog = self.align_dialog(sid)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        model = dialog.clock_model()
        if model is None:
            return False
        self._session.set_clock(sid, model)
        self.clock_changed.emit(sid)
        return True

    def _on_context_menu(self, pos: QPoint) -> None:
        item = self.tree.itemAt(pos)
        if item is None or self._session is None:
            return
        sid = item.data(0, ROLE_SOURCE)
        if item.data(0, ROLE_CHANNEL) is not None:
            menu = self.channel_menu(sid, item.data(0, ROLE_CHANNEL))
        elif item.data(0, ROLE_MASK) is not None:
            menu = self.mask_menu(sid, item.data(0, ROLE_MASK))
        elif item.parent() is None:
            menu = self.source_menu(sid)
        else:
            return
        if not menu.isEmpty():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    # ------------------------------------------------- masks, calculated channels
    def _changed(self, source_id: str) -> None:
        self.set_session(self._session)
        self.data_changed.emit(source_id)

    def mask_dialog(
        self,
        start: float | None = None,
        end: float | None = None,
        channels: list[tuple[str, str]] | None = None,
        source_id: str | None = None,
        index: int | None = None,
    ) -> MaskDialog:
        """Dialog for a new mask (start/end default: the source's span) or mask `index` (not shown yet)."""
        assert self._session is not None
        s = self._session
        mask = s.campaign.source(source_id).masks[index] if index is not None else None
        sid = source_id or (channels[0][0] if channels else None)
        if start is None or end is None:
            span = s.span(sid) if sid is not None else None
            start, end = span if span is not None else (0.0, 0.0)

        def apply(src_id: str, lo: float, hi: float, chans: list[str], reason: str) -> None:
            if index is None:
                s.add_mask(src_id, lo, hi, chans, reason)
            else:
                s.update_mask(src_id, index, lo, hi, chans, reason)

        return MaskDialog(s, start, end, channels or [], mask, sid, apply, self)

    def new_mask(
        self,
        start: float | None = None,
        end: float | None = None,
        channels: list[tuple[str, str]] | None = None,
        source_id: str | None = None,
    ) -> bool:
        dialog = self.mask_dialog(start, end, channels, source_id)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        self._changed(dialog.source.currentData())
        return True

    def edit_mask(self, source_id: str, index: int) -> bool:
        dialog = self.mask_dialog(source_id=source_id, index=index)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        self._changed(source_id)
        return True

    def remove_mask(self, source_id: str, index: int) -> None:
        assert self._session is not None
        self._session.remove_mask(source_id, index)
        self._changed(source_id)

    def derived_dialog(self, source_id: str, name: str | None = None) -> DerivedChannelDialog:
        assert self._session is not None
        s = self._session
        old = s.campaign.source(source_id).derived_channel(name) if name is not None else None

        def apply(channel) -> None:
            if old is None:
                s.add_derived(source_id, channel)
            else:
                s.update_derived(source_id, old.name, channel)

        return DerivedChannelDialog(s, source_id, old, apply, self)

    def new_derived(self, source_id: str) -> bool:
        dialog = self.derived_dialog(source_id)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        self._changed(source_id)
        self.set_checked(source_id, dialog.name_edit.text().strip())
        return True

    def edit_derived(self, source_id: str, name: str) -> bool:
        dialog = self.derived_dialog(source_id, name)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        self._changed(source_id)
        return True

    def remove_derived(self, source_id: str, name: str) -> None:
        assert self._session is not None
        try:
            self._session.remove_derived(source_id, name)
        except ValueError as exc:
            QMessageBox.warning(self, "Delete Calculated Channel", str(exc))
            return
        self._changed(source_id)

    def calibration_dialog(self, meta: dict, origin: str, target: tuple[str, str] | None = None) -> bool:
        """Suggest a calibrated channel from the calibration in `meta`; True if the user applied it."""
        assert self._session is not None
        try:
            dialog = CalibrationDialog(self._session, meta, origin, target, self)
        except ValueError as exc:
            QMessageBox.warning(self, "Calibrated Channel", str(exc))
            return False
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        sid = dialog.source.currentData()
        self._changed(sid)
        self.set_checked(sid, dialog.name_edit.text().strip())
        return True

    def calibration_from_file(self, target: tuple[str, str] | None = None) -> bool:
        """Read a device settings file (not added as a source) and suggest a calibrated channel."""
        path, _ = QFileDialog.getOpenFileName(self, "Device Settings with Calibration", "", SETTINGS_FILTER)
        if not path:
            return False
        try:
            meta = load_source(path).source_meta
        except Exception as exc:  # show, do not crash the window
            QMessageBox.critical(self, "Calibrated Channel", f"{Path(path).name}:\n\n{exc}")
            return False
        return self.calibration_dialog(meta, Path(path).name, target)

    def _on_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        if item.data(0, ROLE_CHANNEL) is not None and self._session is not None:
            self.edit_display(item.data(0, ROLE_SOURCE), item.data(0, ROLE_CHANNEL))

    # -------------------------------------------------------------- handlers
    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if not self._updating and item.data(0, ROLE_CHANNEL) is not None:
            self.selection_changed.emit()

    def _on_remove(self) -> None:
        sid = self.current_source()
        if sid is not None:
            self.remove_requested.emit(sid)

    def _show_clock(self) -> None:
        sid = self.current_source()
        self.clock_box.setEnabled(sid is not None and self._session is not None)
        if sid is None or self._session is None:
            return
        clock = self._session.campaign.source(sid).clock
        self._updating = True
        self.offset_spin.setValue(clock.offset_s)
        self.drift_spin.setValue(clock.drift_ppm)
        self.note_edit.setText(clock.note)
        self.anchor_label.setText(
            f"{len(clock.anchors)} (offset and drift ignored)" if clock.anchors else "none"
        )
        u = f", ± {clock.uncertainty_s:g} s" if clock.uncertainty_s is not None else ""
        self.method_label.setText(f"{clock.method}{u}")
        self._updating = False

    def _apply_clock(self) -> None:
        sid = self.current_source()
        if self._updating or sid is None or self._session is None:
            return
        old = self._session.campaign.source(sid).clock
        new = old.model_copy(update={
            "offset_s": self.offset_spin.value(),
            "drift_ppm": self.drift_spin.value(),
            "note": self.note_edit.text(),
            "method": "manual" if (self.offset_spin.value(), self.drift_spin.value()) != (old.offset_s, old.drift_ppm) else old.method,
        })
        if new != old:
            self._session.set_clock(sid, new)
            self.clock_changed.emit(sid)
