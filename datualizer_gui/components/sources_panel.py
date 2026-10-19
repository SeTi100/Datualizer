"""Source list of a campaign: channels to plot, their display settings and the clock model of each source."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.campaign import ChannelDisplay, CampaignSession, ClockModel
from datualizer_core.schema import PlotStyle
from datualizer_gui.components.channel_display_dialog import STYLE_NAMES, ChannelDisplayDialog
from datualizer_gui.components.clock_sync_dialog import ClockSyncDialog

ROLE_SOURCE = Qt.ItemDataRole.UserRole
ROLE_CHANNEL = Qt.ItemDataRole.UserRole + 1


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
                self.tree.addTopLevelItem(item)
                for ch in (ds.channels if ds is not None else []):
                    alias = src.channel_display(ch).alias
                    child = QTreeWidgetItem([f"{alias} ({ch})" if alias else ch, ds.units.get(ch, ""), "", ""])
                    child.setToolTip(0, f"Channel in the data: {ch}. Right-click or double-click for display settings.")
                    child.setData(0, ROLE_SOURCE, src.id)
                    child.setData(0, ROLE_CHANNEL, ch)
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    on = (src.id, ch) in checked or (not checked and ch == ds.channels[0])
                    child.setCheckState(0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                    item.addChild(child)
                item.setExpanded(True)
            self.warning_label.setText("\n".join(session.warnings))
            self.warning_label.setVisible(bool(session.warnings))
        for col in range(4):
            self.tree.resizeColumnToContents(col)
        self._updating = False
        self._show_clock()

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
        """Context menu of a channel row: display settings and a quick plot style choice."""
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
        if item.data(0, ROLE_CHANNEL) is None:
            sid = item.data(0, ROLE_SOURCE)
            menu = QMenu(self)
            menu.addAction("Align Clock...", lambda: self.align_clock(sid))
            menu.exec(self.tree.viewport().mapToGlobal(pos))
            return
        self.channel_menu(item.data(0, ROLE_SOURCE), item.data(0, ROLE_CHANNEL)).exec(
            self.tree.viewport().mapToGlobal(pos)
        )

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
