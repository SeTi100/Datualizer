"""Dialogs for calculated channels, masks and calibration suggestions of a campaign (Etappe 2b, PR 2).

Times are typed as master wall-clock times in the campaign zone (`dd.mm.yyyy HH:MM:SS`), the
same way manual steps are stored. The dialogs only collect values; the session checks them
(`CampaignSession.add_derived`, `add_mask`, ...) and the caller shows its error, so a broken
formula or an unknown channel fails loudly instead of being stored.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from datualizer_core.calc import FUNCTIONS
from datualizer_core.campaign import (
    CampaignSession,
    DerivedChannel,
    Mask,
    Piece,
    calibration_channel,
    master_datetime,
    to_epoch_s,
)
from datualizer_core.ingestion.calibration import LinearCalibration, calibration_fields, suggest_linear_calibration

TIME_FORMAT = "%d.%m.%Y %H:%M:%S"


def format_master(epoch_s: float, zone: str) -> str:
    return master_datetime(epoch_s, zone).strftime(TIME_FORMAT)


def parse_time(text: str) -> datetime | None:
    """`dd.mm.yyyy HH:MM:SS` (seconds optional) to a naive date-time; empty = None; else ValueError."""
    text = text.strip()
    if not text:
        return None
    for fmt in (TIME_FORMAT, "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"{text!r} is not a time like 15.10.2026 09:24:00")


def _buttons(dialog: QDialog, on_ok: Callable[[], None]) -> QDialogButtonBox:
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(on_ok)
    buttons.rejected.connect(dialog.reject)
    return buttons


def _error_label() -> QLabel:
    label = QLabel()
    label.setWordWrap(True)
    label.setStyleSheet("color: #ff6b6b;")
    label.hide()
    return label


class MaskDialog(QDialog):
    """Start, end, channels and reason of a mask. No channel ticked = all channels of the source.

    `apply(source_id, start_s, end_s, channels, reason)` stores it and raises on bad input; the
    message is shown and the dialog stays open.
    """

    def __init__(
        self,
        session: CampaignSession,
        start: float,
        end: float,
        channels: Sequence[tuple[str, str]] = (),
        mask: Mask | None = None,
        source_id: str | None = None,
        apply: Callable[[str, float, float, list[str], str], object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Mask" if mask is not None else "Mask Range")
        self._session = session
        self._apply = apply
        self._preset = list(mask.channels) if mask is not None else [ch for _sid, ch in channels]
        zone = session.campaign.timezone
        sources = [s for s in session.campaign.sources if s.id in session.datasets]
        wanted = source_id or (channels[0][0] if channels else None)
        self.source = QComboBox()
        for s in sources:
            self.source.addItem(s.label or s.id, s.id)
        if wanted is not None and self.source.findData(wanted) >= 0:
            self.source.setCurrentIndex(self.source.findData(wanted))
        self.source.setEnabled(mask is None)
        self.start_edit = QLineEdit(mask.start.strftime(TIME_FORMAT) if mask else format_master(start, zone))
        self.end_edit = QLineEdit(mask.end.strftime(TIME_FORMAT) if mask else format_master(end, zone))
        self.channel_list = QListWidget()
        self.channel_list.setToolTip("Tick the channels concerned; none ticked = all channels of the source")
        self.reason_edit = QLineEdit(mask.reason if mask else "")
        self.reason_edit.setPlaceholderText("Why, e.g. 'test gas on the FID'")
        self.error = _error_label()

        form = QFormLayout()
        form.addRow("Source", self.source)
        form.addRow(f"Start ({zone})", self.start_edit)
        form.addRow(f"End ({zone})", self.end_edit)
        form.addRow("Channels", self.channel_list)
        form.addRow("Reason", self.reason_edit)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        note = QLabel("Values are flagged 'masked', not deleted; the statistics leave them out "
                      "while 'masked' is an excluded flag.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(note)
        layout.addWidget(self.error)
        layout.addWidget(_buttons(self, self._on_ok))
        self.source.currentIndexChanged.connect(self._fill_channels)
        self._fill_channels()

    def _fill_channels(self) -> None:
        self.channel_list.clear()
        sid = self.source.currentData()
        ds = self._session.datasets.get(sid)
        for ch in ds.channels if ds is not None else []:
            item = QListWidgetItem(ch)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if ch in self._preset else Qt.CheckState.Unchecked)
            self.channel_list.addItem(item)
        self._preset = []  # only for the first source shown

    def values(self) -> tuple[str, float, float, list[str], str]:
        """(source id, start, end as master epoch seconds, channels, reason); ValueError for bad times."""
        zone = self._session.campaign.timezone
        start, end = parse_time(self.start_edit.text()), parse_time(self.end_edit.text())
        if start is None or end is None:
            raise ValueError("Give a start and an end.")
        channels = [
            self.channel_list.item(i).text() for i in range(self.channel_list.count())
            if self.channel_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        return self.source.currentData(), to_epoch_s(start, zone), to_epoch_s(end, zone), channels, self.reason_edit.text().strip()

    def _on_ok(self) -> None:
        try:
            values = self.values()
            if self._apply is not None:
                self._apply(*values)
        except (ValueError, KeyError) as exc:
            self.error.setText(str(exc).strip("'\""))
            self.error.show()
            return
        self.accept()


class DerivedChannelDialog(QDialog):
    """Name, unit, default formula and pieces (formulas for master time spans) of a calculated channel.

    `apply(channel)` stores it (raises on a broken formula, overlapping pieces ...).
    """

    def __init__(
        self,
        session: CampaignSession,
        source_id: str,
        channel: DerivedChannel | None = None,
        apply: Callable[[DerivedChannel], object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Calculated Channel" if channel else "New Calculated Channel")
        self.resize(640, 420)
        self._apply = apply
        src = session.campaign.source(source_id)
        ds = session.datasets.get(source_id)
        names = [n for n in ((ds.channels + ds.parameters) if ds is not None else []) if channel is None or n != channel.name]
        zone = session.campaign.timezone
        self.name_edit = QLineEdit(channel.name if channel else "")
        self.unit_edit = QLineEdit(channel.unit if channel else "")
        self.expr_edit = QLineEdit(channel.expr if channel else "")
        self.expr_edit.setPlaceholderText("e.g. 0.38 * signal - 1.786 (empty: no value outside the pieces)")
        self.note_edit = QLineEdit(channel.note if channel else "")
        self.pieces = QTableWidget(0, 4)
        self.pieces.setHorizontalHeaderLabels([f"Start ({zone})", "End", "Formula", "Note"])
        self.pieces.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.pieces.setToolTip("Formulas for master time spans (start included, end excluded; empty = open)")
        for p in channel.pieces if channel else []:
            self.add_piece(p)
        add, remove = QPushButton("Add Piece"), QPushButton("Remove Piece")
        add.clicked.connect(lambda: self.add_piece())
        remove.clicked.connect(lambda: self.pieces.removeRow(self.pieces.currentRow()))
        hint = QLabel(
            f"Names of {src.label or src.id}: " + ", ".join(names)
            + "\nFunctions: " + ", ".join(FUNCTIONS) + "; + − * / **, comparisons, a if c else b"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaaaaa;")
        hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.error = _error_label()

        form = QFormLayout()
        form.addRow("Name", self.name_edit)
        form.addRow("Unit", self.unit_edit)
        form.addRow("Formula", self.expr_edit)
        form.addRow("Note", self.note_edit)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Pieces"))
        layout.addWidget(self.pieces, stretch=1)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(hint)
        layout.addWidget(self.error)
        layout.addWidget(_buttons(self, self._on_ok))

    def add_piece(self, piece: Piece | None = None) -> None:
        r = self.pieces.rowCount()
        self.pieces.insertRow(r)
        cells = ["", "", "", ""] if piece is None else [
            piece.start.strftime(TIME_FORMAT) if piece.start else "",
            piece.end.strftime(TIME_FORMAT) if piece.end else "",
            piece.expr, piece.note,
        ]
        for c, text in enumerate(cells):
            self.pieces.setItem(r, c, QTableWidgetItem(text))

    def channel(self) -> DerivedChannel:
        """The calculated channel as entered (ValueError for bad times or no formula)."""
        pieces = []
        for r in range(self.pieces.rowCount()):
            cell = lambda c: (self.pieces.item(r, c).text() if self.pieces.item(r, c) else "").strip()  # noqa: E731
            if not any(cell(c) for c in range(4)):
                continue
            if not cell(2):
                raise ValueError(f"Piece {r + 1} has no formula.")
            pieces.append(Piece(start=parse_time(cell(0)), end=parse_time(cell(1)), expr=cell(2), note=cell(3)))
        return DerivedChannel(
            name=self.name_edit.text().strip(), expr=self.expr_edit.text().strip(),
            unit=self.unit_edit.text().strip(), pieces=pieces, note=self.note_edit.text().strip(),
        )

    def _on_ok(self) -> None:
        try:
            channel = self.channel()
            if self._apply is not None:
                self._apply(channel)
        except (ValueError, KeyError) as exc:
            self.error.setText(str(exc))
            self.error.show()
            return
        self.accept()


class CalibrationDialog(QDialog):
    """Suggest a calculated channel from a device calibration in `meta` (e.g. iFID Settings.xml).

    The user gives the factor (e.g. mgC/ppm), picks the raw signal channel and either the whole
    time or a span; a span becomes a piece of the channel (added to an existing channel of the
    same name). Nothing is applied before OK (principle 2).
    """

    def __init__(
        self,
        session: CampaignSession,
        meta: dict,
        origin: str,
        target: tuple[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibrated Channel")
        self._session, self._meta, self._origin = session, meta, origin
        self.fields = calibration_fields(meta)
        if self.fields is None:
            raise ValueError(f"{origin} has no calibration description (profile field 'calibration').")
        f = self.fields
        block = (meta.get("xml") or {}).get(f.block, {})
        info = QLabel("\n".join(f"{k}: {v}" for k, v in block.items()) or f"no block {f.block!r}")
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.factor_edit = QLineEdit()
        self.factor_edit.setValidator(QDoubleValidator())
        per = "/".join(u for u in (f.result_unit, f.concentration_unit) if u) or "result unit per concentration unit"
        self.factor_edit.setPlaceholderText(f"{per}, from the calibration gas (not in the device file)")
        self.source = QComboBox()
        for s in session.campaign.sources:
            ds = session.datasets.get(s.id)
            if ds is not None and ds.channels:
                self.source.addItem(s.label or s.id, s.id)
        self.channel = QComboBox()
        self.name_edit = QLineEdit()
        self.unit_edit = QLineEdit(f.result_unit)
        self.span_check = QCheckBox("Only in a time span (piece)")
        self.start_edit, self.end_edit = QLineEdit(), QLineEdit()
        self.start_edit.setPlaceholderText("open")
        self.end_edit.setPlaceholderText("open")
        self.from_cal = QPushButton("Start at the calibration time")
        self.from_cal.setToolTip("Time of the calibration on the device clock, moved to master time with the source's clock offset")
        self.from_cal.setEnabled(bool(f.time and block.get(f.time)))
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.error = _error_label()

        form = QFormLayout()
        form.addRow(f"Calibration ({origin}, {f.block})", info)
        form.addRow("Factor", self.factor_edit)
        form.addRow("Source", self.source)
        form.addRow("Signal channel", self.channel)
        form.addRow("New channel", self.name_edit)
        form.addRow("Unit", self.unit_edit)
        form.addRow(self.span_check)
        form.addRow("Start", self.start_edit)
        form.addRow("End", self.end_edit)
        form.addRow("", self.from_cal)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.preview)
        layout.addWidget(self.error)
        layout.addWidget(_buttons(self, self._on_ok))

        self.source.currentIndexChanged.connect(self._fill_channels)
        self.channel.currentIndexChanged.connect(self._default_name)
        self.factor_edit.textChanged.connect(self._update_preview)
        self.channel.currentIndexChanged.connect(self._update_preview)
        self.span_check.toggled.connect(self._update_span)
        self.from_cal.clicked.connect(self._start_at_calibration)
        if target is not None and self.source.findData(target[0]) >= 0:
            self.source.setCurrentIndex(self.source.findData(target[0]))
        self._fill_channels(prefer=target[1] if target else None)
        self._update_span(False)

    def _fill_channels(self, *_args, prefer: str | None = None) -> None:
        self.channel.clear()
        ds = self._session.datasets.get(self.source.currentData())
        channels = ds.channels if ds is not None else []
        self.channel.addItems(channels)
        for want in (prefer, self.fields.signal_channel):
            if want in channels:
                self.channel.setCurrentText(want)
                break
        self._default_name()

    def _default_name(self) -> None:
        ch = self.channel.currentText()
        if ch:
            self.name_edit.setText(f"{ch}_{self.fields.result_unit}" if self.fields.result_unit else f"{ch}_cal")

    def _update_span(self, on: bool) -> None:
        for w in (self.start_edit, self.end_edit, self.from_cal):
            w.setEnabled(on and (w is not self.from_cal or bool(self.fields.time)))

    def _start_at_calibration(self) -> None:
        f = self.fields
        text = (self._meta.get("xml") or {}).get(f.block, {}).get(f.time)
        sid = self.source.currentData()
        try:
            device = datetime.fromisoformat(str(text))
        except ValueError:
            self.error.setText(f"Calibration time {text!r} is not a date-time.")
            self.error.show()
            return
        zone = self._session.campaign.timezone
        clock = self._session.campaign.source(sid).clock
        master = float(clock.apply(np.array([to_epoch_s(device, zone)]), zone)[0])
        self.start_edit.setText(format_master(master, zone))

    def calibration(self) -> LinearCalibration:
        text = self.factor_edit.text().replace(",", ".").strip()
        if not text:
            raise ValueError("Give the factor (result unit per concentration unit).")
        return suggest_linear_calibration(self._meta, float(text), self.fields)

    def _update_preview(self) -> None:
        try:
            cal = self.calibration()
        except ValueError as exc:
            self.preview.setText(str(exc))
            return
        self.preview.setText(f"a = {cal.a:.6g}, b = {cal.b:.6g}\n= {cal.formula(self.channel.currentText() or 'signal')}")

    def suggestion(self) -> tuple[str, DerivedChannel, Piece | None]:
        """(source id, suggested channel, piece if only a span): what OK applies."""
        cal = self.calibration()
        ch = self.channel.currentText()
        if not ch:
            raise ValueError("Pick the signal channel.")
        dc = calibration_channel(cal, ch, self.name_edit.text().strip(), self.unit_edit.text().strip(), self._origin)
        piece = None
        if self.span_check.isChecked():
            piece = Piece(start=parse_time(self.start_edit.text()), end=parse_time(self.end_edit.text()),
                          expr=dc.expr, note=dc.note)
        return self.source.currentData(), dc, piece

    def apply(self) -> DerivedChannel:
        """Add the channel, or the piece to the existing channel of that name."""
        sid, dc, piece = self.suggestion()
        src = self._session.campaign.source(sid)
        old = src.derived_channel(dc.name)
        if piece is None:
            if old is not None:
                raise ValueError(f"{dc.name!r} exists; tick 'Only in a time span' to add a piece, or pick another name.")
            return self._session.add_derived(sid, dc)
        if old is None:
            return self._session.add_derived(sid, dc.model_copy(update={"expr": "", "note": "", "pieces": [piece]}))
        new = old.model_copy(update={"pieces": [*old.pieces, piece]})
        return self._session.update_derived(sid, old.name, new)

    def _on_ok(self) -> None:
        try:
            self.apply()
        except (ValueError, KeyError) as exc:
            self.error.setText(str(exc))
            self.error.show()
            return
        self.accept()
