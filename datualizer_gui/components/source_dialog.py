"""Ask what a source needs before it can be loaded: sheet or table, and the date of clock times.

The reader reports the choice machine-readably (`ReaderError.choice` / `.options`); the user's
pick goes into a profile copy (`with_choices`), so a campaign stores and replays it exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QDate
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from datualizer_core import (
    DualModeDataset,
    ReaderError,
    SourceProfile,
    date_in_text,
    detect_profile,
    load_source,
    needs_clock_date,
    with_choices,
)

CHOICE_LABELS = {"excel.sheet": "Sheet", "sqlite.table": "Table"}


@dataclass(frozen=True)
class SourceOptions:
    """The user's answer: the chosen sheet/table (None if none was asked) and the clock date."""

    choice: str | None
    clock_date: date | None


class SourceOptionsDialog(QDialog):
    """Combo box for a sheet or table (if asked) and an optional date for bare times of day."""

    def __init__(
        self,
        file_name: str,
        choice: str | None,
        options: Sequence[str],
        date_hint: date | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Open {file_name}")
        self._file_name = file_name
        self._fallback_hint = date_hint
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.combo: QComboBox | None = None
        if choice is not None:
            what = CHOICE_LABELS.get(choice, choice)
            layout.addWidget(QLabel(f"{file_name} contains several {what.lower()}s. Which one should be loaded?"))
            self.combo = QComboBox()
            self.combo.addItems(list(options))
            self.combo.currentTextChanged.connect(self._suggest_date)
            form.addRow(f"{what}:", self.combo)
        else:
            layout.addWidget(QLabel(f"{file_name} has times of day without a date."))

        self.date_check = QCheckBox(
            "Date of the times of day (if the time column has no date)"
            if choice is not None else "Place the times on this date"
        )
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("dd.MM.yyyy")
        self.date_edit.setDate(QDate.currentDate())
        self.date_check.toggled.connect(self.date_edit.setEnabled)
        form.addRow(self.date_check)
        form.addRow("Date:", self.date_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._suggest_date(self.combo.currentText() if self.combo is not None else "")

    def _suggest_date(self, option: str) -> None:
        """Date from the sheet name (e.g. '12.10.2026'), else from the file name or the hint."""
        hint = date_in_text(option) or self._fallback_hint or date_in_text(Path(self._file_name).stem)
        self.date_check.setChecked(hint is not None or self.combo is None)
        self.date_edit.setEnabled(self.date_check.isChecked())
        if hint is not None:
            self.date_edit.setDate(QDate(hint.year, hint.month, hint.day))

    def options(self) -> SourceOptions:
        d = self.date_edit.date()
        return SourceOptions(
            self.combo.currentText() if self.combo is not None else None,
            date(d.year(), d.month(), d.day()) if self.date_check.isChecked() else None,
        )


def ask_source_options(
    parent: QWidget | None,
    file_name: str,
    choice: str | None,
    options: Sequence[str],
    date_hint: date | None = None,
) -> SourceOptions | None:
    """Show the dialog modally; None if the user cancels. Replaced in tests."""
    dlg = SourceOptionsDialog(file_name, choice, options, date_hint, parent)
    return dlg.options() if dlg.exec() == QDialog.DialogCode.Accepted else None


def load_source_interactive(
    parent: QWidget | None,
    path: str | Path,
    *,
    ask_clock_date: bool = False,
) -> tuple[DualModeDataset, SourceProfile | None] | None:
    """Load a file with its detected profile, asking the user where the source needs a choice.

    - Several sheets or tables: ask which one (and offer a date for bare times of day).
    - `ask_clock_date`: also ask for a date when the time column has only times of day
      (a campaign needs absolute time). Skipping keeps the relative time.
    Returns the dataset and the profile that reproduces it, or None if the user cancelled.
    Other errors propagate, so the caller shows them.
    """
    path = Path(path)
    profile = detect_profile(path)
    try:
        ds = load_source(path, profile)
    except ReaderError as exc:
        if exc.choice not in CHOICE_LABELS or not exc.options:
            raise
        answer = ask_source_options(parent, path.name, exc.choice, exc.options)
        if answer is None:
            return None
        key = "sheet" if exc.choice == "excel.sheet" else "table"
        profile = with_choices(path, profile, **{key: answer.choice})
        ds = load_source(path, profile)
        if answer.clock_date is not None and needs_clock_date(ds):
            profile = with_choices(path, profile, clock_date=answer.clock_date)
            ds = load_source(path, profile)
        return ds, profile

    if ask_clock_date and needs_clock_date(ds):
        hint = date_in_text(str(ds.source_meta.get("sheet", "")))
        answer = ask_source_options(parent, path.name, None, [], hint)
        if answer is not None and answer.clock_date is not None:
            profile = with_choices(path, profile, clock_date=answer.clock_date)
            ds = load_source(path, profile)
    return ds, profile
