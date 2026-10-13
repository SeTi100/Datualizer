"""GUI: sheet/table choice and clock date when opening a file (main window and campaign window).

The modal dialog is replaced by a stub that records what was asked; the dialog widget itself is
tested separately. Multi-sheet workbooks and multi-table databases are built in tmp_path.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from datualizer_gui.campaign_window import CampaignWindow
from datualizer_gui.components import source_dialog
from datualizer_gui.components.source_dialog import SourceOptions, SourceOptionsDialog
from datualizer_gui.main_window import DatualizerMainWindow
from tests.test_source_choices import JOURNAL, SHEETS, make_multi_sheet_journal, make_multi_table_db


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture()
def journal(tmp_path: Path) -> Path:
    return make_multi_sheet_journal(tmp_path / "SA_Labjournal.xlsx")


@pytest.fixture()
def no_error_box(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: pytest.fail(f"error box: {a[2:]}"))


def stub_dialog(monkeypatch: pytest.MonkeyPatch, answer: SourceOptions | None) -> list[tuple]:
    asked: list[tuple] = []

    def ask(parent, file_name, choice, options, date_hint=None):
        asked.append((file_name, choice, list(options), date_hint))
        return answer

    monkeypatch.setattr(source_dialog, "ask_source_options", ask)
    return asked


class TestDialog:
    def test_date_follows_the_sheet_name(self, qapp: QApplication) -> None:
        dlg = SourceOptionsDialog("SA_Labjournal.xlsx", "excel.sheet", SHEETS)
        assert dlg.options() == SourceOptions("12.10.2026", date(2026, 10, 12))
        dlg.combo.setCurrentIndex(1)
        assert dlg.options() == SourceOptions("13.10.2026", date(2026, 10, 13))

    def test_no_date_suggested_without_a_date_in_the_names(self, qapp: QApplication) -> None:
        dlg = SourceOptionsDialog("export.db", "sqlite.table", ["a", "b"])
        assert dlg.options() == SourceOptions("a", None)
        assert not dlg.date_edit.isEnabled()

    def test_date_only(self, qapp: QApplication) -> None:
        dlg = SourceOptionsDialog("journal.xlsx", None, [], date(2026, 10, 12))
        assert dlg.combo is None
        assert dlg.options() == SourceOptions(None, date(2026, 10, 12))


class TestMainWindow:
    def test_multi_sheet_journal(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch, journal: Path, no_error_box: None
    ) -> None:
        asked = stub_dialog(monkeypatch, SourceOptions("13.10.2026", date(2026, 10, 13)))
        win = DatualizerMainWindow(auto_load=False)
        win.load_files([journal])
        qapp.processEvents()
        assert asked == [("SA_Labjournal.xlsx", "excel.sheet", SHEETS, None)]
        ds = win.dataset
        assert ds is not None and ds.source_meta["sheet"] == "13.10.2026"
        assert ds.time_abs.dt.replace_time_zone(None)[1] == datetime(2026, 10, 13, 10, 53)
        assert "[profile: labjournal_xlsx]" in win.status_bar.currentMessage()

    def test_cancel_loads_nothing(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch, journal: Path, no_error_box: None
    ) -> None:
        stub_dialog(monkeypatch, None)
        win = DatualizerMainWindow(auto_load=False)
        win.load_files([journal])
        assert win.dataset is None

    def test_sqlite_without_profile(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_error_box: None
    ) -> None:
        asked = stub_dialog(monkeypatch, SourceOptions("b", None))
        win = DatualizerMainWindow(auto_load=False)
        win.load_files([make_multi_table_db(tmp_path / "x.db")])
        assert asked[0][1:3] == ("sqlite.table", ["a", "b"])
        assert win.dataset is not None and len(win.dataset) == 3

    def test_single_sheet_file_asks_nothing(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch, no_error_box: None
    ) -> None:
        asked = stub_dialog(monkeypatch, None)
        win = DatualizerMainWindow(auto_load=False)
        win.load_files([JOURNAL])
        assert asked == [] and win.dataset is not None


class TestCampaignWindow:
    def test_multi_sheet_journal_is_stored_and_replayed(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch, journal: Path, tmp_path: Path, no_error_box: None
    ) -> None:
        stub_dialog(monkeypatch, SourceOptions("13.10.2026", date(2026, 10, 13)))
        win = CampaignWindow()
        win.add_sources([journal])
        src = win.session.campaign.sources[0]
        assert src.profile.excel.sheet == "13.10.2026"
        assert src.spec.time.clock_date == date(2026, 10, 13)
        span = win.session.span(src.id)
        assert span is not None

        out = tmp_path / "campaign.json"
        win.save_campaign(out)
        stub_dialog(monkeypatch, None)  # reopening must not ask again
        win2 = CampaignWindow()
        win2.open_campaign(out)
        assert win2.session.warnings == []
        assert win2.session.span(src.id) == span

    def test_clock_only_journal_asks_for_the_date(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch, no_error_box: None
    ) -> None:
        asked = stub_dialog(monkeypatch, SourceOptions(None, date(2026, 10, 12)))
        win = CampaignWindow()
        win.add_sources([JOURNAL])
        assert asked == [(JOURNAL.name, None, [], date(2026, 10, 12))]  # suggested from the sheet name
        src = win.session.campaign.sources[0]
        assert src.spec.time.clock_date == date(2026, 10, 12)
        assert win.session.master_time(src.id) is not None

    def test_skipping_the_date_keeps_relative_time(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch, no_error_box: None
    ) -> None:
        stub_dialog(monkeypatch, None)
        win = CampaignWindow()
        win.add_sources([JOURNAL])
        src = win.session.campaign.sources[0]
        assert win.session.master_time(src.id) is None

    def test_cancelled_choice_adds_no_source(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch, journal: Path, no_error_box: None
    ) -> None:
        stub_dialog(monkeypatch, None)
        win = CampaignWindow()
        win.add_sources([journal])
        assert win.session.campaign.sources == []
