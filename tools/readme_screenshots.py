"""Render the README screenshots from the fixtures (docs/img/*.png).

Run from the repository root:  .venv\Scripts\python.exe tools\readme_screenshots.py
Uses a temporary settings folder, so the user's GUI preferences stay untouched.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, tempfile.mkdtemp())

from datualizer_core.ingestion.profiles import detect_profile, get_profile  # noqa: E402
from datualizer_core.ingestion.source_loader import load_source, with_choices  # noqa: E402
from datualizer_gui import campaign_window  # noqa: E402
from datualizer_gui.app import create_app  # noqa: E402
from datualizer_gui.campaign_window import CampaignWindow  # noqa: E402
from datualizer_gui.components.plot_export import ExportOptions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SA = ROOT / "tests" / "fixtures" / "sa_12_10"
RUNS = ROOT / "tests" / "fixtures" / "runs_export"
OUT = ROOT / "docs" / "img"
JOURNAL = SA / "labjournal_12_10_extract.xlsx"
DB = SA / "instrument_reader_12_10_extract.db"
T155_V2 = 19  # "T Einstellung 155", V_MFC = 2: inside the scale recording


def settle(app: QApplication, rounds: int = 5) -> None:
    for _ in range(rounds):
        app.processEvents()
        QTest.qWait(50)


def load(_parent, path, *, ask_clock_date=False):
    """Non-interactive stand-in for the source dialog: journal date given, rest by profile."""
    path = Path(path)
    profile = detect_profile(path)
    if path == JOURNAL:
        profile = with_choices(path, get_profile("labjournal_xlsx"), clock_date=date(2026, 10, 12))
    return load_source(path, profile), profile


def check_only(win: CampaignWindow, wanted: dict[str, set[str]]) -> None:
    tree = win.sources_panel.tree
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        for j in range(top.childCount()):
            child = top.child(j)
            on = any(top.text(0).startswith(src) and (child.text(0) in chans) for src, chans in wanted.items())
            child.setCheckState(0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)


def shot(app: QApplication, widget, name: str) -> None:
    settle(app, 10)
    path = OUT / name
    widget.grab().save(str(path))
    print("wrote", path.relative_to(ROOT))


def main() -> None:
    app, main_win = create_app()
    main_win.resize(1500, 900)
    main_win.load_file(RUNS / "runs_export_20261004_123307_ALL_MESSY.csv")
    main_win.show()
    main_win.inspector.run_table.selectRow(1)  # run 1: one clean run instead of 86 h of gaps
    shot(app, main_win, "hauptfenster.png")
    main_win.close()

    campaign_window.load_source_interactive = load
    win = CampaignWindow()
    win.resize(1500, 900)
    win.show()
    win.add_sources([DB, JOURNAL, SA / "ocr_recording_105012_extract.csv", SA / "fid_ifid_12_10_extract.dat"])
    check_only(win, {
        "instrument_reader": {"masse"},
        "labjournal": {"temperatur", "v_mfc"},
        "ocr_recording": {"mass_g"},
        "fid_ifid": {"conc"},
    })
    win.set_coverage_visible(True)
    shot(app, win, "kampagne_uebersicht.png")

    check_only(win, {"instrument_reader": {"masse", "massenstrom_waage"}, "labjournal": {"temperatur", "v_mfc"}})
    win.set_bands_visible(True)
    win.dock_steps.raise_()
    ivs = win.session.intervals().intervals
    lo, hi = win.session.coverage(DB.stem).span  # the scale recording, several steps long
    win.canvas.set_view_range(lo - 30, hi + 30)
    shot(app, win, "kampagne_schritte.png")

    check_only(win, {"instrument_reader": {"masse", "massenstrom_waage"}})
    win.inspect_step(ivs[T155_V2].id)
    win.dock_window.setFloating(True)  # the docked inspector shares the height with the plots
    win.dock_window.resize(1500, 860)
    settle(app)
    insp = win.window_inspector
    insp.graphics.parent().setSizes([520, 300])
    win.inspect_step(ivs[T155_V2].id)  # rebuild once the plots have their final size
    shot(app, win.dock_window, "fenster_inspektor.png")
    win.dock_window.setFloating(False)
    win.dock_window.hide()

    check_only(win, {"instrument_reader": {"masse", "massenstrom_waage"}, "labjournal": {"temperatur", "v_mfc"}})
    win.canvas.set_view_range(lo - 30, hi + 30)
    settle(app)
    win.set_coverage_visible(False)
    settle(app)
    win.export_plot(OUT / "export_bericht.png", ExportOptions(width_mm=170, dpi=150))
    print("wrote docs/img/export_bericht.png")
    win.close()



if __name__ == "__main__":
    main()
