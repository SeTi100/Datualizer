"""Example campaign and recipe (PR 10) without the real data: valid, in sync with the tool, usable as template.

`examples/campaign_SA_ethanol.json` is written by `tools/sa_example_campaign.py`; the recipe
`examples/rezept_katalyse_voc.json` is its evaluation without sources. The comparison with the
script evaluation on the real data is `test_sa_end_to_end.py` (skipped without DATUALIZER_SA_DIR).
"""

from __future__ import annotations

from datetime import date
import importlib.util
import json
from pathlib import Path

import pytest

from datualizer_core.campaign import Campaign, CampaignSession, Evaluation
from datualizer_core.ingestion.profiles import get_profile
from datualizer_core.ingestion.source_loader import with_choices

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "campaign_SA_ethanol.json"
RECIPE = ROOT / "examples" / "rezept_katalyse_voc.json"
SA = Path(__file__).parent / "fixtures" / "sa_12_10"


def _tool():
    spec = importlib.util.spec_from_file_location("sa_example_campaign", ROOT / "tools" / "sa_example_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_example_campaign_is_valid_and_the_data_stay_outside() -> None:
    c = Campaign.load(EXAMPLE)
    assert {s.id for s in c.sources} >= {"fid_12", "fid_15a", "fid_15b", "fid_17", "journal_12", "journal_17", "runs_15"}
    assert all(s.path.startswith("${DATUALIZER_SA_DIR}/") for s in c.sources)
    assert all(s.sha256 is None and s.spec is None for s in c.sources)  # no fingerprints of private files
    assert c.source("fid_17").clock.cells is not None  # 17.10.: clock from the journal pairs
    assert [m.source_id for m in c.interval_mappings] == ["journal_12", "runs_15", "journal_17"]
    assert len(c.intervals) == 3  # 12.10.: zero phases set by hand


def test_example_is_what_the_tool_writes() -> None:
    tool = _tool()
    assert Campaign.load(EXAMPLE) == Campaign.model_validate(json.loads(json.dumps(tool.campaign().model_dump(mode="json"))))


def test_recipe_is_the_evaluation_of_the_example() -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    assert Evaluation.model_validate(recipe["evaluation"]) == Campaign.load(EXAMPLE).evaluation
    assert set(recipe) == {"name", "description", "stats", "evaluation"}


def test_apply_recipe_to_another_campaign() -> None:
    s = CampaignSession()
    profile = with_choices(SA / "labjournal_12_10_extract.xlsx", get_profile("labjournal_xlsx"), clock_date=date(2026, 10, 12))
    s.add_source(SA / "labjournal_12_10_extract.xlsx", profile, source_id="journal")
    s.add_source(SA / "instrument_reader_12_10_extract.db", source_id="db_12")
    notes = s.apply_recipe(RECIPE)
    assert s.campaign.evaluation == Campaign.load(EXAMPLE).evaluation
    assert s.campaign.stats.reading_u["fid_12/fid_mgc"] == 0.10
    assert [s.id for s in s.campaign.sources] == ["journal", "db_12"]  # sources and steps stay
    assert "recipe: measure 'S' reads fid_12, fid_15a, fid_15b, fid_17, not in this campaign" in notes[0]
    assert any("'mdot_w': ocr_12, db_15, db_17 not in this campaign (only the others are read)" in n for n in notes)
    assert notes == s.warnings[-len(notes):]
    with pytest.raises(ValueError, match="needs an 'evaluation'"):
        s.apply_recipe({"stats": {}})


def test_menu_evaluation_from_template(qtbot_free_app) -> None:
    from datualizer_gui.campaign_window import CampaignWindow

    win = CampaignWindow()
    win.add_sources([SA / "fid_ifid_12_10_extract.dat"])
    notes = win.apply_recipe(RECIPE)
    assert notes and win.results_panel.calc_table.rowCount() == len(Campaign.load(EXAMPLE).evaluation.calculations)
    assert "Evaluation from rezept_katalyse_voc.json" in win.status_bar.currentMessage()
    win.close()


@pytest.fixture()
def qtbot_free_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    return app if app is not None else QApplication([])
