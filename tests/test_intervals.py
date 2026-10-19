"""Measurement steps (Etappe 2b, step 5): from a lab journal by column roles, or set by hand."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from datualizer_core import load_csv
from datualizer_core.campaign import (
    CampaignSession,
    ClockModel,
    IntervalConfigError,
    IntervalMapping,
    WindowRule,
    master_datetime,
    to_epoch_s,
)
from datualizer_core.campaign.intervals import derive_intervals, suggest_mapping
from datualizer_core.ingestion.profiles import get_profile
from datualizer_core.ingestion.source_loader import with_choices
from datualizer_core.ingestion.steps import EndKind, EndRule, StepColumns

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
JOURNAL = SA / "labjournal_12_10_extract.xlsx"
FID_DAT = SA / "fid_ifid_12_10_extract.dat"
ZONE = "Europe/Zurich"


def zurich(*args) -> float:
    return to_epoch_s(datetime(*args), ZONE)


@pytest.fixture()
def session() -> CampaignSession:
    s = CampaignSession()
    profile = with_choices(JOURNAL, get_profile("labjournal_xlsx"), clock_date=date(2026, 10, 12))
    s.add_source(JOURNAL, profile, source_id="journal")
    return s


class TestJournal:
    def test_profile_suggests_the_mapping(self, session: CampaignSession) -> None:
        m = session.suggest_mapping("journal")
        assert m.end == EndRule(kind=EndKind.COLUMN, column="Zeit_1")
        assert (m.label_columns, m.fill_down, m.parameter_columns) == (["col_A"], ["col_A"], ["Temperatur", "V_MFC"])

    def test_data_shape_alone_finds_the_same_roles(self, session: CampaignSession) -> None:
        """Without the profile hint: no column names in code, only the shape of the data."""
        m = suggest_mapping(session.datasets["journal"], "journal")
        assert (m.end.column, m.label_columns, m.fill_down) == ("zeit_1", ["col_a"], ["col_a"])
        assert m.parameter_columns[:2] == ["temperatur", "v_mfc"]

    def test_steps_from_the_journal(self, session: CampaignSession) -> None:
        session.set_mapping(session.suggest_mapping("journal"))
        steps = session.intervals()
        assert len(steps.intervals) == 38
        first = steps.intervals[0]
        assert (first.label, first.start, first.end) == ("Konditionierung", zurich(2026, 10, 12, 10), zurich(2026, 10, 12, 10, 50))
        second = steps.intervals[1]
        assert second.label == "Referenz 1" and second.params == {"temperatur": 200.0, "v_mfc": 1.0}
        third = steps.intervals[2]  # block name written only in the first row of the block
        assert third.label == "Referenz 1" and third.params["v_mfc"] == 10.0
        assert not first.manual and first.origin == "journal:row 11"  # row of the sheet, as in Excel
        # "clearing" rows and empty template rows have no start: reported, not dropped silently
        assert steps.problems == ["journal: rows 29, 47, 51–60: no start time (skipped)."]

    def test_times_on_another_clock(self, session: CampaignSession) -> None:
        """The journal also lists the FID clock's times; they follow the FID clock model."""
        fid = session.add_source(FID_DAT)
        mapping = IntervalMapping(
            source_id="journal", clock_source=fid.id, start_column="FID_Zeit_0",
            end=EndRule(column="FID_Zeit_1"), label_columns=["col_A"], fill_down=["col_A"],
        )
        session.set_mapping(mapping)
        before = session.intervals()
        assert before.intervals[0].label == "Referenz 1"  # the conditioning row has no FID times
        assert before.intervals[0].start == zurich(2026, 10, 12, 10, 36, 50)
        assert "rows 11, 29, 47, 51–60: no start time" in before.problems[0]
        session.set_clock(fid.id, ClockModel(offset_s=970))
        after = session.intervals()
        assert after.intervals[0].start == before.intervals[0].start + 970

    def test_window_rule(self, session: CampaignSession) -> None:
        session.set_mapping(session.suggest_mapping("journal"))
        session.set_window_rule(WindowRule(length_s=120))  # the last 2 min of each step
        iv = session.intervals().intervals[1]
        assert iv.window == (iv.end - 120, iv.end)
        session.set_window_rule(WindowRule(skip_start_s=600))  # longer than most steps
        steps = session.intervals()
        assert steps.intervals[0].window == (steps.intervals[0].start + 600, steps.intervals[0].end)
        assert steps.intervals[1].window is None
        assert any("window rule leaves nothing" in p for p in steps.problems)

    def test_unknown_column_fails_loudly(self, session: CampaignSession) -> None:
        with pytest.raises(IntervalConfigError, match="Zeit_9"):
            session.set_mapping(IntervalMapping(source_id="journal", end=EndRule(column="Zeit_9")))
        assert session.campaign.interval_mappings == []

    def test_save_and_open(self, session: CampaignSession, tmp_path: Path) -> None:
        session.set_mapping(session.suggest_mapping("journal"))
        session.add_interval(datetime(2026, 10, 12, 14, 0), datetime(2026, 10, 12, 14, 5), "Blindwert")
        session.save(tmp_path / "campaign.json")
        again = CampaignSession.open(tmp_path / "campaign.json")
        a, b = session.intervals(), again.intervals()
        assert [(i.id, i.label, i.start, i.end) for i in a.intervals] == [(i.id, i.label, i.start, i.end) for i in b.intervals]
        assert a.problems == b.problems

    def test_removing_the_source_drops_its_mapping(self, session: CampaignSession) -> None:
        session.set_mapping(session.suggest_mapping("journal"))
        session.remove_source("journal")
        assert session.campaign.interval_mappings == [] and session.intervals().intervals == []


class TestManual:
    def test_add_swap_update_remove(self) -> None:
        s = CampaignSession()
        iv = s.add_interval(datetime(2026, 10, 12, 11), datetime(2026, 10, 12, 10, 30))  # set end first
        assert (iv.id, iv.start, iv.end, iv.label) == ("m1", datetime(2026, 10, 12, 10, 30), datetime(2026, 10, 12, 11), "Step 1")
        s.update_interval("m1", label="Blindwert", end=datetime(2026, 10, 12, 11, 5))
        r = s.intervals().get("m1")
        assert r.label == "Blindwert" and r.manual and r.duration_s == 35 * 60
        with pytest.raises(ValueError):
            s.update_interval("m1", end=datetime(2026, 10, 12, 10))
        s.remove_interval("m1")
        assert s.intervals().intervals == []

    def test_master_datetime_roundtrip(self) -> None:
        t = zurich(2026, 10, 12, 14, 27, 2)
        assert master_datetime(t, ZONE) == datetime(2026, 10, 12, 14, 27, 2)
        assert to_epoch_s(master_datetime(t, ZONE), ZONE) == t


class TestEndRules:
    CSV = "time,end,dur,name\n2026-10-12 23:50:00,00:10:00,60,a\n2026-10-13 00:20:00,00:25:00,,b\n2026-10-13 00:40:00,bad,30,c\n"

    def derive(self, end: EndRule) -> tuple[list, list[str]]:
        ds = load_csv(self.CSV)
        return derive_intervals(ds, IntervalMapping(source_id="x", end=end, label_columns=["name"]), ZONE, ClockModel(), WindowRule())

    def test_end_column_over_midnight(self) -> None:
        ivs, problems = self.derive(EndRule(column="end"))
        assert ivs[0].end - ivs[0].start == 20 * 60
        assert [i.label for i in ivs] == ["a", "b"]
        assert problems == ["x: row 4: no readable end (skipped)."]  # header is row 1

    def test_duration_next_start_and_fixed(self) -> None:
        ivs, _ = self.derive(EndRule(kind=EndKind.DURATION, column="dur"))
        assert [(i.label, i.duration_s) for i in ivs] == [("a", 60.0), ("c", 30.0)]
        ivs, problems = self.derive(EndRule(kind=EndKind.NEXT_START))
        assert [i.duration_s for i in ivs] == [30 * 60, 20 * 60]
        assert problems == ["x: row 4: last step has no next start, no end."]
        ivs, _ = self.derive(EndRule(kind=EndKind.FIXED, seconds=90))
        assert [i.duration_s for i in ivs] == [90.0] * 3

    def test_rules_are_validated(self) -> None:
        with pytest.raises(ValueError):
            EndRule(kind=EndKind.COLUMN)
        with pytest.raises(ValueError):
            StepColumns(label_columns=["a"], fill_down=["b"])


def test_window_rule_anchors() -> None:
    assert WindowRule(length_s=10, anchor="start").apply(0, 100) == (0, 10)
    assert WindowRule(skip_start_s=5, skip_end_s=5, length_s=50).apply(0, 100) == (45, 95)
    assert WindowRule(skip_start_s=60, skip_end_s=50).apply(0, 100) is None
    assert np.isclose(WindowRule().apply(1.5, 2.5)[1], 2.5)
