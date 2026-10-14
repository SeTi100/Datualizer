"""Etappe 2b, Schritt 6, Paket D (Kern): Fenster pro Schritt von Hand, Einzelabfrage, Zeichenfelder.

Was der Fenster-Inspektor vom Kern braucht: `session.window_stats` für einen Schritt,
`WindowOverride` in campaign.json (Roundtrip), veraltete Overrides in `problems`, Prüfstufen
der Ampel und die Zeilen, Geraden und Stützstellen zum Zeichnen.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from datualizer_core.campaign import Campaign, CampaignSession, WindowOverride
from datualizer_core.ingestion.profiles import get_profile
from datualizer_core.ingestion.source_loader import load_source, with_choices
from datualizer_core.stats import Check, StatsConfig, WindowSpec, window_stats

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
JOURNAL = SA / "labjournal_12_10_extract.xlsx"
DB = SA / "instrument_reader_12_10_extract.db"


def _session() -> CampaignSession:
    s = CampaignSession(Campaign())
    s.add_source(DB, source_id="scale")
    profile = with_choices(JOURNAL, get_profile("labjournal_xlsx"), clock_date=date(2026, 10, 12))
    s.add_source(JOURNAL, profile, dataset=load_source(JOURNAL, profile), source_id="journal")
    s.set_mapping(s.suggest_mapping("journal"))
    return s


@pytest.fixture()
def session() -> CampaignSession:
    return _session()


def _step(session: CampaignSession, v_mfc: float):
    return next(
        iv for iv in session.intervals().intervals
        if iv.label == "T Einstellung 155" and iv.params["v_mfc"] == v_mfc
    )


# ------------------------------------------------------------- drawing fields
class TestDrawingFields:
    def test_line_without_noise(self) -> None:
        t = np.arange(0.0, 101.0)
        r = window_stats(t, 2.0 + 0.5 * t, np.zeros(0, int), WindowSpec(10, 90), None, StatsConfig())
        assert r.intercept == pytest.approx(7.0) and r.intercept_theil_sen == pytest.approx(7.0)
        assert r.two_point_points == ((10.0, 7.0), (90.0, 47.0))
        assert list(r.used_rows) == list(range(10, 91)) and len(r.excluded_rows) == 0

    def test_rows_index_the_input_even_if_unsorted(self) -> None:
        t = np.array([3.0, 0.0, 2.0, 1.0, 4.0])
        y = np.array([30.0, 0.0, 20.0, 10.0, 40.0])
        flags = np.array([False, False, True, False, False])
        r = window_stats(t, y, np.zeros(0, int), WindowSpec(0, 4), flags, StatsConfig(min_points=2))
        assert sorted(r.used_rows) == [0, 1, 3, 4] and list(r.excluded_rows) == [2]
        assert all(y[i] == 10 * t[i] for i in r.used_rows)

    def test_checks_carry_a_level(self) -> None:
        t = np.arange(0.0, 3.0)
        few = window_stats(t, t, np.zeros(0, int), WindowSpec(0, 2), None, StatsConfig())
        assert few.level == "fail" and few.checks[0].level == "fail"
        assert isinstance(few.checks[0], Check) and few.checks[0].startswith("only 3 points")
        drift = window_stats(np.arange(100.0), np.arange(100.0) ** 1.5, np.zeros(0, int), WindowSpec(0, 99), None, StatsConfig())
        assert drift.level == "warn" and all(c.level == "warn" for c in drift.checks)
        flat = window_stats(np.arange(100.0), np.full(100, 3.0), np.zeros(0, int), WindowSpec(0, 99), None, StatsConfig())
        assert [c for c in flat.checks if c.level == "fail"] == []

    def test_slope_se_eff_is_the_larger_corrected_se(self, session: CampaignSession) -> None:
        r = session.window_stats(_step(session, 2.0), "scale", "masse")
        assert r.slope_se_eff == max(r.slope_se_hac, r.slope_se_ar1)


# ----------------------------------------------------------- one step query
class TestSessionWindowStats:
    def test_same_as_the_table_row(self, session: CampaignSession) -> None:
        iv = _step(session, 2.0)
        r = session.window_stats(iv.id, "scale", "masse")
        row = session.window_table([("scale", "masse")]).filter(lambda_id(iv.id)).row(0, named=True)
        assert r.n == row["n"] == 120
        assert r.slope == pytest.approx(row["slope"]) and r.slope_theil_sen == pytest.approx(row["slope_theil_sen"])
        assert "used_rows" not in row and "two_point_points" not in row  # drawing fields stay out of the table
        assert row["window_override"] is False

    def test_unknown_step_or_channel(self, session: CampaignSession) -> None:
        iv = _step(session, 2.0)
        with pytest.raises(KeyError):
            session.window_stats(iv, "scale", "nope")
        with pytest.raises(KeyError):
            session.window_stats("journal:999", "scale", "masse")


def lambda_id(step_id: str):
    import polars as pl

    return pl.col("step_id") == step_id


# ------------------------------------------------------- override per step
class TestWindowOverride:
    def test_override_replaces_the_rule_for_one_step(self, session: CampaignSession, tmp_path: Path) -> None:
        iv, other = _step(session, 2.0), _step(session, 1.0)
        session.set_window_override(iv.id, iv.end - 30, iv.start + 20)  # swapped: sorted
        moved = session.intervals().get(iv.id)
        assert moved.window == pytest.approx((iv.start + 20, iv.end - 30)) and moved.window_override
        kept = session.intervals().get(other.id)
        assert kept.window == other.window and not kept.window_override
        assert session.window_stats(iv.id, "scale", "masse").n < 120
        table = session.window_table([("scale", "masse")])
        assert table.filter(lambda_id(iv.id))["window_override"][0] is True

        session.set_stats_config(StatsConfig(point_halfwidth_s=2.5, exclude_flags=["spike"]))
        path = tmp_path / "campaign.json"
        session.save(path)
        again = CampaignSession.open(path)
        assert again.campaign.window_overrides == session.campaign.window_overrides
        assert again.campaign.window_overrides[iv.id].label == "T Einstellung 155"
        assert again.campaign.stats.point_halfwidth_s == 2.5
        assert again.intervals().get(iv.id).window == pytest.approx(moved.window)

        again.clear_window_override(iv.id)
        assert again.intervals().get(iv.id).window == iv.window and not again.campaign.window_overrides

    def test_stale_overrides_are_reported(self, session: CampaignSession) -> None:
        iv = session.intervals().intervals[3]
        ov = session.set_window_override(iv.id, iv.start, iv.end - 10)
        session.campaign.window_overrides[iv.id] = ov.model_copy(update={"label": "Old name"})
        session.campaign.window_overrides["gone:99"] = ov
        problems = session.intervals().problems
        assert any("Old name" in p and iv.label in p for p in problems)
        assert any(p.startswith("gone:99: window set by hand, but there is no such step") for p in problems)

    def test_override_must_be_ordered(self) -> None:
        with pytest.raises(ValueError):
            WindowOverride(start=datetime(2026, 10, 12, 12), end=datetime(2026, 10, 12, 11))
