"""Display settings of a campaign (plot style, y scale, alias per channel): model, profile default, roundtrip.

Display only: the datasets are never changed, cut or filtered (AGENTS.md, principle 4).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from datualizer_core.campaign import Campaign, CampaignSession, ChannelDisplay, YScale, YScaleMode
from datualizer_core.ingestion.profiles import get_profile
from datualizer_core.schema import PlotStyle

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
FID_DAT = SA / "fid_ifid_12_10_extract.dat"
DB = SA / "instrument_reader_12_10_extract.db"
JOURNAL = SA / "labjournal_12_10_extract.xlsx"


def journal_profile():
    profile = get_profile("labjournal_xlsx")
    profile.excel.sheet = "12.10.2026"
    profile.ingestion.time.clock_date = date(2026, 10, 12)
    return profile


@pytest.fixture()
def session() -> CampaignSession:
    s = CampaignSession(Campaign())
    s.add_source(FID_DAT)
    s.add_source(DB)
    s.add_source(JOURNAL, journal_profile())
    return s


# ------------------------------------------------------------------ plot style
def test_plot_style_default_comes_from_the_profile(session: CampaignSession) -> None:
    """Labjournal channels are measurement steps: the shipped profile draws them as steps, not as a zigzag."""
    assert get_profile("labjournal_xlsx").plot_style is PlotStyle.STEP
    assert get_profile("testa_ifid_dat").plot_style is PlotStyle.LINE  # field default
    journal = session.campaign.sources[2]
    assert session.track(journal.id, "v_mfc").style is PlotStyle.STEP
    assert session.track(FID_DAT.stem, "conc").style is PlotStyle.LINE


def test_style_per_channel_overrides_the_profile(session: CampaignSession) -> None:
    journal = session.campaign.sources[2].id
    session.set_display(journal, "masse_0", ChannelDisplay(style=PlotStyle.POINTS))
    assert session.track(journal, "masse_0").style is PlotStyle.POINTS
    assert session.track(journal, "masse_1").style is PlotStyle.STEP  # other channels keep the default


# --------------------------------------------------------------------- y scale
def test_robust_scale_ignores_ocr_outliers() -> None:
    """massenstrom_waage: a few OCR misreadings far below the rest must not flatten the curve."""
    y = np.concatenate([np.linspace(10.0, 14.0, 500), [-600.0, -550.0], [np.nan]])
    lo, hi = YScale(mode=YScaleMode.ROBUST).limits(y)
    assert 9.0 < lo < 10.5 and 13.5 < hi < 14.5
    assert YScale().limits(y) is None  # auto: the plot decides
    assert YScale(mode=YScaleMode.FIXED, min=-5, max=20).limits(y) == (-5.0, 20.0)


def test_robust_scale_of_a_constant_channel_is_not_empty() -> None:
    lo, hi = YScale(mode=YScaleMode.ROBUST).limits(np.full(10, 3.0))
    assert lo < 3.0 < hi
    assert YScale(mode=YScaleMode.ROBUST).limits(np.full(3, np.nan)) is None


def test_scale_settings_are_validated() -> None:
    with pytest.raises(ValidationError):
        YScale(mode=YScaleMode.FIXED, min=1.0)  # max missing
    with pytest.raises(ValidationError):
        YScale(mode=YScaleMode.FIXED, min=2.0, max=1.0)
    with pytest.raises(ValidationError):
        YScale(mode=YScaleMode.ROBUST, low_percentile=99, high_percentile=1)


def test_scale_never_changes_the_data(session: CampaignSession) -> None:
    before = session.datasets[DB.stem].df.clone()
    session.set_display(DB.stem, "massenstrom_waage", ChannelDisplay(y_scale=YScale(mode=YScaleMode.FIXED, min=0, max=1)))
    tr = session.track(DB.stem, "massenstrom_waage")
    assert tr.y_scale.mode is YScaleMode.FIXED
    assert len(tr.y) == before.height and np.nanmin(tr.y) < 0  # values outside the range are kept
    assert session.datasets[DB.stem].df.equals(before)


# ----------------------------------------------------------------------- alias
def test_alias_names_the_track_but_not_the_data(session: CampaignSession) -> None:
    """Excel columns whose header was only a unit are called col_8, col_9 …; the alias fixes the display."""
    journal = session.campaign.sources[2]
    session.set_display(journal.id, "col_9", ChannelDisplay(alias="Massenstrom Waage"))
    tr = session.track(journal.id, "col_9")
    assert (tr.channel, tr.name) == ("col_9", "Massenstrom Waage")
    assert tr.label == f"{journal.label}: Massenstrom Waage"
    assert "col_9" in session.datasets[journal.id].df.columns
    assert session.track(journal.id, "col_8").name == "col_8"


def test_changed_display_gives_a_new_track(session: CampaignSession) -> None:
    """The plot reuses subplots of the same Track object, so a changed display must yield a new one."""
    first = session.track(FID_DAT.stem, "conc")
    assert session.track(FID_DAT.stem, "conc") is first
    session.set_display(FID_DAT.stem, "conc", ChannelDisplay(alias="FID"))
    assert session.track(FID_DAT.stem, "conc") is not first
    assert session.track(FID_DAT.stem, "conc").x is first.x  # master time is not recomputed


def test_default_display_is_not_stored(session: CampaignSession) -> None:
    session.set_display(FID_DAT.stem, "conc", ChannelDisplay(alias="FID"))
    session.set_display(FID_DAT.stem, "conc", ChannelDisplay())
    assert session.campaign.source(FID_DAT.stem).display == {}


def test_unknown_channel_is_refused(session: CampaignSession) -> None:
    with pytest.raises(KeyError):
        session.set_display(FID_DAT.stem, "no_such_channel", ChannelDisplay(alias="x"))


# -------------------------------------------------------------------- roundtrip
def test_display_settings_survive_save_and_open(session: CampaignSession, tmp_path: Path) -> None:
    journal = session.campaign.sources[2].id
    displays = {
        (journal, "col_9"): ChannelDisplay(alias="Massenstrom Waage", style=PlotStyle.LINE_POINTS),
        (DB.stem, "massenstrom_waage"): ChannelDisplay(
            y_scale=YScale(mode=YScaleMode.ROBUST, low_percentile=2.5, high_percentile=97.5)
        ),
        (FID_DAT.stem, "conc"): ChannelDisplay(y_scale=YScale(mode=YScaleMode.FIXED, min=0.0, max=250.0)),
    }
    for (sid, ch), d in displays.items():
        session.set_display(sid, ch, d)
    target = tmp_path / "campaign.json"
    session.save(target)
    reopened = CampaignSession.open(target)
    assert reopened.warnings == []
    for (sid, ch), d in displays.items():
        assert reopened.campaign.source(sid).display[ch] == d
    assert reopened.track(journal, "col_9").name == "Massenstrom Waage"
    assert reopened.track(journal, "v_mfc").style is PlotStyle.STEP  # profile default stored with the source
    assert Campaign.load(target) == reopened.campaign  # stable: a second save writes the same file
    reopened.save(target)
    assert Campaign.load(target) == reopened.campaign


def test_campaign_files_without_display_still_open(tmp_path: Path) -> None:
    """campaign.json written before display settings existed: everything at its default."""
    old = '{"name": "alt", "timezone": "Europe/Zurich", "sources": [], "notes": ""}'
    (tmp_path / "c.json").write_text(old, encoding="utf-8")
    assert Campaign.load(tmp_path / "c.json").sources == []
