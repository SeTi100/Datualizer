"""Uhrenabgleich (Etappe 2b, Schritt 7): Versatz aus Zellpaaren und Kreuzkorrelation.

Echter Fall: Labjournal SA, Blatt 17.10.2026. „Start FID“ steht auf der FID-Uhr (C3, D3),
„Start Real“ auf der PC-Uhr (C4, D4): 11:26:00 → 11:42:15 und 11:31:46 → 11:48:00, also
975 s und 974 s; Median 974,5 s, Streuung 0,5 s.
"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pytest

from datualizer_core.campaign import (
    CROSS_CORRELATION,
    JOURNAL_PAIRS,
    Campaign,
    CampaignSession,
    CampaignSource,
    CellPair,
    ClockAnchor,
    ClockCells,
    ClockCellsError,
    ClockModel,
    anchors_from_cells,
    clock_from_cells,
    cross_correlate,
)
from datualizer_core.campaign.sync import cell_datetime, resample
from datualizer_core.ingestion.readers import parse_cell, read_cells

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
PAIRS = SA / "labjournal_17_10_clock_pairs.xlsx"
FID = SA / "fid_ifid_12_10_extract.dat"
ZONE = "Europe/Zurich"


def journal_cells(source_id: str = "journal") -> ClockCells:
    return ClockCells(
        source_id=source_id,
        sheet="17.10.2026",
        day=date(2026, 10, 17),
        pairs=[
            CellPair(source_cell="C3", master_cell="C4", note="Punkt 1"),
            CellPair(source_cell="D3", master_cell="D4", note="Punkt 2"),
        ],
    )


# ------------------------------------------------------------------ cell pairs
def test_read_cells_returns_raw_values() -> None:
    vals = read_cells(PAIRS, "17.10.2026", ["B3", "C3", "D4", "Z99"])
    assert vals == {"B3": "Start FID", "C3": time(11, 26), "D4": time(11, 48), "Z99": None}
    assert parse_cell("aa10") == (9, 26)
    with pytest.raises(ValueError):
        parse_cell("C0")


def test_journal_pairs_give_offset_and_spread() -> None:
    model, anchors = clock_from_cells(PAIRS, journal_cells(), ZONE)
    assert [a.source_time for a in anchors] == [datetime(2026, 10, 17, 11, 26), datetime(2026, 10, 17, 11, 31, 46)]
    assert [a.master_time for a in anchors] == [datetime(2026, 10, 17, 11, 42, 15), datetime(2026, 10, 17, 11, 48)]
    assert model.offset_s == pytest.approx(974.5)
    assert model.uncertainty_s == pytest.approx(0.5)
    assert abs(model.offset_s - 975) <= 5  # plan: 975 s ± 5 s
    assert model.method == JOURNAL_PAIRS
    assert "975, 974" in model.note and "17.10.2026" in model.note
    assert model.cells == journal_cells()
    assert not model.anchors  # pure offset: one wrong pair must not bend the mapping


def test_journal_pairs_through_session_correct_the_fid_clock(tmp_path: Path) -> None:
    session = CampaignSession(Campaign(timezone=ZONE))
    fid = session.add_source(FID)
    journal = session.add_source(PAIRS, source_id="journal")
    before = session.master_time(fid.id).copy()
    model, _ = session.clock_from_cells(journal_cells(journal.id))
    session.set_clock(fid.id, model)
    np.testing.assert_allclose(session.master_time(fid.id) - before, 974.5)

    session.save(tmp_path / "campaign.json")
    again = CampaignSession.open(tmp_path / "campaign.json")
    assert again.campaign.source(fid.id).clock == model  # cells, method and note survive the round trip
    np.testing.assert_allclose(again.master_time(fid.id) - before, 974.5)


def test_single_pair_uses_reading_step_as_uncertainty() -> None:
    cells = journal_cells().model_copy(update={"pairs": [CellPair(source_cell="C3", master_cell="C4")]})
    model, anchors = clock_from_cells(PAIRS, cells, ZONE)
    assert model.offset_s == 975 and model.uncertainty_s == 0.5
    assert anchors[0].note == "C3 → C4"


def test_empty_or_dateless_cells_fail_loudly() -> None:
    bad = journal_cells().model_copy(update={"pairs": [CellPair(source_cell="C3", master_cell="E4")]})
    with pytest.raises(ClockCellsError, match="E4 is empty"):
        anchors_from_cells(PAIRS, bad)
    with pytest.raises(ClockCellsError, match="B3 holds no time"):
        anchors_from_cells(PAIRS, bad.model_copy(update={"pairs": [CellPair(source_cell="B3", master_cell="C4")]}))
    with pytest.raises(ClockCellsError, match="without date"):
        anchors_from_cells(PAIRS, journal_cells().model_copy(update={"day": None}))


def test_cell_values_in_other_forms() -> None:
    day = date(2026, 10, 17)
    assert cell_datetime("11:26", day, "A1") == datetime(2026, 10, 17, 11, 26)
    assert cell_datetime(0.5, day, "A1") == datetime(2026, 10, 17, 12, 0)  # Excel fraction of a day
    assert cell_datetime(datetime(1899, 12, 30, 8, 0), day, "A1") == datetime(2026, 10, 17, 8, 0)
    assert cell_datetime(datetime(2026, 10, 5, 8, 0), day, "A1") == datetime(2026, 10, 5, 8, 0)  # own date wins


def test_shifted_moves_offset_or_anchors() -> None:
    assert ClockModel(offset_s=10).shifted(5.5).offset_s == 15.5
    anchored = ClockModel(anchors=[
        ClockAnchor(source_time=datetime(2026, 1, 1, 10), master_time=datetime(2026, 1, 1, 10, 1)),
        ClockAnchor(source_time=datetime(2026, 1, 1, 11), master_time=datetime(2026, 1, 1, 11, 1)),
    ])
    moved = anchored.shifted(30, method="manual")
    assert [a.master_time.second for a in moved.anchors] == [30, 30] and moved.method == "manual"
    t = np.array([1.7e9])
    np.testing.assert_allclose(moved.apply(t, ZONE) - anchored.apply(t, ZONE), 30)


# ----------------------------------------------------------- cross-correlation
def _random_walk(n: int, seed: int = 1) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(size=n))


def test_resample_never_bridges_gaps() -> None:
    x = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
    y = np.array([0.0, 1.0, np.nan, 10.0, 11.0])
    out = resample(x, y, np.array([0.5, 1.0, 1.5, 5.0, 10.5, 12.0]), gap_rows=np.array([3]))
    np.testing.assert_allclose(out, [0.5, 1.0, np.nan, np.nan, 10.5, np.nan])


def test_synthetic_shift_is_found() -> None:
    rng = np.random.default_rng(7)
    t0 = 1.79e9
    ref_x = t0 + np.arange(0.0, 3600.0, 1.0)
    ref_y = _random_walk(ref_x.size)
    true = 973.4
    mov_x = t0 + np.arange(-1500.0, 3000.0, 0.7)  # other rate, other clock
    mov_y = 3.0 * np.interp(mov_x + true, ref_x, ref_y) + 40.0 + rng.normal(scale=0.2, size=mov_x.size)
    mov_y[1000:1100] = np.nan  # logger dropout
    res = cross_correlate(ref_x, ref_y, mov_x, mov_y, t0 + 1200, t0 + 2400, -1800, 1800)
    assert res.lag_s == pytest.approx(true, abs=res.step_s)
    assert res.r > 0.99 and res.level == "ok"
    assert res.step_s == 1.0  # the coarser sampling step
    assert len(res.lags_s) == len(res.r_curve) == 3601
    best = int(np.nanargmax(res.r_curve))
    assert res.lags_s[best] == pytest.approx(round(true))

    deriv = cross_correlate(ref_x, ref_y, mov_x, mov_y, t0 + 1200, t0 + 2400, -1800, 1800, derivative=True)
    assert deriv.lag_s == pytest.approx(true, abs=deriv.step_s)


def test_derivative_ignores_drift_between_devices() -> None:
    t0 = 1.79e9
    x = t0 + np.arange(0.0, 2000.0, 1.0)
    y = np.sin(x / 37.0) * np.exp(-((x - t0 - 1000) / 300) ** 2)  # one burst
    true = -42.0
    mov_y = np.interp(x + true, x, y) + 0.01 * (x - t0)  # plus a slow drift
    res = cross_correlate(x, y, x, mov_y, t0 + 600, t0 + 1400, -200, 200, derivative=True)
    assert res.lag_s == pytest.approx(true, abs=1.0)


def test_search_range_edge_and_ambiguity_are_reported() -> None:
    t0 = 1.79e9
    x = t0 + np.arange(0.0, 3600.0, 1.0)
    y = _random_walk(x.size, seed=3)
    mov = np.interp(x + 300.0, x, y)
    edge = cross_correlate(x, y, x, mov, t0 + 1200, t0 + 2400, -100, 100, derivative=True)
    assert edge.level == "fail"  # the true shift (300 s) is outside the search range

    periodic = np.sin(2 * np.pi * (x - t0) / 60.0)
    res = cross_correlate(x, periodic, x, np.interp(x + 10, x, periodic), t0 + 1000, t0 + 2000, -100, 100)
    assert (res.lag_s - 10) % 60 == pytest.approx(0, abs=1) or (res.lag_s - 10) % 60 == pytest.approx(60, abs=1)
    assert any("Ambiguous" in c for c in res.checks)


def test_flat_or_empty_span_is_an_error() -> None:
    t0 = 1.79e9
    x = t0 + np.arange(0.0, 100.0)
    with pytest.raises(ValueError, match="variation"):
        cross_correlate(x, np.ones_like(x), x, x - t0, t0 + 10, t0 + 90, -5, 5)
    with pytest.raises(ValueError, match="empty"):
        cross_correlate(x, x, x, x, t0 + 10, t0 + 10, -5, 5)


def test_session_finds_offset_between_two_fid_recordings() -> None:
    """The same FID export twice, the second one with a wrong clock: correlation finds it back."""
    session = CampaignSession(Campaign(timezone=ZONE))
    a = session.add_source(FID, source_id="fid")
    b = session.add_source(FID, source_id="fid_copy")
    session.set_clock(b.id, ClockModel(offset_s=-37.0, method="manual"))
    start, end = session.span(a.id)
    res = session.cross_correlate((a.id, "signal"), (b.id, "signal"), start + 60, end - 60, -120, 120)
    assert res.lag_s == pytest.approx(37.0, abs=res.step_s)
    fixed = session.campaign.source(b.id).clock.shifted(res.lag_s, method=CROSS_CORRELATION, note=res.summary())
    session.set_clock(b.id, fixed)
    np.testing.assert_allclose(session.master_time(b.id), session.master_time(a.id), atol=res.step_s)
    with pytest.raises(ValueError, match="same source"):
        session.cross_correlate((a.id, "signal"), (a.id, "conc"), start, end, -10, 10)


def test_campaign_source_without_dataset_can_hold_the_cells() -> None:
    """The journal does not have to load as a table: only its file is read."""
    session = CampaignSession(Campaign(timezone=ZONE))
    session.campaign.sources.append(CampaignSource(id="journal", path=str(PAIRS)))
    model, _ = session.clock_from_cells(journal_cells())
    assert model.offset_s == pytest.approx(974.5)
