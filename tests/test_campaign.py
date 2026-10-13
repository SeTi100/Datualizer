"""Campaign model and clock models (Etappe 2b, step 3): several sources on one master time axis."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from datualizer_core import TimeConfigError, load_csv
from datualizer_core.campaign import (
    Campaign,
    CampaignSession,
    ClockAnchor,
    ClockModel,
    epoch_seconds,
    to_epoch_s,
)

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
FID_DAT = SA / "fid_ifid_12_10_extract.dat"
PICO = SA / "picolog_12_10_extract.csv"
DB = SA / "instrument_reader_12_10_extract.db"
ZONE = "Europe/Zurich"


def zurich(*args) -> float:
    return to_epoch_s(datetime(*args), ZONE)


class TestClockModel:
    T = np.array([0.0, 100.0, 200.0, np.nan])

    def test_identity_and_offset(self) -> None:
        assert ClockModel().is_identity()
        np.testing.assert_array_equal(ClockModel().apply(self.T, ZONE), self.T)
        np.testing.assert_array_equal(ClockModel(offset_s=970).apply(self.T, ZONE), self.T + 970)

    def test_drift_from_first_sample(self) -> None:
        out = ClockModel(drift_ppm=1000).apply(self.T, ZONE)  # 1 ms per s
        np.testing.assert_allclose(out[:3], [0.0, 100.1, 200.2])
        assert np.isnan(out[3])

    def test_one_anchor_is_an_offset(self) -> None:
        a = ClockAnchor(source_time=datetime(2026, 10, 12, 9, 41), master_time=datetime(2026, 10, 12, 10, 50, 30))
        t = np.array([zurich(2026, 10, 12, 9, 41), zurich(2026, 10, 12, 9, 42)])
        np.testing.assert_allclose(ClockModel(anchors=[a], offset_s=5).apply(t, ZONE) - t, 4170.0)

    def test_two_anchors_piecewise_linear_with_extrapolation(self) -> None:
        # a video whose time axis is stretched: 100 s of source time cover 110 s of master time
        anchors = [
            ClockAnchor(source_time=datetime(2026, 1, 1, 0, 0, 0), master_time=datetime(2026, 1, 1, 0, 0, 10)),
            ClockAnchor(source_time=datetime(2026, 1, 1, 0, 1, 40), master_time=datetime(2026, 1, 1, 0, 2, 0)),
        ]
        t0 = zurich(2026, 1, 1)
        out = ClockModel(anchors=anchors).apply(np.array([t0, t0 + 50, t0 + 100, t0 + 200]), ZONE) - t0
        np.testing.assert_allclose(out, [10.0, 65.0, 120.0, 230.0])

    def test_anchors_need_distinct_source_times(self) -> None:
        a = ClockAnchor(source_time=datetime(2026, 1, 1), master_time=datetime(2026, 1, 1))
        with pytest.raises(ValueError):
            ClockModel(anchors=[a, a])

    def test_json_roundtrip(self) -> None:
        m = ClockModel(offset_s=970, method="cross_correlation", uncertainty_s=20, note="incl. transport delay")
        assert ClockModel.model_validate_json(m.model_dump_json()) == m


class TestEpochSeconds:
    def test_wall_clock_dataset_is_read_in_campaign_zone(self) -> None:
        ds = load_csv("time,v\n2026-10-12 10:00:00,1\n2026-10-12 10:00:05,2\n")
        np.testing.assert_allclose(epoch_seconds(ds, ZONE), [zurich(2026, 10, 12, 10), zurich(2026, 10, 12, 10, 0, 5)])

    def test_zoned_dataset_keeps_its_instants(self) -> None:
        ds = load_csv("time,v\n2026-10-12T08:00:00Z,1\n")
        assert epoch_seconds(ds, "Asia/Tokyo")[0] == datetime(2026, 10, 12, 8, tzinfo=timezone.utc).timestamp()

    def test_no_origin_gives_none(self) -> None:
        assert epoch_seconds(load_csv("t,v\n00:00:00,1\n00:00:01,2\n"), ZONE) is None


@pytest.fixture()
def pico_copy(tmp_path: Path) -> Path:
    f = tmp_path / PICO.name
    shutil.copyfile(PICO, f)
    saved = datetime(2026, 10, 12, 16, 0, 21, tzinfo=timezone.utc).timestamp()
    os.utime(f, (saved, saved))
    return f


class TestCampaignSession:
    def test_sources_share_one_master_axis(self, pico_copy: Path) -> None:
        s = CampaignSession()
        fid = s.add_source(FID_DAT)
        pico = s.add_source(pico_copy)
        assert fid.profile.name == "testa_ifid_dat" and pico.profile.name == "picolog_csv"
        assert s.master_time(fid.id)[0] == zurich(2026, 10, 12, 14, 27, 2)
        assert s.span(pico.id)[1] == datetime(2026, 10, 12, 16, 0, 21, tzinfo=timezone.utc).timestamp()

    def test_clock_offset_moves_only_that_source(self, pico_copy: Path) -> None:
        s = CampaignSession()
        fid, pico = s.add_source(FID_DAT), s.add_source(pico_copy)
        before = s.master_time(pico.id).copy()
        s.set_clock(fid.id, ClockModel(offset_s=970, method="manual", note="FID clock 16:10 behind PC"))
        assert s.master_time(fid.id)[0] == zurich(2026, 10, 12, 14, 43, 12)
        np.testing.assert_array_equal(s.master_time(pico.id), before)

    def test_track_carries_unit_and_gaps(self) -> None:
        s = CampaignSession()
        src = s.add_source(FID_DAT)
        tr = s.track(src.id, "conc")
        assert tr.unit == "mgC"
        assert len(tr.x) == len(tr.y) == 300
        assert len(tr.gap_rows) == 1
        assert np.diff(tr.x)[tr.gap_rows[0] - 1] == 11.0

    def test_unique_ids(self) -> None:
        s = CampaignSession()
        assert [s.add_source(FID_DAT).id, s.add_source(FID_DAT).id] == [FID_DAT.stem, f"{FID_DAT.stem}_2"]
        s.remove_source(FID_DAT.stem)
        assert list(s.datasets) == [f"{FID_DAT.stem}_2"]

    def test_save_and_open_reproduce_everything(self, tmp_path: Path, pico_copy: Path) -> None:
        s = CampaignSession(Campaign(name="SA 12.10."))
        fid, pico, db = s.add_source(FID_DAT), s.add_source(pico_copy), s.add_source(DB)
        s.set_clock(fid.id, ClockModel(offset_s=970))
        os.utime(pico_copy, (0, 0))  # later file time must not move the PicoLog axis
        target = tmp_path / "campaign.json"
        s.save(target)
        assert s.campaign.source(pico.id).path == pico_copy.name  # relative to the campaign file
        again = CampaignSession.open(target)
        assert again.warnings == []
        for sid in (fid.id, pico.id, db.id):
            assert again.datasets[sid].df.equals(s.datasets[sid].df)
            np.testing.assert_array_equal(again.master_time(sid), s.master_time(sid))
        assert again.campaign.source(fid.id).clock.offset_s == 970

    def test_changed_and_missing_files_are_reported(self, tmp_path: Path) -> None:
        f = tmp_path / FID_DAT.name
        shutil.copyfile(FID_DAT, f)
        s = CampaignSession()
        s.add_source(f)
        s.save(tmp_path / "c.json")
        with open(f, "ab") as fh:
            fh.write(b"12.10.2026 14:32:02\t7.0\t10.9\t190\t330\t179\t0\t550\t750\t0\t21.4\n")
        assert "changed" in CampaignSession.open(tmp_path / "c.json").warnings[0]
        f.unlink()
        assert "not found" in CampaignSession.open(tmp_path / "c.json").warnings[0]

    def test_unknown_zone_fails_loudly(self) -> None:
        with pytest.raises(TimeConfigError):
            CampaignSession(Campaign(timezone="Mars/Olympus"))
