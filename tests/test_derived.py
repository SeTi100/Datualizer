"""Calculated channels, masks and device calibrations of a campaign (Etappe 2b, step 8, PR 2)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3

import numpy as np
import polars as pl
import pytest

from datualizer_core.campaign import (
    MASKED,
    Campaign,
    CampaignSession,
    ClockModel,
    DerivedChannel,
    Mask,
    Piece,
    calibration_channel,
    master_datetime,
)
from datualizer_core.ingestion.calibration import CalibrationFields, suggest_linear_calibration
from datualizer_core.ingestion.profiles import ReaderKind, detect_profile, get_profile
from datualizer_core.ingestion.readers import ReaderError, SqliteOptions, XmlOptions, read_sqlite, read_xml_meta
from datualizer_core.ingestion.source_loader import load_source
from datualizer_core.stats.window import StatsConfig

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
FID_DAT = SA / "fid_ifid_12_10_extract.dat"
SETTINGS = SA / "ifid_settings_17_10_extract.xml"
DB = SA / "instrument_reader_12_10_extract.db"
FID_OFFSET_S = 970.0  # FID clock behind the PC (12.10.)


@pytest.fixture()
def fid() -> tuple[CampaignSession, str]:
    s = CampaignSession()
    src = s.add_source(FID_DAT)
    s.set_clock(src.id, ClockModel(offset_s=FID_OFFSET_S))
    return s, src.id


def at(s: CampaignSession, epoch: float) -> datetime:
    return master_datetime(epoch, s.campaign.timezone)


# ------------------------------------------------------------- calculated channels
class TestDerivedChannel:
    def test_formula_on_raw_signal_raw_channels_unchanged(self, fid) -> None:
        s, sid = fid
        raw = s.raw_dataset(sid)
        s.add_derived(sid, DerivedChannel(name="fid_mgC", expr="0.38 * signal - 1.786", unit="mgC"))
        ds = s.datasets[sid]
        assert "fid_mgC" in ds.channels and "fid_mgC" not in raw.columns
        np.testing.assert_allclose(ds.df["fid_mgC"].to_numpy(), 0.38 * raw.df["signal"].to_numpy() - 1.786)
        assert ds.df.drop("fid_mgC").equals(raw.df)  # raw channels exactly as loaded
        assert ds.units["fid_mgC"] == "mgC"
        assert ds.channel_attrs["fid_mgC"]["is_calculated"] is True
        assert "0.38 * signal - 1.786" in ds.channel_attrs["fid_mgC"]["formula"]
        tr = s.track(sid, "fid_mgC")  # a normal channel: track, plot, statistics
        assert tr.unit == "mgC" and len(tr.y) == len(raw)

    def test_pieces_follow_master_time(self, fid) -> None:
        # FID scale changed at a recalibration: other factors before and after an instant
        s, sid = fid
        t = s.master_time(sid)
        cut = float(t[150])
        s.add_derived(sid, DerivedChannel(
            name="fid_mgC", unit="mgC",
            pieces=[Piece(end=at(s, cut), expr="1 * signal"), Piece(start=at(s, cut), expr="2 * signal")],
        ))
        y = s.datasets[sid].df["fid_mgC"].to_numpy()
        sig = s.raw_dataset(sid).df["signal"].to_numpy()
        np.testing.assert_allclose(y[:150], sig[:150])
        np.testing.assert_allclose(y[150:], 2 * sig[150:])  # start included, end excluded

    def test_pieces_move_with_the_clock(self, fid) -> None:
        s, sid = fid
        cut = float(s.master_time(sid)[150])
        s.add_derived(sid, DerivedChannel(name="d", expr="signal", pieces=[Piece(start=at(s, cut), expr="0 * signal")]))
        assert (s.datasets[sid].df["d"].to_numpy()[150:] == 0).all()
        s.set_clock(sid, ClockModel(offset_s=FID_OFFSET_S + 50))  # samples move 50 s later on master time
        y = s.datasets[sid].df["d"].to_numpy()
        after = s.master_time(sid) >= cut
        assert 0 < after.argmax() < 150  # the 11 s gap of the extract lies in between
        assert (y[after] == 0).all() and (y[~after] != 0).all()

    def test_default_formula_outside_pieces_or_none(self, fid) -> None:
        s, sid = fid
        t = s.master_time(sid)
        s.add_derived(sid, DerivedChannel(name="d", pieces=[Piece(start=at(s, t[10]), end=at(s, t[20]), expr="signal")]))
        y = s.datasets[sid].df["d"]
        assert y.null_count() == len(y) - 10
        flags = s.datasets[sid].quality_flags.filter(pl.col("channel") == "d")
        assert "missing_calculated" in flags["flag"].to_list()  # nulls of a formula, not a failed measurement
        assert "missing" not in flags["flag"].to_list()

    def test_overlapping_pieces_fail_loudly(self, fid) -> None:
        s, sid = fid
        t = s.master_time(sid)
        with pytest.raises(ValueError, match="overlap"):
            s.add_derived(sid, DerivedChannel(name="d", pieces=[
                Piece(end=at(s, t[100]), expr="signal"), Piece(start=at(s, t[50]), expr="2 * signal"),
            ]))
        assert s.campaign.source(sid).derived == []

    def test_unknown_name_and_name_in_use(self, fid) -> None:
        s, sid = fid
        with pytest.raises(ValueError, match="sgnal"):
            s.add_derived(sid, DerivedChannel(name="d", expr="sgnal * 2"))
        with pytest.raises(ValueError, match="already has a column"):
            s.add_derived(sid, DerivedChannel(name="signal", expr="conc"))
        with pytest.raises(ValueError, match="no formula"):
            DerivedChannel(name="d")

    def test_inherits_flags_of_its_inputs(self, fid) -> None:
        s, sid = fid
        s.add_derived(sid, DerivedChannel(name="d", expr="0.38 * signal - 1.786"))
        flags = s.datasets[sid].quality_flags
        spikes = lambda ch: set(flags.filter((pl.col("channel") == ch) & (pl.col("flag") == "spike"))["row"].to_list())
        assert spikes("signal") and spikes("signal") <= spikes("d")

    def test_chain_and_remove_rules(self, fid) -> None:
        s, sid = fid
        s.add_derived(sid, DerivedChannel(name="a", expr="signal * 2"))
        s.add_derived(sid, DerivedChannel(name="b", expr="a + 1"))
        np.testing.assert_allclose(s.datasets[sid].df["b"].to_numpy(), 2 * s.raw_dataset(sid).df["signal"].to_numpy() + 1)
        with pytest.raises(ValueError, match="used by b"):
            s.remove_derived(sid, "a")
        s.update_derived(sid, "b", DerivedChannel(name="b", expr="a - 1"))
        np.testing.assert_allclose(s.track(sid, "b").y, 2 * s.raw_dataset(sid).df["signal"].to_numpy() - 1)
        s.remove_derived(sid, "b")
        s.remove_derived(sid, "a")
        assert s.datasets[sid].channels == s.raw_dataset(sid).channels

    def test_broken_channel_in_campaign_file_is_reported_not_fatal(self, fid, tmp_path: Path) -> None:
        s, sid = fid
        s.campaign.source(sid).derived.append(DerivedChannel(name="d", expr="nonexistent * 2"))
        path = tmp_path / "campaign.json"
        s.save(path)
        again = CampaignSession.open(path)
        assert "d" not in again.datasets[sid].channels
        assert any("nonexistent" in p for p in again.derived_problems(sid))


# --------------------------------------------------------------------------- masks
class TestMask:
    def test_masked_rows_and_window_statistics(self, fid) -> None:
        s, sid = fid
        t = s.master_time(sid)
        assert MASKED in StatsConfig().exclude_flags  # left out by default
        step = s.add_interval(at(s, t[0]), at(s, t[-1]))
        before = s.window_stats(step.id, sid, "signal")
        s.add_mask(sid, t[100], t[119], reason="test gas")  # both ends included
        ev = s.datasets[sid].quality_flags.filter(pl.col("flag") == MASKED)
        assert ev["row"].to_list() == list(range(100, 120))
        assert ev["channel"].null_count() == 20  # all channels: row level
        after = s.window_stats(step.id, sid, "signal")
        new = set(range(100, 120)) - _excluded_before(s, sid, "signal")
        assert after.n_excluded - before.n_excluded == len(new)
        assert after.n < before.n
        np.testing.assert_array_equal(s.track(sid, "signal").y, s.raw_dataset(sid).df["signal"].to_numpy())  # nothing deleted

    def test_mask_of_one_channel_and_calculated_channels(self, fid) -> None:
        s, sid = fid
        t = s.master_time(sid)
        s.add_derived(sid, DerivedChannel(name="d", expr="signal * 2"))
        s.add_mask(sid, t[10], t[12], channels=["signal"], reason="spike by hand")
        ev = s.datasets[sid].quality_flags.filter(pl.col("flag") == MASKED)
        assert set(ev["channel"].to_list()) == {"signal", "d"}  # d inherits the mask of its input
        assert sorted(ev.filter(pl.col("channel") == "d")["row"].to_list()) == [10, 11, 12]
        with pytest.raises(KeyError):
            s.add_mask(sid, t[0], t[1], channels=["nope"])
        s.add_mask(sid, t[0], t[1], channels=["d"])
        with pytest.raises(ValueError, match="used by mask 2"):
            s.remove_derived(sid, "d")

    def test_masks_stay_on_master_time_when_the_clock_changes(self, fid) -> None:
        s, sid = fid
        t = s.master_time(sid)
        s.add_mask(sid, t[100], t[109])
        s.set_clock(sid, ClockModel(offset_s=FID_OFFSET_S + 50))
        rows = s.datasets[sid].quality_flags.filter(pl.col("flag") == MASKED)["row"].to_list()
        assert rows == list(range(50, 60))

    def test_update_remove_and_json_round_trip(self, fid, tmp_path: Path) -> None:
        s, sid = fid
        t = s.master_time(sid)
        s.add_derived(sid, DerivedChannel(name="fid_mgC", expr="0.38 * signal - 1.786", unit="mgC"))
        s.add_mask(sid, t[5], t[1], reason="swapped ends")  # start and end may be swapped
        s.add_mask(sid, t[200], t[210], channels=["fid_mgC"], reason="Prüfgas")
        s.update_mask(sid, 0, t[1], t[3], reason="shorter")
        spans = s.mask_spans()
        assert [(m.start, m.end, m.reason) for m in spans][0] == (t[1], t[3], "shorter")
        path = tmp_path / "campaign.json"
        s.save(path)
        again = CampaignSession.open(path)
        src = again.campaign.source(sid)
        assert [m.reason for m in src.masks] == ["shorter", "Prüfgas"] and src.derived[0].unit == "mgC"
        ev = again.datasets[sid].quality_flags.filter(pl.col("flag") == MASKED)
        assert set(ev["row"].to_list()) == {1, 2, 3, *range(200, 211)}
        again.remove_mask(sid, 0)
        assert len(again.campaign.source(sid).masks) == 1
        with pytest.raises(KeyError):
            again.remove_mask(sid, 5)

    def test_mask_validation(self) -> None:
        with pytest.raises(ValueError, match="not after"):
            Mask(start=datetime(2026, 10, 12, 10), end=datetime(2026, 10, 12, 9))


def _excluded_before(s: CampaignSession, sid: str, ch: str) -> set[int]:
    """Rows already left out for other flags (spike, dropout), so the count of new ones is exact."""
    flags = s.datasets[sid].quality_flags.filter(
        pl.col("flag").is_in(["spike", "dropout"]) & (pl.col("channel").is_null() | (pl.col("channel") == ch))
    )
    return set(flags["row"].to_list())


# ------------------------------------------------------------------- calibration
class TestIfidSettings:
    def test_profile_reads_only_the_calibration_blocks(self) -> None:
        assert detect_profile(SETTINGS).name == "testa_ifid_settings_xml"
        ds = load_source(SETTINGS)
        meta = ds.source_meta
        assert meta["reader"] == ReaderKind.XML.value
        assert set(meta["xml"]) == {"calib_data", "calib_init", "rf-select"}
        cal = meta["xml"]["calib_data"]
        assert cal["sens"] == 4.213281 and cal["zero"] == 4.527784 and cal["std_gas"] == 50.0
        assert cal["sig_span"] == 215.1918 and cal["time_last_cal"] == "2026-10-17 11:19:52"
        assert meta["xml"]["rf-select"]["name-eng"] == "Propane"
        assert len(ds) == 0 and ds.time_origin is None and ds.channels == []

    def test_secrets_in_other_elements_never_reach_the_metadata(self, tmp_path: Path) -> None:
        # the real file holds a VNC password hash, a PIN and a system id next to the calibration
        f = tmp_path / "Settings.xml"
        f.write_text(
            "<settings><vnc-pw>SECRET-HASH</vnc-pw><secure-mode-pin>4711</secure-mode-pin>"
            "<system-info>SYSTEM-ID</system-info>"
            "<calib_data><sens>4.0e+000</sens><zero>1.0e+000</zero><nested><x>SECRET-NESTED</x></nested></calib_data>"
            "</settings>",
            encoding="utf-8",
        )
        ds = load_source(f)
        text = repr(ds.source_meta)
        for secret in ("SECRET-HASH", "4711", "SYSTEM-ID", "SECRET-NESTED"):
            assert secret not in text
        assert ds.source_meta["xml_missing_blocks"] == ["calib_init", "rf-select"]
        assert read_xml_meta(f).meta["xml"] == {}  # without blocks nothing is kept

    def test_not_xml_raises_reader_error(self, tmp_path: Path) -> None:
        f = tmp_path / "broken.xml"
        f.write_text("<settings><calib_data>", encoding="utf-8")
        with pytest.raises(ReaderError, match="not a readable XML"):
            read_xml_meta(f, XmlOptions(blocks=["calib_data"]))

    def test_suggest_linear_calibration(self) -> None:
        cal = suggest_linear_calibration(load_source(SETTINGS).source_meta, 1.616)
        assert cal.sens == 4.213281 and cal.zero == 4.527784
        assert cal.a == pytest.approx(0.38355, abs=5e-6)
        assert cal.b == pytest.approx(-1.7366, abs=5e-5)
        assert cal.time == "2026-10-17 11:19:52"
        assert "4.213281" in cal.note() and "1.616 mgC/ppm" in cal.note()

    def test_suggestion_as_calculated_channel_on_fid_data(self, fid) -> None:
        s, sid = fid
        cal = suggest_linear_calibration(load_source(SETTINGS).source_meta, 1.616)
        fields = cal.fields
        dc = calibration_channel(cal, fields.signal_channel, "fid_mgC", source=SETTINGS.name)
        assert dc.unit == "mgC" and SETTINGS.name in dc.note
        s.add_derived(sid, dc)
        sig = s.raw_dataset(sid).df["signal"].to_numpy()
        np.testing.assert_allclose(s.datasets[sid].df["fid_mgC"].to_numpy(), cal.a * sig + cal.b, rtol=1e-12)

    def test_missing_values_fail_loudly(self) -> None:
        meta = load_source(SETTINGS).source_meta
        with pytest.raises(ValueError, match="no calibration block"):
            suggest_linear_calibration(meta, 1.616, CalibrationFields(block="other", sens="sens", zero="zero"))
        with pytest.raises(ValueError, match="'gain' is missing"):
            suggest_linear_calibration(meta, 1.616, CalibrationFields(block="calib_data", sens="gain", zero="zero"))
        with pytest.raises(ValueError, match="no calibration description"):
            suggest_linear_calibration({"xml": meta["xml"]}, 1.616)


# -------------------------------------------------------------------- JSON columns
class TestJsonColumns:
    def test_run_parameters_split_missing_key_is_empty(self) -> None:
        raw = read_sqlite(DB, SqliteOptions(table="runs", json_columns=["parameters_json"]))
        df = raw.df
        cols = df.columns
        assert cols[cols.index("parameters_json") + 1: cols.index("parameters_json") + 3] == ["Temperatur", "V_MFC"]
        run1 = df.filter(pl.col("id") == "1").row(0, named=True)
        assert run1["Temperatur"] == "150" and run1["V_MFC"] is None  # null phase without V_MFC
        assert df.filter(pl.col("id") == "12")["V_MFC"][0] == "10"
        assert raw.meta["json_columns"]["parameters_json"]["invalid_rows"] == 0

    def test_profile_gives_parameters_per_run(self) -> None:
        ds = load_source(DB)
        assert get_profile("instrument_reader_db").sqlite.json_columns == ["run_parameters"]
        assert {"temperatur", "v_mfc"} <= set(ds.parameters)
        per_run = ds.df.select("run_id", "v_mfc").unique().sort("run_id")
        assert per_run.rows() == [(9, 1.0), (10, 2.0)]

    def test_other_keys_invalid_json_and_name_clash(self, tmp_path: Path) -> None:
        db = tmp_path / "x.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE runs (id INTEGER, V_MFC REAL, p TEXT)")
        con.executemany("INSERT INTO runs VALUES (?, ?, ?)", [
            (1, 0, '{"V_MFC": "5", "Volumenstrom": 100, "f_Fan": 35.5, "x": {"a": 1}}'),
            (2, 0, "not json"),
            (3, 0, None),
            (4, 0, "[1, 2]"),
        ])
        con.commit()
        con.close()
        raw = read_sqlite(db, SqliteOptions(table="runs", json_columns=["p"]))
        assert raw.df.columns == ["id", "V_MFC", "p", "p_V_MFC", "Volumenstrom", "f_Fan", "x"]
        assert raw.df.row(0) == ("1", "0.0", raw.df["p"][0], "5", "100", "35.5", '{"a": 1}')
        assert raw.df["Volumenstrom"].to_list()[1:] == [None, None, None]
        info = raw.meta["json_columns"]["p"]
        assert info["invalid_rows"] == 2 and info["invalid_examples"] == [1, 3]
        with pytest.raises(ReaderError, match="json_columns"):
            read_sqlite(db, SqliteOptions(table="runs", json_columns=["nope"]))


def test_with_calculated_needs_the_same_table() -> None:
    ds = load_source(FID_DAT)
    with pytest.raises(ValueError, match="same rows"):
        ds.with_calculated(ds.df.head(3).with_columns(pl.lit(1.0).alias("x")))
    with pytest.raises(ValueError, match="same rows"):
        ds.with_calculated(ds.df.select(reversed(ds.df.columns)))
    out = ds.with_calculated(ds.df.with_columns((pl.col("signal") * 2).alias("x")), units={"x": "pA"})
    assert out.channels == [*ds.channels, "x"] and out.units["x"] == "pA"
    assert ds.channels[-1] != "x"  # the original is not changed
