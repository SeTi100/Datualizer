"""Etappe 2b, Schritt 8, PR 3: Schrittarten, benannte Fenster, Schritt-Tabelle, praktische Stationarität.

Labjournal und Waage (OCR, SQLite) vom 12.10.: Schritte bekommen eine Klasse aus einer Formel über
ihre Parameter, Messgrössen kommen aus benannten Fenstern mit ihrer Standardunsicherheit. Der
FID-Ausschnitt (flaches Plateau) zeigt den Äquivalenztest: statistisch nicht stationär, praktisch schon.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from datualizer_core.calc import ExprError
from datualizer_core.campaign import (
    Campaign,
    CampaignSession,
    Evaluation,
    Measure,
    NamedWindow,
    StepClass,
    WindowOverride,
    WindowRule,
    classify_steps,
    master_datetime,
)
from datualizer_core.ingestion.profiles import get_profile
from datualizer_core.ingestion.source_loader import load_source, with_choices
from datualizer_core.stats import StatsConfig, WindowSpec, window_stats

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
JOURNAL = SA / "labjournal_12_10_extract.xlsx"
DB = SA / "instrument_reader_12_10_extract.db"
FID = SA / "fid_ifid_12_10_extract.dat"
T155_V1, T155_V2 = "journal:19", "journal:20"  # the steps the scale recording covers

CLASSES = [StepClass(name="null", when="v_mfc == 0"), StepClass(name="dosierung", when="v_mfc > 0")]


def _session() -> CampaignSession:
    s = CampaignSession(Campaign())
    s.add_source(DB, source_id="scale")
    profile = with_choices(JOURNAL, get_profile("labjournal_xlsx"), clock_date=date(2026, 10, 12))
    s.add_source(JOURNAL, profile, dataset=load_source(JOURNAL, profile), source_id="journal")
    s.set_mapping(s.suggest_mapping("journal"))
    return s


@pytest.fixture(scope="module")
def loaded() -> CampaignSession:
    return _session()


@pytest.fixture()
def session(loaded: CampaignSession) -> CampaignSession:
    """A fresh campaign on the loaded data (loading the journal takes a while)."""
    s = CampaignSession(Campaign())
    s.campaign = Campaign.model_validate(loaded.campaign.model_dump())
    s._raw = dict(loaded._raw)
    s._datasets = dict(loaded._datasets)
    return s


# ------------------------------------------------------------------ classes
class TestStepClasses:
    def test_formula_over_parameters(self, session: CampaignSession) -> None:
        session.set_evaluation(Evaluation(step_classes=CLASSES))
        cls = session.step_classes()
        assert cls.of("journal:0") == "null"  # conditioning, V_MFC = 0
        assert cls.of(T155_V1) == "dosierung"
        assert cls.problems == [] and cls.notes == {}
        table = session.step_table()
        assert table["klasse"].to_list().count("null") == 1
        assert table.filter(table["step_id"] == T155_V2)["klasse"][0] == "dosierung"

    def test_no_match_and_several_matches_are_reported(self, session: CampaignSession) -> None:
        classes = [StepClass(name="viel", when="v_mfc >= 10"), StepClass(name="dosiert", when="v_mfc > 0")]
        session.set_evaluation(Evaluation(step_classes=classes))
        cls = session.step_classes()
        assert cls.of("journal:0") is None and cls.notes["journal:0"] == "no class matches"
        assert cls.of("journal:7") == "viel"  # V_MFC = 10: both match, the first wins
        assert cls.notes["journal:7"] == "matches viel, dosiert; 'viel' taken (first in the list)"
        assert cls.of(T155_V1) == "dosiert" and T155_V1 not in cls.notes
        assert "journal:0: no class matches." in cls.problems
        row = session.step_table().filter(lambda_id("journal:0")).row(0, named=True)
        assert row["klasse"] is None and row["klasse_hinweis"] == "no class matches"

    def test_text_parameters_and_step_names(self, session: CampaignSession) -> None:
        session.campaign.interval_mappings = []
        a = session.add_interval(datetime(2026, 10, 12, 10, 0), datetime(2026, 10, 12, 10, 5), params={"Probe": "Eingang"})
        b = session.add_interval(datetime(2026, 10, 12, 10, 5), datetime(2026, 10, 12, 10, 9), params={"Probe": "Ausgang"})
        c = session.add_interval(datetime(2026, 10, 12, 10, 9), datetime(2026, 10, 12, 10, 10), label="Spülen")
        session.set_evaluation(Evaluation(step_classes=[
            StepClass(name="ein", when="Probe == 'Eingang'"),
            StepClass(name="aus", when='Probe != "Eingang" and dauer_s >= 200'),
            StepClass(name="spuelen", when="step_label == 'Spülen'"),
        ]))
        cls = session.step_classes()
        assert [cls.of(iv.id) for iv in (a, b, c)] == ["ein", "aus", "spuelen"]  # c has no Probe: neither == nor !=

    def test_wrong_formula_fails_loudly_with_a_suggestion(self, session: CampaignSession) -> None:
        session.set_evaluation(Evaluation(step_classes=[StepClass(name="dosierung", when="V_MFC > 0")]))
        with pytest.raises(ExprError) as info:
            session.step_classes()
        assert "step class 'dosierung'" in info.value.message and "did you mean 'v_mfc'" in info.value.message
        session.set_evaluation(Evaluation(step_classes=[StepClass(name="x", when="v_mfc == 'a'")]))
        with pytest.raises(ExprError, match="compares text with a number"):  # v_mfc holds numbers
            session.step_classes()
        with pytest.raises(ValidationError, match="syntax error"):
            StepClass(name="x", when="v_mfc >")
        with pytest.raises(ValidationError, match="'<' needs numbers"):
            StepClass(name="x", when="v_mfc < 'a'")

    def test_no_classes_no_hints(self, session: CampaignSession) -> None:
        table = session.step_table()
        assert table["klasse"].null_count() == len(table) and table["klasse_hinweis"].null_count() == len(table)
        assert classify_steps([], CLASSES).classes == {}


# ------------------------------------------------------------------ windows
class TestNamedWindows:
    def test_default_is_the_window_rule(self, session: CampaignSession) -> None:
        session.set_window_rule(WindowRule(length_s=60))
        sw = session.step_window(T155_V2)
        assert sw.name == "default" and sw.window == session.intervals().get(T155_V2).window
        iv = session.intervals().get(T155_V2)
        assert sw.window == pytest.approx((iv.end - 60, iv.end))

    def test_named_window_and_classes(self, session: CampaignSession) -> None:
        session.set_evaluation(Evaluation(
            step_classes=CLASSES,
            windows=[
                NamedWindow(name="masse", rule=WindowRule(skip_start_s=60)),
                NamedWindow(name="blank", rule=WindowRule(length_s=180), classes=["null"]),
            ],
        ))
        iv = session.intervals().get(T155_V2)
        assert session.step_window(iv, "masse").window == pytest.approx((iv.start + 60, iv.end))
        blank = session.step_window(iv, "blank")
        assert not blank.applies and blank.window is None and "'dosierung'" in blank.note
        zero = session.intervals().get("journal:0")
        assert session.step_window(zero, "blank").window == pytest.approx((zero.end - 180, zero.end))
        with pytest.raises(ValueError, match="window 'blank' is for null"):
            session.window_stats(iv, "scale", "masse", window="blank")
        with pytest.raises(KeyError, match="No window 'nope'"):
            session.step_window(iv, "nope")

    def test_override_per_named_window(self, session: CampaignSession, tmp_path: Path) -> None:
        session.set_evaluation(Evaluation(windows=[NamedWindow(name="masse", rule=WindowRule(skip_start_s=60))]))
        iv = session.intervals().get(T155_V2)
        ov = session.set_window_override(iv.id, iv.start + 10, iv.start + 100, window="masse")
        assert ov.window == "masse" and f"{iv.id}@masse" in session.campaign.window_overrides
        sw = session.step_window(iv, "masse")
        assert sw.override and sw.window == pytest.approx((iv.start + 10, iv.start + 100))
        assert not session.intervals().get(iv.id).window_override  # the default window is untouched
        assert session.window_stats(iv.id, "scale", "masse", window="masse").window_s == pytest.approx(90)

        session.save(tmp_path / "campaign.json")
        again = CampaignSession.open(tmp_path / "campaign.json")
        assert again.step_window(iv.id, "masse").window == pytest.approx(sw.window)
        again.clear_window_override(iv.id, "masse")
        assert not again.step_window(iv.id, "masse").override

    def test_stale_named_overrides_are_reported(self, session: CampaignSession) -> None:
        iv = session.intervals().get(T155_V2)
        ov = WindowOverride(start=datetime(2026, 10, 12, 11, 0), end=datetime(2026, 10, 12, 11, 1), window="weg")
        session.campaign.window_overrides[f"{iv.id}@weg"] = ov
        assert f"{iv.id}: window set by hand ('weg'), but there is no such window (kept in the campaign)." in (
            session.intervals().problems
        )
        with pytest.raises(ValidationError, match="key must end in '@weg'"):
            Campaign(window_overrides={iv.id: ov})

    def test_old_campaign_json_reads_unchanged(self, session: CampaignSession) -> None:
        iv = session.intervals().get(T155_V2)
        session.set_window_override(iv.id, iv.start + 20, iv.end - 20)
        data = json.loads(session.campaign.model_dump_json())
        del data["evaluation"]  # a campaign.json from before PR 3
        for ov in data["window_overrides"].values():
            del ov["window"]
        old = Campaign.model_validate(data)
        assert old.evaluation == Evaluation() and old.window_overrides[iv.id].window is None
        assert WindowOverride(start=ov["start"], end=ov["end"], window="default").window is None

    def test_validation(self) -> None:
        with pytest.raises(ValidationError, match="'default' is the campaign's window rule"):
            NamedWindow(name="default")
        with pytest.raises(ValidationError, match="unknown class"):
            Evaluation(windows=[NamedWindow(name="w", classes=["null"])])
        with pytest.raises(ValidationError, match="unknown window 'w'"):
            Evaluation(measures=[Measure(alias="S", source="a", channel="b", window="w")])
        with pytest.raises(ValidationError, match="Unknown statistic 'mittel'"):
            Measure(alias="S", source="a", channel="b", stat="mittel")
        with pytest.raises(ValidationError, match="usable in formulas"):
            Measure(alias="S b", source="a", channel="b")
        with pytest.raises(ValidationError, match="column 'u_S' is already"):
            Evaluation(measures=[Measure(alias="S", source="a", channel="b"), Measure(alias="u_S", source="a", channel="b")])
        with pytest.raises(ValidationError, match="'klasse' is already a step table column"):
            Evaluation(measures=[Measure(alias="klasse", source="a", channel="b")])


# ------------------------------------------------------------------ measures
def lambda_id(step_id: str):
    import polars as pl

    return pl.col("step_id") == step_id


class TestStepTable:
    def _spec(self) -> Evaluation:
        return Evaluation(
            step_classes=CLASSES,
            windows=[
                NamedWindow(name="masse", rule=WindowRule(skip_start_s=60)),
                NamedWindow(name="ende", rule=WindowRule(length_s=60)),
                NamedWindow(name="blank", rule=WindowRule(length_s=180), classes=["null"]),
            ],
            measures=[
                Measure(alias="m", source="scale", channel="masse", window="ende", stat="mean"),
                Measure(alias="mdot", source="scale", channel="masse", window="masse", stat="slope"),
                Measure(alias="d2", source="scale", channel="masse", window="masse", stat="two_point"),
                Measure(alias="integ", source="scale", channel="masse", window="ende", stat="integral"),
                Measure(alias="med", source="scale", channel="masse", window="ende", stat="median"),
                Measure(alias="mb", source="scale", channel="masse", window="blank", stat="mean"),
                Measure(alias="flat", source="scale", channel="masse", window="masse", stat="stationary_practical"),
            ],
        )

    def test_values_and_uncertainties(self, session: CampaignSession) -> None:
        session.set_stats_config(StatsConfig(reading_u={"scale/masse": 0.5}, equivalence={"scale/masse": 1 / 60}))
        session.set_evaluation(self._spec())
        table = session.step_table()
        assert len(table) == len(session.intervals().intervals)
        assert table.columns[:5] == ["step_id", "step_label", "step_start", "step_end", "dauer_s"]
        assert {"temperatur", "v_mfc", "klasse", "klasse_hinweis", "m", "u_m", "checks_m"} <= set(table.columns)
        row = table.filter(lambda_id(T155_V2)).row(0, named=True)
        iv = session.intervals().get(T155_V2)
        assert row["dauer_s"] == pytest.approx(iv.duration_s)
        end = session.window_stats(iv, "scale", "masse", window="ende")
        mass = session.window_stats(iv, "scale", "masse", window="masse")
        assert row["m"] == pytest.approx(end.mean)
        assert row["u_m"] == pytest.approx(math.hypot(end.sem_eff, 0.5))  # sem_eff ⊕ reading_u
        assert row["mdot"] == pytest.approx(mass.slope) and row["u_mdot"] == pytest.approx(mass.slope_se_eff)
        assert row["d2"] == pytest.approx(mass.two_point) and row["u_d2"] == pytest.approx(mass.two_point_u)
        assert row["integ"] == pytest.approx(end.integral) and row["u_integ"] == pytest.approx(end.integral_u)
        assert row["med"] == pytest.approx(end.median) and row["u_med"] is None  # no u defined for a median
        assert row["flat"] in (0.0, 1.0) and row["flat"] == float(mass.stationary_practical)
        assert row["checks_mdot"] == "; ".join(mass.checks)
        assert row["mb"] is None and row["u_mb"] is None and row["checks_mb"] is None  # blank is for zero phases
        assert table["u_m"].dtype == table["m"].dtype  # Float64 even where everything is null

    def test_steps_without_data_say_why(self, session: CampaignSession) -> None:
        session.set_evaluation(self._spec())
        table = session.step_table()
        zero = table.filter(lambda_id("journal:0")).row(0, named=True)  # conditioning: before the scale recording
        assert zero["klasse"] == "null" and zero["mb"] is None and zero["checks_mb"] == "no data in the window"
        short = WindowRule(skip_start_s=400)
        spec = self._spec().model_copy(update={"windows": [NamedWindow(name="masse", rule=short),
                                                            *self._spec().windows[1:]]})
        row = session.step_table(spec).filter(lambda_id(T155_V2)).row(0, named=True)
        assert row["mdot"] is None and row["checks_mdot"] == "window rule 'masse' leaves nothing"

    def test_spec_does_not_change_the_campaign(self, session: CampaignSession) -> None:
        before = session.campaign.model_dump_json()
        table = session.step_table(self._spec())
        assert "mdot" in table.columns and session.campaign.model_dump_json() == before
        assert session.step_table().columns[-2:] == ["klasse", "klasse_hinweis"]

    def test_alias_equal_to_a_parameter_fails(self, session: CampaignSession) -> None:
        spec = Evaluation(measures=[Measure(alias="v_mfc", source="scale", channel="masse")])
        with pytest.raises(ValueError, match="also step parameters"):
            session.step_table(spec)
        with pytest.raises(KeyError, match="no channel 'nope'"):
            session.step_table(Evaluation(measures=[Measure(alias="x", source="scale", channel="nope")]))

    def test_measure_from_several_sources(self, session: CampaignSession) -> None:
        # PR 10d: the scale of 12.10. in the database (afternoon) and in a video (morning, clock 15 s late);
        # invalid OCR readings blanked out with nan (PR 10b) in a calculated channel of the same name
        from datualizer_core.campaign import ClockModel
        from datualizer_core.campaign.derived import DerivedChannel

        session.add_source(SA / "ocr_recording_105012_extract.csv", source_id="ocr")
        session.set_clock("ocr", ClockModel(offset_s=-15, method="manual"))
        session.add_derived("ocr", DerivedChannel(
            name="masse", expr="where(is_valid == 1 and mass_g > 9600 and mass_g < 9700, mass_g, nan)", unit="g"))
        m = Measure(alias="mdot", sources=["scale", "ocr"], channel="masse", stat="slope", min_n=30)
        table = session.step_table(Evaluation(step_classes=CLASSES, measures=[m]))
        by_id = {r["step_id"]: r for r in table.iter_rows(named=True)}
        assert by_id["journal:2"]["mdot_quelle"] == "ocr" and by_id["journal:2"]["mdot"] < 0
        assert by_id[T155_V1]["mdot_quelle"] == "scale" and by_id[T155_V1]["u_mdot"] > 0
        assert "not taken: scale: n = 0" in by_id["journal:2"]["checks_mdot"]
        few = by_id["journal:1"]  # 7 OCR readings at the end of the step: not enough
        assert few["mdot"] is None and few["mdot_quelle"] is None
        assert few["checks_mdot"].startswith("no source with enough points: scale: n = 0")
        assert "ocr: n = 7" in few["checks_mdot"] and "needs n ≥ 30" in few["checks_mdot"]

        # a single source with a threshold: too few points leave the value empty
        one = Measure(alias="m1", source="ocr", channel="masse", stat="slope", min_coverage=0.9)
        row = session.step_table(Evaluation(measures=[one])).filter(lambda_id("journal:2")).row(0, named=True)
        assert row["m1"] is None and row["checks_m1"].startswith("too few points: ocr: n = ")
        assert "m1_quelle" not in session.step_table(Evaluation(measures=[one])).columns

    def test_measure_needs_one_way_to_its_source(self) -> None:
        with pytest.raises(ValidationError, match="either source or sources"):
            Measure(alias="m", channel="masse")
        with pytest.raises(ValidationError, match="either source or sources"):
            Measure(alias="m", source="a", sources=["b"], channel="masse")
        with pytest.raises(ValidationError, match="listed twice"):
            Measure(alias="m", sources=["a", "a"], channel="masse")
        with pytest.raises(ValidationError, match="m_quelle"):
            Evaluation(measures=[Measure(alias="m", sources=["a"], channel="x"),
                                 Measure(alias="m_quelle", source="a", channel="x")])

    def test_campaign_json_roundtrip(self, session: CampaignSession, tmp_path: Path) -> None:
        session.set_evaluation(self._spec())
        session.save(tmp_path / "campaign.json")
        again = CampaignSession.open(tmp_path / "campaign.json")
        assert again.campaign.evaluation == session.campaign.evaluation


# ------------------------------------------------------------ practical stationarity
class TestEquivalence:
    def _fid(self):
        s = CampaignSession(Campaign())
        s.add_source(FID, source_id="fid")
        return s, s.track("fid", "conc")

    def test_flat_fid_plateau_is_practically_stationary(self) -> None:
        _, tr = self._fid()
        delta = 0.1 / 60  # 0.1 mgC/min
        t0 = tr.x[0]
        for lo, hi in [(0, 120), (0, tr.x[-1] - t0)]:  # the reference window (120 s) and the whole extract
            r = window_stats(tr.x, tr.y, tr.gap_rows, WindowSpec(t0 + lo, t0 + hi), None, StatsConfig(), equivalence=delta)
            assert r.stationary is False  # the slope (≈ −0.05 resp. −0.03 mgC/min) is significant
            assert r.stationary_practical is True and r.equivalence_delta == pytest.approx(delta)
            assert any(a.startswith("stationary_practical: TOST at 5% per side, 90% interval") for a in r.assumptions)
            assert not any("practically" in c for c in r.checks)

    def test_tighter_tolerance_fails_and_says_so(self) -> None:
        _, tr = self._fid()
        r = window_stats(tr.x, tr.y, tr.gap_rows, WindowSpec(tr.x[0], tr.x[0] + 120), None, StatsConfig(),
                         equivalence=0.02 / 60)
        assert r.stationary_practical is False
        assert any(c.startswith("not practically stationary: 90% interval of the slope") for c in r.checks)

    def test_not_tested_without_delta(self) -> None:
        _, tr = self._fid()
        r = window_stats(tr.x, tr.y, tr.gap_rows, WindowSpec(tr.x[0], tr.x[0] + 120), None, StatsConfig())
        assert r.stationary_practical is None and r.equivalence_delta is None
        assert not any("stationary_practical" in a for a in r.assumptions)

    def test_delta_from_the_stats_config(self) -> None:
        s, tr = self._fid()
        s.set_stats_config(StatsConfig(equivalence={"fid/conc": 0.1 / 60}))
        zone = s.campaign.timezone  # not the local zone of the machine (CI runs in UTC)
        iv = s.add_interval(*(master_datetime(t, zone) for t in (tr.x[0], tr.x[0] + 120)))
        res = s.window_stats(iv.id, "fid", "conc")
        assert res.stationary_practical is True
        table = s.window_table([("fid", "conc")])
        assert table["stationary_practical"].to_list() == [True]
