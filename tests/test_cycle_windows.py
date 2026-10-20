"""Zyklus-Fenster und Integrale für die Kohlenstoff-Bilanz (PR 11).

Eine Dosierstufe und die folgenden Null-Phasen bilden einen Zyklus: das Fenster reicht bis zum
Start der nächsten Dosierung (`NamedWindow.until_next`), höchstens `max_extend_s` über das Ende.
Integral und abgedeckte Zeit kommen aus derselben Statistik, so lässt sich ein Blindwert exakt
abziehen. Echter FID-Ausschnitt vom 12.10. (300 Zeilen mit der einzigen Lücke der Datei, 11 s).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from datualizer_core.campaign import (
    Calculation,
    Campaign,
    CampaignSession,
    Evaluation,
    Measure,
    NamedWindow,
    StepClass,
    master_datetime,
)
from datualizer_core.stats import StatsConfig

FID = Path(__file__).parent / "fixtures" / "sa_12_10" / "fid_ifid_12_10_extract.dat"
CLASSES = [StepClass(name="dosierung", when="v_mfc > 0"), StepClass(name="null", when="v_mfc == 0")]


@pytest.fixture()
def session() -> CampaignSession:
    """FID extract with four steps by hand: dose, zero, dose, zero (seconds after the first sample)."""
    s = CampaignSession(Campaign())
    s.add_source(FID, source_id="fid")
    t0 = s.track("fid", "conc").x[0]
    zone = s.campaign.timezone
    for (a, b), v in zip([(0, 60), (60, 150), (150, 200), (200, 290)], (1.0, 0.0, 2.0, 0.0)):
        s.add_interval(master_datetime(t0 + a, zone), master_datetime(t0 + b, zone), params={"v_mfc": v})
    s.t0 = t0  # type: ignore[attr-defined]
    return s


def _cycle(max_extend_s: float | None = None) -> NamedWindow:
    return NamedWindow(name="zyklus", classes=["dosierung"], until_next="dosierung", max_extend_s=max_extend_s)


class TestCycleWindow:
    def test_reaches_the_next_dosing_step(self, session: CampaignSession) -> None:
        session.set_evaluation(Evaluation(step_classes=CLASSES, windows=[_cycle(3600)]))
        t0 = session.t0  # type: ignore[attr-defined]
        first, zero, second, _ = session.intervals().intervals
        sw = session.step_window(first, "zyklus")
        assert sw.window == pytest.approx((t0, t0 + 150)) and sw.note == ""
        # no later dosing step: max_extend_s past the end, said in the note
        last = session.step_window(second, "zyklus")
        assert last.window == pytest.approx((t0 + 150, t0 + 200 + 3600))
        assert "no later 'dosierung' step" in last.note
        assert not session.step_window(zero, "zyklus").applies  # only for dosing steps

    def test_capped_by_max_extend(self, session: CampaignSession) -> None:
        session.set_evaluation(Evaluation(step_classes=CLASSES, windows=[_cycle(30)]))
        t0 = session.t0  # type: ignore[attr-defined]
        sw = session.step_window(session.intervals().intervals[0], "zyklus")
        assert sw.window == pytest.approx((t0, t0 + 90))
        assert "ends 30 s after the step" in sw.note

    def test_without_cap_and_without_next_step_ends_with_the_step(self, session: CampaignSession) -> None:
        session.set_evaluation(Evaluation(step_classes=CLASSES, windows=[_cycle()]))
        second = session.intervals().intervals[2]
        sw = session.step_window(second, "zyklus")
        assert sw.window == pytest.approx((second.start, second.end)) and "ends with the step" in sw.note

    def test_validation(self) -> None:
        with pytest.raises(ValidationError, match="until_next names unknown class 'dosis'"):
            Evaluation(step_classes=CLASSES, windows=[NamedWindow(name="z", until_next="dosis")])
        with pytest.raises(ValidationError, match="max_extend_s only applies with until_next"):
            NamedWindow(name="z", max_extend_s=60)


class TestIntegralOverTheCycle:
    def _spec(self, **measure) -> Evaluation:
        return Evaluation(
            step_classes=CLASSES, windows=[_cycle(3600)],
            measures=[
                Measure(alias="I", source="fid", channel="conc", window="zyklus", stat="integral", **measure),
                Measure(alias="T", source="fid", channel="conc", window="zyklus", stat="covered_s", **measure),
            ],
            calculations=[Calculation(name="netto", expr="I - 1.5 * T")],
        )

    def test_integral_and_covered_time_without_bridging_the_gap(self, session: CampaignSession) -> None:
        session.set_evaluation(self._spec())
        row = session.results().row(0, named=True)
        ref = session.window_stats(session.intervals().intervals[0], "fid", "conc", window="zyklus")
        assert ref.window_s == pytest.approx(150)
        assert row["I"] == pytest.approx(ref.integral) and row["T"] == pytest.approx(ref.covered_s)
        assert row["T"] < 150 - 10  # the 11 s gap of the file is not covered
        assert row["u_T"] == 0.0 and row["u_I"] is not None and row["u_I"] > 0
        # a baseline subtracted over the covered time is exact: ∫(c − B) dt = I − B · T
        assert row["netto"] == pytest.approx(ref.integral - 1.5 * ref.covered_s)
        assert row["u_netto"] == pytest.approx(row["u_I"])

    def test_masked_span_left_out_unless_the_measure_keeps_it(self, session: CampaignSession) -> None:
        t0 = session.t0  # type: ignore[attr-defined]
        session.add_mask("fid", t0 + 20, t0 + 40, reason="Prüfgas")
        session.set_evaluation(self._spec(exclude_flags=["masked"]))
        masked = session.results().row(0, named=True)
        session.set_evaluation(self._spec(exclude_flags=[]))
        kept = session.results().row(0, named=True)
        assert kept["T"] - masked["T"] == pytest.approx(20, abs=2)
        assert kept["I"] > masked["I"]
        assert Measure(alias="a", source="s", channel="c").stats_config(StatsConfig()) == StatsConfig()
