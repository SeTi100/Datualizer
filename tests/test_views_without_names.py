"""Ansichten ohne fest verdrahtete Spaltennamen (PR 13).

Vorher: Facetten-Titel „≈ … °C“ für jede Zahl (die Reproduzierbarkeit zeigte „≈ 5 °C“ für 5 Nl/min),
Balken beschriftet über Spalten `T_bett`/`V_MFC`, Budget-Spalten über Namensmuster (dabei kamen
`u_X_syst`/`u_X_zufall` zusätzlich zu den Einzelbeiträgen in die Anteile), `aligned` suchte
`sequenz`/`beladung`/`tag` und füllte fehlende Kanäle mit dem Schrittwert oder 1.0. Jetzt sind es
Vorlagen und ausdrückliche Spalten; was fehlt, wird nicht gezeichnet.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import polars as pl
import pyqtgraph as pg
import pytest
from pydantic import ValidationError
from PySide6.QtWidgets import QApplication

from datualizer_core.calc.relations import CumulativeRelation, apply_relation, cumulative
from datualizer_core.campaign import AlignedViewSpec, BarsViewSpec, XYViewSpec
from datualizer_gui.results_window import (
    CAT,
    AlignedViewWidget,
    BarsViewWidget,
    XYViewWidget,
    part_colors,
)

T0 = datetime(2026, 10, 12, 10, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


def _table() -> pl.DataFrame:
    """Two loading runs: zero phase (resets), dose, dose | zero phase, dose."""
    mins = [(0, 10), (10, 20), (20, 30), (40, 50), (50, 60)]
    return pl.DataFrame({
        "step_id": ["n1", "d1", "d2", "n2", "d3"],
        "step_start": [T0 + timedelta(minutes=a) for a, _ in mins],
        "step_end": [T0 + timedelta(minutes=b) for _, b in mins],
        "klasse": ["null", "dosierung", "dosierung", "null", "dosierung"],
        "lauf": [1, 1, 1, 2, 2],
        "v_mfc": [0.0, 10.0, 20.0, 0.0, 10.0],
        "T_bett": [150.0, 151.0, 152.0, 200.0, 201.0],
        "c": [0.0, 2.0, 3.0, 0.0, 4.0],
        "X": [np.nan, 0.4, 0.5, np.nan, 0.9],
        "u_X": [np.nan, 0.05, np.nan, np.nan, 0.03],
        "u_X_zufall": [np.nan, 0.04, np.nan, np.nan, 0.02],
        "u_X_syst": [np.nan, 0.03, np.nan, np.nan, 0.02],
        "b_X__S": [np.nan, 0.04, 0.01, np.nan, 0.02],
        "b_X__f": [np.nan, 0.03, 0.01, np.nan, 0.02],
    })


def _titles(w) -> list[str]:
    return [it.titleLabel.text for it in w.glw.ci.items if isinstance(it, pg.PlotItem) and it.titleLabel.text]


# ------------------------------------------------------------------ facets
def test_facet_titles_from_template_or_column_name(qapp) -> None:
    w = XYViewWidget(XYViewSpec(name="xy", x="T_bett", y="X", facet_by="v_mfc"))
    w.set_data(_table())
    assert _titles(w) == ["v_mfc = 10", "v_mfc = 20"]  # no unit made up
    w.spec = XYViewSpec(name="xy", x="T_bett", y="X", facet_by="v_mfc", facet_label="{v_mfc:g} Nl/min · {T_bett|max:.0f} °C")
    w.set_data(_table())
    assert _titles(w) == ["10 Nl/min · 201 °C", "20 Nl/min · 152 °C"]  # filled from the facet's steps
    w.deleteLater()
    qapp.processEvents()


def test_linked_facets_show_the_data_of_all_facets(qapp) -> None:
    w = XYViewWidget(XYViewSpec(name="xy", x="T_bett", y="X", facet_by="v_mfc"))
    w.set_data(_table())
    (x0, x1), (y0, y1) = w.glw.ci.getItem(0, 0).getViewBox().viewRange()
    assert x0 < 151 and x1 > 201 and y0 < 0.4 and y1 > 0.9  # not only the first facet
    w.deleteLater()
    qapp.processEvents()


# ------------------------------------------------------------------ bars
def test_bars_take_the_budget_columns_and_the_label_template(qapp) -> None:
    w = BarsViewWidget(BarsViewSpec(name="b", quantity="X", filter="klasse == 'dosierung'", sort_by=["T_bett"],
                                    bar_label="{T_bett:.0f} °C\n{v_mfc:g}"))
    w.set_data(_table())
    p0, p1 = w.glw.ci.getItem(0, 0), w.glw.ci.getItem(1, 0)
    legend = [it for it in w.glw.ci.items if isinstance(it, pg.LegendItem)][0]
    # only b_X__*: u_X_zufall and u_X_syst are sums of these, they would count twice
    assert [label.text for _, label in legend.items] == ["<b>Quelle</b>", "f", "S"]
    assert dict(p1.getAxis("bottom")._tickLevels[0]) == {0: "151 °C\n10", 1: "152 °C\n20", 2: "201 °C\n10"}
    assert p1.getAxis("left").width() == p0.getAxis("left").width()  # bars of both rows aligned
    assert p1.getAxis("bottom").style["autoExpandTextSpace"] is False  # two lines fit before the first paint
    totals = [it for it in p0.items if isinstance(it, pg.BarGraphItem)][0]
    assert list(totals.opts["x"]) == [0, 2]  # d2 has no u(X): no bar, not a bar of height 0
    assert p1.getAxis("left").labelText == "Anteil an u²(X)"
    w.deleteLater()
    qapp.processEvents()


def test_bars_label_defaults_to_step_id(qapp) -> None:
    w = BarsViewWidget(BarsViewSpec(name="b", quantity="X", filter="klasse == 'dosierung'"))
    w.set_data(_table())
    assert dict(w.glw.ci.getItem(1, 0).getAxis("bottom")._tickLevels[0]) == {0: "d1", 1: "d2", 2: "d3"}
    w.deleteLater()
    qapp.processEvents()


def test_part_colors_do_not_repeat() -> None:
    colors = part_colors(12)
    assert colors[:8] == CAT and len(set(colors)) == 12


# ------------------------------------------------------------------ aligned
def test_sequence_start_needs_the_column() -> None:
    with pytest.raises(ValidationError, match="needs `sequence_col`"):
        AlignedViewSpec(name="a", expr="X", t0="sequence_start")
    assert AlignedViewSpec(name="a", expr="X").t0 == "step_start"


class _Session:
    """Two sources with channel `s`, one per 'day' (as one FID file per measuring day)."""

    def __init__(self) -> None:
        base = T0.timestamp()
        self._t = {"day1": base + np.arange(0, 1800, 10.0), "day2": base + np.arange(2400, 3600, 10.0)}
        self.datasets = {k: SimpleNamespace(channels=["s"], df=pl.DataFrame({"s": np.full(len(t), 2.0 if k == "day1" else 4.0)}))
                         for k, t in self._t.items()}
        self.campaign = SimpleNamespace(evaluation=None)

    def master_time(self, source_id: str) -> np.ndarray:
        return self._t[source_id]


def _curves(w) -> list[tuple[np.ndarray, np.ndarray]]:
    return [it.getData() for it in w.glw.scene().items() if isinstance(it, pg.PlotDataItem)]


def test_aligned_channel_from_the_source_covering_the_step(qapp) -> None:
    spec = AlignedViewSpec(name="a", expr="s / c", filter="klasse == 'dosierung'", t0="sequence_start",
                           sequence_col="lauf", sequence_steps="klasse == 'dosierung'")
    w = AlignedViewWidget(spec, session=_Session())
    w.set_data(_table())
    # (minutes since the run's first dosing, value): d1 and d2 from day1 (s = 2), d3 from day2 (s = 4),
    # not the first source for every step; t0 is d1 resp. d3, not the zero phase that reset the run
    starts = sorted((round(float(x[0])), round(float(y[0]), 3)) for x, y in _curves(w))
    assert starts == [(0, 1.0), (0, 1.0), (10, 0.667)]
    w.deleteLater()
    qapp.processEvents()


def test_aligned_draws_nothing_for_a_missing_name(qapp) -> None:
    w = AlignedViewWidget(AlignedViewSpec(name="a", expr="fid / c", filter="klasse == 'dosierung'"), session=_Session())
    w.set_data(_table())
    assert _curves(w) == [] and "3 step(s) without data" in w.status.text()  # no curve at 1.0 made up
    w.deleteLater()
    qapp.processEvents()


# ------------------------------------------------------------------ cumulative runs
def test_cumulative_numbers_the_runs() -> None:
    t = _table()
    s, runs = cumulative(t, name="beladung", expr="c", reset_when="klasse == 'null'", gap_s=1e9, runs=True)
    assert s.to_list() == [0.0, 2.0, 5.0, 0.0, 4.0]
    assert runs.name == "beladung_lauf" and runs.to_list() == [1, 1, 1, 2, 2]
    out = apply_relation(t, CumulativeRelation(name="beladung", expr="c", reset_when="klasse == 'null'"))
    assert out["beladung_lauf"].to_list() == [1, 1, 1, 2, 2]
    # a gap starts a new run as well
    assert cumulative(t, name="b", expr="c", gap_s=300, runs=True)[1].to_list() == [1, 1, 1, 2, 2]
