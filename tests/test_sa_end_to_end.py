"""End-to-end acceptance (PR 10): the SA campaign as configuration against the script evaluation.

`examples/campaign_SA_ethanol.json` on the real data of 12., 15. and 17.10.2026 must give what
`SA/auswertung/sa_auswertung.py` wrote to `ergebnisse_auto/<tag>/schritte.csv` (plan, section 7):
X ± 0.005, c_in ± 0.5 %, u(X) ± 10 %, T_bett ± 0.2 K, rating and stationarity equal, and across
all days T50 (10 Nl/min 179 °C, 20 Nl/min 203 °C) and Ea 21 ± 4 kJ/mol from 8 steps.

The data never go into the repository: the test runs only where DATUALIZER_SA_DIR names the SA
folder (with `auswertung/ergebnisse_auto`), otherwise it is skipped. Steps the reference names
Z<row> (journal) or Run<id> (database) are matched by that name.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

import numpy as np
import polars as pl
import pytest

from datualizer_core.campaign import CampaignSession

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "campaign_SA_ethanol.json"
SA_DIR = os.environ.get("DATUALIZER_SA_DIR")
REFERENCE = Path(SA_DIR) / "auswertung" / "ergebnisse_auto" if SA_DIR else None
DAYS = ("2026-10-12", "2026-10-15", "2026-10-17")

pytestmark = pytest.mark.skipif(
    REFERENCE is None or not REFERENCE.is_dir(),
    reason="DATUALIZER_SA_DIR is not set to the SA folder (real data stay outside the repository)",
)


def reference_name(step_id: str, origin_row: int | None, label: str) -> str:
    """The script's name of a step: Z<sheet row> for the journal, Run<id> for database runs."""
    if step_id.startswith("runs_"):
        return "Run" + re.sub(r"\.0$", "", label)
    return f"Z{origin_row}"


@pytest.fixture(scope="module")
def session() -> CampaignSession:
    return CampaignSession.open(EXAMPLE)


@pytest.fixture(scope="module")
def results(session: CampaignSession) -> pl.DataFrame:
    rows = {iv.id: iv for iv in session.intervals().intervals}
    res = session.results()
    names = [
        reference_name(sid, int(rows[sid].origin.rsplit("row ", 1)[1]) if not rows[sid].manual else None, label)
        for sid, label in zip(res["step_id"], res["step_label"])
    ]
    return res.with_columns(pl.Series("schritt", names))


@pytest.fixture(scope="module")
def comparison(results: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for day in DAYS:
        ref = pl.read_csv(REFERENCE / day / "schritte.csv", separator=";", decimal_comma=True)
        ref = ref.select(
            pl.lit(day).alias("tag"), "schritt", pl.col("c_in_mg_m3").alias("c_in_ref"), pl.col("X").alias("X_ref"),
            pl.col("u_X").alias("u_X_ref"), pl.col("T_bett").alias("T_bett_ref"),
            pl.col("blank_mgC").alias("B_ref"), pl.col("bewertung").alias("bewertung_ref"),
            pl.col("stationaer").cast(pl.String).str.to_lowercase().eq("true").alias("stationaer_ref"),
            pl.col("mdot_quelle").alias("mdot_quelle_ref"),
        )
        parts.append(ref)
    ref = pl.concat(parts)
    ours = results.filter(pl.col("klasse") == "dosierung").select(
        "tag", "schritt", "c_in", "X", "u_X", "T_bett", "B", "bewertung", "stationaer", "mdot_w_quelle",
    )
    return ref.join(ours, on=["tag", "schritt"], how="full", coalesce=True)


def test_every_dosing_step_of_the_script_is_found(comparison: pl.DataFrame) -> None:
    missing = comparison.filter(pl.col("X").is_null() | pl.col("X_ref").is_null())
    assert missing.is_empty(), missing.select("tag", "schritt", "X", "X_ref")


def _off(comparison: pl.DataFrame, condition: pl.Expr, *cols: str) -> list[dict]:
    return comparison.filter(condition).select("tag", "schritt", *cols).to_dicts()


def test_conversion_and_inlet_concentration(comparison: pl.DataFrame) -> None:
    off_x = _off(comparison, (pl.col("X") - pl.col("X_ref")).abs() > 0.005, "X", "X_ref")
    off_c = _off(comparison, (pl.col("c_in") / pl.col("c_in_ref") - 1).abs() > 0.005, "c_in", "c_in_ref")
    assert off_x == [] and off_c == []


def test_uncertainty_of_the_conversion(comparison: pl.DataFrame) -> None:
    off = _off(comparison, (pl.col("u_X") / pl.col("u_X_ref") - 1).abs() > 0.10, "u_X", "u_X_ref")
    assert off == []


def test_bed_temperature(comparison: pl.DataFrame) -> None:
    assert _off(comparison, (pl.col("T_bett") - pl.col("T_bett_ref")).abs() > 0.2, "T_bett", "T_bett_ref") == []


def test_rating_and_stationarity(comparison: pl.DataFrame) -> None:
    off = _off(comparison, (pl.col("bewertung") != pl.col("bewertung_ref")) | (pl.col("stationaer") != pl.col("stationaer_ref")),
               "bewertung", "bewertung_ref", "stationaer", "stationaer_ref")
    assert off == []


def _script_arrhenius() -> tuple[int, float, float]:
    """Ea as katalyse_plots.arrhenius_fit computes it from the script's table of all days (n, Ea, u)."""
    st = pl.read_csv(REFERENCE / "gesamt" / "schritte.csv", separator=";", decimal_comma=True, infer_schema_length=None)
    luft = st["luft_m3_h"].mode()[0]
    d = st.filter(
        (pl.col("c_in_mg_m3") >= 60) & (pl.col("bewertung") != "unsicher") & (pl.col("luft_m3_h") == luft)
        & pl.col("stationaer").cast(pl.String).str.to_lowercase().eq("true")
        & (pl.col("X") > 0.05) & (pl.col("X") < 0.8) & pl.col("k_1_s").is_not_null()
    )
    t = d["T_bett"].to_numpy() + 273.15
    x, ux = 1000 / t, 1000 * d["u_T_bett"].to_numpy() / t**2
    y, uy = np.log(d["k_1_s"].to_numpy()), d["u_lnk_zufall"].to_numpy()
    m = np.polyfit(x, y, 1)[0]
    for _ in range(5):
        sig = np.sqrt(uy**2 + (m * ux) ** 2)
        (m, b), cov = np.polyfit(x, y, 1, w=1 / sig, cov="unscaled")
    chi2 = float(np.sum(((y - (m * x + b)) / sig) ** 2) / max(1, len(x) - 2))
    return len(d), -m * 8.314, float(np.sqrt(cov[0, 0]) * max(1.0, np.sqrt(chi2))) * 8.314


def test_light_off_over_all_days(session: CampaignSession) -> None:
    ref = pl.read_csv(REFERENCE / "katalyse" / "light_off_kennwerte.csv", separator=";", decimal_comma=True)
    an = session.analyses()
    for level in ("T50", "T90"):
        ours = {float(k): g.wert for k, g in an[level].groups.items()}
        theirs = dict(zip(ref["V_MFC"].cast(pl.Float64), ref[level]))
        for v_mfc, text in theirs.items():
            if text.endswith("°C") and text[0].isdigit():  # a crossing, not "< 185 °C"
                assert abs(float(ours[v_mfc].split()[0]) - float(text.split()[0])) <= 1, (level, v_mfc, ours[v_mfc], text)
            else:
                assert ours[v_mfc][0] in "<>", (level, v_mfc, ours[v_mfc], text)


def test_arrhenius_over_all_days(session: CampaignSession) -> None:
    """The script's fit (random uncertainties only) is `derived_zufall`; the GLS fit with the
    correlation of systematic inputs within a day (scope "tag") is reported next to it."""
    n, ea, u_ea = _script_arrhenius()
    fit = session.analyses()["Arrhenius"]
    assert fit.n == n == 8
    ours, u_ours = fit.derived_zufall["Ea_kJ_mol"]
    assert ours == pytest.approx(ea, abs=0.5) and u_ours == pytest.approx(u_ea, abs=0.5)
    gls, u_gls, _ = fit.derived["Ea_kJ_mol"]
    assert abs(gls - ea) < u_gls  # the two readings agree within the uncertainty
