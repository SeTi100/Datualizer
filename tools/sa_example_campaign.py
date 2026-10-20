r"""Writes examples/campaign_SA_ethanol.json and examples/rezept_katalyse_voc.json (Etappe 2b, PR 10, 11).

The SA measuring campaign (catalytic ethanol oxidation, 12., 15. and 17.10.2026) as configuration:
every source of the three days, clocks, calculated channels, masks, steps and the full evaluation.
Nothing of it is code in the Datualizer; this script only writes the JSON so it stays readable and
reviewable (run it again after changing it):

    .venv\Scripts\python.exe tools\sa_example_campaign.py            (writes the two JSON files)
    .venv\Scripts\python.exe tools\sa_example_campaign.py --bilder   (also docs/img/sa_*.png and the
                                                                     report, needs DATUALIZER_SA_DIR)

The data stay outside the repository: source paths start with ${DATUALIZER_SA_DIR} (the SA folder
with Data-2026-10-1x, the PicoLog files and the lab journal). The recipe is the same evaluation
without sources, for a new campaign of the same kind.
"""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout, not an installed copy

from datualizer_core.calc import Constant, CumulativeRelation, NearestReferenceRelation
from datualizer_core.calc.analyses import CrossingAnalysis, FitLineAnalysis
from datualizer_core.calc.relations import BinRelation
from datualizer_core.campaign import (
    AlignedViewSpec,
    BarsViewSpec,
    Calculation,
    Campaign,
    CampaignSource,
    CellPair,
    ClockCells,
    ClockModel,
    Criterion,
    Evaluation,
    Interval,
    IntervalMapping,
    MatrixViewSpec,
    Measure,
    NamedWindow,
    ParameterLookup,
    Rating,
    StepClass,
    TableViewSpec,
    WindowRule,
    XYViewSpec,
)
from datualizer_core.campaign.derived import DerivedChannel, Mask
from datualizer_core.campaign.intervals import CellValue
from datualizer_core.campaign.views import (
    RefLine,
    ReportViewSpec,
    SecondaryAxis,
    TimelineMarks,
    TimelinePanel,
    TimelineTrace,
    TimelineViewSpec,
)
from datualizer_core.ingestion.config import IngestionConfig, QualityConfig, TimeConfig
from datualizer_core.ingestion.profiles import OriginRule, ReaderKind, SourceProfile, get_profile
from datualizer_core.ingestion.readers.sqlite import SqliteOptions
from datualizer_core.ingestion.steps import EndKind, EndRule, RowSelection, StopRule
from datualizer_core.stats import StatsConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples"
SA = "${DATUALIZER_SA_DIR}"
ZONE = "Europe/Zurich"
JOURNAL = f"{SA}/SA_Labjournal_xlsx.xlsx"
DAYS = ("2026-10-12", "2026-10-15", "2026-10-17")


# ---------------------------------------------------------------- profiles (data, not code)
def journal(sheet: str, day: date) -> SourceProfile:
    prof = get_profile("labjournal_xlsx")
    prof.excel.sheet = sheet
    prof.ingestion.time.clock_date = day
    return prof


def picolog(start: datetime) -> SourceProfile:
    """PicoLog start as in the day configuration (save time minus duration), fixed: file times change on copies."""
    prof = get_profile("picolog_csv")
    prof.origin_rule = OriginRule.NONE
    prof.ingestion.time.origin = start
    return prof


def scale_db() -> SourceProfile:
    """OCR balance with the outlier rule of the script: |deviation from the 31 s median| > 0.05 g."""
    prof = get_profile("instrument_reader_db")
    prof.ingestion.quality = QualityConfig(
        spike_window_s=31.0, spike_k=1.0, spike_channels=["masse"], spike_min_scale={"masse": 0.05}
    )
    return prof


def runs_db() -> SourceProfile:
    """Steps of 15.10.: runs of experiments 7 and 8; set points from parameters_json under uniform names."""
    return SourceProfile(
        name="instrument_reader_runs", reader=ReaderKind.SQLITE,
        description="Runs der Experimente 7 (Leertest) und 8 (Ethanol); started_at in Lokalzeit.",
        sqlite=SqliteOptions(query=(
            "SELECT id AS run_id, started_at, ended_at, target_temperature AS temperatur, "
            "json_extract(parameters_json, '$.V_MFC') AS v_mfc, "
            "json_extract(parameters_json, '$.Volumenstrom') AS luft "
            "FROM runs WHERE experiment_id IN (7, 8) ORDER BY started_at"
        )),
        ingestion=IngestionConfig(time_column="started_at", time=TimeConfig(timezone=ZONE)),
    )


def fid(path: str, offset_s: float, a: float, b: float, scale_note: str, **clock) -> CampaignSource:
    sid = "fid_" + path.split("Data-2026-10-")[1][:2]
    return CampaignSource(
        id=sid, path=f"{SA}/{path}",
        clock=ClockModel(offset_s=offset_s, **({"method": "manual", "note": "wie Tageskonfiguration"} | clock)),
        derived=[DerivedChannel(name="fid_mgc", expr=f"{a} * signal + ({b})", unit="mgC", note=scale_note)],
    )


# ---------------------------------------------------------------- sources
def sources() -> list[CampaignSource]:
    fid12 = fid("Data-2026-10-12/data-10-32-50.dat", 970, 0.38384, -2.36953,
                "Anzeige am 12.10. 3x zu hoch; Skala aus dem Rohsignal mit der Kalibrierung vom 05.10.")
    fid15a = fid("Data-2026-10-15/data-08-15-38.csv", 975, 0.38, -1.786, "Skala nach der Kalibrierung um 09:24")
    fid15a.id = "fid_15a"
    fid15a.masks = [Mask(start=datetime(2026, 10, 15, 9, 5), end=datetime(2026, 10, 15, 9, 25), reason="Prüfgas")]
    fid15b = fid("Data-2026-10-15/data-09-33-16.csv", 975, 0.38, -1.786, "Skala nach der Kalibrierung um 09:24")
    fid15b.id = "fid_15b"
    cells = ClockCells(source_id="journal_17", sheet="17.10.2026", day=date(2026, 10, 17), pairs=[
        CellPair(source_label="Start FID", master_label="Start Real", nth=n) for n in (1, 2)])
    fid17 = fid("Data-2026-10-17/data-11-24-51.csv", 974.5, 0.38355, -1.7366,
                "Settings_17_10.xml: 1.616 mgC/ppm / sens 4.2133 pA/ppm, zero 4.528 pA (suggest_linear_calibration)",
                method="journal_pairs", uncertainty_s=0.5, cells=cells,
                note="Journal: Start FID 11:26:00 / 11:31:46 = Start Real 11:42:15 / 11:48:00")
    fid17.masks = [Mask(start=datetime(2026, 10, 17, 11, 30), end=datetime(2026, 10, 17, 11, 43, 30),
                        reason="Ende Prüfgas am Dateianfang")]
    ocr = CampaignSource(
        id="ocr_12", path=f"{SA}/../SA_Analyse/results/ocr_recording_20261012_105012.csv",
        clock=ClockModel(offset_s=-15, method="manual", note="Video-Zeitstempel ca. 15 s zu spät"),
        derived=[DerivedChannel(
            name="masse", unit="g", note="nur gültige und plausible Lesungen (Bereich 9600-9700 g wie im Skript)",
            expr="where(is_valid == 1 and mass_g > 9600 and mass_g < 9700, mass_g, nan)")],
    )
    return [
        CampaignSource(id="journal_12", path=JOURNAL, profile=journal("12.10.2026", date(2026, 10, 12))),
        CampaignSource(id="journal_17", path=JOURNAL, profile=journal("17.10.2026", date(2026, 10, 17))),
        fid12, fid15a, fid15b, fid17,
        CampaignSource(id="settings_17", path=f"{SA}/Settings_17_10.xml", label="iFID-Kalibrierung 17.10."),
        CampaignSource(id="pico_12", path=f"{SA}/SA_Messung_Picolog_12_10_2026.csv",
                       profile=picolog(datetime(2026, 10, 12, 8, 42, 58))),
        CampaignSource(id="pico_15", path=f"{SA}/Data-2026-10-15/SA_Messung_Picolog_15_10_2026.csv",
                       profile=picolog(datetime(2026, 10, 15, 8, 34, 14))),
        CampaignSource(id="pico_17", path=f"{SA}/SA_Messung_Picolog_17_10_2026.csv",
                       profile=picolog(datetime(2026, 10, 17, 11, 19, 20))),
        CampaignSource(id="db_12", path=f"{SA}/instrument_reader.db", profile=scale_db()),
        ocr,
        CampaignSource(id="db_15", path=f"{SA}/Data-2026-10-15/instrument_reader.db", profile=scale_db()),
        CampaignSource(id="runs_15", path=f"{SA}/Data-2026-10-15/instrument_reader.db", profile=runs_db()),
        CampaignSource(id="db_17", path=f"{SA}/Data-2026-10-17/instrument_reader_auswertung.db", profile=scale_db()),
    ]


# ---------------------------------------------------------------- steps
JOURNAL_ROLES = dict(
    start_column="Zeit_0", parameter_columns=["Temperatur", "V_MFC", "Masse_0", "Masse_1"],
    rows=RowSelection(stop=StopRule.TIME_JUMP),
)


def mappings() -> list[IntervalMapping]:
    masses_15 = ParameterLookup(
        source_id="journal_17", sheet="15.10.2026_1", day=date(2026, 10, 15), end=EndRule(column="Zeit_1"),
        **{**JOURNAL_ROLES, "parameter_columns": ["V_MFC", "Masse_0", "Masse_1"]},
        match=["v_mfc"], take=["Masse_0", "Masse_1"], duration_as="dauer_journal_s", tolerance_s=240,
        when="v_mfc > 0",
    )
    return [
        IntervalMapping(source_id="journal_12", end=EndRule(column="Zeit_1"), **JOURNAL_ROLES,
                        parameter_defaults={"luft": 100, "tag": "2026-10-12", "raumluft": 1.7}),
        IntervalMapping(source_id="runs_15", end=EndRule(column="ended_at"), label_columns=["run_id"],
                        parameter_columns=["temperatur", "v_mfc", "luft"],
                        parameter_defaults={"luft": 100, "v_mfc": 0, "tag": "2026-10-15", "raumluft": 1.9},
                        lookups=[masses_15]),
        IntervalMapping(source_id="journal_17", end=EndRule(column="Zeit_1", fallback=EndRule(kind=EndKind.NEXT_START)),
                        **JOURNAL_ROLES, parameter_defaults={"luft": 100, "tag": "2026-10-17"},
                        cell_values=[CellValue(name="raumluft", label="Baseline mgC")]),
    ]


def manual_steps() -> list[Interval]:
    """12.10.: three zero phases without journal row (as in the day configuration)."""
    params = {"v_mfc": 0.0, "luft": 100.0, "tag": "2026-10-12", "raumluft": 1.7}
    return [
        Interval(id=f"m{i + 1}", label=label, start=datetime(2026, 10, 12, *a), end=datetime(2026, 10, 12, *b),
                 params={**params, "temperatur": t}, note="Null-Phase ohne Journal-Eintrag")
        for i, (label, a, b, t) in enumerate([
            ("Ende Konditionierung", (10, 49, 5), (10, 52, 0), 200.0),
            ("Clearing 250", (13, 40, 0), (13, 57, 0), 250.0),
            ("Pause 150", (14, 40, 0), (15, 12, 0), 150.0),
        ])
    ]


# ---------------------------------------------------------------- evaluation (plan, section 4)
FID = ["fid_12", "fid_15a", "fid_15b", "fid_17"]
PICO = ["pico_12", "pico_15", "pico_17"]
SCALE = ["db_12", "ocr_12", "db_15", "db_17"]
# mass flow (g/s, negative = the reservoir loses mass): journal start/end masses (mean over the step),
# else the regression of the balance; written out once more for the loading (relations come before calculations)
DAUER_M = "where(isfinite(dauer_journal_s), dauer_journal_s, dauer_s)"
MDOT_J = f"-(masse_0 - masse_1) / {DAUER_M}"
MDOT = f"where(isfinite({MDOT_J}), {MDOT_J}, mdot_w)"
# baseline of the carbon balance: the blank before the step, else room air
B_ZYK = "where(isfinite(B_vor), B_vor, raumluft)"
# steps the script counts for light-off and Arrhenius: stationary, not 'unsicher', c_in >= 60, standard air flow
STAT_OK = "stationaer and bewertung != 'unsicher' and c_ok and luft == 100"
# random part of u(ln k) for the Arrhenius fit (in the results table, where u_X_zufall exists)
U_LNK = "u_X_zufall / ((1 - X) * -log(1 - X))"


def evaluation() -> Evaluation:
    return Evaluation(
        constants={
            "f": Constant(value=2.701, u=0.135, kind="syst", note="mgC -> mg Ethanol, RF 0.71 (5 %)"),
            "k_kal": Constant(value=1.0, u=0.02, kind="syst", note="Prüfgas-Zertifikat 2 %"),
            "k_luft": Constant(value=1.0, u=0.072, kind="syst", scope="tag",
                               note="Venturi/DS200 vs. Testo: Rechteck ±12.5 % -> 7.2 %"),
            "k_mdot": Constant(value=1.0, u=0.03, kind="syst", note="Massenstrom OCR gegen Journal ±3 %"),
            "T_ref": Constant(value=24.4, u=3.0, kind="syst", unit="°C", note="Einstell-Temperatur des Luftstroms"),
            "V_kat": Constant(value=0.0135, unit="m3", note="4 Monolithe 150³ mm"),
            "u_masse": Constant(value=0.0029, unit="g", note="Journal-Ablesung: 0.01 g / √12"),
            "u_zeit": Constant(value=5.0, unit="s", note="Journal-Ablesezeit je Zeitpunkt"),
            "drift_h": Constant(value=10.0, unit="min", note="Horizont der Restdrift"),
        },
        step_classes=[StepClass(name="dosierung", when="v_mfc > 0"), StepClass(name="null", when="v_mfc == 0")],
        windows=[
            NamedWindow(name="ende", rule=WindowRule(length_s=120, anchor="end")),
            NamedWindow(name="blank", rule=WindowRule(length_s=180, anchor="end"), classes=["null"]),
            NamedWindow(name="steig", rule=WindowRule(length_s=300, anchor="end")),
            NamedWindow(name="tdrift", rule=WindowRule(length_s=600, anchor="end")),
            NamedWindow(name="masse", rule=WindowRule(skip_start_s=60), classes=["dosierung"]),
            # carbon balance: dosing step and the zero phases after it, up to the next dosing (at most 60 min)
            NamedWindow(name="zyklus", classes=["dosierung"], until_next="dosierung", max_extend_s=3600),
        ],
        measures=[
            Measure(alias="S", sources=FID, channel="fid_mgc", window="ende", stat="mean", min_coverage=0.5),
            Measure(alias="S_b", sources=FID, channel="fid_mgc", window="blank", stat="mean", min_coverage=0.5),
            Measure(alias="dS", sources=FID, channel="fid_mgc", window="steig", stat="slope", min_coverage=0.5),
            Measure(alias="K1", sources=PICO, channel="kanal_1_kat_oben", window="steig", stat="mean"),
            Measure(alias="K2", sources=PICO, channel="kanal_2_kat_unten", window="steig", stat="mean"),
            Measure(alias="dK1", sources=PICO, channel="kanal_1_kat_oben", window="tdrift", stat="slope"),
            Measure(alias="mdot_w", sources=SCALE, channel="masse", window="masse", stat="slope",
                    min_n=30, min_coverage=0.3),
            Measure(alias="I_zyk", sources=FID, channel="fid_mgc", window="zyklus", stat="integral", min_n=30,
                    exclude_flags=["masked"]),
            Measure(alias="t_zyk", sources=FID, channel="fid_mgc", window="zyklus", stat="covered_s", min_n=30,
                    exclude_flags=["masked"]),
        ],
        relations=[
            NearestReferenceRelation(
                name="B", value="S_b", where="klasse == 'null' and abs(dS * 60) <= 0.1",
                match=["temperatur", "luft", "tag"], u_min=0.3, u_one=1.0,
                fallback={"value": "raumluft", "u": 2.0},
            ),
            CumulativeRelation(name="beladung", expr=f"-({MDOT}) * dauer_s",
                               reset_when="(klasse == 'null' and dauer_s >= 300) or changed(temperatur)", gap_s=300.0),
        ],
        calculations=[
            Calculation(name="mdot_j", expr=MDOT_J, unit="g/s", note="Journal: Start- und Endmasse",
                        u_expr=f"sqrt((sqrt(2) * u_masse / {DAUER_M})**2 + ((masse_0 - masse_1) * sqrt(2) * u_zeit / {DAUER_M}**2)**2)"),
            Calculation(name="mdot", expr=f"where(isfinite(mdot_j), mdot_j, mdot_w) * k_mdot", unit="g/s",
                        note="Journal, sonst Regression der Waage"),
            Calculation(name="V_N", expr="luft * k_luft * 273.15 / (T_ref + 273.15)", unit="Nm3/h"),
            Calculation(name="c_in", expr="-mdot * 3600 * 1000 / V_N", unit="mg/Nm3"),
            Calculation(name="dXdt", expr="-dS * 60 * f / c_in", unit="1/min"),
            Calculation(name="drift", expr="0", u_expr="where(isfinite(dXdt), abs(dXdt) * drift_h, 0)",
                        note="Restdrift: so weit kann X nach dem Stufenende noch wandern"),
            Calculation(name="X", expr="1 - (S - B) * f * k_kal / c_in + drift"),
            Calculation(name="T_bett", expr="(K1 + K2) / 2", unit="°C",
                        u_expr="sqrt(1.5**2 + (K1 - K2)**2 / 12)", note="Thermoelement ⊕ Bett-Gradient (Rechteck)"),
            # carbon balance of the cycle: what the FID sees above the blank before the step, against the dosed mass
            Calculation(name="dosiert", expr="-mdot * 1000 * dauer_s", unit="mg", note="dosierte Masse Ethanol"),
            Calculation(name="sichtbar", expr=f"(I_zyk - {B_ZYK} * t_zyk) * f * V_N / 3600", unit="mg",
                        note="∫(FID − Blindwert vorher) · f · V̇_N dt über den Zyklus"),
            Calculation(name="anteil", expr="sichtbar / dosiert", note="FID-sichtbar / dosiert"),
            Calculation(name="k", expr="V_N * (T_bett + 273.15) / 273.15 / V_kat / 3600 * -log(1 - X)", unit="1/s"),
        ],
        criteria=[
            Criterion(name="c_ok", expr="c_in >= 60", text="c_in < 60"),
            Criterion(name="flach", expr="abs(dS * 60) <= 0.1 or abs(dXdt) <= 0.004", text="nicht stationär: FID nicht flach"),
            Criterion(name="t_ok", expr="not (abs(dK1 * 600) > 5)", text="nicht stationär: Temperatur driftet"),
            Criterion(name="geladen", expr="beladung >= 3", text="nicht stationär: Speicherphase (< 3 g dosiert)"),
            Criterion(name="stationaer", expr="flach and t_ok and geladen", text="nicht stationär"),
            Criterion(name="b_ok", expr="B_art != 'keiner'", text="kein Blindwert"),
            Criterion(name="spread_ok",
                      expr="(abs(B_vor - B_nach) if B_art == 'eingeklammert' else 1) * f / c_in <= 0.05",
                      text="Blindwahl ändert X > Grenze"),
        ],
        ratings=[
            Rating(label="gut", all_of=["c_ok", "stationaer", "b_ok", "spread_ok"]),
            Rating(label="brauchbar", all_of=["c_ok"], any_of=["stationaer", "b_ok"]),
            Rating(label="unsicher"),
        ],
        analyses=[
            CrossingAnalysis(name="T50", x="T_bett", y="X", level=0.5, group="v_mfc", u_y="u_X", unit="°C",
                             where=STAT_OK),
            CrossingAnalysis(name="T90", x="T_bett", y="X", level=0.9, group="v_mfc", u_y="u_X", unit="°C",
                             where=STAT_OK),
            FitLineAnalysis(name="Arrhenius", x_expr="1000 / (T_bett + 273.15)", y_expr="log(k)",
                            u_x="1000 * u_T_bett / (T_bett + 273.15)**2", u_y=U_LNK,
                            where=STAT_OK + " and X > 0.05 and X < 0.8",
                            derived={"Ea_kJ_mol": "-m * 8.314"}),
        ],
        views=views(),
    )


def views():
    return [
        TableViewSpec(name="ergebnisse", title="Alle stationären Ergebnisse",
                      columns=["tag", "step_label", "temperatur", "v_mfc", "T_bett", "c_in", "X", "u_X", "B", "B_art",
                               "bewertung", "grund"],
                      filter="klasse == 'dosierung'", sort_by=["tag", "step_start"]),
        MatrixViewSpec(name="umsatz_matrix", title="Umsatz je Temperatur und Dosierstufe", rows="T_klasse",
                       cols="v_mfc", value="X", status="stationaer and bewertung != 'unsicher'", subtext="{c_in:.0f} mg/Nm³",
                       filter="klasse == 'dosierung' and luft == 100 and c_ok",
                       x_label="V_MFC [Nl/min]", y_label="T_bett (25 K)",
                       footer="* nur nicht stationäre Stufen: Wert am Stufenende, zur Orientierung"),
        XYViewSpec(name="light_off", title="Light-off", x="T_bett", y="X", u_y="u_X", color_by="v_mfc",
                   color_mode="ordinal", marker_by="tag", filled="stationaer",
                   filter="klasse == 'dosierung' and luft == 100 and c_ok and bewertung != 'unsicher'",
                   x_label="T_bett [°C]", y_label="X", y_lim=(0.0, 1.05),
                   ref_lines=[RefLine(value=0.5, label="T50", linestyle="--"), RefLine(value=0.9, label="T90", linestyle=":")]),
        XYViewSpec(name="konzentration", title="Umsatz gegen c_ein", x="c_in", y="X", u_y="u_X", facet_by="T_klasse",
                   color_by="v_mfc", color_mode="ordinal", marker_by="tag", filled="stationaer",
                   filter="klasse == 'dosierung' and luft == 100 and c_ok", x_label="c_ein [mg/Nm³]", y_label="X"),
        AlignedViewSpec(name="durchbruch", title="Durchbruch seit Beginn der Beladung",
                        expr="1 - (fid_mgc - B) * f / c_in", t0="sequence_start", rolling_s=10.0,
                        facet_by="T_klasse", color_by="v_mfc", marker_by="tag", filled="stationaer",
                        filter="klasse == 'dosierung' and c_ok"),
        XYViewSpec(name="arrhenius", title="Arrhenius (scheinbare Ea)", x_expr="1000 / (T_bett + 273.15)",
                   y_expr="log(k)", u_y=U_LNK, marker_by="tag", color_by="v_mfc", color_mode="ordinal",
                   filter=STAT_OK + " and X > 0.05 and X < 0.8",
                   x_label="1000 / T [1/K]", y_label="ln k",
                   secondary_x=SecondaryAxis(label="T [°C]", forward_expr="1000 / x - 273.15",
                                             inverse_expr="1000 / (x + 273.15)")),
        XYViewSpec(name="blindwerte", title="FID am Ende der Null-Phasen", x="temperatur", y="S_b", marker_by="tag",
                   filled="abs(dS * 60) <= 0.1", filter="klasse == 'null'", x_label="T_soll [°C]", y_label="FID [mgC]"),
        XYViewSpec(name="reproduzierbarkeit", title="Gleiche Bedingungen an verschiedenen Tagen", x="T_bett", y="X",
                   u_y="u_X", facet_by="v_mfc", marker_by="tag", color_by="tag", color_mode="categorical",
                   filled="stationaer", filter="klasse == 'dosierung' and luft == 100 and c_ok",
                   x_label="T_bett [°C]", y_label="X"),
        TimelineViewSpec(
            name="verlauf", title="Messtag", page_by="tag", page_title="Messtag {tag}",
            subtitle="FID, Bett-Temperaturen und Umsatz je Dosierstufe · Flächen: Dosierung (Farbe = Dosierstufe, Zahl = Nl/min)",
            bands="klasse == 'dosierung'", band_color_by="v_mfc", band_label="{v_mfc:g}", band_legend="Nl/min",
            panels=[
                TimelinePanel(traces=[TimelineTrace(channel="fid_mgc", label="FID (Rohsignal)")], y_label="FID [mgC]",
                              y_robust=99.7, height=2.2,
                              marks=[TimelineMarks(y="S_b", at="window", window="blank", filter="klasse == 'null'",
                                                   filled="abs(dS * 60) <= 0.1", label="Blindwert (FID flach)",
                                                   label_open="Null-Phase mit Desorption")]),
                TimelinePanel(traces=[TimelineTrace(channel="kanal_1_kat_oben", label="Kat oben", color="#eb6834"),
                                      TimelineTrace(channel="kanal_2_kat_unten", label="Kat unten", color="#1baf7a")],
                              y_label="T [°C]", height=1.2),
                TimelinePanel(marks=[TimelineMarks(y="X", u_y="u_X", filter="klasse == 'dosierung' and c_ok",
                                                   filled=STAT_OK, label="X stationär", label_open="X nicht stationär")],
                              y_label="Umsatz X", y_lim=(-0.05, 1.1), height=1.3, ref_lines=[RefLine(value=1.0)]),
            ],
        ),
        XYViewSpec(name="bilanz", title="Kohlenstoff-Bilanz",
                   subtitle="Anteil des dosierten Ethanols, der als organischer Kohlenstoff am FID erscheint "
                            "(Dosierstufe + folgende Null-Phasen)",
                   x="T_bett", y="anteil", color_by="v_mfc", color_mode="ordinal", marker_by="tag",
                   filter="klasse == 'dosierung' and c_ok and luft == 100",
                   x_label="Bett-Temperatur [°C]", y_label="FID-sichtbar / dosiert",
                   ref_lines=[RefLine(value=0.0), RefLine(value=1.0, label="alles sichtbar")]),
        BarsViewSpec(name="unsicherheit", title="Unsicherheitsbudget von X", quantity="X",
                     filter="stationaer and klasse == 'dosierung'"),
        ReportViewSpec(name="bericht", title="Messkampagne Ethanol", subtitle="SA, 12./15./17.10.2026",
                       views=["umsatz_matrix", "light_off", "konzentration", "durchbruch", "arrhenius", "verlauf",
                              "blindwerte", "bilanz", "reproduzierbarkeit", "unsicherheit", "ergebnisse"]),
    ]


def with_classes(ev: Evaluation) -> Evaluation:
    """Temperature class: bed temperature rounded to 25 K (relations come before calculations, so from K1, K2)."""
    rel = [*ev.relations, BinRelation(name="T_klasse", expr="(K1 + K2) / 2", width=25)]
    return ev.model_copy(update={"relations": rel})


def campaign() -> Campaign:
    return Campaign(
        name="SA Ethanol-Oxidation", timezone=ZONE, sources=sources(), interval_mappings=mappings(),
        intervals=manual_steps(), stats=StatsConfig(reading_u={f"{s}/fid_mgc": 0.10 for s in FID}),
        evaluation=with_classes(evaluation()),
        notes=("Beispielkampagne zu docs/PLAN_AUSWERTUNG_KAMPAGNE.md (PR 10). Daten liegen ausserhalb des Repos: "
               "DATUALIZER_SA_DIR auf den SA-Ordner setzen. Referenz: SA/auswertung/sa_auswertung.py."),
    )


def compact(ev: Evaluation) -> dict:
    """The evaluation without default values, but with the `kind` that tells relations, analyses and views apart."""
    out = ev.model_dump(mode="json", exclude_defaults=True)
    for key in ("relations", "analyses", "views"):
        for item, model in zip(out.get(key, []), getattr(ev, key)):
            item["kind"] = model.kind
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    c = campaign()
    data = c.model_dump(mode="json", exclude_defaults=True)
    data["evaluation"] = compact(c.evaluation)
    Campaign.model_validate(data)  # what is written reads back
    (OUT / "campaign_SA_ethanol.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    recipe = {
        "name": "Katalyse VOC (Umsatz am Monolith, FID vor/nach)",
        "description": "Auswertung ohne Quellen: Messgrössen lesen Quellen-IDs (fid_*, pico_*, db_*), die eine neue "
                       "Kampagne gleich benennt oder in 'measures' anpasst. Konstanten je Anlage ändern.",
        "stats": c.stats.model_dump(mode="json", exclude_defaults=True),
        "evaluation": data["evaluation"],
    }
    (OUT / "rezept_katalyse_voc.json").write_text(json.dumps(recipe, indent=2, ensure_ascii=False), encoding="utf-8")
    print("written:", *(p.name for p in OUT.glob("*.json")))


IMAGES = {"umsatz_matrix": 105, "light_off": 105, "arrhenius": 105, "verlauf": 150}  # name: height in mm


def images(report: Path | None = None) -> None:
    """docs/img/sa_<view>.png from the real data (and the report PDF if a path is given); settings isolated."""
    import os
    import tempfile

    from PySide6.QtCore import QSettings

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, tempfile.mkdtemp())
    from PySide6.QtWidgets import QApplication

    from datualizer_gui.campaign_window import CampaignWindow
    from datualizer_gui.components.plot_export import ExportOptions

    if not os.environ.get("DATUALIZER_SA_DIR"):
        raise SystemExit("Set DATUALIZER_SA_DIR to the SA folder first.")
    app = QApplication.instance() or QApplication([])
    win = CampaignWindow()
    win.resize(1600, 1000)
    win.open_campaign(OUT / "campaign_SA_ethanol.json")
    win.open_results_window()
    app.processEvents()
    for name, height in IMAGES.items():
        target = ROOT / "docs" / "img" / f"sa_{name}.png"
        widget = win._results_window._view_widgets[name]
        widget.export(target, ExportOptions(width_mm=170 if height < 150 else 250, height_mm=height, dpi=150))
        print("wrote", target.relative_to(ROOT))
    if report is not None:
        print("wrote", win.export_report(report))
    win.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bilder", action="store_true", help="render docs/img/sa_*.png from the real data")
    ap.add_argument("--bericht", type=Path, help="also export the report PDF to this path")
    args = ap.parse_args()
    main()
    if args.bilder or args.bericht:
        images(args.bericht)
