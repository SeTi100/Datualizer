# Plan: Auswertung einer Messkampagne im Datualizer (Etappe 2b, Schritte 7, 8, 10)

> Stand 2026-10-19. Grundlage für die Umsetzung durch spätere Agents; die Übergabeprompts je PR stehen in [`UEBERGABE_AUSWERTUNG.md`](UEBERGABE_AUSWERTUNG.md).
> Ziel: Der Datualizer erzeugt aus Rohdaten, einer `campaign.json` und Einstellungen in der GUI **alle Kennwerte und Grafiken**, die heute das Skript `SA/auswertung/sa_auswertung.py` plus `katalyse_plots.py` liefert (Ethanol-Oxidation am Monolith, Messtage 12., 15. und 17.10.2026). Nichts davon ist SA-spezifisch im Code: Die SA-Auswertung ist eine Konfiguration (Prinzip 1).

Referenz-Implementierung (nur lesen, nicht kopieren):
- `C:\Users\timkl\OneDrive - OST\Dokumente\SA\auswertung\sa_auswertung.py`: Laden, Fenster, Formeln, Blindwert, Beladung, Kriterien, Fehlerfortpflanzung (`evaluate`, `loading`, `mass_flow`, `t_crossing`, `fit_arrhenius`).
- `…\SA\auswertung\katalyse_plots.py`: alle Grafiken der Kampagne.
- `…\SA\auswertung\AUSWERTUNG_LIESMICH.md`, `FEHLERFORTPFLANZUNG.md`: Kriterien und Unsicherheiten in Worten.
- `…\SA\auswertung\tage\2026-10-1{2,5,7}.json`: die Tageskonfigurationen.
- `…\SA\auswertung\ergebnisse_auto\<tag>\schritte.csv`: das Orakel für PR 10.

## 1. Was die Skript-Auswertung tut, und was der Datualizer davon kann

✅ vorhanden · ◐ teilweise · ✗ fehlt

### 1.1 Einlesen und Zeit
| Operation | Datualizer |
|---|---|
| FID (`.dat`/`.csv`), PicoLog, OCR-Datenbank, Journal lesen | ✅ Reader + Profile |
| Uhrversatz FID (+970/975 s), OCR-Video (−15 s) | ✅ `ClockModel.offset_s` |
| PicoLog-Start = Speicherzeit − Dauer | ✅ `OriginRule.FILE_MTIME_END` |
| Versatz aus Journal-Paaren („Start FID“/„Start Real“) | ✅ `ClockCells` → `session.clock_from_cells` (PR 9) |
| Journal-Zeilen als Schritte, fehlendes Ende = nächster Start, Sollwerte als Parameter | ✅ `IntervalMapping`, `EndKind.NEXT_START` |
| DB-Runs als Schritte, `V_MFC`/`Volumenstrom` aus `parameters_json` | ✅ `SqliteOptions.json_columns` (PR 2), Spalten `v_mfc`, `volumenstrom` |
| OCR-Fehllesungen der Waage | ✅ `spike` (Hampel) |
| FID-Skala aus dem Rohsignal: mgC = a·pA + b, a/b aus der Kalibrierung (Settings-XML: `sens`, `zero`; 1.616 mgC/ppm Propan), am 15.10. vor/nach 09:24 verschieden | ✗ Rechenkanal mit Zeitstücken, XML-Reader → PR 1, 2 |
| Prüfgas-Fenster ausblenden | ✗ Maske → PR 2 |

### 1.2 Statistik pro Fenster
| Operation | Datualizer |
|---|---|
| FID-Endwert: Mittel der letzten 120 s; u = sd/√(n/10) ⊕ Nullpunkt 0.1 mgC | ◐ `mean`, `sem_eff`; Nullpunkt über `reading_u` |
| Blindwert: Mittel der letzten 180 s einer Null-Phase | ✅ benanntes Fenster nur für Null-Phasen (PR 3); Auswahl der Null-Phase → PR 4 |
| Steigung der letzten 300 s (Stationarität) | ✅ OLS, HAC, Theil-Sen |
| Massenstrom: OLS-Steigung der Masse, erste 60 s weg; Ersatz Journal 2 Punkte (GUM) | ✅ besser als im Skript (HAC/AR(1), `two_point`) |
| Bett-T = (K1 + K2)/2 der letzten 300 s; K1-Drift der letzten 10 min | ◐ Kanäle kombinieren → PR 3/5 |
| Verschiedene Fenster je Grösse | ✅ `NamedWindow`, `Measure` (PR 3) |

### 1.3 Grössen pro Schritt
- V̇_N = V̇ · 273.15/(T_ref + 273.15); c_in = ṁ/V̇_N (mg/Nm³)
- X = 1 − (S − B)·f/c_in mit f = 2.701 (mgC → mg Ethanol); X_vor, X_nach, X_ohne_Blindwert
- dX/dt = −Steigung_FID·f/c_in
- T_bett = (K1 + K2)/2, u(T) = √(1.5² + (K1 − K2)²/12)
- GHSV = V̇_N/V_Kat; k = V̇_N·(T + 273.15)/273.15/V_Kat · (−ln(1 − X)); u(ln k) = u(X)/((1 − X)(−ln(1 − X)))
- u(X): Beiträge einzeln (FID, Blindwert, Restdrift = |dX/dt|·10 min, ṁ zufällig, ṁ syst. 3 %, Luft syst. 7.2 % ⊕ T_ref, RF 5 %, Prüfgas 2 %), Summen zufällig/systematisch getrennt; lineare GUM-Fortpflanzung.

### 1.4 Bezüge zwischen Schritten
- Schrittart: `V_MFC > 0` → Dosierung, sonst Null-Phase.
- **Blindwert**: nächste Null-Phase gleicher `T_soll` und gleichen Luftstroms, deren FID am Ende flach ist, vor und nach dem Schritt. Beide → Mittel, u = max(|Δ|/2/√3, 0.3 mgC); eine → u = 1 mgC; keine → Raumluft, u = 2 mgC.
- **Beladung**: Σ ṁ·Dauer seit der letzten Null-Phase ≥ 5 min, dem letzten Wechsel von `T_soll` oder einer Lücke ≥ 5 min.
- **Kriterien**: c_in ≥ 60; stationär = (|Steigung| ≤ 0.1 mgC/min oder |dX/dt| ≤ 0.004/min) und |K1-Drift (10 min)| ≤ 5 K und Beladung ≥ 3 g; Blindwert vorhanden; Blindwahl ändert X ≤ 0.05. Bewertung gut/brauchbar/unsicher, Spalte `grund` mit jeder verletzten Bedingung.
- Bilanz: ∫(FID − Basis)·f·V̇_N dt über Dosierstufe + folgende Null-Phasen / dosierte Masse.
- Sprungantwort: Totzeit (10 Werte in Folge über Basis + max(5σ, 0.3)), t63.

### 1.5 Auswertungen über alle Schritte und Tage
- T50/T90: lineare Interpolation in X(T_bett) je Dosierstufe, Bereich aus X ± u(X).
- Arrhenius: ln k gegen 1000/T, gewichteter Geradenfit mit Unsicherheit auf beiden Achsen (effektive Varianz, 5 Iterationen), χ²_red, Birge-Faktor; Ea = −m·R.
- Klassen: T_bett auf 25 K gerundet; Matrix = Mittel der stationären Stufen je Klasse × `V_MFC`; Stufen mit anderem Luftstrom ausgeschlossen.
- X(t) seit Schritt- bzw. Beladungsbeginn (Schrittkonstanten c_in, B auf der Zeitachse, 10-s-Mittel).
- Unsicherheitsbudget als Varianzanteile.

### 1.6 Grafiken
Tagesverlauf (geht im Kampagnenfenster schon fast), Umsatz-Matrix, Light-off, Konzentration (Kleinvielfache je T-Klasse), Durchbruch, Arrhenius mit °C-Zweitachse, Blindwerte (flach/Desorption), Bilanz, Reproduzierbarkeit, Budget, Ergebnistabelle, mehrseitiges PDF. Lesart: Blau-Abstufung = Dosierstufe, Form = Messtag, gefüllt = stationär.

## 2. Zielarchitektur

```
Quellen ─ Kanäle ─ Rechenkanäle, Masken                 (PR 1, 2)
   │
Schritte ─ Schrittarten                                  (PR 3)
   │
benannte Fenster × Kanal → window_stats → Schritt-Tabelle (PR 3)
   │
Bezüge: Blindwert, Beladung, Klassen                     (PR 4)
   │
Rechengrössen mit Unsicherheit, Kriterien, Szenarien     (PR 5)
   │
Auswertungen: Durchgang, Geradenfit, Aggregation         (PR 6)
   │
Ansichten (ViewSpec) ─ Ergebnisfenster ─ Bericht         (PR 7, 8)
```

Alles, was der Nutzer festlegt, steht in `campaign.json` (Abschnitt `evaluation`). Neues Modul `datualizer_core/calc/` (keine GUI-Abhängigkeit), GUI in `datualizer_gui/results_window.py` und `components/`.

## 3. Bausteine

### 3.1 Formelkern (`calc/expr.py`, `calc/uncertain.py`) – PR 1
- Parser auf Python-`ast` mit Positivliste: Zahlen, Namen (auch `a.b` als ein Name), `+ − * / **`, unäres Minus, Vergleiche, `and/or/not`, `x if c else y`, Funktionen `log, log10, exp, sqrt, abs, min, max, where, isfinite, round`. **Kein `eval`.** Fehler mit Position und Liste bekannter Namen.
- Ein Ausdruck, zwei Ziele: `Expr.to_polars()` (Rechenkanäle auf Rohdaten) und `Expr.evaluate(env)` (numpy, Schritt-Tabelle).
- Unsicherheit: `UArray(value, grads: dict[input, ndarray])` als Dual-Zahl über alle unsicheren Eingänge. Ein Eingang hat `u`, `kind` (`zufall`/`syst`) und `group`. Gleiche `group` heisst gemeinsame Fehlerquelle (ein Prüfgas, eine Waagenkalibrierung), voll korreliert innerhalb einer Zeile; verschiedene Grössen desselben Tages bekommen **keine** gemeinsame Gruppe. Ergebnis: Wert, u gesamt, u zufällig, u syst., Beitrag je Eingang. (PR 1 ✅)
- Korrelation **zwischen Zeilen** (dieselbe Konstante in allen Schritten eines Tages) bildet `UArray` nicht ab; sie gehört zu den Auswertungen in PR 6 (`scope` der Konstante, Kovarianz im Fit).
- Text in Formeln (PR 3 ✅): Text-Literale und Textspalten nur mit `==`/`!=` (`klasse == 'null'`); fehlender Text ist weder gleich noch ungleich (beides falsch).
- `Constant(name, value, u=0, kind="syst", group=None, unit="", note="")`.
- Monte Carlo (GUM S1) optional, nur für Auswertungen in PR 6.

### 3.2 Rechenkanäle, Masken, Gerätemetadaten – PR 2
- `CampaignSource.derived: list[DerivedChannel]` mit `name`, `expr`, `unit`, `pieces: list[Piece(start, end, expr)]` (stückweise nach Zeit, z. B. andere Kalibrierung vor 09:24). Rechenkanäle sind normale Kanäle für Plot, Statistik und Qualitäts-Flags; Rohkanäle bleiben unverändert.
- `CampaignSource.masks: list[Mask(start, end, channels, reason)]` → Flag `masked` (in `StatsConfig.exclude_flags` standardmässig enthalten), im Plot schraffiert.
- Reader für Testa-iFID-`Settings*.xml`: Kalibrierblock (`time_last_cal`, `std_gas`, `sig_span`, `sens`, `zero`, `rsp-fac`) als `source_meta`; Vorschlag für die Kalibrierung `a = k/sens`, `b = −zero·a` mit `k` (mgC/ppm) als Eingabe. Vorschlag, nicht Automatik (Prinzip 2). Datei enthält Passwort-Hash, PIN und Systemkennung: Reader liest nur den Kalibrierblock, Fixture nur gekürzt.
- `SqliteOptions.json_columns: list[str]`: JSON-Spalten in Spalten zerlegen (`parameters_json` → `V_MFC`, `Volumenstrom`, `f_Fan`; nach der Namensbereinigung `v_mfc`, `volumenstrom`, `f_fan`).

### 3.3 Schrittarten, benannte Fenster, Schritt-Tabelle, praktische Stationarität – PR 3 ✅

Umgesetzt in `datualizer_core/campaign/evaluation.py` und `CampaignSession.step_table`. Spaltennamen der Schritt-Tabelle: `step_id`, `step_label`, `step_start`, `step_end`, `dauer_s`, Parameter, `klasse`, `klasse_hinweis`, je Messgrösse `<alias>`, `u_<alias>`, `checks_<alias>`. Klassenformeln kennen zusätzlich `step_id`, `step_label` und `dauer_s`. Das 90-%-Intervall des Äquivalenztests nimmt den grösseren der beiden autokorrelationsfesten SE (`slope_se_eff`), Normalquantile wie die Ampel.

```python
class StepClass(BaseModel):  name: str; when: str            # "dosierung": "v_mfc > 0"
class NamedWindow(BaseModel): name: str; rule: WindowRule; classes: list[str] = []
class Measure(BaseModel):    alias: str; source: str; channel: str; window: str; stat: str
    # stat: Feld von WindowResult (mean, slope, two_point, integral, …); Unsicherheit automatisch
    # (mean → sem_eff ⊕ reading_u, slope → slope_se_eff, two_point → two_point_u)
session.step_table(spec) -> pl.DataFrame   # eine Zeile pro Schritt: Parameter, Klasse, <alias>, u_<alias>, Prüfungen
```
- Bestehende `WindowRule` bleibt als Fenster `default`; `WindowOverride` bekommt optional den Fensternamen.
- **Praktische Stationarität (Äquivalenztest, TOST)**: `StatsConfig.equivalence: dict["quelle/kanal", float]` (δ in Einheit/s). Stationär, wenn das 90-%-Intervall der Steigung (HAC-SE) ganz in ±δ liegt. Neues Feld `stationary_practical` neben der statistischen Ampel `stationary`. Grund: Bei 1-Hz-Daten ist fast jede Steigung signifikant; fachlich zählt, ob sie klein ist.

### 3.4 Bezüge zwischen Schritten (`calc/relations.py`) – PR 4
```python
nearest_reference(table, value, where, match, side="both", combine="mean_rect",
                  u_min=0.3, u_one=1.0, fallback=None) -> (Wert, u, Quelle-Schritt(e), Art)
cumulative(table, expr, reset_when, gap_s=300) -> Series   # Beladung; Funktion changed(col) im Ausdruck
bin(expr, width) -> Series                                 # T-Klasse
```
Jede Zeile nennt den benutzten Bezugsschritt (Prinzip 3).

### 3.5 Rechengrössen, Kriterien, Szenarien – PR 5
```python
class Calculation(BaseModel): name: str; expr: str; unit: str = ""; note: str = ""
class Criterion(BaseModel):   name: str; expr: str; text: str     # Text, wenn verletzt
class Rating(BaseModel):      label: str; all_of: list[str]; any_of: list[str] = []
class Scenario(BaseModel):    name: str; constants: dict[str, Constant]
class Evaluation(BaseModel):  constants, step_classes, windows, measures, relations,
                              calculations, criteria, ratings, scenarios, analyses, views
Campaign.evaluation: Evaluation | None
session.results(scenario=None) -> pl.DataFrame              # Schritt-Tabelle + Rechengrössen + u + Budget + Bewertung
```
Dock „Results“: Konstanten, Formeln (mit Prüfung beim Tippen), Kriterien, Szenario-Umschalter, Tabelle mit Filter und CSV-Export, Klick auf Zeile → Fenster-Inspektor.

### 3.6 Auswertungen (`calc/analyses.py`) – PR 6
- `crossing(table, x, y, level, group, u_y, where)` → Wert und Bereich je Gruppe („< 185 °C“, wenn kein Durchgang).
- `fit_line(table, x_expr, y_expr, u_x, u_y, where, method="effective_variance")` → m, b, u_m, χ²_red, Birge-Faktor; abgeleitete Grössen als Formel (Ea = −m·8.314).
- `aggregate(table, rows, cols, value, where, how="mean")`.
- Ergebnisse gehören in `campaign.json` (`analyses`), Ausgabe als Tabelle.

### 3.7 Ansichten – PR 7, 8
`ViewSpec` (in `campaign.json`), gerendert mit pyqtgraph, Export über den bestehenden Plot-Export (PNG/SVG/PDF):
| Art | Felder | ersetzt |
|---|---|---|
| `table` | Spalten, Filter, Sortierung | Ergebnistabelle |
| `xy` | x, y, u_x, u_y, color_by (Ordinal-Rampe), marker_by, filled (Ausdruck), connect_by, facet_by, ref_lines, x_expr/y_expr, zweite Achse | Light-off, Konzentration, Blindwerte, Bilanz, Reproduzierbarkeit, Arrhenius |
| `matrix` | rows, cols, value, status-Ausdruck, Zelltext | Umsatz-Matrix |
| `aligned` | Ausdruck über Kanal + Schrittkonstanten, t0 (`step_start`/`sequence_start`), facet_by, color_by | Durchbruch |
| `bars` | gestapelte Spalten (Budget) | Unsicherheit |
| `report` | Liste von Ansichten → mehrseitiges PDF | Kampagnenbericht |

`aligned` braucht `broadcast_steps(table, columns)`: Schrittwerte als stückweise konstante Spur auf Master-Zeit (join_asof), damit X(t) = 1 − (FID(t) − B)·f/c_in ein Rechenkanal wird.

Farben nach `docs`-Konvention des Kampagnenfensters; Ordinal-Rampe (Dosierstufe) blau hell → dunkel, Messtag als Form, stationär gefüllt.

### 3.8 Abgleich-Werkzeuge – PR 9 ✅
- Uhr-Anker aus Zellpaaren einer Quelle (Journal: „Start FID“/„Start Real“): `ClockAnchor` aus konfigurierbaren Zellen (Blatt, Zeile, Spalte), Median und Streuung als `offset_s` und `uncertainty_s`.
- Kreuzkorrelation zweier Spuren (Signal oder Ableitung) auf gemeinsamem Raster → Versatz, Kurve, Güte. Dialog mit Überlagerung, Schieber, „Übernehmen“.

Umgesetzt (`datualizer_core/campaign/sync.py`, Dialog `datualizer_gui/components/clock_sync_dialog.py`):
- `ClockCells` (Quelle mit der Arbeitsmappe, Blatt, Datum, Paare `CellPair(source_cell, master_cell)`, Ablese-Schritt) steht im `ClockModel` (`cells`) und damit in `campaign.json`. Ergebnis: reiner Versatz `offset_s` = Median, `uncertainty_s` = max(Median der absoluten Abweichungen, halber Ablese-Schritt), `method="journal_pairs"`. Die Anker selbst werden nur angezeigt, nicht als `anchors` gespeichert: Ein falsch abgeschriebenes Paar soll die Abbildung nicht verbiegen. Journal 17.10.2026: 975 s und 974 s → 974,5 s ± 0,5 s. Leere oder unlesbare Zellen und Uhrzeiten ohne Datum schlagen laut fehl (`ClockCellsError`).
- `cross_correlate`: beide Spuren linear auf ein Raster (Standard: die gröbere mittlere Abtastung im Ausschnitt), Lücken und fehlende Werte werden nie überbrückt. Pearson-r für jeden Versatz im Suchbereich, nur über gemeinsame Punkte (maskierte Summen per FFT); Versätze mit weniger als 50 % Überlappung bleiben leer. Bester Versatz mit Parabel zwischen den Rasterpunkten. Prüfungen mit Ampel: Spitze am Rand des Suchbereichs (fail), r < 0,3 (fail), r < 0,7 (warn), zweite Spitze innerhalb 0,05 von r (warn, periodisches Signal), wenig Überlappung (warn). Die Unsicherheit ist der halbe Rasterschritt; eine falsche Spitze deckt sie nicht ab.
- Grenze: Zufallsartige, driftende Signale korrelieren auch bei falschem Versatz deutlich (r ≈ 0,7), wenn der wahre Versatz ausserhalb des Suchbereichs liegt. Dann Ableitung vergleichen und den Ausschnitt um ein klares Ereignis (Sprung, Spitze) legen.

## 4. Die SA-Auswertung als Konfiguration (Auszug, Zielbild)

```json
"evaluation": {
  "constants": {
    "f": {"value": 2.701, "note": "mgC -> mg Ethanol (RF 0.71)", "u": 0.135, "kind": "syst"},
    "V_kat": {"value": 0.0135, "unit": "m3"},
    "luft_m3_h": {"value": 100, "u": 7.2, "kind": "syst", "scope": "tag"},
    "T_ref": {"value": 24.4, "u": 3, "kind": "syst"}
  },
  "step_classes": [{"name": "dosierung", "when": "v_mfc > 0"}, {"name": "null", "when": "v_mfc == 0"}],
  "windows": [
    {"name": "ende",  "rule": {"length_s": 120, "anchor": "end"}},
    {"name": "blank", "rule": {"length_s": 180, "anchor": "end"}, "classes": ["null"]},
    {"name": "steig", "rule": {"length_s": 300, "anchor": "end"}},
    {"name": "tdrift","rule": {"length_s": 600, "anchor": "end"}},
    {"name": "masse", "rule": {"skip_start_s": 60}}
  ],
  "measures": [
    {"alias": "S",    "source": "fid",  "channel": "fid_mgC", "window": "ende",  "stat": "mean"},
    {"alias": "S_b",  "source": "fid",  "channel": "fid_mgC", "window": "blank", "stat": "mean"},
    {"alias": "dS",   "source": "fid",  "channel": "fid_mgC", "window": "steig", "stat": "slope"},
    {"alias": "mdot", "source": "db",   "channel": "Masse",   "window": "masse", "stat": "slope"},
    {"alias": "K1",   "source": "pico", "channel": "K1",      "window": "steig", "stat": "mean"},
    {"alias": "K2",   "source": "pico", "channel": "K2",      "window": "steig", "stat": "mean"},
    {"alias": "dK1",  "source": "pico", "channel": "K1",      "window": "tdrift","stat": "two_point"}
  ],
  "relations": [
    {"name": "B", "kind": "nearest_reference", "value": "S_b",
     "where": "klasse == 'null' and abs(dS*60) <= 0.1", "match": ["temperatur", "luft"],
     "u_min": 0.3, "u_one": 1.0, "fallback": {"value": 2.09, "u": 2.0}},
    {"name": "beladung", "kind": "cumulative", "expr": "-mdot * dauer_s",
     "reset_when": "(klasse == 'null' and dauer_s >= 300) or changed(temperatur)"},
    {"name": "T_klasse", "kind": "bin", "expr": "T_bett", "width": 25}
  ],
  "calculations": [
    {"name": "V_N",    "expr": "luft_m3_h * 273.15 / (T_ref + 273.15)", "unit": "Nm3/h"},
    {"name": "c_in",   "expr": "-mdot*3600*1000 / V_N", "unit": "mg/Nm3"},
    {"name": "X",      "expr": "1 - (S - B) * f / c_in"},
    {"name": "dXdt",   "expr": "-dS*60 * f / c_in", "unit": "1/min"},
    {"name": "T_bett", "expr": "(K1 + K2) / 2", "unit": "°C"},
    {"name": "k",      "expr": "V_N*(T_bett+273.15)/273.15 / V_kat / 3600 * -log(1 - X)", "unit": "1/s"}
  ],
  "criteria": [
    {"name": "c_ok",   "expr": "c_in >= 60", "text": "c_in < 60"},
    {"name": "flach",  "expr": "abs(dS*60) <= 0.1 or abs(dXdt) <= 0.004", "text": "nicht stationär: FID nicht flach"},
    {"name": "t_ok",   "expr": "abs(dK1) <= 5", "text": "nicht stationär: Temperatur driftet"},
    {"name": "geladen","expr": "beladung >= 3", "text": "nicht stationär: Speicherphase (< 3 g dosiert)"}
  ],
  "ratings": [{"label": "gut", "all_of": ["c_ok", "flach", "t_ok", "geladen", "b_ok", "spread_ok"]}, "…"],
  "analyses": [{"name": "T50", "kind": "crossing", "x": "T_bett", "y": "X", "level": 0.5, "group": "v_mfc",
                "where": "stationaer and bewertung != 'unsicher'"}],
  "views": ["…"]
}
```
Die Namen der Kanäle/Quellen sind Beispiele. Geladene Spalten stehen in bereinigter Schreibweise (`v_mfc`, `temperatur`, siehe `clean_names`), selbst vergebene Namen (`T_bett`, `X`) wie angegeben. Die Restdrift (|dX/dt|·10 min) ist ein zusätzlicher zufälliger Eingang in X.

## 5. Abwägungen

| Frage | Optionen | Empfehlung |
|---|---|---|
| Formelmaschine | eigener AST · sympy · numexpr/asteval | eigener AST: klein, sicher, Polars + numpy, Ableitungen über Dual-Zahlen. sympy wäre schwer; numexpr/asteval liefern keine Ableitungen |
| Fehlerfortpflanzung | linear (Dual-Zahlen) · Monte Carlo | linear für Schrittgrössen; Monte Carlo optional für T50 und Fits |
| Korrelation | innerhalb einer Zeile · zwischen Zeilen | `group` (PR 1) nur für Eingänge mit gemeinsamer Fehlerquelle; zwischen Zeilen `scope` je Konstante (z. B. `tag`), ausgewertet in PR 6: dieselbe systematische Konstante wirkt auf alle Schritte eines Tages gleich, dadurch hängt Ea korrekt nicht vom Luftstrom ab |
| Stationarität | statistische Ampel · Äquivalenztest | beide anzeigen; Kriterien nutzen den Äquivalenztest |
| Mehrere Tage | eine Kampagne mit allen Quellen · Sammlung mehrerer `campaign.json` | zuerst eine Kampagne (Master-Zeit kann mehrere Tage); Sammlung später |
| Bericht | pyqtgraph-Export · matplotlib | pyqtgraph (keine neue Abhängigkeit, Vektor); matplotlib höchstens als Extra `[report]` |
| Wiederverwendung | SA im Code · Rezept | `examples/rezept_katalyse_voc.json`: Fenster, Formeln, Bezüge, Kriterien, Ansichten; neue Kampagne übernimmt es und ändert nur Konstanten. Keim der Recipe-AST (Etappe 3) |

## 6. PR-Folge

| PR | Inhalt | braucht | Masterplan |
|---|---|---|---|
| 1 ✅ | Formelkern + Dual-Zahlen + Konstanten | – | 2b-8 |
| 2 | Rechenkanäle (Zeitstücke), Masken, Settings-XML, `json_columns` | 1 | 2b-8, 2b-2 |
| 3 ✅ | Text-Literale in Formeln, Schrittarten, benannte Fenster, `step_table`, Äquivalenztest | 1 | 2b-6/8 |
| 4 ✅ | Bezüge: Blindwert, Beladung, Klassen | 1, 3 | 2b-8 |
| 5 ✅ | Rechengrössen, Kriterien, Szenarien, `Evaluation`, Dock „Results“ | 1, 3, 4 | 2b-8 |
| 6 ✅ | Auswertungen: Durchgang, Geradenfit, Aggregation, Korrelation zwischen Zeilen (`scope`) | 5 | 2b-8 |
| 7 ✅ | Ergebnisfenster: Ansichten `table`, `xy`, `matrix`, Kleinvielfache | 5, 6 | 2b-8, Etappe 5 |
| 8 | `aligned` + `broadcast_steps`, `bars`, Bericht als PDF | 2, 7 | 2b-8 |
| 9 ✅ | Uhr-Anker aus Zellpaaren, Kreuzkorrelation, Dialog | – | 2b-7 |
| 10 | SA-Kampagne (3 Tage) als Beispiel + Rezept, End-to-End-Orakel, Doku | 1–9 | 2b-10 |

Parallel möglich: PR 2, 3 und 9 (PR 1 ist in main).

## 7. Abnahme (PR 10)

Mit `examples/campaign_SA_ethanol.json` auf die echten Dateien (lokal, Test überspringt ohne Daten) muss `session.results()` mit `SA/auswertung/ergebnisse_auto/<tag>/schritte.csv` übereinstimmen:
- X ± 0.005, c_in ± 0.5 %, u(X) ± 10 %, T_bett ± 0.2 K, Bewertung und Stationarität gleich (Abweichungen begründet in der PR-Beschreibung).
- Stichproben 17.10.: Z16 (200 °C, 10 Nl/min) c_in 144.4, X 0.560, u 0.047, gut; Z23 (150 °C) X 0.392; Z29 (220 °C) X 0.642, Blindwert „keiner“.
- Light-off: T50 179 °C (10 Nl/min), 203 °C (20 Nl/min); Arrhenius Ea 21 ± 4 kJ/mol aus 8 Stufen.
- Alle Ansichten aus Abschnitt 3.7 lassen sich aus der Kampagne erzeugen; der Bericht hat dieselben Seiten wie `katalyse_bericht.pdf`.

Unterschiede zwischen Skript und Datualizer, die erwartet sind und dokumentiert werden: Massenstrom-SE (HAC statt ×√10), FID-u (`sem_eff` statt n/10).
