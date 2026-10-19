# Übergabeprompts: Auswertung einer Messkampagne (PR 1–10)

> Zu [`PLAN_AUSWERTUNG_KAMPAGNE.md`](PLAN_AUSWERTUNG_KAMPAGNE.md). Ein Prompt pro PR, jeweils vollständig in eine neue Agent-Sitzung kopieren. Reihenfolge und Abhängigkeiten: Plan, Abschnitt 6. PR 1 ist in main (#36); PR 2, 3 und 9 können parallel starten.
>
> Nachtrag nach PR 1 (2026-10-19): `group` in `UArray` heisst gemeinsame Fehlerquelle innerhalb einer Zeile, nicht „gleicher Messtag“ (Plan 3.1). Text in Formeln kommt mit PR 3, die Korrelation zwischen Zeilen mit PR 6. Die Prompts unten sind darauf angepasst.

Gemeinsamer Rahmen (steht in jedem Prompt, hier zum Nachlesen):
- `AGENTS.md` und `CLAUDE.md` gelten: Branch von aktuellem `main`, zuerst der Test mit dem echten Problemfall, `pytest` grün, GUI-Smoke-Test offscreen bei GUI-Änderungen, Docs im selben PR, PR nach `main`.
- Keine fest verdrahteten Konventionen (Prinzip 1): nichts SA-Spezifisches im Code, alles über Konfiguration in `campaign.json`.
- Nicht destruktiv, ehrliches Audit: Rohdaten bleiben unverändert, jede Annahme steht beim Ergebnis.
- Fixtures werden nie verändert; neue Ausschnitte nur aus den SA-Daten unter `C:\Users\timkl\OneDrive - OST\Dokumente\SA\`, klein, ohne Passwörter oder Kennungen.
- Keine Pakete ungefragt installieren.
- Die Skript-Auswertung `…\SA\auswertung\sa_auswertung.py` ist die fachliche Referenz (nur lesen).

---

## PR 1 – Formelkern mit Fehlerfortpflanzung (✅ #36, Prompt nur noch zur Nachvollziehbarkeit)

```text
Du arbeitest im Repo C:\Users\timkl\Desktop\Coding\EEU\Datualizer (Python, Polars, PySide6).
Lies zuerst AGENTS.md, CLAUDE.md und docs/PLAN_AUSWERTUNG_KAMPAGNE.md (Abschnitte 1.3, 3.1, 4, 5).

Aufgabe (Etappe 2b, Schritt 8, PR 1 von 10): Baue den Formelkern im neuen Paket
datualizer_core/calc/ – ohne GUI, ohne Kampagnenbezug.

1. calc/expr.py: Parser für Formeln auf Basis von Pythons `ast` mit Positivliste:
   Zahlen, Namen (auch mit Punkt, z. B. `fid.mean`, als ein Name), + − * / **, unäres Minus,
   Vergleiche, and/or/not, `a if c else b`, Funktionen log, log10, exp, sqrt, abs, min, max,
   where, isfinite, round. Kein eval/exec, keine Attribute, Indizes, Lambdas, Comprehensions.
   `compile_expr(text, names) -> Expr`; Fehler als `ExprError` mit Position (Spalte) und den
   bekannten Namen. `Expr.names`, `Expr.to_polars() -> pl.Expr`, `Expr.evaluate(env) -> UArray`.
2. calc/uncertain.py: `Constant` (pydantic: name, value, u=0, kind "zufall"|"syst", group,
   unit, note) und `UArray`: Wert (ndarray) plus Gradienten je unsicherem Eingang (Dual-Zahlen).
   Operationen und Funktionen aus 1. propagieren die Ableitungen exakt. Methoden: `u()`,
   `u_random()`, `u_syst()`, `budget() -> dict[eingang, ndarray]` (∂f/∂x·u). Eingänge mit
   gleicher `group` und kind "syst" sind voll korreliert (Beiträge addieren sich linear,
   nicht quadratisch); Doku im Docstring.
3. Tests (tests/test_calc_expr.py, tests/test_calc_uncertain.py):
   - Parser lehnt `__import__('os')`, `a.b()`, `x[0]`, Lambda ab, mit verständlicher Meldung.
   - Polars- und numpy-Auswertung liefern dasselbe.
   - Fortpflanzung gegen Handrechnung: X = 1 − (S − B)·f/c_in mit den Zahlen von 17.10. Z16
     (S 25.807 ± 0.10, B 2.266 ± 1.0, f 2.701, c_in 144.361; erwartet X 0.560, Blindwert-Beitrag
     2.701/144.361·1.0 = 0.0187); Vergleich mit numerischer Ableitung.
   - Korrelierte syst. Eingänge: f(x)=a·x mit a, b in derselben Gruppe.
4. Docs: README-Abschnitt "Formeln" kurz, MASTER_ETAPPEN_PLAN Schritt 8 "PR 1 ✅".

Nicht in diesem PR: Kampagnenintegration, GUI, Monte Carlo.
Abnahme: pytest grün; Beispiel im PR-Text, wie eine Formel samt Budget aussieht.
```

---

## PR 2 – Rechenkanäle, Masken, Gerätemetadaten

```text
Repo: C:\Users\timkl\Desktop\Coding\EEU\Datualizer. Lies AGENTS.md, CLAUDE.md,
docs/PLAN_AUSWERTUNG_KAMPAGNE.md (1.1, 3.2) und datualizer_core/calc/ (aus PR 1, muss in main sein).

Aufgabe (PR 2 von 10):
1. `DerivedChannel(name, expr, unit, pieces: list[Piece(start, end, expr)])` in
   `CampaignSource.derived` (campaign.json). Rechenkanäle entstehen mit `Expr.to_polars()` aus
   Kanälen derselben Quelle; `pieces` gelten nach Master-Zeit (z. B. FID-Skala 15.10. vor und
   nach 09:24). Sie erscheinen als normale Kanäle (Plot, `track`, `window_stats`, Qualitäts-Flags),
   Rohkanäle bleiben unverändert. Einheit und Herkunft (Formel) im Kanal-Tooltip.
2. `Mask(start, end, channels=[], reason)` in `CampaignSource.masks` → Flag `masked` in
   `quality_flags`; `StatsConfig.exclude_flags` enthält `masked` standardmässig; im
   Kampagnenplot schraffiert mit Grund im Tooltip. Bearbeiten über Rechtsklick im Plot
   („Bereich ausblenden…“) und im Quellen-Dock.
3. Reader/Profil für Testa-iFID-`Settings*.xml`: nur den Block `calib_data`/`calib_init`
   (`time_last_cal`, `std_gas`, `sig_span`, `sens`, `zero`) und `rf-select` als `source_meta`;
   alles andere ignorieren (die Datei enthält VNC-Passwort-Hash, PIN, Systemkennung).
   Funktion `suggest_linear_calibration(meta, mgc_per_ppm)` → a = mgc_per_ppm/sens,
   b = −zero·a; im Quellen-Dock als Vorschlag für einen Rechenkanal (Nutzer bestätigt).
   Fixture: gekürzte Kopie von `SA\Settings_17_10.xml` nur mit diesen Blöcken.
   Erwartung: sens 4.213281, zero 4.527784, mgC/ppm 1.616 → a 0.38355, b −1.7366.
4. `SqliteOptions.json_columns: list[str]`: JSON-Textspalten in Spalten zerlegen
   (`parameters_json` → `V_MFC`, `Temperatur`, `Volumenstrom`, `f_Fan`); fehlende Schlüssel = leer.
   Test mit tests/fixtures/sa_12_10/instrument_reader_12_10_extract.db.
5. Tests für alles; Docs (BENUTZUNG.md, DATENKATALOG_SA_KAMPAGNE.md, Masterplan).

Nicht in diesem PR: Schritt-Tabelle, Statistik-Änderungen.
```

---

## PR 3 – Text in Formeln, Schrittarten, benannte Fenster, Schritt-Tabelle, Äquivalenztest

```text
Repo: C:\Users\timkl\Desktop\Coding\EEU\Datualizer. Lies AGENTS.md, CLAUDE.md,
docs/PLAN_AUSWERTUNG_KAMPAGNE.md (1.2, 1.4, 3.1, 3.3), docs/RESEARCH_FENSTER_STATISTIK.md,
datualizer_core/calc/ (Formelkern aus PR 1, in main), datualizer_core/stats/window.py und
datualizer_core/campaign/campaign.py (window_table).

Aufgabe (PR 3 von 10):
0. Text in Formeln (Erweiterung von datualizer_core/calc/expr.py): Text-Literale in einfachen
   oder doppelten Anführungszeichen und Textspalten, nur mit `==` und `!=` (Ergebnis Wahrheits-
   wert, Ableitung 0); jeder andere Gebrauch von Text (Arithmetik, <, Funktionen) ist ein
   `ExprError` mit Position. Polars und numpy rechnen gleich; null/fehlender Text ist weder
   gleich noch ungleich einem Literal (Ergebnis falsch, wie NaN bei Zahlen) – im Docstring
   festhalten. `UArray` bleibt zahlenbasiert: Textspalten kommen als eigener, unsicherheitsfreier
   Eingang (z. B. numpy-Array dtype object/str). Tests: `klasse == 'null'`, `klasse != "null"
   and V_MFC > 0`, Fehler bei `klasse + 1` und `klasse < 'a'`, Polars = numpy mit null.
   Nebenbei: Der Vorschlag bei Tippfehlern bevorzugt gleichen Anfangsbuchstaben bzw. Gross/klein
   (heute schlägt `Bx` bei bekannten `B` und `x` den Namen `x` vor); Test dazu.
1. `StepClass(name, when)` – `when` ist eine Formel des Formelkerns über Schrittparametern
   (Zahlen und Text, z. B. `V_MFC > 0`, `Probe == 'Eingang'`).
   Spalte `klasse` je Schritt; Schritte ohne Treffer: Klasse leer + Hinweis; Treffer in
   mehreren Klassen: erste Klasse + Hinweis.
2. `NamedWindow(name, rule: WindowRule, classes=[])`; die bestehende `window_rule` ist das
   Fenster `default` (Rückwärtskompatibilität von campaign.json!). `WindowOverride` bekommt
   optional `window` (Name); Fenster-Inspektor zeigt eine Auswahl der Fenster.
3. `Measure(alias, source, channel, window, stat)` und `session.step_table(spec)`: eine Zeile
   pro Schritt (Parameter, Klasse, Dauer, je Measure Wert `alias` und Standardunsicherheit
   `u_alias`: mean→ sem_eff ⊕ reading_u, slope→ slope_se_eff, two_point→ two_point_u,
   integral→ integral_u), dazu `checks_alias`. Fenster, die zur Klasse nicht passen, bleiben leer.
4. Praktische Stationarität (Äquivalenztest TOST): `StatsConfig.equivalence:
   dict["quelle/kanal", float]` (δ in Einheit/s); neues Feld `stationary_practical` in
   `WindowResult`: True, wenn das 90-%-Intervall der Steigung mit slope_se_eff ganz in ±δ liegt.
   Begründung in checks/assumptions. Die statistische Ampel bleibt. Test: FID-Ausschnitt mit
   flachem Plateau (12.10. Fixture) ist praktisch stationär bei δ = 0.1 mgC/min, statistisch
   evtl. nicht; Beitrag im PR-Text.
5. Steps-Dock: Klassen-Spalte; Window-Dock: Fensterauswahl. Docs + Masterplan.

Referenzfenster der SA-Auswertung: FID-Ende 120 s, Blindwert 180 s, Steigung 300 s,
T-Drift 600 s, Masse ganze Stufe ohne die ersten 60 s.
```

---

## PR 4 – Bezüge zwischen Schritten

```text
Repo: C:\Users\timkl\Desktop\Coding\EEU\Datualizer. Lies AGENTS.md, CLAUDE.md,
docs/PLAN_AUSWERTUNG_KAMPAGNE.md (1.4, 3.4) und in
C:\Users\timkl\OneDrive - OST\Dokumente\SA\auswertung\sa_auswertung.py die Funktionen
`evaluate` (Blindwert-Auswahl) und `loading` (Beladung) als fachliche Referenz.
Voraussetzung: PR 1 und PR 3 in main.

Aufgabe (PR 4 von 10): neues Modul datualizer_core/calc/relations.py, arbeitet auf der
Schritt-Tabelle (Polars) und Ausdrücken aus dem Formelkern.
1. `nearest_reference(table, value, where, match, side="both", combine="mean_rect",
   u_min, u_one, fallback)`: für jeden Schritt den nächsten passenden Schritt davor/danach
   (where = Ausdruck, match = gleiche Werte in diesen Spalten). Beide → Mittel, u =
   max(|Δ|/2/√3, u_min); einer → u = u_one; keiner → fallback (Wert, u) oder leer.
   Spalten: `<name>`, `u_<name>`, `<name>_quelle` (Schritt-IDs), `<name>_art`
   ("eingeklammert"/"einseitig"/"keiner"), `<name>_vor`, `<name>_nach`.
2. `cumulative(table, expr, reset_when, gap_s)`: laufende Summe in Zeitreihenfolge,
   Rücksetzen bei `reset_when` (Ausdruck; Funktion `changed(spalte)` = Wert anders als
   in der vorigen Dosierzeile) und bei Lücken ≥ gap_s zwischen Schritten.
3. `bin(expr, width)`.
4. Konfiguration als `Relation`-Modelle (pydantic, discriminated union nach `kind`).
5. Tests mit einer kleinen synthetischen Schritttabelle, die die Fälle von 17.10. nachbildet:
   Null-Phasen am Morgen flach (2.27), Null-Phase nach Dosierung fällt noch (nicht nutzbar),
   Dosierung bei 220 °C ohne passende Null-Phase (fallback). Beladung: Z23 (150/10, 12.544 g/h,
   19.65 min) → 4.108 g; Z24 direkt danach (17.291 g/h, 11.217 min) → 7.340 g; Reset bei T-Wechsel.
6. Docs + Masterplan.
```

---

## PR 5 – Rechengrössen, Kriterien, Szenarien, Dock „Results“

```text
Repo: C:\Users\timkl\Desktop\Coding\EEU\Datualizer. Lies AGENTS.md, CLAUDE.md,
docs/PLAN_AUSWERTUNG_KAMPAGNE.md (1.3, 1.4, 3.5, 4), dazu
C:\Users\timkl\OneDrive - OST\Dokumente\SA\auswertung\FEHLERFORTPFLANZUNG.md.
Voraussetzung: PR 1, 3, 4 in main.

Aufgabe (PR 5 von 10):
1. `Evaluation` in `Campaign.evaluation` (optional, rückwärtskompatibel): constants,
   step_classes, windows, measures, relations, calculations, criteria, ratings, scenarios.
   Reihenfolge: step_table → relations → calculations (in Listenreihenfolge, jede darf die
   vorigen nutzen) → criteria → ratings.
2. `session.results(scenario=None) -> pl.DataFrame`: Schritt-Tabelle + Rechengrössen mit
   `u_`, `u_<name>_zufall`, `u_<name>_syst`, Budget-Spalten `b_<name>__<eingang>`,
   Kriterien als Bool, `bewertung`, `grund` (Texte der verletzten Kriterien, "; "-getrennt).
   Unsicherheiten der Measures und Relations sind zufällige Eingänge; Konstanten nach `kind`.
   Szenario = Satz von Konstanten, der die Grundwerte überschreibt.
   Konstanten bekommen zusätzlich `scope` (z. B. "tag"), nur gespeichert und in der
   Tabelle der Konstanten gezeigt; ausgewertet wird es in PR 6. `group` nur für Eingänge mit
   gemeinsamer Fehlerquelle (Plan 3.1); die SA-Konstanten aus Plan 4 haben keine `group`.
3. Dock „Results“ im Kampagnenfenster: Konstanten (Wert, u, Art), Formeln mit Prüfung beim
   Tippen, Kriterien, Bewertungsregeln, Szenario-Umschalter, Ergebnistabelle (Filter,
   Sortierung, CSV-Export, Spaltenauswahl); Doppelklick auf eine Zeile → Fenster-Inspektor.
4. Tests: Rechnung von 17.10. Z16 mit den Konstanten aus dem Plan, Abschnitt 4
   (c_in 144.36 mg/Nm³ bei ṁ 13.252 g/h und V̇_N 91.80 Nm³/h, X 0.560 ± 0.047); Szenario
   „Luft 115 m³/h“ verschiebt X auf ≈ 0.494. GUI-Smoke offscreen.
5. Docs: BENUTZUNG.md (Abschnitt Auswertung), Masterplan.
```

---

## PR 6 – Auswertungen über Schritte

```text
Repo: C:\Users\timkl\Desktop\Coding\EEU\Datualizer. Lies AGENTS.md, CLAUDE.md,
docs/PLAN_AUSWERTUNG_KAMPAGNE.md (1.5, 3.6) und in sa_auswertung.py (Pfad siehe Plan)
`t_crossing` und `fit_arrhenius`. Voraussetzung: PR 5 in main.

Aufgabe (PR 6 von 10): datualizer_core/calc/analyses.py, Konfiguration `Analysis`
(discriminated union) in `Evaluation.analyses`, Ergebnisse über `session.analyses()`.
1. `crossing(table, x, y, level, group, u_y, where)`: lineare Interpolation; Bereich aus
   y ± u_y; ohne Durchgang Text "< x_min" bzw. "> x_max".
2. `fit_line(table, x_expr, y_expr, u_x, u_y, where, method="effective_variance")`:
   σ² = u_y² + m²·u_x², 5 Iterationen, χ²_red, u_m mit Birge-Faktor max(1, √χ²_red);
   abgeleitete Grössen als Formel (z. B. Ea = −m·8.314).
3. `aggregate(table, rows, cols, value, where, how)` (mean, median, n, min, max).
4. Korrelation zwischen Zeilen: `Constant.scope` (z. B. "tag" = eine Konstante gilt für alle
   Schritte desselben Messtags gleich; Messtag aus dem Schrittstart in der Kampagnenzone).
   Aus den Budget-Spalten der Ergebnis-Tabelle (PR 5: Beitrag je Eingang und Zeile) die
   Kovarianzmatrix bilden: zufällige Beiträge nur auf der Diagonale, systematische Beiträge
   derselben Konstante innerhalb desselben scope voll korreliert (c_i·c_j), über scope-Grenzen
   unabhängig. `group` aus PR 1 bleibt die Korrelation verschiedener Eingänge innerhalb einer
   Zeile und wird nicht umgedeutet.
   - `fit_line`: verallgemeinerte kleinste Quadrate mit dieser Kovarianz (auf y nach
     Transformation, also Beiträge über die Ableitung von y_expr umrechnen) plus u_x über die
     effektive Varianz. Ausweisen: m und u_m nur zufällig, u_m mit Korrelation, u_b. Erwartung
     und Test: ein systematischer Faktor auf alle k eines Tages (Luftstrom) verschiebt nur b,
     u_m bleibt gleich; ein Faktor, der nur einen Tag trifft, wirkt auf m, wenn Punkte mehrerer
     Tage im Fit sind.
   - `crossing`: Bereich zusätzlich mit allen systematischen Beiträgen desselben scope gemeinsam
     verschoben (statt Punkt für Punkt ± u), getrennt als `bereich_syst` ausgewiesen.
5. Optional: Monte Carlo (Option `mc_samples`) für crossing und fit_line, mit derselben
   Kovarianz (multivariate Normalverteilung).
6. Tests: SA-Light-off 10 Nl/min aus Plan 7 (Werte aus
   …\SA\auswertung\ergebnisse_auto\katalyse\kampagne_ergebnisse.csv): T50 179 °C,
   Arrhenius Ea 21 ± 4 kJ/mol aus 8 Punkten. Fit gegen scipy.odr als Kontrolle (nur im Test,
   falls vorhanden; sonst Handrechnung).
7. Anzeige in einem einfachen Tabellen-Tab des Results-Docks; Docs (BENUTZUNG.md: Unterschied
   `group` innerhalb einer Zeile und `scope` zwischen Zeilen, mit Beispiel).
```

---

## PR 7 – Ergebnisfenster: Tabelle, XY, Matrix, Kleinvielfache

```text
Repo: C:\Users\timkl\Desktop\Coding\EEU\Datualizer. Lies AGENTS.md, CLAUDE.md,
docs/PLAN_AUSWERTUNG_KAMPAGNE.md (1.6, 3.7) und schau dir die PNGs in
C:\Users\timkl\OneDrive - OST\Dokumente\SA\auswertung\ergebnisse_auto\katalyse\
an (01_umsatz_matrix, 02_light_off, 03_konzentration, 05_arrhenius, 07_blindwerte,
09_reproduzierbarkeit). Voraussetzung: PR 5, 6 in main.

Aufgabe (PR 7 von 10): `ViewSpec` (discriminated union) in `Evaluation.views`, neues
Fenster datualizer_gui/results_window.py mit Ansichten als Tabs, gerendert mit pyqtgraph.
1. `table`: Spalten, Filter (Ausdruck), Sortierung.
2. `xy`: x, y (oder x_expr/y_expr), u_x, u_y (Fehlerbalken), color_by (ordinal: blaue Rampe
   hell → dunkel; kategorial: feste Palette), marker_by, filled (Ausdruck; leer = Rand),
   connect_by (Linien, sortiert nach x), facet_by (Kleinvielfache mit gemeinsamen Achsen),
   ref_lines, zweite x-Achse als Funktion (z. B. °C zu 1000/T). Legende neben dem Plot.
   Tooltip je Punkt mit Schritt-ID; Klick → Fenster-Inspektor auf den Schritt.
3. `matrix`: rows, cols, value, status-Ausdruck (Zelle grau + "*" wenn nicht erfüllt),
   Zelltext (Wert, Untertext), sequentielle Farbskala mit Legende, Textfarbe nach Helligkeit.
4. Export jeder Ansicht über den bestehenden Plot-Export (PNG/SVG/PDF, mm, pt).
5. Tests: Ansichten aus einer synthetischen Ergebnis-Tabelle, Offscreen-Smoke, Export-Datei
   vorhanden. Docs mit Screenshot (tools/readme_screenshots.py erweitern).
```

---

## PR 8 – Ausgerichtete Zeitreihen, Budget, Bericht

```text
Repo: C:\Users\timkl\Desktop\Coding\EEU\Datualizer. Lies AGENTS.md, CLAUDE.md,
docs/PLAN_AUSWERTUNG_KAMPAGNE.md (1.5, 3.7) und die Seiten 04_durchbruch, 06_verlauf_*,
10_unsicherheit sowie katalyse_bericht.pdf in
C:\Users\timkl\OneDrive - OST\Dokumente\SA\auswertung\ergebnisse_auto\katalyse\.
Voraussetzung: PR 2, 7 in main.

Aufgabe (PR 8 von 10):
1. `broadcast_steps(results, columns)` → stückweise konstante Spuren auf Master-Zeit
   (join_asof nach Schrittstart, leer ausserhalb der Schritte); nutzbar in Rechenkanälen.
2. Ansicht `aligned`: Ausdruck über einen Kanal und Schrittkonstanten (z. B.
   1 − (fid_mgC − B)·f/c_in), t0 = step_start oder sequence_start (Beginn der Beladung aus
   der Relation `cumulative`), facet_by, color_by; Endpunkt je Schritt markiert (gefüllt nach
   Ausdruck). Optionales Mitteln (z. B. 10 s).
3. Ansicht `bars`: gestapelte Anteile (Budget-Spalten aus PR 5), darüber u gesamt als eigene
   Balkenreihe (kein zweites y).
4. Ansicht `report`: geordnete Liste von Ansichten + Titelseite (Kampagnenname, Messtage,
   Kennwerte aus `analyses`) → mehrseitiges PDF (A4 quer). Menü Campaign → Export Report….
5. Tests, Offscreen-Smoke, Docs.
```

---

## PR 9 – Abgleich-Werkzeuge (Etappe 2b, Schritt 7)

```text
Repo: C:\Users\timkl\Desktop\Coding\EEU\Datualizer. Lies AGENTS.md, CLAUDE.md,
docs/PLAN_AUSWERTUNG_KAMPAGNE.md (3.8), datualizer_core/campaign/clock.py und
docs/DATENKATALOG_SA_KAMPAGNE.md (Uhrenprobleme).

Aufgabe (PR 9 von 10):
1. Uhr-Anker aus Zellpaaren einer Quelle: Konfiguration (Quelle, Blatt, Zelle Quellzeit,
   Zelle Masterzeit, Datum) → `ClockAnchor`; aus mehreren Paaren Median-Versatz und Streuung
   als `ClockModel(offset_s, uncertainty_s, method="journal_pairs")`. Beispiel: Journal-Blatt
   17.10.2026, C3/C4 und D3/D4 („Start FID“ 11:26:00/11:31:46, „Start Real“ 11:42:15/11:48:00)
   → 974.5 s ± 0.5 s. Fixture: Ausschnitt aus labjournal_12_10_extract.xlsx nutzen oder neue
   kleine Datei aus dem SA-Journal (nur diese Zellen).
2. Kreuzkorrelation zweier Spuren (Signal oder Ableitung) auf gemeinsamem Raster im
   gewählten Zeitbereich → Versatz, Korrelationskurve, Güte; Grenzen des Suchbereichs.
3. Dialog im Quellen-Dock: zwei Spuren überlagert, Schieber, Korrelationskurve, Anker-Liste,
   „Übernehmen“ schreibt das ClockModel (mit method und note).
4. Tests: synthetische Verschiebung wird gefunden; Journal-Paare ergeben 975 s ± 5 s.
   Docs + Masterplan (Schritt 7).
```

---

## PR 10 – SA-Kampagne als Beispiel, End-to-End-Abnahme

```text
Repo: C:\Users\timkl\Desktop\Coding\EEU\Datualizer. Lies AGENTS.md, CLAUDE.md,
docs/PLAN_AUSWERTUNG_KAMPAGNE.md vollständig (insbesondere 4 und 7) und
C:\Users\timkl\OneDrive - OST\Dokumente\SA\auswertung\AUSWERTUNG_LIESMICH.md sowie
…\auswertung\tage\2026-10-1{2,5,7}.json. Voraussetzung: PR 1–9 in main.

Aufgabe (PR 10 von 10):
1. examples/campaign_SA_ethanol.json: Quellen der drei Messtage (Pfade relativ, wie
   campaign_SA.json), Uhren, Rechenkanäle (FID-Skala je Tag aus den Tageskonfigurationen),
   Masken, Schritt-Zuordnung (12.10. und 17.10. Journal, 15.10. DB-Runs), zusätzliche
   Null-Phasen des 12.10., vollständige `evaluation` (Konstanten, Fenster, Measures,
   Relations, Calculations, Criteria, Ratings, Analyses, Views).
2. examples/rezept_katalyse_voc.json: dieselbe `evaluation` ohne Quellen als Vorlage;
   Funktion/Menü „Auswertung aus Vorlage übernehmen“.
3. Test tests/test_sa_end_to_end.py (übersprungen, wenn die SA-Daten fehlen): vergleicht
   `session.results()` mit …\SA\auswertung\ergebnisse_auto\<tag>\schritte.csv
   (X ± 0.005, c_in ± 0.5 %, u(X) ± 10 %, T_bett ± 0.2 K, Bewertung, Stationarität gleich)
   und die Analysen (T50 179/203 °C, Ea 21 ± 4 kJ/mol). Jede Abweichung erklärt im PR-Text.
4. Bericht aus der Kampagne erzeugen und mit katalyse_bericht.pdf vergleichen (Seiten,
   Werte); Screenshots in die Doku.
5. Docs: BENUTZUNG.md (Rundgang Auswertung), DATENKATALOG_SA_KAMPAGNE.md,
   MASTER_ETAPPEN_PLAN (Schritte 7, 8, 10 ✅), AGENTS.md Abschnitt 6.
```
