# Recherche: Fenster-Statistik und Fenster-Inspektor (Etappe 2b, Schritt 6)

> Stand 2026-10-13. Grundlage für die Umsetzung durch spätere Agents. Dieses Dokument legt fest, **welche** Kennwerte ein Messschritt-Fenster braucht, **wie** sie gerechnet werden, **unter welchen Annahmen** sie gelten und **woran** man erkennt, dass sie falsch sind.
> Fenster kommen aus Schritt 5: `CampaignSession.intervals()` liefert pro Schritt `start`/`end` und das Teilfenster `window` (nach `WindowRule`) auf Master-Zeit. Kanäle kommen als `Track` (`session.track(quelle, kanal)`: `x` Master-Epoch-Sekunden, `y`, `gap_rows`).

## 1. Bedarf aus der SA-Kampagne

Die bisherige Skript-Auswertung der Ethanol-Oxidation braucht pro Messschritt (Temperatur, Dosierstufe `V_MFC`):

| Grösse | Quelle | Kennwert im Fenster | Kritische Annahme |
|---|---|---|---|
| Massenstrom Ethanol | Waage (OCR, SQLite) | **Steigung** der Masse über der Zeit (g/s → g/h) | Fenster ohne Nachfüllen (Sprung), OCR-Fehllesungen ausgeschlossen, Waage stationär |
| Massenstrom (Kontrolle) | Labjournal `Masse_0`/`Masse_1`, `Zeit_0`/`Zeit_1` | **2-Punkt-Differenz** Δm/Δt | Ablesung zu genau diesen Zeiten, Uhren abgeglichen |
| Konzentration nach Katalysator | FID (`conc`, mgC) | **Mittelwert/Median** im eingeschwungenen Teil | Stationarität, Uhrenversatz FID ↔ PC (16 min), Response-Faktor |
| Blindwert | FID in Referenzschritten | Mittelwert in Referenzfenstern | Gleiche Skala, gleiche Drift |
| Umsatz | abgeleitet | 1 − c_aus / c_ein (Schritt 8) | Unsicherheiten der Mittelwerte und Steigungen |
| Drift | FID, Temperaturen | Steigung über den Schritt, Vergleich Referenz vorher/nachher | Drift linear im Fenster |
| Dosierte Menge | Massenstrom | **Integral** über das Fenster | Lücken nicht überbrückt |

Folgerung: Jeder Kennwert braucht eine **Unsicherheit** und eine **Gültigkeitsprüfung** (genug Punkte, keine Lücke, stationär, keine Spikes). Die Annahmen stehen beim Ergebnis (Prinzip 3: ehrlich statt still).

## 2. Funktionskatalog

Bezeichnungen: Fensterpunkte (tᵢ, yᵢ), i = 1…n, nur endliche Werte; Punkte mit `spike`/`dropout`-Flag werden je nach Einstellung ausgeschlossen und **gezählt**.

### 2.1 Lage und Streuung (immer)
- n, Abdeckungsgrad = Summe der Abstände ohne Lücke / Fensterlänge (Lücken aus `coverage`/`gap_rows`), Anteil ausgeschlossener Punkte.
- Mittelwert ȳ, Median, SD s, Min/Max, Quantile 5/95 %.
- **Standardfehler des Mittelwerts:** SEM = s/√n gilt nur für unabhängige Werte. Sensorwerte im Sekundentakt sind stark autokorreliert. Deshalb mit effektiver Stichprobengrösse n_eff = n·(1−ρ₁)/(1+ρ₁) (AR(1)-Näherung, ρ₁ = Lag-1-Autokorrelation der Abweichungen) rechnen: SEM_eff = s/√n_eff. ρ₁ ≤ 0 → n_eff = n. Ausweisen: ρ₁, n_eff.
- Fehlerbild: Bei Drift im Fenster ist ρ₁ ≈ 1 und n_eff winzig. Das ist richtig so: Der Mittelwert ist dann kaum belastbar, und die Stationaritätsprüfung (2.5) schlägt an.

### 2.2 Steigung (Massenstrom, Drift)
- **OLS-Steigung** b mit Achsenabschnitt, R², Residuen-SD, klassischer SE(b) = s_res / √Σ(tᵢ − t̄)².
- **Autokorrelation:** Klassischer SE(b) ist bei positiver Autokorrelation zu klein. Zwei Korrekturen, beide ausweisen:
  1. **Newey-West (HAC)**: Punktschätzung bleibt OLS, nur der SE ändert sich; Bartlett-Gewichte 1 − j/(L+1), Lag-Faustregel L = ⌊4·(n/100)^(2/9)⌋. Robust gegen allgemeine Autokorrelation.
  2. **AR(1)-Näherung**: SE·√((1+ρ)/(1−ρ)) mit ρ = Lag-1-Autokorrelation der Residuen. Einfach, aber nur eine Näherung (hängt auch von der Autokorrelation des Regressors ab).
  Prüfgrösse: Durbin-Watson bzw. ρ₁ der Residuen.
- **Robuste Steigung (Theil-Sen):** Median aller paarweisen Steigungen, unempfindlich gegen OCR-Fehllesungen. Konfidenzintervall über die Kendall-Rangverteilung (Normalnäherung, mit Bindungskorrektur) oder Bootstrap. Aufwand O(n²): für n > ~2000 Teilstichprobe oder Repeated-Median-Variante; das muss der Umsetzer messen, nicht schätzen.
- Vergleich OLS ↔ Theil-Sen als Warnsignal: Weichen sie um mehr als ~2 SE ab, gibt es Ausreisser oder einen Knick im Fenster.
- Gravimetrische Praxis: Regression über viele Ablesungen ist robuster als Anfang/Ende. Die Restscatter der Regression ist die Typ-A-Unsicherheit. Typ-B-Anteile der Waage (Auflösung d → d/√12, Linearität, Drift, Auftrieb) kommen aus Gerätedaten und sind **Eingaben**, nicht Code-Konstanten.

### 2.3 Zwei-Punkt-Differenz (Journal-Kontrolle)
- Δy/Δt aus Werten an Fensteranfang und -ende. Wert an einem Zeitpunkt: nächster Messpunkt (Abstand ausweisen) oder Median der Punkte in ±h um den Zeitpunkt (h einstellbar).
- Unsicherheit nach GUM: u²(Δy/Δt) = (u²(y₁)+u²(y₂))/Δt² + (Δy/Δt²)²·u²(Δt). u(Δt) enthält die Uhrenunsicherheit (`ClockModel.uncertainty_s`) beider Zeitpunkte.
- Fehlerbild: Ein Spike an einem Endpunkt verfälscht das Ergebnis voll. Deshalb immer neben der Regressionssteigung zeigen.

### 2.4 Integral
- Trapezregel über das Fenster **ohne Überbrückung von Lücken**: Segmente an `gap_rows` und an NaN trennen, jedes Segment einzeln integrieren. Die nicht abgedeckte Zeit wird ausgewiesen und nicht interpoliert. Wer interpolieren will, muss das ausdrücklich einschalten, und es steht dann als Annahme beim Ergebnis.
- Randbehandlung: Werte an Fensteranfang und -ende linear zwischen den Nachbarpunkten interpolieren, sofern beide Nachbarn innerhalb derselben Lücke-freien Strecke liegen.
- Unsicherheit: aus SEM_eff × Dauer als Näherung, dazu Anteil der Lückenzeit.

### 2.5 Stationarität und Drift im Fenster
- **Steigungstest:** |b|·Fensterlänge im Verhältnis zu s, bzw. ob das Konfidenzintervall der (HAC-)Steigung die Null enthält. Einfach und ausreichend als erste Ampel.
- **Hälftenvergleich:** Mittelwerte der ersten und zweiten Fensterhälfte, Differenz relativ zu SEM_eff.
- **R-Statistik (Cao/Rhinehart 1995):** Verhältnis zweier Varianzschätzungen mit exponentiellen Filtern; R ≈ 1 heisst stationär, kritische Werte tabelliert (J. Process Control 1997). Optional, weil drei Filterparameter zu wählen sind.
- Ergebnis als Ampel mit Begründung (z. B. „Steigung 0,8 mgC/min, signifikant“), nie als stilles Verwerfen.

### 2.6 Zweiseitiges `spike`-Flag (P31)
- **Hampel-Filter:** Punkt ist Spike, wenn |yᵢ − Median_Fenster| > k · 1,4826 · MAD_Fenster. Startwerte k = 3, zentriertes Fenster 2·h+1 Punkte (h = 3…10); als Zeitfenster in Sekunden angeben, weil die Abtastrate je Quelle verschieden ist.
- MAD = 0 (quantisierte oder konstante Werte, z. B. Waage) → nichts markieren und das melden; optional Mindest-Skala (Auflösung des Geräts) als Untergrenze.
- Einbau wie die übrigen Flags: `QualityConfig` erhält `spike_window_s`, `spike_k`, Kanalfilter; die Auflösung landet in der `ingestion_spec`. Der Replay-Test (`tests/test_ingestion_config.py`) muss grün bleiben. Werte werden **markiert**, nie ersetzt (Prinzip 4).
- Abgrenzung zu `dropout` (heute: Sprung weg und zurück auf das alte Niveau): `spike` ist die zweiseitige, kurze Form; `dropout` bleibt für längere Ausfälle.

### 2.7 Wahl der Mittelungszeit (Hilfsmittel, später)
- **Allan-Abweichung** (überlappend) über τ: Minimum = sinnvollste Mittelungszeit; rechts davon dominiert Drift. Nützlich, um `WindowRule.length_s` zu begründen, statt sie zu raten. Optional im Inspektor.

### 2.8 Bewusst nicht in Schritt 6
- Kalibrierung, Response-Faktoren, Umsatz und Formeln → Schritt 8 (Rechengrössen, Referenz-Fits).
- Kreuzkorrelation für Uhrenversatz → Schritt 7.
- Monte-Carlo-Fortpflanzung (GUM Supplement 1) → bei Bedarf in Schritt 8.

## 3. API-Vorschlag

Neues Modul `datualizer_core/stats/window.py` (ohne GUI-Import, ohne Kampagnen-Import im Rechenkern; die Kampagne ruft es auf):

```python
@dataclass(frozen=True)
class WindowSpec:          # aus ResolvedInterval.window oder (start, end)
    start: float; end: float

class StatsConfig(BaseModel):   # serialisierbar, landet in campaign.json
    exclude_flags: list[str] = ["spike", "dropout"]
    point_halfwidth_s: float = 0.0     # 2-Punkt: 0 = nächster Messpunkt
    hac_lag: int | None = None         # None = ⌊4·(n/100)^(2/9)⌋
    robust_slope: bool = True
    min_points: int = 5
    min_coverage: float = 0.8

@dataclass(frozen=True)
class WindowResult:
    n: int; n_excluded: int; coverage: float
    mean: float; median: float; sd: float; sem: float; sem_eff: float; rho1: float; n_eff: float
    slope: float; slope_se: float; slope_se_hac: float; intercept: float; r2: float; durbin_watson: float
    slope_theil_sen: float | None; slope_ts_ci: tuple[float, float] | None
    two_point: float; two_point_u: float
    integral: float; integral_gap_s: float
    stationary: bool | None; checks: list[str]      # Begründungen, Warnungen
    assumptions: list[str]                           # was gilt (z. B. "AR(1)-Näherung für n_eff")

def window_stats(x: np.ndarray, y: np.ndarray, gap_rows: np.ndarray, window: WindowSpec,
                 flags: np.ndarray | None, config: StatsConfig) -> WindowResult: ...
```

`CampaignSession.window_table(channels, config) -> pl.DataFrame`: eine Zeile pro Schritt × Kanal, Spalten aus `WindowResult` plus Schritt-Label und Parametern. Grundlage für die Fenstertabelle mit Export (Schritt 8).

**Abhängigkeiten:** Entschieden (Paket B): scipy ist Abhängigkeit, vom Nutzer freigegeben, und wird nur für t- und Normal-Quantile genutzt.

## 4. Fenster-Inspektor (GUI)

- Doppelklick auf einen Schritt (Steps-Dock) oder Rechtsklick → „Inspect step“ öffnet ein Dock „Window“:
  - **Oben:** Ausschnitt der gewählten Kanäle, Fenster hervorgehoben, Regressionsgerade (OLS und Theil-Sen), ausgeschlossene Punkte (Spikes) als Kreuze, 2-Punkt-Stützstellen markiert.
  - **Unten:** Tabelle der Kennwerte mit Unsicherheit, Ampel (Stationarität, Abdeckung, Ausschlüsse) und die Annahmen im Klartext.
  - Fensterränder im Inspektor ziehbar; das ändert nur dieses Teilfenster (Override pro Schritt) und wird in `campaign.json` gespeichert.
- Alles per Tastatur weiterschaltbar (nächster/voriger Schritt), damit 38 Schritte schnell durchgesehen sind.

## 5. Arbeitspakete für spätere Agents

| Paket | Inhalt | Abnahme |
|---|---|---|
| **A. Kern-Statistik** ✅ erledigt | `stats/window.py`: Lage/Streuung, OLS, 2-Punkt, Integral mit Lücken, Abdeckung, `WindowResult`, `window_table` | Tests mit synthetischen Daten bekannter Lösung, Waagen-Fixture (`instrument_reader_12_10_extract.db`) im Schritt „T Einstellung 155“ |
| **B. Unsicherheit + Autokorrelation** ✅ erledigt (Abschnitt 7) | n_eff, SEM_eff, Newey-West, AR(1)-Korrektur, Durbin-Watson, Theil-Sen + CI, Stationaritäts-Ampel | Simulierte AR(1)-Reihen: Abdeckung der Intervalle nahe Sollwert (Monte-Carlo-Test mit festem Seed) |
| **C. `spike`-Flag** ✅ erledigt (PR #19) | Hampel in `quality.py`, Einstellungen in `QualityConfig` und Spec, Abgrenzung zu `dropout` | OCR-Fixture (`ocr_recording_105012_extract.csv`, Fehllesungen bis 9620 g), Replay-Test grün |
| **D. Fenster-Inspektor** ✅ erledigt (Abschnitt 8) | Dock, Plot-Ausschnitt mit Fits, Kennwerttabelle, Ampel, Override pro Schritt | GUI-Tests offscreen, Screenshot im PR |

Reihenfolge: A → B → D; C unabhängig.

## 6. Quellen

- Autokorrelation und effektive Stichprobengrösse: [Standard error of the mean for time-series data](https://ljmartin.github.io/technical-notes/stats/estimators-autocorrelated/), [Autocorrelation: The Error That Fakes Significance](https://metricgate.com/blogs/autocorrelation-ignored-time-series/), [Issues Using OLS with Time Series Data (Montana State)](https://www.montana.edu/cstoddard/562/Autocorrelation.pdf)
- Newey-West: [panelbox: Newey-West](https://panelbox.readthedocs.io/en/latest/inference/newey-west/), [SAS PANEL Procedure](https://support.sas.com/documentation/cdl/en/etsug/68148/HTML/default/etsug_panel_details41.htm), [EViews: Long-run Covariance Estimation](https://help.eviews.com/content/lrvar-Technical_Discussion.html)
- Theil-Sen: [Wikipedia](https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator), [SciPy theilslopes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html), [Real Statistics: Kendall-Theil-Sen](https://real-statistics.com/regression/kendall-theil-sen-regression/)
- Hampel-Filter: [MathWorks hampel](https://www.mathworks.com/help/signal/ref/hampel.html), [Real Statistics: Hampel](https://real-statistics.com/time-series-analysis/stochastic-processes/hampel-filter-outliers/), [daxs filters](https://daxs.readthedocs.io/latest/modules/filters.html)
- Gravimetrische Durchflussmessung und GUM: [Uncertainty in microflow measurements: GUM, GUM-S1, Bayes (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9756327/), [Gravimetrische Regressionsmethode im Sub-Mikroliter-Bereich](https://www.researchgate.net/publication/235945541_Novel_gravimetric_measurement_technique_for_quantitative_volume_calibration_in_the_sub-microliter_range)
- Stationarität: [Turan et al. 2023, Steady state detection](https://jaschke.folk.ntnu.no/preprints/2023/ESCAPE2023/Turan2023_Steady_state_detection_ESCAPE33_revised.pdf), [Steady State and Transient State Identification (OSU)](https://shareok.org/handle/11244/11161), [Critical values for a steady-state identifier](https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/3532355)
- Allan-Abweichung: [NIST](https://tf.nist.gov/general/pdf/3052.pdf), [Inertial Sense: Allan deviation](https://docs.inertialsense.com/user-manual/imx/performance/allan_deviation/)
- FID (EN 12619): [EN 12619:2013 Übersicht](https://standards.iteh.ai/catalog/standards/cen/99ba2781-f335-490d-ae16-f9cd39748f5b/en-12619-2013). Mittelungszeit und Driftgrenzen stehen nur im Normtext und waren nicht frei zugänglich; vor Übernahme konkreter Grenzwerte die Norm selbst prüfen.

**Hinweis zur Belastbarkeit:** Die Formeln oben sind Standardverfahren. Die AR(1)-Korrektur ist ausdrücklich eine Näherung. Die Faustregeln (k = 3 beim Hampel-Filter, Lag-Regel bei Newey-West) sind Startwerte und müssen an den Fixtures geprüft werden.

## 7. Umsetzung Paket B: Entscheidungen und Messwerte

Stand nach der Umsetzung (`datualizer_core/stats/window.py`, Tests in `tests/test_window_uncertainty.py`).

- **Neue Felder** in `WindowResult` zusätzlich zum Vorschlag in Abschnitt 3: `slope_se_ar1` (AR(1)-korrigierter SE, Abschnitt 2.2 Punkt 2) und `halves_diff` (Hälftenvergleich). `integral_u` = `sem_eff` · abgedeckte Zeit, als Näherung in `assumptions`.
- **n_eff:** ρ₁ aus aufeinanderfolgenden nutzbaren Punkten (gleiche Abstände angenommen), n_eff nie unter 1. Ab n_eff < 10 steht eine Warnung in `checks`.
- **Newey-West:** Bartlett-Gewichte, Lag nach Faustregel oder `StatsConfig.hac_lag`, Kleinstichproben-Faktor n/(n−2).
- **Theil-Sen-Intervall:** Kendall-Normalnäherung mit Bindungskorrektur in t und y (wie `scipy.stats.theilslopes`, im Test gegen scipy geprüft). Bei positiv autokorrelierten Residuen wird Var(S) mit (1+ρ)/(1−ρ) verbreitert (Näherung, steht in `assumptions`). Ohne diese Verbreiterung wäre das Intervall bei Sensordaten viel zu schmal.
- **Laufzeit Theil-Sen** (alle Paare, numpy, ein Kern, gemessen): n = 1000 ≈ 23 ms und 4 MB, n = 2000 ≈ 0,2 s und 16 MB, n = 5000 ≈ 1 s und 100 MB. Grenze `THEIL_SEN_MAX_POINTS = 1000`: Darüber wird eine Stichprobe von 499 500 Paaren mit festem Seed gezogen (reproduzierbar, in `assumptions`). Ein ganzes `window_stats` mit n = 5000 braucht damit ≈ 50 ms. Repeated Median wurde nicht gewählt: Er ist naiv ebenfalls O(n²) und hat kein einfaches Kendall-Intervall.
- **Warnung OLS ↔ Theil-Sen:** Abweichung > 2 · SE der Theil-Sen-Steigung (aus ihrem Intervall). Der OLS-SE taugt dafür nicht, weil genau die Ausreisser ihn aufblähen (OCR-Fixture: OLS −1680 g/h, HAC-SE ≈ 8500 g/h; Theil-Sen −17,6 g/h).
- **Stationaritäts-Ampel:** Steigungstest mit dem grösseren von HAC- und AR(1)-SE (die Lag-Regel ist bei starker Autokorrelation zu kurz) und Hälftenvergleich mit sem_eff je Hälfte, je auf 97,5 % (Bonferroni, zusammen ≈ 95 %), **Normal-Quantile**. t-Quantile mit n_eff-Freiheitsgraden bestrafen die Autokorrelation doppelt: Auf der OCR-Waage (Residuen eine glatte Kurve, Durbin-Watson 0,02) galt eine Drift von 6 SE sonst als stationär. Die R-Statistik ist nicht umgesetzt (optional).
- **Monte-Carlo** (300 AR(1)-Reihen à 200 Punkte, Seed 1, Soll 95 %):

| φ | Mittelwert (t, n_eff) | Steigung AR(1)-SE | Theil-Sen | Newey-West | klassisch | Ampel-Fehlalarm |
|---|---|---|---|---|---|---|
| 0,5 | 0,94 | 0,93 | 0,95 | 0,89 | 0,79 | 0,07 |
| 0,8 | 0,91 | 0,92 | 0,93 | 0,75 | 0,53 | 0,12 |

  Bei φ = 0,95 (separat gemessen, 400 Reihen) fallen Mittelwert, AR(1)-SE und Theil-Sen auf 0,80–0,87, Newey-West auf 0,45, und der Fehlalarm steigt auf ≈ 0,18; dort ist n_eff < 10 und die Warnung in `checks` steht. Newey-West mit der Faustregel deckt bei starker Autokorrelation deutlich zu wenig ab. Wer sich auf `slope_se_hac` stützt, sollte bei Durbin-Watson ≪ 2 `hac_lag` grösser setzen oder `slope_se_ar1` nehmen.

## 8. Umsetzung Paket D: Fenster-Inspektor

- **Dock „Window“** (`datualizer_gui/components/window_inspector.py`). Öffnet per Doppelklick im Steps-Dock (zoomt weiterhin), über das Kontextmenü der Steps-Tabelle oder per Rechtsklick im Plot → „Inspect step“. Zeigt die im Quellen-Dock gewählten Kanäle, höchstens 8 als Plot, weitere nur in der Tabelle. Zeitachse in Sekunden ab Schrittbeginn.
- **Kern-Ergänzungen** für die Darstellung, ohne die Rechnung zu ändern:
  - `WindowResult.used_rows`/`excluded_rows`: Zeilen der Eingabe, für die Kreuze
  - `intercept_theil_sen`: Theil-Sen-Gerade am Fensterbeginn
  - `two_point_points`: Stützstellen der 2-Punkt-Differenz
  - `slope_se_eff`: der grössere korrigierte SE, wie in der Ampel
  - `Check(str)` mit `level` und `WindowResult.level`

  Die Felder zum Zeichnen erscheinen nicht in `window_table`.
- **Override pro Schritt.** `Campaign.window_overrides` (Schritt-ID → Start, Ende, Name des Schritts beim Setzen) ersetzt die `WindowRule` nur für diesen Schritt. Im Inspektor lässt sich der Bereich nur innerhalb des Schritts ziehen. Bekommt die ID später einen anderen Namen (Zeilen im Journal eingefügt) oder fehlt sie, steht das in `intervals().problems`. Das Override wird dann weder still angewendet noch still verworfen.
- **Werte in der Tabelle** sind auf zwei signifikante Stellen der Unsicherheit gerundet.
- **Absturz.** Die Szene des Inspektors läuft ohne BSP-Index. Mit Index stürzte das Zeichnen nach dem Neuaufbau ab (Segfault im BSP-Baum von Qt, Linux offscreen, sobald vorher ein anderer Inspektor existierte). Die genaue Ursache ist nicht geklärt.
