# Datualizer benutzen: ausführliche Anleitung

Diese Seite sammelt die Details zur Bibliothek und zur GUI, die früher alle im README standen. Den Überblick mit Bildern gibt das [README](../README.md).

## Als Bibliothek nutzen

```python
from datualizer_core import load_csv

ds = load_csv("SA_testmessung_1.csv")      # PreScan + Laden + Audit in einem Rutsch

ds.shape                                   # (4931, 9)
ds.columns                                 # ['time_seconds', 'kanal_1', 'kanal_2', ...]

t = ds.to_numpy("time_seconds", allow_copy=False)   # Zero-Copy-View für Plots
long = ds.to_long()                                  # [time_seconds, channel, value]

if ds.audit_log.has_errors():
    print(ds.audit_log.to_dataframe())     # Welche Zelle war kaputt, und warum?
```

## Automatik schlägt vor, du entscheidest

Jede Heuristik (Zeitspalte, Spaltentypen, Long-Format, Messwert vs. Parameter, Namens-Vokabulare, Schwellen) ist über eine `IngestionConfig` überschreibbar. Und jedes geladene Dataset verrät, was die Automatik entschieden hat:

```python
from datualizer_core import ColumnKind, IngestionConfig, LongFormatConfig, load_csv

ds = load_csv("export.csv")
spec = ds.ingestion_spec                     # alle Entscheidungen, ausgeschrieben
print(spec.model_dump_json(indent=2))        # → speichern, prüfen, korrigieren …

cfg = IngestionConfig(
    column_kinds={"Versuch": ColumnKind.IDENTIFIER,           # Spaltentyp erzwingen
                  "Temperatur": ColumnKind.NUMERIC},          # konstanter Sensor bleibt Messkanal
    long_format=LongFormatConfig(variable_col="Groesse",      # Long-Rollen selbst festlegen
                                 value_col="Betrag",
                                 channel_attr_cols=["Masseinheit"]),
)
ds = load_csv("export.csv", config=cfg)      # … und reproduzierbar neu laden
```

**Messwert oder Parameter?** Eine numerische Spalte, die in jedem Run konstant ist (Sollwerte wie `Rotameter` oder `target_temperature`), wird als `ColumnKind.PARAMETER` eingestuft. Sie bleibt in `ds.df`, steht in `ds.parameters` und wird nicht als Kanal geplottet (`ds.channels`). Runs erkennt der Loader an Spalten wie `run_id` (`Vocabulary.run_tokens`), oder du gibst sie vor: `IngestionConfig(roles=RoleConfig(run_columns=["Charge"]))`. Ein Sensor, der zufällig konstant misst, sieht genauso aus wie ein Sollwert. Den stufst du per `column_kinds` zurück auf `NUMERIC`.

## Was der Loader erkennt

**Leere Spalten** werden markiert, nicht gelöscht: `ds.fill_ratio` liefert den Füllgrad pro Spalte, `ds.empty_columns` die Spalten ohne jeden Wert. Der Inspector zeigt leere Kanäle grau und abgewählt an.

**Runs** sind eine eigene Dimension: `ds.runs` zeigt pro Run Samples, Dauer, Median-Δt, Parameterwerte und ein `aborted`-Flag für Mini-Runs (Schwelle über `IngestionConfig(runs=RunConfig(aborted_fraction=0.1))`). `ds.channel_availability` liefert den Füllgrad je Run und Kanal, `ds.select_run((2,))` ein neues Dataset nur mit Run 2 und run-relativer Zeit. In der GUI plottet ein Klick in die Run-Tabelle nur diesen Run.

**Mehrere Dateien zusammenführen:** `load_csvs([alt, neu])` lädt mehrere Exporte als ein Dataset. Byte-identische Dateien erkennt der Loader am SHA-256 und lädt sie nur einmal (Audit: `duplicate_file`). Überlappende Snapshots werden über den Schlüssel dedupliziert (Zeit, bei Long-Tabellen Zeit und Variable): Die später angegebene Datei gewinnt, abweichende Werte der früheren Datei stehen als `merge_conflict` im Audit. Mit `IngestionConfig(merge=MergeConfig(conflict=MergeConflictMode.ERROR))` schlägt jeder Konflikt laut fehl, `MergeConfig(key_columns=[...])` setzt den Schlüssel selbst. `ds.sources` zeigt pro Datei Hash, übernommene, ersetzte und widersprüchliche Zeilen. Wide und Long oder verschiedene Dezimaltrenner werden nicht gemischt (`MergeError`). In der GUI: Mehrfachauswahl unter *Open CSV…* oder *File → Add CSV (merge)…*.

**Qualitäts-Flags** markieren verdächtige Samples, ohne die Daten anzufassen: `ds.quality_flags` ist eine Ereignistabelle (`row`, `channel`, `flag`) mit `gap` (Session-Lücke), `missing` bzw. `missing_calculated` (Leerfeld in einem Kanal, der im Run sonst Werte hat; Rechenkanäle getrennt), `stuck` (eingefrorener Sensor), `jump` (z. B. Nachfüllen), `dropout` (kurzer Ausreißer, der aufs alte Niveau zurückkehrt, z. B. Waage meldet 0.0), `out_of_range` und `spike` (Hampel-Ausreißer in beide Richtungen, z. B. OCR liest 9628 statt 9672,8 g). Dazu gibt es `ds.quality_summary`, `ds.flag_mask(["dropout"], channel="masse")` und `ds.with_flag_columns()` (neue Tabelle mit einer Flag-Spalte pro Kanal). Schwellen stehen in `IngestionConfig(quality=QualityConfig(...))`; Sprungschwellen werden pro Kanal aus den Daten vorgeschlagen und landen ausgeschrieben in der Spec. Plausible Bereiche kennt nur der Nutzer: `QualityConfig(ranges={"Massenstrom Waage (g/s)": (0.0, None)})`. Der `spike`-Detektor vergleicht jeden Wert mit dem Median eines zentrierten Zeitfensters (`spike_window_s`, ohne Vorgabe 21 typische Abtastschritte) und markiert ihn, wenn er mehr als `spike_k` · 1,4826 · MAD (Start k = 3) davon abweicht; `spike_channels` schränkt die Kanäle ein, `spike_window_s=0` schaltet ab. Als Untergrenze der Streuung dient `spike_min_scale` (ohne Vorgabe die kleinste Stufe zwischen den Werten, also die Auflösung, z. B. 0,01 g), sonst würde bei quantisierten Werten schon eine Ziffer zum Spike. Mit Untergrenze 0 bleiben Fenster mit MAD = 0 unbeurteilt und stehen in `ds.quality_notes` (`spike_mad_zero`). Der Plot verbindet keine Linien über Lücken, der Inspector zählt die Flags pro Kanal.

**Einheiten** gehen nicht verloren: `Massenstrom Waage (g/s)` wird zur Spalte `massenstrom_waage`, und `ds.units["massenstrom_waage"]` ist `"g/s"`. Bei Long-Exporten kommt die Einheit aus der Einheiten-Spalte. `ds.unitless_columns` zeigt Kanäle und Parameter ohne Einheit. Eigene Einheiten oder andere Schreibweisen im Header: `IngestionConfig(units=UnitConfig(units={"snad2": "°C"}, header_patterns=[r"/\s*(?P<unit>\S+)\s*$"]))`. Die Plot-Achsen zeigen die Einheit an.

**Andere Formate und Geräteprofile:** `load_source(pfad)` lädt Text (`.csv`, `.tsv`, `.txt`, `.dat`), Excel (`.xlsx`, `.xls`) und SQLite-Datenbanken (`.db`). Ein Profil beschreibt pro Gerät Reader, Optionen und `IngestionConfig` als JSON (`datualizer_core/profiles/`). `detect_profile` wählt es anhand von Dateimuster und Hinweisen (Kopfzeile, Blatt- oder Tabellennamen). Mitgeliefert sind Profile für Testa iFID (`.dat`, `.xls`), PicoLog-CSV, instrument_reader (`.db` und OCR-CSV) und ein Labjournal. Ohne passendes Profil verhält sich `load_source` wie `load_csv`. Was der Reader zusätzlich findet, steht in `ds.source_meta`: Excel-Nebenblätter mit Geräteeinstellungen und Kalibrierung, gelöschte IDs in der Datenbank, die Annahme hinter einer berechneten Startzeit. Eigene Profile: `load_source(pfad, load_profile("mein_geraet.json"))`. Hat eine Excel-Datei mehrere Blätter oder eine Datenbank ohne Profil mehrere Tabellen, nennt der `ReaderError` die Auswahl maschinenlesbar (`exc.choice` = `"excel.sheet"` bzw. `"sqlite.table"`, `exc.options`). `with_choices(pfad, profil, sheet=…, table=…, clock_date=…)` legt die Wahl in einer Profilkopie ab, `needs_clock_date(ds)` sagt, ob reine Uhrzeiten noch ein Datum brauchen, `date_in_text("12.10.2026")` liefert einen Datumsvorschlag aus Blatt- oder Dateinamen. In der GUI öffnet *Open…* alle diese Formate und fragt bei Bedarf nach Blatt bzw. Tabelle und dem Datum für Uhrzeiten.

## Messkampagne: mehrere Geräte auf einer Zeitachse

**Messkampagne: mehrere Geräte auf einer Zeitachse.** `CampaignSession` lädt mehrere Quellen (jede mit ihrem Profil) und legt sie auf eine gemeinsame Master-Zeit in der Zone der Kampagne. Pro Quelle korrigiert ein `ClockModel` die Geräteuhr: Versatz (`offset_s`), Drift (`drift_ppm`) oder stückweise linear über Ankerpunkte (`ClockAnchor`, z. B. eine im Video abgelesene Uhrzeit), mit Methode, Unsicherheit und Notiz zur Annahme. `session.track(quelle, kanal)` liefert einen Kanal auf Master-Zeit. Die Kampagne wird als `campaign.json` gespeichert, mit relativen Pfaden, Profil, aufgelöster Spec und Hash pro Datei; beim Öffnen werden geänderte oder fehlende Dateien gemeldet (`session.warnings`).

```python
from datualizer_core.campaign import CampaignSession, ClockModel
s = CampaignSession()
fid = s.add_source("data-10-32-50.dat")                 # Profil testa_ifid_dat erkannt
pico = s.add_source("SA_Messung_Picolog.csv")           # Start aus Speicherzeit (Annahme)
s.set_clock(fid.id, ClockModel(offset_s=970, method="cross_correlation", note="FID-Uhr 16:10 hinter PC"))
cov = s.coverage(pico.id)                                # wann die Quelle Werte hat, wo Lücken sind
[(g.reason, g.duration_s) for g in cov.gaps]             # Logger-Aussetzer als "no_values"
s.save("campaign.json")
```

**Abdeckung:** `session.coverage(quelle)` liefert pro Quelle die Zeitspanne (`span`), die Abschnitte mit Werten (`segments`) und die Lücken (`gaps`) auf Master-Zeit. Eine Lücke ist entweder ein als `gap` markierter Zeitsprung (`time_gap`, Schwelle `QualityConfig.gap_factor`) oder eine Strecke, in der die Zeit weiterläuft, aber kein Kanal einen Wert hat (`no_values`, z. B. Logger-Aussetzer oder ungültige OCR-Lesungen), auch am Anfang oder Ende. `coverage(quelle, ["kanal"])` zählt nur die genannten Kanäle. Quellen ohne absolute Zeit haben keine Spanne. Die Daten selbst bleiben unverändert.

**Messschritte:** Welche Spalten einer Quelle (z. B. des Labjournals) Start, Ende, Name und Sollwerte eines Schritts sind, beschreibt eine `IntervalMapping` (Ende als Zeitspalte, Dauerspalte, „bis zum nächsten Start“ oder feste Länge; Blocknamen, die nur in der ersten Zeile stehen, über `fill_down`; Zeiten optional auf der Uhr einer anderen Quelle). Nichts davon steht im Code: `session.suggest_mapping(quelle)` nimmt den Vorschlag des Profils (`SourceProfile.intervals`) oder errät die Rollen aus der Datenform. `session.intervals()` liefert alle Schritte auf Master-Zeit, dazu `problems` (Zeilen ohne Start, unlesbares Ende …). Schritte von Hand: `session.add_interval(start, ende, "Blindwert")`. `WindowRule` legt fest, welcher Teil jedes Schritts ausgewertet wird (Einschwingzeit abschneiden, nur die letzten N Sekunden). Alles steht in `campaign.json`.

```python
s.set_mapping(s.suggest_mapping(journal.id))           # Vorschlag aus dem Profil labjournal_xlsx
s.set_window_rule(WindowRule(length_s=120))            # nur die letzten 2 min jedes Schritts
for step in s.intervals().intervals:
    print(step.label, step.window, step.params)
```

**Fenster-Statistik** (Etappe 2b, Schritt 6, Paket A): `session.window_table([(quelle, kanal), …])` liefert pro Schritt × Kanal eine Zeile mit Schritt, Fenster, Sollwerten und den Kennwerten aus `datualizer_core.stats.window_stats`: Anzahl Punkte, ausgeschlossene (`StatsConfig.exclude_flags`, Standard `spike`, `dropout`) und leere Werte, Abdeckungsgrad, Mittelwert, Median, SD, SEM, Min/Max, 5-/95-%-Quantil, OLS-Steigung pro Sekunde mit klassischem Standardfehler, R² und Residuen-SD, 2-Punkt-Differenz Δy/Δt mit Unsicherheit nach GUM (Ablesung aus `StatsConfig.reading_u`, Uhr aus `ClockModel.uncertainty_s`) und das Trapez-Integral. Lücken, leere und ausgeschlossene Werte werden nie überbrückt: Die nicht abgedeckte Zeit steht in `integral_gap_s`. `checks` nennt, was auffällt (zu wenige Punkte, geringe Abdeckung, 2-Punkt weicht von der Steigung ab), `assumptions`, worauf die Werte beruhen. SEM und klassischer Steigungs-SE setzen unabhängige Werte voraus und sind bei dicht abgetasteten Sensoren zu klein. Deshalb gibt es zusätzlich (Paket B): `rho1`, `n_eff` und `sem_eff` (AR(1)-Näherung), `slope_se_hac` (Newey-West, Lag `StatsConfig.hac_lag`, sonst ⌊4·(n/100)^(2/9)⌋), `slope_se_ar1`, `durbin_watson`, die robuste Theil-Sen-Steigung `slope_theil_sen` mit 95-%-Intervall `ts_ci_low`/`ts_ci_high` (Kendall, bei Autokorrelation verbreitert; ab 1000 Punkten aus einer festen Stichprobe von Paaren), `integral_u` und die Ampel `stationary` (Steigungstest und Hälftenvergleich `halves_diff`, Begründung in `checks`). Weichen OLS und Theil-Sen um mehr als 2 SE ab, steht das in `checks` (Ausreisser oder Knick). Die Einstellungen stehen als `stats` in `campaign.json`.

```python
from datualizer_core.stats import StatsConfig
s.set_stats_config(StatsConfig(reading_u={"instrument_reader/masse": 0.01 / 12**0.5}))  # Auflösung 0,01 g
t = s.window_table([("instrument_reader", "masse")])
t.select("step_label", "v_mfc", "n", "coverage", (t["slope"] * 3600).alias("g_h"), "slope_se", "checks")
```

Einen einzelnen Schritt liefert `s.window_stats(schritt, quelle, kanal)` als `WindowResult`. Jeder Eintrag in `checks` hat zusätzlich eine Stufe (`check.level`: `fail` bei keinen oder zu wenigen Punkten, sonst `warn`); `result.level` fasst sie zur Ampel zusammen (`ok` ohne Befund). Das Fenster eines einzelnen Schritts lässt sich von Hand setzen: `s.set_window_override(schritt_id, start, ende)` ersetzt dort die Fenster-Regel, `clear_window_override` hebt das wieder auf. Overrides stehen in `campaign.json` (`window_overrides`), in `window_table` zeigt die Spalte `window_override` sie an. Bekommt ein Journal-Schritt später einen anderen Namen oder fehlt er, steht das in `intervals().problems`.

Im Kampagnenfenster zeigt das Dock *Steps* die Schritte (Doppelklick zoomt und öffnet den Fenster-Inspektor, Rechtsklick bietet beides einzeln), die Probleme und die Fenster-Regel; *Steps from Source…* öffnet die Zuordnung mit Vorschau. Als Bänder über allen Plots erscheinen die Schritte, wenn *Settings → Tools → Interval Bands* eingeschaltet ist (Standard: aus, die Wahl wird gemerkt). Eigene Schritte: Rechtsklick → *Marker ○ ○*; der erste Klick setzt den Start (erster Kreis grün, gestrichelte Linie), der zweite das Ende. Eigene Bänder lassen sich ziehen, umbenennen und löschen; Esc verwirft einen offenen Marker.

Der **Fenster-Inspektor** (Dock *Window*, auch per Rechtsklick im Plot → *Inspect step*) zeigt die gewählten Kanäle rund um einen Schritt:
- das ausgewertete Fenster hell hinterlegt,
- die OLS-Gerade (durchgezogen) und die Theil-Sen-Gerade (gestrichelt),
- ausgelassene Samples als rote Kreuze,
- die Stützstellen der 2-Punkt-Differenz als grüne Kreise.

Darunter stehen die Kennwerte mit Unsicherheit (auf zwei Stellen der Unsicherheit gerundet), die Befunde mit Ampel pro Kanal und die Annahmen. Die Fensterränder lassen sich ziehen. Das gilt nur für diesen Schritt und wird in `campaign.json` gespeichert: Die Kante wird orange, im Steps-Dock steht „window ✎“. *Reset Window* überlässt den Schritt wieder der Fenster-Regel. Bild ab / Bild auf schaltet zum nächsten bzw. vorigen Schritt.

In der GUI öffnet *Campaign → Open Campaign Window…* (Ctrl+K) die Kampagnenansicht: links die Quellen mit Profil, Zeitspanne und Uhr, darunter der Uhren-Editor; rechts oben, falls unter *Settings → Tools → Coverage Info* eingeschaltet (Standard: aus, die Wahl wird gemerkt), die Abdeckungsspur (eine Zeile pro Quelle, Abschnitte mit Werten gefüllt, Lücken schraffiert, die Infozeile nennt Dauer und Art der Lücke unter dem Fadenkreuz), darunter ein Subplot pro gewähltem Kanal, jeder mit eigenem x-Array auf einer gemeinsamen Datumsachse (verlinkt, ohne Resampling). Die Ansicht zeigt immer alle Spuren; eigenes Zoomen bleibt bis *Reset Zoom* erhalten. Beim Hinzufügen fragt das Fenster nach Blatt bzw. Tabelle und, bei reinen Uhrzeiten, nach dem Datum (Vorschlag aus dem Blattnamen); die Wahl steht im Profil und in der Spec der Quelle in `campaign.json` und wird beim Öffnen ohne Rückfrage wieder angewendet.

**Darstellung pro Kanal** (nur Anzeige, die Daten bleiben unverändert): Rechtsklick oder Doppelklick auf einen Kanal im Quellen-Baum oder *Settings → Channel Display…* (Ctrl+D) öffnet die Einstellungen. Wählbar sind der Anzeigename (Alias, z. B. für Excel-Spalten `col_9`, deren Überschrift nur eine Einheit war; er erscheint in Achse, Baum und Infozeile, der Kanalname im Datensatz bleibt), der Stil (Linie, Punkte, Treppe, Linie + Punkte) und die Y-Skalierung: *Auto*, *robust* (1.–99. Perzentil, einstellbar, damit seltene OCR-Ausreisser den Rest nicht flach drücken) oder *fest* (min, max). Werte ausserhalb des Bereichs werden nicht verworfen, sondern am oberen oder unteren Rand mit einem Dreieck markiert. Der Standardstil kommt aus dem Profil (`SourceProfile.plot_style`; das Labjournal-Profil zeichnet Messschritte als Treppe). Die Einstellungen stehen pro Quelle unter `display` in `campaign.json`, nur Abweichungen vom Standard werden gespeichert:

```python
from datualizer_core.campaign import ChannelDisplay, YScale, YScaleMode
s.set_display("instrument_reader", "massenstrom_waage", ChannelDisplay(y_scale=YScale(mode=YScaleMode.ROBUST)))
s.set_display("SA_Labjournal", "col_9", ChannelDisplay(alias="Massenstrom Waage", style="line_points"))
s.track("SA_Labjournal", "col_9").name                   # "Massenstrom Waage"
```

**Zeit und Zoom im Kampagnenfenster** (Menü *Settings*): Das Mausrad zoomt standardmässig nur die Zeitachse (über einer Y-Achse zoomt es diese), und automatisch skalierte Subplots passen Y an den sichtbaren Zeitausschnitt an; beides ist abschaltbar. Die Zeitachse zeigt wahlweise automatisch gewählte Formate, die Uhrzeit, Datum + Uhrzeit oder die Zeit relativ zu einem Bezugspunkt (t − t0). t0 setzt *Time Axis → Set t0 at Crosshair* an der Mausposition; ohne t0 gilt der erste angezeigte Messpunkt. *Campaign Time Zone…* wählt die Zone der Kampagne: Achse und Infozeile folgen ihr, Quellen ohne eigene Zone werden in ihr gelesen. Unter *Minimum Subplot Height…* (Standard 120 px, 0 = aus) bekommen Subplots eine Mindesthöhe; reicht der Platz nicht, scrollt die Ansicht, und jeder Subplot zeigt seine eigene Zeitachse. Zeitzone und t0 gehören zur Kampagne und stehen in `campaign.json` (`timezone`, `time_reference`); Achsenformat, Zoom-Verhalten und Mindesthöhe sind persönliche Vorlieben (QSettings).

**Subplots und Farben** (*Settings → Subplots and Colours*): ein Subplot pro Kanal (Standard), Kanäle gleicher Einheit oder gleicher Quelle in einem gemeinsamen Subplot, dann mit Legende. Farben nach Kanal, nach Quelle (dieselbe Farbe wie die Zeile der Quelle in der Abdeckungsspur) oder manuell: Die Farbe pro Kanal wählt man im Dialog *Channel Display* und sie steht als `color` in `campaign.json`. Haben alle Kanäle eines gemeinsamen Subplots einen festen oder robusten Y-Bereich, zeigt er die Vereinigung dieser Bereiche; sonst skaliert er automatisch. Gruppierung und Farbschema sind persönliche Vorlieben (QSettings).

**Werte unter dem Mauszeiger** (*Settings → Values under Cursor*): als Leiste über den Plots (Standard; was nicht passt, wird abgeschnitten), als Tooltip am Mauszeiger (eine Zeile pro Kanal) oder als Tabelle im Dock *Values* (Kanal, Wert, Einheit, darüber Zeit und Lücken). *Decimals…* legt die Nachkommastellen fest; −1 heisst automatisch (4 signifikante Stellen).

**Export für den Bericht** (*Campaign → Export Plot…*, Ctrl+E): schreibt den gezeigten Zeitausschnitt mit den gezeigten Y-Bereichen, Subplots, Farben und der Abdeckungsspur als PNG, SVG oder PDF. Einstellbar sind Breite und Höhe in mm (Höhe automatisch: pro Subplot eine feste Höhe), Schriftgrösse und Linienbreite in pt, die Auflösung (nur PNG; SVG und PDF sind Vektordateien) und ein helles oder dunkles Theme. Am Bildschirm bleiben die Linien 1 px breit (Geschwindigkeit); dickere Linien und grössere Punkte gibt es nur in der Datei. Die letzten Einstellungen werden gemerkt.

```python
from datualizer_gui.components.plot_export import ExportOptions
window.export_plot("bericht_12_10.pdf", ExportOptions(width_mm=170, font_pt=8, line_width_pt=0.75))
```

## Formeln mit Unsicherheit

`datualizer_core/calc/` ist der Formelkern für Rechengrössen (Etappe 2b, Schritt 8). Er kennt weder Kampagne noch GUI; die Anbindung an Schritt-Tabelle und Rechenkanäle folgt.

**Was eine Formel enthalten darf:** Zahlen, Namen (auch mit Punkt, `fid.mean` ist *ein* Name), `+ - * / **`, unäres Minus, Vergleiche (auch verkettet, `0 < X <= 1`), `and`, `or`, `not`, `a if c else b` und die Funktionen `log` (natürlich), `log10`, `exp`, `sqrt`, `abs`, `min`, `max` (zwei oder mehr Argumente), `where(c, a, b)`, `isfinite`, `round(x, n=0)`. Die Zeichen − · × werden als `-` und `*` gelesen. Alles andere (Attribute, Methodenaufrufe, Indizes, Lambdas, Comprehensions, Text) lehnt `compile_expr` mit einem `ExprError` ab, der die Spalte und die bekannten Namen nennt:

```python
from datualizer_core.calc import ExprError, compile_expr

try:
    compile_expr("1 - (S - Bx) * f", names=["S", "B", "f"])
except ExprError as e:
    print(e)
# unknown name 'Bx'; did you mean 'B'? (column 10)
#     1 - (S - Bx) * f
#              ^
# Known names: B, S, f
```

**Zwei Ziele, eine Bedeutung:** `expr.to_polars(columns=…, constants=…)` liefert einen `pl.Expr` (Namen werden Spalten, Konstanten Literale, ohne Unsicherheit). `expr.evaluate(env)` rechnet mit numpy; `env` ordnet jedem Namen einen `UArray`, eine `Constant`, eine Zahl, ein Array oder eine Series zu. Beide liefern dieselben Werte: `/` ist echte Division, Division durch null gibt ±inf oder NaN, NaN pflanzt sich durch `min`/`max` fort und ist mit nichts gleich. Polars kennt zusätzlich null (fehlend), das sich durch alles fortpflanzt.

**Unsicherheit:** Ein `UArray` trägt neben dem Wert die exakte Ableitung nach jedem unsicheren Eingang (Dual-Zahlen). Daraus folgen nach GUM (linear, erste Ordnung):

| Methode | Bedeutung |
|---|---|
| `budget()` | Beitrag cᵢ = ∂f/∂xᵢ · uᵢ jedes Eingangs, mit Vorzeichen |
| `u_random()` | √(Σ cᵢ²) über die Eingänge `kind="zufall"` (unabhängig) |
| `u_syst()` | Eingänge `kind="syst"` mit gleicher `group` sind voll korreliert: ihre Beiträge addieren sich linear, Gruppen und Eingänge ohne Gruppe quadratisch |
| `u()` | √(u_random² + u_syst²) |

Eine `Constant` (`name`, `value`, `u`, `kind`, `group`, `unit`, `note`) ist serialisierbar und kommt später aus `campaign.json`. Gemessene Werte mit einer Unsicherheit pro Schritt werden zu `UArray.input(name, werte, u, kind="zufall")`. `round` und Vergleiche haben die Ableitung 0; `abs`, `min`, `max`, `where` und `if` nehmen die Ableitung des benutzten Zweigs.

```python
from datualizer_core.calc import Constant, UArray, evaluate

a = Constant(name="a", value=5.0, u=0.2, kind="syst", group="waage")
b = Constant(name="b", value=3.0, u=0.2, kind="syst", group="waage")
evaluate("a - b", {"a": a, "b": b}).u()     # 0.0: derselbe Fehler hebt sich in der Differenz auf
```

## Absolute Zeit

**Absolute Zeit** bleibt erhalten: `time_seconds` zählt ab dem ersten gültigen Sample, `ds.time_origin` ist der absolute Zeitpunkt dazu, `ds.time_abs` die absolute Zeit jeder Zeile und `ds.with_time_abs()` die Tabelle mit Spalte `time_abs`. Damit lassen sich später mehrere Geräte einer Messkampagne auf eine Zeitachse legen (Etappe 2b). Datums- und Zeitwerte ohne Zeitzone gelten als Wanduhrzeit. Mit `IngestionConfig(time=TimeConfig(timezone="Europe/Zurich"))` werden sie in dieser Zone gelesen, dann bleiben die Sekunden auch über die Sommerzeitumstellung korrekt (doppelte oder fehlende Uhrzeiten stehen im Audit). Logger, die nur die verstrichene Zeit exportieren (z. B. PicoLog `00:00:00`), bekommen ihren Startzeitpunkt über `TimeConfig(origin=datetime(...))`. Uhrzeiten ohne Datum (z. B. im Labjournal) bekommen ihr Datum über `TimeConfig(clock_date=date(...))`. Weitere Datumsformate: `TimeConfig(datetime_formats=[...])`; ein erzwungenes `TimeConfig(format=TimeFormat.DATETIME)`, das nicht passt, schlägt laut fehl (`TimeConfigError`).

Die mitgelieferten Namenslisten (`Vocabulary`) sind nur Startwerte und keine Konvention, an die sich deine Daten halten müssen.
