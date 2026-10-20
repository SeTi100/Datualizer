# AGENTS.md: Leitfaden für alle KI-Agenten in diesem Repo

> Diese Datei lesen alle Agenten (Claude Code, Codex, Copilot, Cursor, Gemini/Antigravity …), bevor sie hier arbeiten.
> Sie ist verbindlich. Wenn du etwas Grundsätzliches änderst, aktualisierst du diese Datei im selben PR.

---

## 1. Das große Ziel

**Datualizer** ist ein Desktop-Werkzeug (Python, Polars, PySide6, PyQtGraph), das **chaotische Mess- und Sensordaten** robust einliest, nachvollziehbar in **Tidy Data** überführt und schnell visualisiert. Kurz: *PowerQuery-Ergonomie für Ingenieursdaten, mit wissenschaftlicher Reproduzierbarkeit.*

Endprodukt (Beta, Etappe 7):

- Jeder reale Geräteexport wird geladen, ohne Absturz und ohne stilles Verfälschen.
- Jeder Bereinigungsschritt ist ein serialisierbarer Rezept-Knoten (JSON-AST). Daraus folgen Undo/Redo, Replay auf neue Messungen und Code-Export nach Polars, DuckDB-SQL und R.
- Native 60-FPS-Plots mit verlinkten Achsen, auch bei > 1 Mio. Punkten.
- Eine portable Windows-`.exe`.

Fahrplan und Architektur: [`docs/MASTER_ETAPPEN_PLAN.md`](docs/MASTER_ETAPPEN_PLAN.md).
Was echte Daten an Problemen mitbringen: [`docs/DATENKATALOG_RUNS_EXPORT.md`](docs/DATENKATALOG_RUNS_EXPORT.md).

---

## 2. Unverhandelbare Prinzipien

1. **Keine fest verdrahteten Konventionen.** Spaltennamen, Einheiten und Bezeichner unterscheiden sich von Anlage zu Anlage. Eine Heuristik darf Namenshinweise nutzen, aber nur über die konfigurierbare `Vocabulary` in `IngestionConfig`. Niemals `if col == "unit"` o. Ä. im Code.
2. **Automatik schlägt vor, der Nutzer entscheidet.**
   - Jede automatische Entscheidung ist über `IngestionConfig` überschreibbar. Explizite Vorgaben schlagen immer die Heuristik.
   - Jede Entscheidung landet ausgeschrieben in `DualModeDataset.ingestion_spec`. Diese Spec ist ein **Fixpunkt**: Erneut geladen ergibt sie exakt dieselbe Tabelle.
   - Neue Heuristiken erweitern `IngestionConfig` *und* die aufgelöste Spec. Der Replay-Test in `tests/test_ingestion_config.py` muss weiter grün sein.
3. **Ehrliches Audit statt stiller Korrektur.**
   - Was nicht geparst werden kann, wird protokolliert (`AuditLog`), nicht still zu `null` gemacht oder „repariert“.
   - Erzwungene, aber unpassende Vorgaben schlagen **laut** fehl (Exception), statt falsche Daten zu liefern.
   - Daten, die verworfen werden, stehen im Audit (z. B. `dropped_long_column`).
4. **Nicht destruktiv.** Keine In-place-Mutation von DataFrames. Jede Transformation ist ein nachvollziehbarer, umkehrbarer Schritt. Qualitätsprobleme werden *markiert*, nicht gelöscht.
5. **Realdaten zuerst.** Jedes Ingestion-Feature braucht einen Testfall mit dem echten Problem. Gibt es eine passende Fixture, wird sie benutzt, sonst kommt eine minimale synthetische CSV in den Test. Fixtures werden **nie** verändert.
6. **Wide und Long.** Wide (`DualModeDataset.df`) ist der primäre Modus für Mathematik und Zero-Copy-Plotting. Long gibt es auf Abruf über `to_long()`. Metadaten (`metadata_columns`) und Parameter (`parameters`) werden nie mitgeschmolzen oder geplottet. Nur `channels` sind Messkanäle.

---

## 3. Architektur in 60 Sekunden

```
datualizer_core/                  # Engine, KEINE GUI-Abhängigkeiten
├── schema.py                     # ColumnKind (TIME/NUMERIC/PARAMETER/IDENTIFIER/CATEGORICAL), importzyklusfrei
├── dataset.py                    # DualModeDataset, AuditLog
├── runs.py                       # Run-Übersicht, Kanalverfügbarkeit, Zeit pro Run
├── quality.py                    # Qualitäts-Flags als Ereignistabelle (markieren, nie ändern), inkl. Hampel-`spike`
├── timebase.py                   # Absolute Zeit: Datumsformate, Zeitzonen, Sommerzeit (über Polars, importzyklusfrei)
├── campaign/                     # Messkampagne: Campaign (JSON), CampaignSession, ClockModel (Versatz, Drift, Anker, ClockCells), Uhrenabgleich (sync.py: Zellpaare, Kreuzkorrelation), Track, Abdeckung (coverage.py), Messschritte (intervals.py), Darstellung pro Kanal (display.py), Schrittarten/benannte Fenster/Messgrössen (evaluation.py, Schritt-Tabelle)
├── calc/                         # Formelkern: expr.py (sicherer Parser, Polars + numpy), uncertain.py (UArray/Dual-Zahlen, Constant); ohne Kampagnen- und GUI-Import
├── stats/window.py               # Fenster-Statistik eines Messschritts: StatsConfig, window_stats, theil_sen (ohne Kampagnen- und GUI-Import; numpy, scipy nur für Quantile)
├── ingestion/
│   ├── config.py                 # IngestionConfig, TimeConfig, Vocabulary, LongFormatConfig, RoleConfig, RunConfig, UnitConfig, QualityConfig, MergeConfig  ← alle Stellschrauben
│   ├── pre_scanner.py            # Delimiter, Dezimal, BOM, Header, Footer
│   ├── type_inference.py         # Spaltentyp pro Spalte
│   ├── long_format.py            # Long erkennen + nach wide pivotieren
│   ├── roles.py                  # Messwert vs. Parameter (konstant pro Run)
│   ├── quality_rules.py          # QualityConfig auflösen: Namen übersetzen, Sprungschwellen vorschlagen
│   ├── steps.py                  # StepColumns/EndRule/RowSelection/CellRef: Spaltenrollen und Zeilenauswahl für Messschritte (auch im Profil)
│   ├── grid.py                   # Rohes Raster eines Blatts/einer Tabelle: Überschriften-Abgleich, Kopfzeilensuche, Zellen über Beschriftung
│   ├── units.py                  # Einheiten aus Header, Long-Attribut oder Config (P20)
│   ├── loader.py                 # CSV → DualModeDataset (orchestriert alles, baut ingestion_spec)
│   ├── sources.py                # Quelle lesen, SHA-256-Fingerabdruck
│   ├── readers/                  # Text, Excel (.xlsx/.xls), SQLite → String-Rohtabelle + source_meta
│   ├── profiles.py               # SourceProfile (JSON), detect_profile
│   ├── source_loader.py          # load_source: Reader + Profil → load_frame
│   └── merge.py                  # load_csvs: mehrere Dateien → ein Dataset (Duplikate, Snapshots)
├── pipeline/operators.py         # clean_names, drop_footer, unpivot
└── profiles/*.json               # Geräteprofile (Daten, kein Code)
datualizer_gui/                   # PySide6-App, Einstieg: python -m datualizer_gui.app
                                  # campaign_window.py: Kampagnenansicht (Quellen-Dock, Uhren-Editor, Plot auf Master-Zeit)
                                  # components/interval_bands.py + intervals_panel.py: Schritt-Bänder, Marker ○ ○, Steps-Dock
                                  # components/clock_sync_dialog.py: Dialog „Align Clock“ (Kreuzkorrelation, Journal-Paare)
                                  # components/window_inspector.py: Dock „Window“ (Fenster-Inspektor, Kennwerte, Override pro Schritt)
                                  # components/coverage_strip.py: Abdeckungsspur (Lücken schraffiert, in Pixeln gezeichnet)
                                  # components/time_axis.py: Zeitachsen (Uhrzeit, Datum + Uhrzeit, t − t0), Mausrad nur auf der Zeit
                                  # components/curves.py: schnelle Kurven (Peak-Downsampling, Clip-to-View, Cache) für alle Plots
                                  # components/step_mapping_dialog.py: Dialog „Steps from Source“ am Tabellenblatt (Rollen, Zeilen, Zellen anklicken)
tools/readme_screenshots.py       # rendert die README-Bilder (docs/img/) aus den Fixtures
tools/sa_example_campaign.py      # schreibt examples/campaign_SA_ethanol.json und das Rezept; --bilder rendert docs/img/sa_*.png (echte Daten)
examples/                         # SA-Beispielkampagne (Daten über ${DATUALIZER_SA_DIR}, nie im Repo) und Rezept rezept_katalyse_voc.json
tests/fixtures/runs_export/       # 13 echte Pipeline-Exporte (unveränderlich!)
tests/fixtures/sa_12_10/          # Ausschnitte einer echten Mehrgeräte-Messkampagne (unveränderlich!)
```

Ablauf: `load_csv(path, config=…)` → PreScanner → Zeitspalte → Typ-Inferenz (Overrides zuerst) → Casting mit Audit → Long-Erkennung und Pivot → Rollen (Parameter) → Qualitäts-Einstellungen und Einheiten auflösen → `DualModeDataset` (mit `column_kinds`, `channels`, `parameters`, `empty_columns`, `runs`, `channel_attrs`, `source_format`, `ingestion_spec`, `sources`, `quality_flags`, `units`, `time_origin`, `time_zone`).
Zeit: `time_seconds` zählt ab dem ersten gültigen Sample, `time_origin` ist dessen absoluter Zeitpunkt (`time_abs` wird daraus abgeleitet, nicht als Spalte gespeichert). Zeitzonen rechnet Polars; Python braucht unter Windows dafür das Paket `tzdata`, sonst stürzt die Ausgabe zonenbehafteter Werte ab.
Andere Formate: `load_source(path, profile=…)` liest über einen Reader (Text, Excel, SQLite) eine String-Rohtabelle und gibt sie an denselben `CSVLoader.load_frame`; gerätespezifische Einstellungen kommen aus Profilen, nie aus dem Code.
Mehrere Dateien: `load_csvs(paths, config=…)` liest jede Datei roh ein, prüft gleiches Layout, dedupliziert über den Schlüssel (spätere Datei gewinnt) und lädt die gemischte Rohtabelle einmal durch denselben Loader.

**Import-Regel:** `dataset.py` darf nichts aus `ingestion/` zur Laufzeit importieren, sonst entsteht ein Zyklus. Gemeinsame Typen gehören nach `schema.py`. Typ-Hinweise gehen über `TYPE_CHECKING`.

---

## 4. Entwicklungsumgebung (Windows)

| Aufgabe | Befehl |
|---|---|
| Tests | `.venv\Scripts\python.exe -m pytest -q` |
| GUI starten | `.venv\Scripts\python.exe -m datualizer_gui.app` |
| GUI headless prüfen | `QT_QPA_PLATFORM=offscreen` setzen |
| Ausgabe mit `°`, `³`, Umlauten | `PYTHONIOENCODING=utf-8` setzen, sonst `UnicodeEncodeError` auf der Konsole |

**Fallstricke:**
- Das lokale `.venv` wurde mit uv erstellt; `uv` ist nicht im PATH. Ein uv-venv hat kein pip, es wurde mit Zustimmung des Nutzers per `.venv\Scripts\python.exe -m ensurepip` nachgerüstet. Keine Pakete ungefragt installieren.
- **Immer `python -m pip`, nie `pip`.** Fehlt `pip.exe` in `.venv\Scripts`, startet `pip` auch bei aktivem venv das pip der globalen Store-Python und installiert in deren Nutzerordner. Die App bricht dann mit `ModuleNotFoundError` ab (so geschehen mit scipy). Abhängigkeiten nach neuen Einträgen in `pyproject.toml` mit `.venv\Scripts\python.exe -m pip install -e ".[dev]"` nachziehen. Anleitungen für den Nutzer (README, Antworten) nennen ebenfalls `python -m pip`.
- `python -m datualizer_gui.main_window` startet **nichts**. Der Einstieg ist `datualizer_gui.app`.
- CI (`.github/workflows/pytest.yml`, Ubuntu, Python 3.11) installiert per `pip install -e ".[dev]"` und testet damit auch die Paket-Metadaten.
- **pyqtgraph ≥ 0.14** ist die getestete Version (`pyproject.toml`).
- **scipy** ist Abhängigkeit (vom Nutzer freigegeben, Paket B der Fenster-Statistik), aber nur für t-/Normal-Quantile. Rechenkerne bleiben numpy.
- Kurven nie mit `p.plot(...)` bauen, sondern über `curves.add_curve` (`FastCurve`): eigenes Peak-Downsampling, das nie über eine Lücke hinweg verbindet (pyqtgraphs eigenes `setDownsampling` zog Linien über Logger-Aussetzer), Clip-to-View, 1-px-Stift, Pixmap-Cache. Die Spalten der Reduktion sind eine Bildschirmspalte der Ansicht breit und fest auf der Zeitachse verankert, nie über die abgeschnittenen Daten gespannt: Ein Nachbarpunkt vor einer langen Lücke machte sie sonst riesig (Sägezahn), und beim Verschieben zitterte die Kurve. Gitter über `curves.light_grid`, Cache-Grösse über `curves.ensure_pixmap_cache` (Qt-Standard 10 MB reicht nur für wenige Subplots, danach wird jedes Bild komplett neu gezeichnet). `getData()` liefert die *angezeigten* Daten, alle Punkte gibt `getOriginalDataset()`. `add_curve` blendet pyqtgraphs Kontextmenü *Plot Options* aus (Alpha, FFT, Downsampling, Log rufen `PlotDataItem`-Methoden auf, die `FastCurve` nicht hat → `AttributeError`); das ViewBox-Menü bleibt. Jeder Stil (`PlotStyle`: Linie, Punkte, Treppe, Linie + Punkte) läuft durch dieselbe Reduktion: Treppen werden vorher aufgefaltet, Punkte zeichnet `FastScatter` (`curve.markers`). Werte ausserhalb eines festen oder robusten Y-Bereichs markiert `add_edge_markers` am Rand, statt sie wegzulassen. Nach jeder Ansichtsänderung verwirft `drop_cached_pixmap` das gecachte Bild der Kurve: Qt verschiebt sonst beim Ziehen ein veraltetes Pixmap (Treppen erschienen zeitlich verschoben oder doppelt); `update()` allein reicht nicht.
- Mit `setXLink` verlinkte Plots gleicht pyqtgraph in Bildschirm-Pixeln ab. Unterschiedlich breite linke Achsen ergeben deshalb unterschiedliche x-Bereiche. Im Kampagnenfenster bekommen alle linken Achsen dieselbe feste Breite (`coverage_strip.left_axis_width`).
- Viele gleichartige Markierungen in Plots (Schritt-Bänder) nicht als je ein `LinearRegionItem` pro Plot bauen, sondern als ein `GraphicsObject`, das nur die sichtbaren in Pixeln zeichnet (`interval_bands.StepsItem`, Muster wie `coverage_strip.CoverageItem`). 38 Regionen × 7 Plots kosteten 64 statt 13 ms pro Frame.
- Closures, die an pyqtgraph-Objekte gehängt werden (z. B. ersetztes `raiseContextMenu`), halten diese nur per `weakref`. Ein Zyklus hält den Python-Wrapper eines von Qt gelöschten Plots am Leben, verlinkte Views rufen dann hinein: Segfault unter Linux, unter Windows nur eine ausgegebene RuntimeError.
- Plots, die bei jedem Wechsel neu aufgebaut werden (Fenster-Inspektor), brauchen `scene().setItemIndexMethod(NoIndex)`. Mit dem BSP-Index von Qt stürzte das Zeichnen nach `GraphicsLayoutWidget.clear()` ab (Segfault, Linux offscreen).
- Spalten ohne Überschrift heissen `col_<Buchstabe>` und hängen als einzige an der Position. Eine eingefügte Spalte davor verschiebt sie; die Zuordnung scheitert dann laut, statt still die falsche Spalte zu lesen.
- Wechselnde Anzeigetexte (Fadenkreuz-Werte) über `components/readout.ReadoutLabel`: Ein normales `QLabel` meldet seine Textbreite als Mindestbreite und zieht beim Hovern das ganze Fenster mit. `QToolTip.hideText()` blendet erst nach einer kurzen Verzögerung aus; Tests warten mit `QTest.qWait`.
- Nutzer-Einstellungen der GUI (Menü *Settings*) liegen in `datualizer_gui/settings.py` (`QSettings`, INI pro Nutzer). Tests leiten die Datei über `tests/conftest.py` in ein temporäres Verzeichnis um, damit sie nie die echten Einstellungen lesen oder ändern.
- Zwei Ebenen von Einstellungen im Kampagnenfenster: Was zu einem Kanal oder einer Quelle gehört (Stil, Y-Skalierung, Alias), steht als pydantic-Feld in `datualizer_core/campaign/` und damit in `campaign.json`; persönliche Vorlieben gehören in `datualizer_gui/settings.py`. Beides betrifft nur die Darstellung. Standardwerte pro Gerät kommen aus dem Profil (`SourceProfile.plot_style`), nie aus dem Code.
- Plot-Export (`plot_export`): Jedes Gerät mit logisch 96 dpi bemalen und über den Painter skalieren. pyqtgraph zeichnet Achsen in ein `QPicture`, das Qt auf einem Gerät mit anderer Auflösung noch einmal skaliert (Achsen doppelt so gross). Die PNG-dpi erst nach dem Malen setzen, Item-Caches abschalten (sonst landen Bitmaps im PDF), Stifte kosmetisch lassen und ihre Breite in Gerätepixeln angeben. Qt-Rich-Text nimmt nur ganze `px`.
- Qt gibt `str`-Enums aus `itemData`/`QAction.data()` als einfachen `str` zurück. Deshalb `.value` speichern und beim Lesen wieder in das Enum umwandeln.
- **PySide6 6.12.0** bricht die GUI-Tests mit einem Speicherfehler ab („deallocating None“). `pyproject.toml` begrenzt deshalb auf `<6.12`. Vor dem Anheben die GUI-Tests mit der neuen Version mehrfach laufen lassen.
- Ein Dialog mit eigenem `QTimer` (z. B. Debounce für Vorschauen) muss den Timer bei `closeEvent`, `accept`, `reject` und `_remove` stoppen (`self._timer.stop()`). Läuft ein Timer bei Prozessende weiter oder wird ein im Test erzeugter Dialog nicht per `deleteLater()` aufgeräumt, warnt Qt beim Pytest-Teardown („QObject: shared QObject was deleted directly“) und stürzt unter Linux mit Segfault (Exit-Code 139) bzw. unter Windows mit 0xC0000409 ab.
- Änderungen an `.github/workflows/` brauchen beim Push den OAuth-Scope `workflow` (`gh auth refresh -s workflow`).

---

## 5. Arbeitsablauf

1. **Branch von aktuellem `main`.** Nie direkt auf `main` committen.
2. **Zuerst den Test** mit dem echten Problemfall, dann die Implementierung.
3. `pytest` lokal grün, und zusätzlich bei GUI-relevanten Änderungen ein Offscreen-Smoke-Test mit einer Fixture.
4. **Docs im selben PR nachziehen:**
   - Status im Datenkatalog (✅/offen)
   - Master-Plan
   - README, falls sich die API ändert (Details mit Code-Beispielen stehen in `docs/BENUTZUNG.md`, das README bleibt ein Überblick mit Bildern)
   - README-Bilder neu rendern, wenn sich die GUI sichtbar ändert: `.venv\Scripts\python.exe tools\readme_screenshots.py`
   - diese Datei, falls sich Prinzipien ändern
5. **PR nach `main`**, CI muss grün sein. Die PR-Beschreibung erklärt *was*, *warum* und was Reviewer wissen müssen (Verhaltensänderungen, Heuristik-Grenzen, offene Punkte).

**Sprache:** Docs, Commit-Messages und PR-Texte auf Deutsch. Code, Bezeichner und Docstrings auf Englisch, passend zum Bestand.
**Ehrlich berichten:** Was nicht getestet werden konnte, steht ausdrücklich im PR. Keine Leistungsversprechen (FPS, Zeilenzahlen) ohne Messung.

---

## 6. Wo wir stehen und was als Nächstes kommt

**Fertig:**
- Etappe 1 (Tracer Bullet)
- Etappe 2 bisher:
  - Typ-Inferenz pro Spalte
  - Long-Format-Erkennung und Pivot
  - `IngestionConfig` mit replaybarer `ingestion_spec`
  - Rollen Messwert vs. Parameter (P7): `ColumnKind.PARAMETER`, `RoleConfig`
  - Leere Spalten markieren (P9, P19): `fill_ratio`, `empty_columns`
  - Run-Segmentierung (P10, P11, P18): `runs`, `channel_availability`, `select_run`, `RunConfig`, Run-Tabelle im Inspector
  - Multi-File-Merge mit Dedupe (P2, P3): `load_csvs`, `MergeConfig`, `sources`, „Add CSV (merge)“ in der GUI
  - Qualitäts-Flags (P12–P17): `quality_flags`, `flag_mask`, `with_flag_columns`, `QualityConfig`, Plot ohne Linien über Lücken
  - Einheiten (P20, Rest von P9): `units`, `unitless_columns`, `UnitConfig`, Einheit an der Plot-Achse

- Etappe 2b (Messkampagne) bisher:
  - Schritt 1, absolute Zeit: `TimeConfig` (Format, Zeitzone, Ursprung, Datumsformate), `time_origin`, `time_zone`, `time_abs`, `with_time_abs()`, DST-sichere Sekunden, `TimeConfigError`
  - Schritt 2, Reader und Profile (P21–P28, siehe [Datenkatalog SA](docs/DATENKATALOG_SA_KAMPAGNE.md)): `load_source`, Excel- und SQLite-Reader, `source_meta`, `SourceProfile` mit sechs mitgelieferten Profilen, `TimeConfig.clock_date`, Tabellenfehler als Sentinels, GUI öffnet alle Formate
  - Schritt 3, Kampagne und Uhren: `Campaign`/`CampaignSession` (campaign.json mit relativen Pfaden, Spec, Hash), `ClockModel` (Versatz, Drift, Anker), Kampagnenfenster mit eigenem x-Array pro Spur und Datumsachse
  - Nachtrag: Blatt-/Tabellenwahl und Datum für Uhrzeiten beim Öffnen (`ReaderError.choice`/`options`, `with_choices`, `needs_clock_date`, `date_in_text`, Dialog `datualizer_gui/components/source_dialog.py`)
  - Schritt 4, Abdeckung: `session.coverage()` (`SourceCoverage`, `CoverageGap` mit `time_gap`/`no_values`), Abdeckungsspur im Kampagnenfenster (Menü *Settings → Tools → Coverage Info*, gespeichert über `datualizer_gui/settings.py`)
  - Schritt 5, Messschritte: `IntervalMapping` (Spaltenrollen aus Profil oder Datenform, nie im Code), `session.intervals()` mit `problems`, manuelle Schritte (Rechtsklick „Marker ○ ○“), `WindowRule`, Bänder (Standard aus, *Settings → Tools → Interval Bands*) und Steps-Dock
  - Schritt 6 recherchiert: [`docs/RESEARCH_FENSTER_STATISTIK.md`](docs/RESEARCH_FENSTER_STATISTIK.md) mit Arbeitspaketen A–D
  - Schritt 6, Paket C: zweiseitiges `spike`-Flag (Hampel, P31) in `quality.py`; `QualityConfig.spike_window_s` (Sekunden, ohne Vorgabe 21 Abtastschritte), `spike_k`, `spike_channels`, `spike_min_scale` (ohne Vorgabe die Auflösung), alles aufgelöst in der Spec; nicht Beurteilbares (MAD = 0) in `ds.quality_notes`
  - Schritt 6, Paket A: `datualizer_core/stats/window.py` (`StatsConfig`, `WindowSpec`, `WindowResult`, `window_stats`, nur numpy) und `session.window_table(kanäle)`; Lücken und leere oder ausgeschlossene Werte werden nie überbrückt, Prüfungen und Annahmen stehen als Klartext beim Ergebnis; `Campaign.stats` in `campaign.json`
  - Schritt 6, Paket B: `n_eff`/`sem_eff` (AR(1)), Newey-West- und AR(1)-SE der Steigung, Durbin-Watson, Theil-Sen mit Kendall-Intervall (ab 1000 Punkten Teilstichprobe der Paare mit festem Seed, gemessen), Stationaritäts-Ampel (Steigungstest und Hälftenvergleich); Monte-Carlo-Test mit AR(1)-Reihen. scipy ist mit Zustimmung des Nutzers Abhängigkeit (Quantile)
  - Schritt 6, Paket D: Dock „Window“ (`components/window_inspector.py`): öffnet per Doppelklick im Steps-Dock oder Rechtsklick „Inspect step“; Ausschnitt mit Fenster, OLS- und Theil-Sen-Gerade, ausgelassenen Punkten und 2-Punkt-Stützstellen, Kennwerttabelle mit Ampel (`Check.level`, `WindowResult.level`) und Annahmen, Bild ab/auf; Fensterränder ziehbar als `WindowOverride` pro Schritt in `campaign.json`; `session.window_stats` für einen Schritt
  - Einstellungen des Kampagnenfensters, Teil 1: Darstellung pro Kanal (`ChannelDisplay`: Alias, `PlotStyle`, `YScale` auto/robust/fest) in `campaign.json`, Standardstil aus `SourceProfile.plot_style`, Dialog `components/channel_display_dialog.py`, Randmarker für Werte ausserhalb des Bereichs
  - Einstellungen des Kampagnenfensters, Teil 2: Zoom nur auf der Zeitachse und Y passend zum sichtbaren Ausschnitt (`components/time_axis.TimeViewBox`), Zeitachse als Uhrzeit, Datum + Uhrzeit oder t − t0 (`Campaign.time_reference`), Zone der Kampagne (`session.set_timezone`), Mindesthöhe der Subplots mit Scrollen
  - Einstellungen des Kampagnenfensters, Teil 3: Subplots pro Kanal, nach Einheit oder nach Quelle mit Legende, Farben nach Kanal, Quelle oder manuell (`components/plot_groups.py`, `ChannelDisplay.color`). `canvas.plot_items` ist nach Subplot geschlüsselt (bei „pro Kanal“ gleich dem Spur-Schlüssel `quelle/kanal`), `canvas.curves` und `edge_markers` immer nach Spur
  - Einstellungen des Kampagnenfensters, Teil 4: Werte unter dem Mauszeiger als Leiste, Tooltip oder Tabelle im Dock (`readout.ReadoutMode`, `ReadoutTable`), Nachkommastellen. Angezeigt wird, was die Kurve zeichnet (`FastCurve.value_at`): Treppen halten ihren Wert bis zum nächsten Punkt, sonst der nächste Messpunkt, in einer Unterbrechung „—“
  - Einstellungen des Kampagnenfensters, Teil 5: Export als PNG, SVG, PDF (`components/plot_export.py`, `canvas.export_copy()`, Dialog `components/export_dialog.py`)
  - Schritt 7 (PR 9 im Plan Auswertung): Uhrenabgleich in `campaign/sync.py`. Versatz aus Uhrenpaaren in Zellen (`ClockCells`/`CellPair` im `ClockModel.cells`, `session.clock_from_cells`, Median ± Streuung, `method="journal_pairs"`; Zellen sind Konfiguration, nie im Code), Kreuzkorrelation (`session.cross_correlate` → `XCorrResult`, maskiertes Pearson-r per FFT, Lücken nie überbrückt, Ampel in `checks`), `ClockModel.shifted`, `read_cells`/`sheet_names` im Excel-Reader, Dialog *Align Clock…* im Quellen-Dock
  - Schritt 8, PR 1 (siehe [Plan Auswertung](docs/PLAN_AUSWERTUNG_KAMPAGNE.md)): Formelkern `datualizer_core/calc/`. `compile_expr` prüft gegen eine Positivliste (nie `eval`), `Expr.to_polars()` und `Expr.evaluate()` rechnen dasselbe (NaN-Vergleiche nach IEEE auch in Polars); `UArray` pflanzt Unsicherheiten exakt linear fort (`budget`, `u_random`, `u_syst`, syst. Eingänge gleicher `group` voll korreliert); `Constant`
  - Schritt 8, PR 3: Text in Formeln nur mit `==`/`!=` (fehlender Text: beides falsch; `text_names`), `campaign/evaluation.py` mit `StepClass`, `NamedWindow` (`window_rule` = Fenster `default`; Override eines benannten Fensters unter `"<id>@<fenster>"`), `Measure`; `Campaign.evaluation`, `session.step_classes()`, `step_window()`, `step_table()`; TOST `StatsConfig.equivalence` → `WindowResult.stationary_practical`; Klassen-Spalte im Steps-Dock, Fensterauswahl im Fenster-Inspektor
  - Schritt 8, PR 2: Rechenkanäle (`DerivedChannel` mit `pieces` nach Master-Zeit), Masken (`Mask`, `CampaignSource.masks`, Flag `masked`, Schraffur im Kampagnenplot, Ausschluss in Statistik), XML-Reader für Testa-iFID `Settings*.xml` und `suggest_linear_calibration`, `SqliteOptions.json_columns` (`parameters_json` zerlegen)
  - Schritt 8, PR 4: Bezüge zwischen Schritten (`datualizer_core/calc/relations.py`): `nearest_reference` (Blindwert mit Einklammerung/Mittel, u = max(|Δ|/2/√3, u_min), einseitig, fallback; Spalten `<name>`, `u_<name>`, `<name>_quelle`, `<name>_art`, `<name>_vor`, `<name>_nach`), `cumulative` (laufende Summe in Zeitreihenfolge, Reset bei Lücke ≥ gap_s oder `reset_when` mit `changed(col)` = Wert anders als in der vorigen Dosierzeile; Beladung), `bin` (Gitter-Rundung round(x/w)*w), Pydantic `Relation` (discriminated union: NearestReferenceRelation, CumulativeRelation, BinRelation), `apply_relations` in `step_table` und `Evaluation.relations` in `campaign.json`.
  - PR 9b (Plan Auswertung): robuste Schrittzuordnung. Schritte kommen aus dem rohen Raster der Quelle (`session.step_grid`, `ingestion/grid.py`; Blattzeilen wie in Excel, bei CSV/SQLite Kopfzeile = Zeile 1), Spalten nur über die Überschrift (ohne Gross/klein, Leerzeichen, Einheit; fehlend → `IntervalConfigError` mit Vorschlag), Kopfzeile = erste Zeile mit allen Überschriften, `RowSelection` (Bereich, `stop` time_jump/first_empty_start, `where`, `skip_rows`), `EndRule.fallback`, `parameter_defaults`, `id_column`, `CellValue`/`CellPair` über die Beschriftung, `session.derive_mapping` (Zustand jeder Zeile), Änderungsbericht beim Öffnen (`IntervalMapping.resolved` → `session.warnings`), Overrides folgen ihrem Schritt (`relink_overrides`), Dialog *Steps from Source* am Blatt. Schritt-IDs ohne `id_column` bleiben `<quelle>:<Index unter der Kopfzeile>`; `origin` und `problems` nennen die Blattzeile.
  - PR 12 (Bericht): Legenden von `xy`, `aligned`, `bars` als `pg.LegendItem` im Grafik-Layout (`results_window.add_layout_legend`, damit in Export und PDF); Deckblatt-Kacheln `ReportViewSpec.tiles` (`ReportTile`, `report_tiles`), Matrix-Zeilen `MatrixViewSpec.row_label`; Vorlagen-Felder mit Aggregat `{formel|min:.0f}` (`AGGREGATES`: mean, median, min, max, sum, n)
  - PR 13 (Ansichten ohne Namen): `facet_label` (xy, aligned), `BarsViewSpec.bar_label`, Budget aus `b_<grösse>__*`, `AlignedViewSpec.sequence_col` (Pflicht bei `sequence_start`) und `sequence_steps`, Spalte `<name>_lauf` der laufenden Summe; Kanäle je Schritt aus der abdeckenden Quelle, fehlende Namen → keine Kurve; `results_window.fit_tick_text` (Tick-Texte vor dem ersten Malen einpassen), `shared_range` (verlinkte Facetten)
  - Schritt 8, PR 5: Rechengrössen, Kriterien, Szenarien (`Evaluation` in `campaign.json`: constants, calculations, criteria, ratings, scenarios, analyses, views), `session.results(scenario)` (Schritt-Tabelle + Rechengrössen mit `u_`, `u_*_zufall`, `u_*_syst`, Budget-Spalten `b_<name>__<eingang>`, Kriterien als Bool, `bewertung`, `grund`), `Constant.scope` (z. B. „tag“ für PR 6), Dock „Results“ im Kampagnenfenster (`ResultsPanel`: Ergebnistabelle mit Filter, Sortierung, CSV-Export, Spaltenauswahl; Doppelklick → Inspektor; Formeleditor mit Live-Syntaxprüfung beim Tippen, Tabellen für Konstanten, Kriterien, Bewertungsregeln, Szenario-Umschalter).
  - Schritt 8, PR 6–8: Auswertungen über Schritte (`crossing`, `fit_line`, `aggregate`, Korrelation je `scope`), Ergebnisfenster mit Ansichten, `aligned`, `bars`, PDF-Bericht.
  - Schritt 10, PR 10: SA-Kampagne als Konfiguration (`examples/campaign_SA_ethanol.json`, Generator `tools/sa_example_campaign.py`, Rezept `examples/rezept_katalyse_voc.json`, `session.apply_recipe`, Menü *Evaluation from Template…*), Orakeltest `tests/test_sa_end_to_end.py` (nur mit `DATUALIZER_SA_DIR`). Neu dafür: Profil `testa_ifid_csv`, `nan`/`inf` als Zahlen im Formelkern (`NUMBER_NAMES`), `Calculation.u_expr`/`kind`/`group`, `Measure.sources` mit `min_n`/`min_coverage` und Spalte `<alias>_quelle`, `FallbackValue` als Formel, `ParameterLookup` in `IntervalMapping.lookups`, Pfade mit `${VAR}` (bleiben beim Speichern stehen), `FitLineResult.m_zufall`/`derived_zufall`; Budget-Beitrag 0, wo ein Eingang nicht wirkt (sonst machte ein NaN-u im ungenutzten `where`-Zweig u(X) zu NaN).
  - PR 11: fehlende Berichtsseiten. Ansicht `timeline` (`TimelineViewSpec`, Panels mit Kanälen und Marken pro Schritt, Bänder, `page_by` = eine Bericht-Seite je Wert; Widget `TimelineViewWidget`, Bänder als ein `ColorBandsItem`), Zyklus-Fenster `NamedWindow.until_next`/`max_extend_s` (bis zum Start des nächsten Schritts einer Klasse), Statistik `covered_s` (u = 0) für ∫(c − B) dt = Integral − B · abgedeckte Zeit, `Measure.exclude_flags`, Vorlagen `{formel:format}` (`render_template`; Matrix-`subtext`, `band_label`, `page_title`), Matrix mit grösster Zeile oben und allen Zeilenbeschriftungen, `CampaignWindow.export_report` folgt der `report`-Ansicht. Bericht-Seiten in `painter.save()/restore()`: Der weisse Pinsel der Deckblatt-Kacheln füllte sonst alle Kurven der folgenden Seiten.

**Nächste offene Punkte:**

1. **Später zu den Einstellungen** (mit Etappe 2b, Schritt 5/6): Qualitäts-Flags als Marker, Lücken-Schwelle, Standardauswahl der Kanäle beim Hinzufügen einer Quelle.
2. **Etappe 2b, Schritt 9** (Video-Spur).
3. **Wert und Einheit in einer Zelle (P6)** (`Konzentration = "20 °C"`) und widersprüchliche Metadaten anzeigen (P8).
4. Danach: Block-Segmentation und Ragged-Healer (ursprünglicher Etappe-2-Plan), sobald Gerätedaten mit Kopfblöcken vorliegen.

Langfristig ist `IngestionConfig` die Keimzelle des **Recipe-AST** (Etappe 3) und des **Ingestion-Wizards** (Etappe 6). Neue Einstellungen deshalb sauber typisiert und serialisierbar halten.
