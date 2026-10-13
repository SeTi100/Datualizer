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
├── campaign/                     # Messkampagne: Campaign (JSON), CampaignSession, ClockModel (Versatz, Drift, Anker), Track, Abdeckung (coverage.py), Messschritte (intervals.py), Darstellung pro Kanal (display.py)
├── ingestion/
│   ├── config.py                 # IngestionConfig, TimeConfig, Vocabulary, LongFormatConfig, RoleConfig, RunConfig, UnitConfig, QualityConfig, MergeConfig  ← alle Stellschrauben
│   ├── pre_scanner.py            # Delimiter, Dezimal, BOM, Header, Footer
│   ├── type_inference.py         # Spaltentyp pro Spalte
│   ├── long_format.py            # Long erkennen + nach wide pivotieren
│   ├── roles.py                  # Messwert vs. Parameter (konstant pro Run)
│   ├── quality_rules.py          # QualityConfig auflösen: Namen übersetzen, Sprungschwellen vorschlagen
│   ├── steps.py                  # StepColumns/EndRule: Spaltenrollen für Messschritte (auch im Profil)
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
                                  # components/coverage_strip.py: Abdeckungsspur (Lücken schraffiert, in Pixeln gezeichnet)
                                  # components/curves.py: schnelle Kurven (Peak-Downsampling, Clip-to-View, Cache) für alle Plots
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
- Das lokale `.venv` wurde mit uv erstellt; `uv` ist nicht im PATH. pip wurde mit Zustimmung des Nutzers per `python -m ensurepip` nachgerüstet. Keine Pakete ungefragt installieren.
- `python -m datualizer_gui.main_window` startet **nichts**. Der Einstieg ist `datualizer_gui.app`.
- CI (`.github/workflows/pytest.yml`, Ubuntu, Python 3.11) installiert per `pip install -e ".[dev]"` und testet damit auch die Paket-Metadaten.
- **pyqtgraph ≥ 0.14** ist die getestete Version (`pyproject.toml`).
- Kurven nie mit `p.plot(...)` bauen, sondern über `curves.add_curve` (`FastCurve`): eigenes Peak-Downsampling, das nie über eine Lücke hinweg verbindet (pyqtgraphs eigenes `setDownsampling` zog Linien über Logger-Aussetzer), Clip-to-View, 1-px-Stift, Pixmap-Cache. Gitter über `curves.light_grid`, Cache-Grösse über `curves.ensure_pixmap_cache` (Qt-Standard 10 MB reicht nur für wenige Subplots, danach wird jedes Bild komplett neu gezeichnet). `getData()` liefert die *angezeigten* Daten, alle Punkte gibt `getOriginalDataset()`. `add_curve` blendet pyqtgraphs Kontextmenü *Plot Options* aus (Alpha, FFT, Downsampling, Log rufen `PlotDataItem`-Methoden auf, die `FastCurve` nicht hat → `AttributeError`); das ViewBox-Menü bleibt. Jeder Stil (`PlotStyle`: Linie, Punkte, Treppe, Linie + Punkte) läuft durch dieselbe Reduktion: Treppen werden vorher aufgefaltet, Punkte zeichnet `FastScatter` (`curve.markers`). Werte ausserhalb eines festen oder robusten Y-Bereichs markiert `add_edge_markers` am Rand, statt sie wegzulassen.
- Mit `setXLink` verlinkte Plots gleicht pyqtgraph in Bildschirm-Pixeln ab. Unterschiedlich breite linke Achsen ergeben deshalb unterschiedliche x-Bereiche. Im Kampagnenfenster bekommen alle linken Achsen dieselbe feste Breite (`coverage_strip.left_axis_width`).
- Viele gleichartige Markierungen in Plots (Schritt-Bänder) nicht als je ein `LinearRegionItem` pro Plot bauen, sondern als ein `GraphicsObject`, das nur die sichtbaren in Pixeln zeichnet (`interval_bands.StepsItem`, Muster wie `coverage_strip.CoverageItem`). 38 Regionen × 7 Plots kosteten 64 statt 13 ms pro Frame.
- Closures, die an pyqtgraph-Objekte gehängt werden (z. B. ersetztes `raiseContextMenu`), halten diese nur per `weakref`. Ein Zyklus hält den Python-Wrapper eines von Qt gelöschten Plots am Leben, verlinkte Views rufen dann hinein: Segfault unter Linux, unter Windows nur eine ausgegebene RuntimeError.
- Wechselnde Anzeigetexte (Fadenkreuz-Werte) über `components/readout.ReadoutLabel`: Ein normales `QLabel` meldet seine Textbreite als Mindestbreite und zieht beim Hovern das ganze Fenster mit.
- Nutzer-Einstellungen der GUI (Menü *Settings*) liegen in `datualizer_gui/settings.py` (`QSettings`, INI pro Nutzer). Tests leiten die Datei über `tests/conftest.py` in ein temporäres Verzeichnis um, damit sie nie die echten Einstellungen lesen oder ändern.
- Zwei Ebenen von Einstellungen im Kampagnenfenster: Was zu einem Kanal oder einer Quelle gehört (Stil, Y-Skalierung, Alias), steht als pydantic-Feld in `datualizer_core/campaign/` und damit in `campaign.json`; persönliche Vorlieben gehören in `datualizer_gui/settings.py`. Beides betrifft nur die Darstellung. Standardwerte pro Gerät kommen aus dem Profil (`SourceProfile.plot_style`), nie aus dem Code.
- Qt gibt `str`-Enums aus `itemData`/`QAction.data()` als einfachen `str` zurück. Deshalb `.value` speichern und beim Lesen wieder in das Enum umwandeln.
- **PySide6 6.12.0** bricht die GUI-Tests mit einem Speicherfehler ab („deallocating None“). `pyproject.toml` begrenzt deshalb auf `<6.12`. Vor dem Anheben die GUI-Tests mit der neuen Version mehrfach laufen lassen.
- Änderungen an `.github/workflows/` brauchen beim Push den OAuth-Scope `workflow` (`gh auth refresh -s workflow`).

---

## 5. Arbeitsablauf

1. **Branch von aktuellem `main`.** Nie direkt auf `main` committen.
2. **Zuerst den Test** mit dem echten Problemfall, dann die Implementierung.
3. `pytest` lokal grün, und zusätzlich bei GUI-relevanten Änderungen ein Offscreen-Smoke-Test mit einer Fixture.
4. **Docs im selben PR nachziehen:**
   - Status im Datenkatalog (✅/offen)
   - Master-Plan
   - README, falls sich die API ändert
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
  - Einstellungen des Kampagnenfensters, Teil 1: Darstellung pro Kanal (`ChannelDisplay`: Alias, `PlotStyle`, `YScale` auto/robust/fest) in `campaign.json`, Standardstil aus `SourceProfile.plot_style`, Dialog `components/channel_display_dialog.py`, Randmarker für Werte ausserhalb des Bereichs

**Nächste offene Punkte:**

1. **Einstellungen des Kampagnenfensters, Punkte 4–8** (Zoom, Gruppierung und Farben, Infozeile, Zeitachse, Export; siehe Masterplan unter Schritt 4).
2. **Etappe 2b, Schritte 6–10** (siehe [Masterplan](docs/MASTER_ETAPPEN_PLAN.md)): Fenster-Statistik (Pakete A, B, D laut Recherche; `spike` ist fertig), Abgleich-Werkzeuge, Rechengrössen und Szenarien, Video-Spur.
3. **Wert und Einheit in einer Zelle (P6)** (`Konzentration = "20 °C"`) und widersprüchliche Metadaten anzeigen (P8).
4. Danach: Block-Segmentation und Ragged-Healer (ursprünglicher Etappe-2-Plan), sobald Gerätedaten mit Kopfblöcken vorliegen.

Langfristig ist `IngestionConfig` die Keimzelle des **Recipe-AST** (Etappe 3) und des **Ingestion-Wizards** (Etappe 6). Neue Einstellungen deshalb sauber typisiert und serialisierbar halten.
