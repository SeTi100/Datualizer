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
├── ingestion/
│   ├── config.py                 # IngestionConfig, Vocabulary, LongFormatConfig, RoleConfig, RunConfig  ← alle Stellschrauben
│   ├── pre_scanner.py            # Delimiter, Dezimal, BOM, Header, Footer
│   ├── type_inference.py         # Spaltentyp pro Spalte
│   ├── long_format.py            # Long erkennen + nach wide pivotieren
│   ├── roles.py                  # Messwert vs. Parameter (konstant pro Run)
│   ├── loader.py                 # CSV → DualModeDataset (orchestriert alles, baut ingestion_spec)
│   ├── sources.py                # Quelle lesen, SHA-256-Fingerabdruck
│   └── merge.py                  # load_csvs: mehrere Dateien → ein Dataset (Duplikate, Snapshots)
└── pipeline/operators.py         # clean_names, drop_footer, unpivot
datualizer_gui/                   # PySide6-App, Einstieg: python -m datualizer_gui.app
tests/fixtures/runs_export/       # 13 echte Pipeline-Exporte (unveränderlich!)
```

Ablauf: `load_csv(path, config=…)` → PreScanner → Zeitspalte → Typ-Inferenz (Overrides zuerst) → Casting mit Audit → Long-Erkennung und Pivot → Rollen (Parameter) → `DualModeDataset` (mit `column_kinds`, `channels`, `parameters`, `empty_columns`, `runs`, `channel_attrs`, `source_format`, `ingestion_spec`, `sources`).
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
- Das lokale `.venv` wurde mit uv erstellt und hat **kein pip**. `uv` ist nicht im PATH. Keine Pakete ungefragt installieren.
- `python -m datualizer_gui.main_window` startet **nichts**. Der Einstieg ist `datualizer_gui.app`.
- CI (`.github/workflows/pytest.yml`, Ubuntu, Python 3.11) installiert per `pip install -e ".[dev]"` und testet damit auch die Paket-Metadaten.
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

**Nächste offene Punkte** (Reihenfolge nach [Datenkatalog §6](docs/DATENKATALOG_RUNS_EXPORT.md)):

1. **Qualitäts-Flags (P12–P17)** als separate Flag-Spalten: Lücken, Dropout als 0.0, Nachfüll-Sprünge, eingefrorene Sensoren, unplausible Werte.
2. **Einheiten aus Headern extrahieren (P20)** nach `channel_attrs`, danach Parameter ohne Einheit markieren (Rest von P9).
3. Danach: Block-Segmentation und Ragged-Healer (ursprünglicher Etappe-2-Plan), sobald Gerätedaten mit Kopfblöcken vorliegen.

Langfristig ist `IngestionConfig` die Keimzelle des **Recipe-AST** (Etappe 3) und des **Ingestion-Wizards** (Etappe 6). Neue Einstellungen deshalb sauber typisiert und serialisierbar halten.
