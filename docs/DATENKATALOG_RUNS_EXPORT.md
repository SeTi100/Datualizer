# Datenkatalog: `runs_export`-Korpus (Stand 2026-10-04)

> Zweck: Konkrete Anforderungen für **Etappe 2 (Ingestion-Festung)** aus echten Pipeline-Exporten ableiten.
> Fixtures: `tests/fixtures/runs_export/` (13 Dateien, 2,7 MB). Alle Befunde sind mit Polars nachgeprüft.

## 1. Was die Daten fachlich sind

Es sind Verdampfungs- und Konzentrationsmessungen von VOCs (Ethanol, Toluol).

- **Hierarchie:** `experiment` → `run` (Parameter-Set) → `phase_status` (`IDLE`, `STAGE_1`, `STAGE_TRANSITION`, `STAGE_2`) → Zeitreihe.
- **Rohkanäle:** `Masse` [g] von einer Waage, `Temperatur` [°C], `V_VOC`, `Volumenstrom_VOC` [Nl/h].
- **Rechenkanäle** (`is_calculated = 1`): `Massenstrom Waage (g/s)`, `Normkonzentration (g/Nm³)`, `Abgaskonzentration Betriebszustand (g/m³)`.
- **Parameter/Sollwerte** sind pro Run konstant und liegen als Spalten vor: `Temperatur`, `param_Temperatur`, `Rotameter`, `Volumenstrom`, `Konzentration`, `snad`, `snad2`. Im long-Format kommen `target_temperature` und `voc_type` dazu.

## 2. Zwei Export-Schemata

| | wide | long |
|---|---|---|
| Schlüssel | `timestamp` (eindeutig) | `(timestamp, roi_name)` (eindeutig, geprüft) |
| Metadaten | `run_id, experiment_name, phase_status` | zusätzlich `experiment_id, voc_type, target_temperature, unit, is_calculated` |
| Einheiten | nur teilweise im Spaltennamen, z. B. `(g/s)` | eigene Spalte `unit` |
| Spaltenzahl | variabel (8–17), je nach enthaltenen Runs | fix (11) |

**Der Dateiname taugt nicht zur Erkennung des Formats.** `…_long.csv` / `…_lang.csv` ist teils wide, `…115346.csv` ohne Suffix ist long. Das Format muss aus dem Header erkannt werden: Sind `roi_name` und `parsed_value` vorhanden, ist es long.

## 3. Pathologien → Anforderungen

| # | Befund (mit Fundstelle) | Anforderung an Datualizer |
|---|---|---|
| P1 | **Format-Lüge im Dateinamen** (siehe oben) | Schema-Sniffing über den Header statt über den Dateinamen |
| P2 | **Byte-identische Duplikate** (`rand_params` = `rand_params_long`, `ALL_MESSY` = `ALL_MESSY_lang`) | Hash beim Import, Warnung „Datei schon geladen“ |
| P3 | **Überlappende Snapshots**: `konst_T_V_long` = `konst_T_V` + 104 neue Zeilen; `115351_long` ⊃ `115346` | Mehrere Dateien mergen: über den Schlüssel deduplizieren, die neuere Version gewinnt, Konflikte ins Audit |
| P4 | **Text-Spalten werden zu Float gecastet**: `phase_status`, `experiment_name`, `roi_name`, `unit` landen als Null im Datensatz und erzeugen **tausende falsche Audit-Einträge** (z. B. 12.913 bei `ALL_MESSY`) | Typen pro Spalte erkennen (kategorial / numerisch / Zeit). Fehlschläge nur bei Spalten melden, die als numerisch erkannt wurden. **Das ist ein aktueller Bug in `CSVLoader`.** |
| P5 | **Numerischer Experimentname**: `experiment_name = "250"` (Run 5/6) wird zur Zahl 250.0 | IDs und Namen bleiben immer String, egal wie sie aussehen |
| P6 | **Wert mit Einheit in einer Zelle, Einheit fachlich falsch**: `Konzentration = "20 °C"` (Run 7, 306 Zeilen) | `split_column` für Wert und Einheit, außerdem ein Plausibilitäts-Flag (Einheit passt nicht zur Größe) |
| P7 | **Dieselbe Bezeichnung mit verschiedener Bedeutung**: `Temperatur` ist in Run 1 ein Sensor (25.0 °C), in `konst_T_V` ein Sollwert (100). Im Gesamtexport wurde Letzterer zu `param_Temperatur` umbenannt | Rollen pro Spalte (Messwert / Parameter / Metadatum), Heuristik: konstant pro Run → Parameter |
| P8 | **Widersprüchliche Metadaten**: Experiment „Ethanol 50°“ enthält Runs mit `target_temperature` 100 und 180. `voc_type` ist leer, obwohl der Name „Ethanol“ sagt. In Test2 gilt `target_temperature = 20`, der Sensor misst aber 25 | Metadaten nie stillschweigend „korrigieren“, nur als Inkonsistenz anzeigen |
| P9 | **Kryptische und leere Parameter**: `snad` hat 0 % Füllgrad, `snad2 = 180` ohne Einheit | Leere Spalten erkennen, Parameter ohne Einheit markieren |
| P10 | **Sparse Wide-Matrix**: Spalten existieren nur für einige Runs (Füllgrad pro Run 0 % oder 100 %) | Runs nicht über ein gemeinsames Schema zwingen, sondern Kanalverfügbarkeit pro Run anzeigen |
| P11 | **Gemischte Abtastraten**: Runs 1–6 mit 1 Hz, Runs 7–20 mit 10 Hz in einer Datei | `time_seconds` pro Run und global, kein fixes Δt annehmen (wichtig für Etappe 3, Resampling) |
| P12 | **Session-Lücken**: 2 Sprünge > 5 s (30.09. → 04.10., 11:55 → 12:23) | Lücken erkennen und im Plot nicht durchverbinden (`connect="finite"`) |
| P13 | **Sensor-Dropout als 0.0**: `Masse = 0.0` für 12 Samples zwischen 172 g (Run 1, 22:00:17–28), davor ein Leerfeld | Plausibilitätsregel „physikalisch unmöglicher Sprung → als Ausfall markieren“, *keine* echte Null |
| P14 | **Nachfüll-Sprung**: `Masse` 162,7 → 172,9 g (22:04:41) | Ereignis erkennen, Massenstrom nicht über den Sprung ableiten |
| P15 | **Eingefrorener Sensor**: `Masse` 162.704 über ~65 s konstant (Run 1) | Stuck-Value-Detektor (n gleiche Werte in Folge) |
| P16 | **Lücken in Rechenkanälen**: `Normkonzentration` und `Abgaskonzentration` sind leer, wenn sich `Masse` nicht ändert (Δm = 0 → Division) | Null aus Berechnung ≠ Messfehler, im Audit getrennt führen |
| P17 | **Physikalisch unplausible Rechenwerte**: `Massenstrom` < 0 (454 Werte, Rauschen am Run-Ende) | Toleranzband statt hartem Filter, Werte markieren und behalten |
| P18 | **Abgebrochene bzw. Mini-Runs**: Run 2 (11 Samples), Run 5 (12), Run 17 komplett ohne Rechenwerte | Mindestlänge bzw. Qualitäts-Score pro Run, „abgebrochen“-Badge im Inspector |
| P19 | **Long-Format mit Null-Werten**: Zeilen mit leerem `parsed_value` (z. B. `Volumenstrom_VOC`: 4 Zeilen, alle leer) | Beim Pivot nach wide keine Phantom-Spalten aus Kanälen ohne jeden Wert |
| P20 | **Sonderzeichen in Namen**: `°`, `³`, Klammern und Leerzeichen (`Normkonzentration (g/Nm³)`) | `clean_names` muss die Einheit *extrahieren* (→ Metadaten) statt sie einfach zu verwerfen |

## 4. Was *nicht* vorkommt

Das ist genauso wichtig für die Priorisierung:

- Encoding ist durchgehend UTF-8 ohne BOM, Trennzeichen `,`, Dezimalpunkt `.`, ISO-8601-Zeitstempel mit µs.
- Es gibt weder Metadaten-Kopfblöcke noch mehrzeilige Header, Footer-Statistiken oder abgeschnittene Zeilen.

→ **Konsequenz für den Etappenplan:** Bei diesem Datentyp liegen die Probleme nicht in der Dateistruktur (Block-Segmentation, Ragged-Rows). Sie liegen in **Semantik und Qualität**: Typen, Rollen, Duplikate, Plausibilität. Etappe 2 sollte entsprechend umgewichtet werden (siehe Abschnitt 6).

## 5. Grundsatz: Automatik schlägt vor, Nutzer entscheidet

Alle Namenshinweise in diesem Katalog (`roi_name`, `unit`, `is_calculated`, `snad` …) stammen aus *diesem* Korpus. Andere Anlagen benennen dieselben Dinge anders. Deshalb gilt für jede Heuristik:

- Ihre Vokabulare und Schwellen sind konfigurierbar (`IngestionConfig.vocabulary`, `numeric_ratio_threshold`).
- Explizite Vorgaben (`column_kinds`, `long_format.*`, `time_column`) schlagen jede Heuristik.
- Das Ergebnis trägt die aufgelöste Konfiguration (`DualModeDataset.ingestion_spec`). Sie ist ein Fixpunkt: Erneut geladen ergibt sie dieselbe Tabelle (getestet für alle Fixtures).
- Erzwungene, aber unpassende Vorgaben schlagen laut fehl (`LongFormatError`) statt still falsche Daten zu liefern.

## 6. Vorgeschlagene Umpriorisierung von Etappe 2

1. **Typ- und Rollen-Inferenz** (P4, P5, P7, P9): Behebt den akuten Audit-Bug und ist Voraussetzung für alles Weitere.
   *✅ Typ-Teil erledigt:* `ingestion/type_inference.py` unterscheidet NUMERIC, IDENTIFIER und CATEGORICAL; `DualModeDataset.channels` und `metadata_columns` sind neu. *✅ Rollen-Teil erledigt (P7):* `ingestion/roles.py` stuft numerische Spalten, die innerhalb jedes Runs konstant sind, als `ColumnKind.PARAMETER` ein (`DualModeDataset.parameters`, nicht geplottet, nicht mitgeschmolzen). Das betrifft u. a. `target_temperature` aus Long-Exporten, `Rotameter`, `Volumenstrom`, `param_Temperatur`, `snad2`. *Grenze:* `Temperatur` in Run 1 (Sensor, konstant 25.0) ist aus den Werten nicht von einem Sollwert zu unterscheiden und wird als Parameter vorgeschlagen; Korrektur per `column_kinds`. *✅ Leere Spalten erledigt (P9):* `DualModeDataset.fill_ratio` (Füllgrad pro Spalte) und `empty_columns` markieren Spalten ohne jeden Wert, z. B. `snad`, `Volumenstrom_VOC` und das Metadatum `voc_type`. Sie bleiben in `df` und behalten ihre Rolle. Der Inspector listet leere Kanäle grau und abgewählt, sie werden nicht geplottet. *Offen:* „Parameter ohne Einheit markieren“ (`snad2`) setzt die Einheiten-Extraktion aus Headern voraus (P20).
2. **Schema-Sniffer wide/long** + `pivot_wider` für long (P1, P19): Aus Etappe 3 vorziehen, als Minimalversion.
   *✅ Erledigt:* `ingestion/long_format.py` erkennt Long-Tabellen am Header und an der Datenform und pivotiert sie nach wide. `unit` und `is_calculated` landen in `DualModeDataset.channel_attrs`, doppelte Schlüssel im Audit. Wide- und Long-Export derselben Messung ergeben nachweislich dieselbe Tabelle. *✅ P19:* Kanäle ganz ohne Werte bleiben wie im Wide-Format als leere Spalte erhalten und stehen in `empty_columns`; der Plot zeigt sie standardmäßig nicht an.
3. **Run-Segmentierung** (`run_id` als erstklassige Dimension, P10, P11, P18): Der Inspector zeigt Runs statt einer flachen Tabelle.
4. **Multi-File-Merge mit Dedupe** (P2, P3).
5. **Qualitäts-Flags** (P12–P17) als separate Flag-Spalten, keine destruktive Bereinigung. Das passt zum Nicht-destruktiv-Prinzip.
6. Block-Segmentation und Ragged-Healer (ursprünglicher Plan) **danach**, sobald Gerätedaten mit Kopfblöcken vorliegen.
