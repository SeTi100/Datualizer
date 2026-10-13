# Datenkatalog: Messkampagne SA (Stand 2026-10-13)

> Zweck: Anforderungen für **Etappe 2b (Messkampagne)** aus einer echten Mehrgeräte-Messung ableiten.
> Fixtures: [`tests/fixtures/sa_12_10/`](../tests/fixtures/sa_12_10/README.md). Die Nummerierung schliesst an [`DATENKATALOG_RUNS_EXPORT.md`](DATENKATALOG_RUNS_EXPORT.md) an (P1–P20).

## 1. Was die Daten fachlich sind

Katalytische Oxidation von Ethanol in einer Versuchsanlage. Pro Solltemperatur werden mehrere Dosierstufen gefahren; gemessen wird parallel mit:

| Quelle | Inhalt | Format | Uhr |
|---|---|---|---|
| FID (Testa iFID Mobile) | Kohlenstoff-Konzentration nach Katalysator, Rohsignal in pA | `.dat` (Tab) bzw. `.xls` mit Metadatenblättern | Geräteuhr, ≈ 16 min hinter der PC-Uhr |
| PicoLog | 8 Temperaturen | CSV, nur verstrichene Zeit | keine absolute Zeit |
| instrument_reader | Waage per Kamera-OCR, daraus Massenstrom | SQLite (`experiments`, `runs`, `readings`) | Lokalzeit, Metadaten in UTC |
| Video-OCR | Waage aus Aufnahmen, nachträglich ausgewertet | CSV | aus Dateiname und Dateizeit (MP4) bzw. Frame-Zeitstempeln (MKV) |
| Labjournal | Messschritte, Sollwerte, abgelesene Massen und Zeiten | Excel, ein Blatt pro Tag | PC-Uhr, nur Uhrzeit |

## 2. Pathologien → Anforderungen

| # | Befund | Anforderung | Status |
|---|---|---|---|
| P21 | Jede Quelle hat ihre eigene Uhr: FID-Versatz 16:10, PicoLog nur relativ, Waagen-Uhr im Video −69 min | Absolute Zeit pro Datensatz, Uhrenmodell pro Quelle, Abgleich | ✅ absolute Zeit (`time_origin`, `TimeConfig`), Uhrenmodell (`ClockModel`: Versatz, Drift, Anker), gemeinsame Master-Zeit; ⏳ automatischer Abgleich (Schritt 7) |
| P22 | Gemischte Zeitbasen in einer Datenbank (`started_at` UTC, `timestamp` lokal) | Zeitzone pro Zeitspalte; Zeitzonen DST-sicher | ✅ Zeitzonen über `TimeConfig.timezone`; ⏳ pro Spalte |
| P23 | Logger exportiert nur verstrichene Zeit | Start aus Speicherzeit der Datei oder Vorgabe, als Annahme markiert | ✅ `TimeConfig.origin`, Profilregel `file_mtime_end` mit Hinweis in `source_meta` |
| P24 | Logger-Aussetzer: Zeit läuft weiter, Werte leer (78 min) | Leere Zeilen behalten, nicht als Footer verwerfen | ✅ in der Mitte; ⚠ am Dateiende verwirft der Pre-Scanner sie als Footer (Warnung bei der Startzeit-Regel); ✅ als schraffierte Lücke (`no_values`) in der Abdeckungsspur |
| P25 | Excel: Kopfzeile nicht in Zeile 1, Spalten ohne Überschrift, Uhrzeiten ohne Datum, `#DIV/0!` in Vorlagezeilen | Kopfzeilen-Erkennung, Platzhalternamen, `clock_date`, Tabellenfehler als Sentinel | ✅ `read_excel`, `TimeConfig.clock_date`, Sentinels; GUI fragt Blatt und Datum ab (Vorschlag aus dem Blattnamen) |
| P26 | Geräteeinstellungen und Kalibrierung stecken in Nebenblättern | Als Metadaten mitnehmen statt verwerfen | ✅ `ExcelOptions.meta_sheets`, `ds.source_meta` |
| P27 | Gelöschte Experimente/Runs in der Datenbank | Fehlende IDs anzeigen | ✅ `source_meta["missing_ids"]` |
| P28 | Ungültige OCR-Lesungen (`is_valid = 0`) | Als leer markieren, nicht als Messwert | ✅ Profil `instrument_reader_db` |
| P29 | Gleiches Gerät, andere Anzeige-Skala an verschiedenen Tagen (Faktor 2.95 bei gleichem pA-Signal) | Rohsignal behalten, Skalen per Referenz-Fit vergleichen | ⏳ Schritt 8 |
| P30 | Messschritte stehen nur im Labjournal | Intervalltabelle aus Journal, Fenster über allen Spuren | ✅ `IntervalMapping` (Start `Zeit_0`, Ende `Zeit_1`, Blockname aus Spalte A per `fill_down`, Sollwerte), Bänder, manuelle Schritte; „clearing“- und Vorlagezeilen ohne Start stehen in `problems` |
| P31 | OCR-Fehllesungen in beide Richtungen | Zweiseitiges `spike`-Flag | ✅ Hampel-Detektor (`spike`): Zeitfenster in Sekunden, Schwelle k · 1,4826 · MAD, Auflösung als Untergrenze, alles in `QualityConfig` und Spec; im Video-OCR-Auszug genau die 33 Fehllesungen markiert (28 mit `is_valid = 0`, 5 vom Leser für gültig gehalten, z. B. 9628 statt 9672,8 g) |
| P32 | Video ohne Frame-Zeitstempel (MP4) | Zeitachse über Anker (im Bild abgelesene Uhr) | ⏳ Schritt 9 |

## 3. Grundsatz

Wie bei P1–P20: Automatik schlägt vor, Nutzer entscheidet. Gerätespezifisches steht in Profilen (`datualizer_core/profiles/*.json`), nie im Code.
