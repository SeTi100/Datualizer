# Fixture-Korpus `sa_12_10`

Ausschnitte einer echten Messkampagne: SA, katalytische Ethanol-Oxidation, Heissmessung am 12.10.2026, Kaltmessung am 05.10.2026. Mehrere Geräte zeichneten parallel auf, jedes mit eigener Uhr und eigenem Format.
Die Dateien sind **unverändert** aus den Originalen herausgeschnitten: ganze Zeilen bzw. Tabellenzeilen, Kopfzeile behalten. Fixtures werden nie verändert.

Die Probleme der ganzen Kampagne stehen in [`docs/DATENKATALOG_SA_KAMPAGNE.md`](../../../docs/DATENKATALOG_SA_KAMPAGNE.md).

| Datei | Gerät / Quelle | Ausschnitt | Besonderheit |
|---|---|---|---|
| `fid_ifid_12_10_extract.dat` | Testa iFID Mobile, Text-Export | Datenzeilen 14050–14349 (300 Zeilen) | Tab-getrennt, deutsches Datum, Geräteuhr ≈ 16 min hinter der PC-Uhr; enthält die einzige Lücke der Datei (11 s, FID-Zeit 14:29:05–14:29:16) |
| `fid_ifid_17_10_extract.csv` | Testa iFID Mobile, CSV-Export (ab 15.10.) | Kopfzeile und Datenzeilen 1–30 von `data-11-24-51.csv` | Semikolon, jeder Wert in Anführungszeichen, sonst gleiche Spalten wie `.dat`; Geräteuhr ≈ 16 min hinter der PC-Uhr |
| `fid_ifid_05_10.xls` | Testa iFID Mobile, Excel-Export (BIFF) | vollständig (6512 Zeilen) | Messwerte im Blatt `Data01`, Geräteeinstellungen und Kalibrierung (Prüfgas, Sensitivität in pA/ppm) in eigenen Blättern. Vollständig, weil sich `.xls` ohne Zusatzpaket nicht kürzen lässt |
| `picolog_12_10_extract.csv` | PicoLog (Thermoelement-Logger) | Datenzeilen 25260–25419 (160 Zeilen) | Nur verstrichene Zeit `HH:MM:SS`, Dezimalkomma, Kopf mit `Letzte (C)`; Ende eines 78-min-Aussetzers (76 leere Zeilen, Zeit läuft weiter) |
| `instrument_reader_12_10_extract.db` | instrument_reader (OCR der Waage), SQLite | Schema, alle `experiments` und `runs`, `readings` nur von Run 9 und 10, `roi_presets` Nr. 7 | `runs.started_at` in UTC, `readings.timestamp` in Lokalzeit; Experiment 2 und Runs 4–7 gelöscht (Lücken in den IDs) |
| `ocr_recording_105012_extract.csv` | OCR-Auswertung eines Videos der Waage (instrument-reader offline) | Zeitstempel 10:58:00–11:08:00 | OCR-Fehllesungen nach unten (bis 9620 g statt ≈ 9672 g); Zeitachse des MP4 aus Dateiname und Dateizeit rekonstruiert |
| `labjournal_17_10_clock_pairs.xlsx` | Labjournal (Excel) | Blatt `17.10.2026`, nur Zellen B2:D4 (Werte und Zahlenformat übernommen) | Uhrenpaare: „Start FID“ (C3, D3) auf der FID-Uhr, „Start Real“ (C4, D4) auf der PC-Uhr, Uhrzeiten ohne Datum; Versätze 975 s und 974 s |
| `labjournal_12_10_extract.xlsx` | Labjournal (Excel) | Blatt `12.10.2026`, Zeilen 1–60, Formeln als Werte | Kopfzeile erst in Zeile 10, Spalte A ohne Überschrift (Blocknamen), Uhrzeiten ohne Datum, leere Vorlagezeilen mit `#DIV/0!` |
| `labjournal_17_10_extract.xlsx` | Labjournal (Excel), Stand 19.10.2026 | Blatt `17.10.2026`, Spalten A–J, Zeilen 1–62, nur Werte (Uhrzeitformat übernommen) | Neue Spalte C „Index“ (Daten eine Spalte weiter rechts als vor dem 19.10.), Uhrenpaare jetzt D3/E3 und D4/E4, „Baseline mgC“ in C5 mit dem Wert in E5 (D5 leer); Kopfzeile 10, Überschrift `V_MFC [Nl/min]` mit Einheit; Zeilen 11–30 die 20 Schritte des 17.10. (9 Dosierungen), ab Zeile 31 Schritte des 15.10. mit früheren Uhrzeiten; Endzeit fehlt in Zeilen 11, 12, 21, 31, 35, 39 |
