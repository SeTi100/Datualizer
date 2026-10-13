# Fixture-Korpus `sa_12_10`

Ausschnitte einer echten Messkampagne: SA, katalytische Ethanol-Oxidation, Heissmessung am 12.10.2026, Kaltmessung am 05.10.2026. Mehrere Geräte zeichneten parallel auf, jedes mit eigener Uhr und eigenem Format.
Die Dateien sind **unverändert** aus den Originalen herausgeschnitten: ganze Zeilen bzw. Tabellenzeilen, Kopfzeile behalten. Fixtures werden nie verändert.

Die Probleme der ganzen Kampagne stehen in [`docs/DATENKATALOG_SA_KAMPAGNE.md`](../../../docs/DATENKATALOG_SA_KAMPAGNE.md).

| Datei | Gerät / Quelle | Ausschnitt | Besonderheit |
|---|---|---|---|
| `fid_ifid_12_10_extract.dat` | Testa iFID Mobile, Text-Export | Datenzeilen 14050–14349 (300 Zeilen) | Tab-getrennt, deutsches Datum, Geräteuhr ≈ 16 min hinter der PC-Uhr; enthält die einzige Lücke der Datei (11 s, FID-Zeit 14:29:05–14:29:16) |
| `fid_ifid_05_10.xls` | Testa iFID Mobile, Excel-Export (BIFF) | vollständig (6512 Zeilen) | Messwerte im Blatt `Data01`, Geräteeinstellungen und Kalibrierung (Prüfgas, Sensitivität in pA/ppm) in eigenen Blättern. Vollständig, weil sich `.xls` ohne Zusatzpaket nicht kürzen lässt |
| `picolog_12_10_extract.csv` | PicoLog (Thermoelement-Logger) | Datenzeilen 25260–25419 (160 Zeilen) | Nur verstrichene Zeit `HH:MM:SS`, Dezimalkomma, Kopf mit `Letzte (C)`; Ende eines 78-min-Aussetzers (76 leere Zeilen, Zeit läuft weiter) |
| `instrument_reader_12_10_extract.db` | instrument_reader (OCR der Waage), SQLite | Schema, alle `experiments` und `runs`, `readings` nur von Run 9 und 10, `roi_presets` Nr. 7 | `runs.started_at` in UTC, `readings.timestamp` in Lokalzeit; Experiment 2 und Runs 4–7 gelöscht (Lücken in den IDs) |
| `ocr_recording_105012_extract.csv` | OCR-Auswertung eines Videos der Waage (instrument-reader offline) | Zeitstempel 10:58:00–11:08:00 | OCR-Fehllesungen nach unten (bis 9620 g statt ≈ 9672 g); Zeitachse des MP4 aus Dateiname und Dateizeit rekonstruiert |
| `labjournal_12_10_extract.xlsx` | Labjournal (Excel) | Blatt `12.10.2026`, Zeilen 1–60, Formeln als Werte | Kopfzeile erst in Zeile 10, Spalte A ohne Überschrift (Blocknamen), Uhrzeiten ohne Datum, leere Vorlagezeilen mit `#DIV/0!` |
