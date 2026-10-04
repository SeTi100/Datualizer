# Fixture-Korpus `runs_export`

Exporte aus der echten Mess-Pipeline (Signale per Simulation erzeugt, Export-Code produktiv).
Die Dateien werden **unverändert** abgelegt. Dass Inhalt und Dateiname nicht zusammenpassen, ist bereits einer der Testfälle.

Die vollständige Analyse der Pathologien steht in [`docs/DATENKATALOG_RUNS_EXPORT.md`](../../../docs/DATENKATALOG_RUNS_EXPORT.md).

| Datei | Format (tatsächlich) | Zeilen | Runs | Besonderheit |
|---|---|---:|---|---|
| `…115339.csv` | wide | 244 | 2–4 | Rechenkanäle erst ab Run 3/4 befüllt |
| `…115346.csv` | **long** (trotz fehlendem Suffix) | 520 | 2–4 | Teil-Snapshot von `…115351_long` |
| `…115351_long.csv` | long | 556 | 2–4 | Obermenge von `…115346` |
| `…115422.csv` | wide | 536 | 1 | `Masse` = 0.0 (Sensor-Dropout), Nachfüll-Sprung +10 g |
| `…115428_long.csv` | long | 1595 | 1 | wie `…115422`, aber long |
| `…115512.csv` | wide | 23 | 5–6 | Mini-Export, `experiment_name = 250` (numerisch!) |
| `…115519_long.csv` | long | 156 | 5–6 | wie `…115512`, aber long |
| `…123102_konst_T_V.csv` | wide | 2488 | 15–20 | 10 Hz, Parameterspalten |
| `…123115_konst_T_V_long.csv` | **wide** (trotz `_long`) | 2592 | 15–20 | Obermenge von `…123102` (+104 Zeilen) |
| `…123129_rand_params.csv` | wide | 1861 | 8–14 | Spalte `snad2`, wechselnde Parameter-Sets |
| `…123138_rand_params_long.csv` | **wide**, byte-identisch zu `…123129` | 1861 | 8–14 | Duplikat |
| `…123307_ALL_MESSY.csv` | wide | 6328 | 1–20 | Gesamtexport, 17 Spalten, 1 Hz und 10 Hz gemischt |
| `…123316_ALL_MESSY_lang.csv` | **wide**, byte-identisch zu `…123307` | 6328 | 1–20 | Duplikat |
