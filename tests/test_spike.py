"""Etappe 2b, Schritt 6, Paket C: zweiseitiges `spike`-Flag (Hampel, Datenkatalog P31).

Markieren, nie verändern. Fenster in Sekunden, Schwelle k · 1,4826 · MAD, MAD = 0 wird gemeldet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from datualizer_core import (
    DualModeDataset,
    IngestionConfig,
    QualityConfig,
    QualityConfigError,
    load_csv,
    load_source,
)
from datualizer_core.quality import suggest_resolution

SA = Path(__file__).parent / "fixtures" / "sa_12_10"
OCR = SA / "ocr_recording_105012_extract.csv"
ALL_MESSY = Path(__file__).parent / "fixtures" / "runs_export" / "runs_export_20261004_123307_ALL_MESSY.csv"

# Fehllesungen, die der OCR-Leser selbst erkannt hat (is_valid = 0) ...
OCR_INVALID = [12, 41, 42, 47, 48, 56, 76, 124, 126, 128, 129, 131, 135, 146, 153, 163, 167, 173,
               257, 258, 259, 280, 283, 292, 293, 294, 296, 297]
# ... und solche, die er für gültig hielt: 962.8 → 9628, 962.4 → 9624 (zweimal), 962.2, 961.7
OCR_VALID_MISREADS = [118, 154, 155, 172, 272]


def _rows(ds: DualModeDataset, flag: str, channel: str) -> list[int]:
    ev = ds.quality_flags.filter((pl.col("flag") == flag) & (pl.col("channel") == channel))
    return ev["row"].to_list()


@pytest.fixture(scope="module")
def ocr() -> DualModeDataset:
    return load_source(OCR)


class TestOcrFixture:
    def test_misreads_are_spikes(self, ocr: DualModeDataset) -> None:
        """P31: Fehllesungen in beide Richtungen (bis 9617 g und 67405 g statt ≈ 9672 g)."""
        assert ocr["is_valid"].to_list().count(0.0) == len(OCR_INVALID)  # Fixture wie erwartet
        spikes = set(_rows(ocr, "spike", "mass_g"))
        assert set(OCR_INVALID) <= spikes
        assert set(OCR_VALID_MISREADS) <= spikes

    def test_normal_values_are_not_spikes(self, ocr: DualModeDataset) -> None:
        spikes = set(_rows(ocr, "spike", "mass_g"))
        assert spikes == set(OCR_INVALID) | set(OCR_VALID_MISREADS)
        normal = ocr.df.with_row_index().filter((pl.col("mass_g") - 9672.5).abs() < 2.5)
        assert not spikes & set(normal["index"].to_list())

    def test_data_unchanged(self, ocr: DualModeDataset) -> None:
        plain = load_source(OCR, config=IngestionConfig(quality=QualityConfig(detect=False)))
        assert ocr["mass_g"].equals(plain["mass_g"])
        assert ocr["mass_g"][259] == 67195.0

    def test_flag_mask(self, ocr: DualModeDataset) -> None:
        mask = ocr.flag_mask(["spike"], channel="mass_g")
        assert mask.sum() == len(OCR_INVALID) + len(OCR_VALID_MISREADS)

    def test_window_in_spec_follows_sample_rate(self, ocr: DualModeDataset) -> None:
        """Ohne Vorgabe: 21 typische Abtastschritte (hier 2,008 s → 42,2 s), ausgeschrieben in der Spec."""
        q = ocr.ingestion_spec.quality
        assert q.spike_window_s == pytest.approx(21 * 2.008)
        assert q.spike_k == 3.0
        assert q.spike_channels == ocr.channels

    def test_resolution_is_the_scale_floor(self, ocr: DualModeDataset) -> None:
        """Waage zeigt 0,01-g-Schritte: MAD im Fenster oft 0 oder 0,005 g, die Auflösung ist die Untergrenze."""
        floor = ocr.ingestion_spec.quality.spike_min_scale
        assert floor == {"seconds_from_start": 2.0, "mass_g": 0.01, "is_valid": 1.0}
        assert ocr.quality_notes.is_empty()

    def test_without_floor_quantisation_misleads(self) -> None:
        """Ohne Untergrenze wird 9674,10 neben 9674,08 zum Spike (MAD = 0,005 g) und MAD = 0 gemeldet."""
        cfg = IngestionConfig(quality=QualityConfig(spike_min_scale={"mass_g": 0, "is_valid": 0}))
        ds = load_source(OCR, config=cfg)
        assert 1 in _rows(ds, "spike", "mass_g")
        assert ds.quality_notes.filter(pl.col("channel") == "is_valid")["note"].to_list() == ["spike_mad_zero"]

    def test_spec_replays(self, ocr: DualModeDataset) -> None:
        spec = IngestionConfig.model_validate_json(ocr.ingestion_spec.model_dump_json())
        again = load_source(OCR, config=spec)
        assert again.ingestion_spec == spec
        assert again.quality_flags.equals(ocr.quality_flags)


class TestOverrides:
    def test_window_zero_switches_off(self) -> None:
        ds = load_source(OCR, config=IngestionConfig(quality=QualityConfig(spike_window_s=0)))
        assert not _rows(ds, "spike", "mass_g")
        assert ds.ingestion_spec.quality.spike_window_s == 0

    def test_channel_filter(self) -> None:
        ds = load_source(OCR, config=IngestionConfig(quality=QualityConfig(spike_channels=[])))
        assert ds.quality_flags.filter(pl.col("flag") == "spike").is_empty()
        ds = load_source(OCR, config=IngestionConfig(quality=QualityConfig(spike_channels=["mass_g"])))
        assert ds.ingestion_spec.quality.spike_channels == ["mass_g"]
        assert set(ds.quality_flags.filter(pl.col("flag") == "spike")["channel"]) == {"mass_g"}

    def test_larger_k_flags_less(self, ocr: DualModeDataset) -> None:
        ds = load_source(OCR, config=IngestionConfig(quality=QualityConfig(spike_k=1e6)))
        rows = _rows(ds, "spike", "mass_g")
        assert set(rows) < set(_rows(ocr, "spike", "mass_g"))
        assert 259 in rows  # 67405 statt 9671.95 bleibt auch bei riesigem k ein Ausreisser

    def test_unknown_channel_fails_loudly(self) -> None:
        cfg = IngestionConfig(quality=QualityConfig(spike_channels=["gibtsnicht"]))
        with pytest.raises(QualityConfigError, match="spike_channels"):
            load_source(OCR, config=cfg)
        cfg = IngestionConfig(quality=QualityConfig(spike_min_scale={"gibtsnicht": 0.01}))
        with pytest.raises(QualityConfigError, match="spike_min_scale"):
            load_source(OCR, config=cfg)

    def test_invalid_k_rejected(self) -> None:
        with pytest.raises(ValueError):
            QualityConfig(spike_k=0)


class TestMadZero:
    TEXT = "time,x\n" + "".join(f"00:00:{s:02d},{5.0 if s == 10 else 1.0}\n" for s in range(21))

    def test_constant_window_marks_nothing_and_reports(self) -> None:
        """Quantisierte Werte (Waage): MAD = 0 und keine Untergrenze → nichts markieren, aber melden."""
        ds = load_csv(self.TEXT, config=IngestionConfig(quality=QualityConfig(spike_min_scale={"x": 0})))
        assert not _rows(ds, "spike", "x")
        notes = ds.quality_notes.filter(pl.col("channel") == "x")
        assert notes.to_dicts() == [{"channel": "x", "note": "spike_mad_zero", "n": 1}]

    def test_min_scale_gives_a_floor(self) -> None:
        """Mit Geräteauflösung als Mindest-Skala wird der Ausreisser beurteilt."""
        cfg = IngestionConfig(quality=QualityConfig(spike_min_scale={"x": 0.1}))
        ds = load_csv(self.TEXT, config=cfg)
        assert _rows(ds, "spike", "x") == [10]
        assert ds.quality_notes.filter(pl.col("channel") == "x").is_empty()
        assert ds.ingestion_spec.quality.spike_min_scale == {"x": 0.1}

    def test_default_floor_is_the_step_between_values(self) -> None:
        """Nur zwei Werte: die geschätzte Auflösung ist der Ausreisser selbst. Kein Spike, aber in der Spec."""
        ds = load_csv(self.TEXT)
        assert ds.ingestion_spec.quality.spike_min_scale == {"x": 4.0}
        assert not _rows(ds, "spike", "x")
        assert ds.quality_notes.is_empty()

    def test_min_scale_suppresses_resolution_steps(self) -> None:
        text = "time,x\n" + "".join(f"00:00:{s:02d},{1.01 if s == 10 else 1.0}\n" for s in range(21))
        cfg = IngestionConfig(quality=QualityConfig(spike_min_scale={"x": 0.01}))
        ds = load_csv(text, config=cfg)
        assert not _rows(ds, "spike", "x")


class TestSynthetic:
    def test_two_sided(self) -> None:
        values = [10.0 + 0.1 * ((s * 7) % 5) for s in range(40)]
        values[10], values[25] = 30.0, -10.0
        text = "time,x\n" + "".join(f"00:00:{s:02d},{v}\n" for s, v in enumerate(values))
        assert _rows(load_csv(text), "spike", "x") == [10, 25]

    def test_level_change_is_not_a_spike(self) -> None:
        """Ein bleibender Sprung (Nachfüllen) ist `jump`, kein `spike`."""
        values = [10.0 + 0.1 * ((s * 7) % 5) for s in range(20)] + [50.0 + 0.1 * ((s * 7) % 5) for s in range(20)]
        text = "time,x\n" + "".join(f"00:00:{s:02d},{v}\n" for s, v in enumerate(values))
        assert not _rows(load_csv(text), "spike", "x")

    def test_window_in_seconds_not_samples(self) -> None:
        """Gleiche Zeitspanne, doppelte Abtastrate: das Fenster bleibt in Sekunden gleich breit."""
        cfg = IngestionConfig(quality=QualityConfig(spike_window_s=10))
        fast = "time,x\n" + "".join(f"{s * 0.5},{30.0 if s == 40 else 10.0 + 0.1 * (s % 3)}\n" for s in range(80))
        ds = load_csv(fast, config=cfg)
        assert _rows(ds, "spike", "x") == [40]
        assert ds.ingestion_spec.quality.spike_window_s == 10


def test_resolution_ignores_non_finite_values() -> None:
    df = pl.DataFrame({"x": [1.0, 1.5, float("inf"), float("nan"), None, 2.5]})
    assert suggest_resolution(df, "x") == 0.5
    assert suggest_resolution(pl.DataFrame({"x": [3.0, 3.0]}), "x") == 0.0


def test_messy_fixture_still_replays() -> None:
    first = load_csv(ALL_MESSY)
    again = load_csv(ALL_MESSY, config=first.ingestion_spec)
    assert again.ingestion_spec == first.ingestion_spec
    assert again.quality_flags.equals(first.quality_flags)
