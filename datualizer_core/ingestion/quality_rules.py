"""Resolve `QualityConfig` against a loaded table (Datenkatalog P12–P17).

User-given channel names are translated to loaded names, unknown names fail loudly, and every
automatic decision (jump thresholds, calculated channels) is written out for the spec.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import polars as pl

from datualizer_core.ingestion.config import QualityConfig, Vocabulary
from datualizer_core.quality import QualitySettings, suggest_jump_threshold


class QualityConfigError(ValueError):
    """A quality setting names a column that is not a measurement channel."""


def resolve_quality(
    cfg: QualityConfig,
    df: pl.DataFrame,
    channels: Sequence[str],
    run_columns: Sequence[str],
    channel_attrs: Mapping[str, Mapping[str, Any]],
    names: Mapping[str, str],
    vocabulary: Vocabulary,
) -> QualityConfig:
    """Return a copy of `cfg` with loaded channel names and all automatic values filled in."""
    resolved = cfg.model_copy(deep=True)
    if not cfg.detect:
        return resolved

    def channel(name: str, setting: str) -> str:
        loaded = names.get(name, name)
        if loaded not in channels:
            raise QualityConfigError(
                f"quality.{setting} names '{name}', which is not a measurement channel {list(channels)}."
            )
        return loaded

    explicit = {channel(n, "jump_thresholds"): v for n, v in cfg.jump_thresholds.items()}
    resolved.jump_thresholds = {
        ch: explicit[ch] if ch in explicit else suggest_jump_threshold(df, ch, run_columns, cfg.jump_factor)
        for ch in channels
    }
    resolved.ranges = {channel(n, "ranges"): r for n, r in cfg.ranges.items()}
    if cfg.calculated_channels is not None:
        resolved.calculated_channels = [channel(n, "calculated_channels") for n in cfg.calculated_channels]
    else:
        tokens = [t.lower() for t in vocabulary.calculated_tokens]
        resolved.calculated_channels = [
            ch for ch in channels
            if any(
                any(t in key.lower() for t in tokens) and _truthy(value)
                for key, value in channel_attrs.get(ch, {}).items()
            )
        ]
    return resolved


def to_settings(cfg: QualityConfig) -> QualitySettings | None:
    """Convert a resolved config into the detector settings (None = detection off)."""
    if not cfg.detect:
        return None
    return QualitySettings(
        gap_factor=cfg.gap_factor,
        stuck_min_duration_s=cfg.stuck_min_duration_s,
        stuck_min_samples=cfg.stuck_min_samples,
        jump_thresholds=dict(cfg.jump_thresholds),
        dropout_max_duration_s=cfg.dropout_max_duration_s,
        ranges=dict(cfg.ranges),
        calculated_channels=tuple(cfg.calculated_channels or ()),
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "1.0", "true", "yes", "ja", "y", "x"}
    return bool(value) if value is not None else False
