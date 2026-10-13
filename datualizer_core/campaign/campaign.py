"""Measurement campaign: several sources on one master time axis (Etappe 2b, step 3).

`Campaign` is the serializable description (`campaign.json`): which files, read how (profile
and resolved ingestion spec), and how each clock maps to master time. `CampaignSession` holds
the loaded datasets and computes master times. The JSON is the seed of the recipe AST (Etappe 3).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from datualizer_core.campaign.clock import ClockModel, epoch_seconds
from datualizer_core.dataset import DualModeDataset
from datualizer_core.ingestion.config import IngestionConfig
from datualizer_core.ingestion.profiles import SourceProfile, detect_profile
from datualizer_core.ingestion.source_loader import load_source
from datualizer_core.timebase import check_zone


class CampaignSource(BaseModel):
    """One file of a campaign and how it is read and aligned."""

    model_config = ConfigDict(extra="forbid")

    id: str
    path: str = Field(description="File path; relative paths are relative to the campaign file.")
    label: str = ""
    profile: SourceProfile | None = Field(None, description="Profile as used (incl. sheet, clock date …).")
    spec: IngestionConfig | None = Field(None, description="Resolved ingestion spec for an exact replay.")
    sha256: str | None = Field(None, description="Content hash when added; a changed file is reported.")
    clock: ClockModel = Field(default_factory=ClockModel)
    enabled: bool = True


class Campaign(BaseModel):
    """Serializable description of a measurement campaign."""

    model_config = ConfigDict(extra="forbid")

    name: str = "Campaign"
    timezone: str = Field("Europe/Zurich", description="Master zone; date-times without zone are read in it.")
    sources: list[CampaignSource] = Field(default_factory=list)
    notes: str = ""

    def source(self, source_id: str) -> CampaignSource:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(f"No source {source_id!r} in campaign {self.name!r}.")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Campaign":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Track:
    """One channel of one source on master time, ready to plot."""

    source_id: str
    channel: str
    label: str
    unit: str
    x: np.ndarray  # master epoch seconds
    y: np.ndarray
    gap_rows: np.ndarray  # rows where a time gap starts (no line drawn into them)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CampaignSession:
    """Loaded campaign: datasets per source plus master-time helpers.

    `base_dir` resolves relative source paths (the campaign file's folder).
    `warnings` collects problems found while loading (missing or changed files).
    """

    def __init__(self, campaign: Campaign | None = None, base_dir: str | Path | None = None) -> None:
        self.campaign = campaign if campaign is not None else Campaign()
        check_zone(self.campaign.timezone)
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.datasets: dict[str, DualModeDataset] = {}
        self.warnings: list[str] = []
        for src in self.campaign.sources:
            self._load(src)

    # ---------------------------------------------------------------- sources
    def resolve(self, src: CampaignSource) -> Path:
        p = Path(src.path)
        return p if p.is_absolute() else self.base_dir / p

    def _load(self, src: CampaignSource) -> None:
        path = self.resolve(src)
        if not path.is_file():
            self.warnings.append(f"{src.id}: file not found: {path}")
            return
        if src.sha256 and _sha256(path) != src.sha256:
            self.warnings.append(f"{src.id}: file changed since it was added ({path.name})")
        self.datasets[src.id] = load_source(path, src.profile, config=src.spec)

    def add_source(
        self,
        path: str | Path,
        profile: SourceProfile | None = None,
        source_id: str | None = None,
        label: str = "",
        dataset: DualModeDataset | None = None,
    ) -> CampaignSource:
        """Load a file and add it to the campaign (profile detected if not given).

        `dataset`: the result of `load_source(path, profile)` if the caller already loaded it
        (e.g. while asking for a sheet), so the file is not read twice.
        """
        path = Path(path)
        profile = profile if profile is not None else detect_profile(path)
        ds = dataset if dataset is not None else load_source(path, profile)
        ids = {s.id for s in self.campaign.sources}
        base = source_id or path.stem
        sid, n = base, 2
        while sid in ids:
            sid, n = f"{base}_{n}", n + 1
        src = CampaignSource(
            id=sid,
            path=str(path),
            label=label or path.name,
            profile=profile,
            spec=ds.ingestion_spec,
            sha256=_sha256(path),
        )
        self.campaign.sources.append(src)
        self.datasets[sid] = ds
        return src

    def remove_source(self, source_id: str) -> None:
        self.campaign.sources = [s for s in self.campaign.sources if s.id != source_id]
        self.datasets.pop(source_id, None)

    def set_clock(self, source_id: str, clock: ClockModel) -> None:
        self.campaign.source(source_id).clock = clock

    # ------------------------------------------------------------ master time
    def master_time(self, source_id: str) -> np.ndarray | None:
        """Master epoch seconds of every row of a source (None if it has no absolute time)."""
        ds = self.datasets.get(source_id)
        if ds is None:
            return None
        t = epoch_seconds(ds, self.campaign.timezone)
        if t is None:
            return None
        return self.campaign.source(source_id).clock.apply(t, self.campaign.timezone)

    def span(self, source_id: str) -> tuple[float, float] | None:
        """First and last master time of a source."""
        t = self.master_time(source_id)
        if t is None or not np.isfinite(t).any():
            return None
        return float(np.nanmin(t)), float(np.nanmax(t))

    def track(self, source_id: str, channel: str) -> Track:
        """One channel of a source on master time."""
        ds = self.datasets[source_id]
        x = self.master_time(source_id)
        if x is None:
            raise ValueError(f"Source {source_id!r} has no absolute time; set TimeConfig.origin.")
        y = ds.df[channel].cast(float).to_numpy().astype(float)
        gap = np.flatnonzero(ds.flag_mask(["gap"]).to_numpy())
        src = self.campaign.source(source_id)
        return Track(source_id, channel, f"{src.label or src.id}: {channel}", ds.units.get(channel, ""), x, y, gap)

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Write campaign.json; source paths become relative to its folder where possible."""
        path = Path(path).resolve()
        out = self.campaign.model_copy(deep=True)
        for src in out.sources:
            p = self.resolve(src).resolve()
            try:
                src.path = Path(os.path.relpath(p, path.parent)).as_posix()
            except ValueError:  # other drive on Windows
                src.path = str(p)
        out.save(path)
        self.base_dir = path.parent
        self.campaign = out

    @classmethod
    def open(cls, path: str | Path) -> "CampaignSession":
        path = Path(path).resolve()
        return cls(Campaign.load(path), base_dir=path.parent)
