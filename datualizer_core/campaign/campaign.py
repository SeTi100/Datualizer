"""Measurement campaign: several sources on one master time axis (Etappe 2b, step 3).

`Campaign` is the serializable description (`campaign.json`): which files, read how (profile
and resolved ingestion spec), and how each clock maps to master time. `CampaignSession` holds
the loaded datasets and computes master times. The JSON is the seed of the recipe AST (Etappe 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from datualizer_core.campaign.clock import ClockModel, epoch_seconds
from datualizer_core.campaign.coverage import SourceCoverage, coverage_intervals, value_rows
from datualizer_core.campaign.display import ChannelDisplay, YScale
from datualizer_core.dataset import DualModeDataset
from datualizer_core.ingestion.config import IngestionConfig
from datualizer_core.ingestion.profiles import SourceProfile, detect_profile
from datualizer_core.ingestion.source_loader import load_source
from datualizer_core.schema import PlotStyle
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
    display: dict[str, ChannelDisplay] = Field(
        default_factory=dict, description="Display settings per channel (style, y scale, alias); display only."
    )

    def channel_display(self, channel: str) -> ChannelDisplay:
        """Stored display settings of a channel, or the defaults."""
        return self.display.get(channel) or ChannelDisplay()

    def plot_style(self, channel: str) -> PlotStyle:
        """Style of a channel: its own setting, else the profile default, else a line."""
        own = self.channel_display(channel).style
        if own is not None:
            return own
        return self.profile.plot_style if self.profile is not None else PlotStyle.LINE


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
    name: str = ""  # display name: alias, else the channel name
    style: PlotStyle = PlotStyle.LINE
    y_scale: YScale = field(default_factory=YScale)

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.channel)


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
        # Derived per source, dropped when the source or its clock changes
        self._master_cache: dict[str, tuple[ClockModel, str, np.ndarray | None]] = {}
        self._track_cache: dict[tuple[str, str], Track] = {}
        self._coverage_cache: dict[tuple[str, tuple[str, ...] | None], tuple[np.ndarray | None, SourceCoverage]] = {}
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
        self._forget(sid)
        return src

    def remove_source(self, source_id: str) -> None:
        self.campaign.sources = [s for s in self.campaign.sources if s.id != source_id]
        self.datasets.pop(source_id, None)
        self._forget(source_id)

    def set_clock(self, source_id: str, clock: ClockModel) -> None:
        self.campaign.source(source_id).clock = clock
        self._forget(source_id)

    def set_display(self, source_id: str, channel: str, display: ChannelDisplay) -> None:
        """Change how one channel is drawn (display only; the dataset stays as loaded).

        Defaults are not stored, so campaign.json lists only what the user changed.
        """
        src = self.campaign.source(source_id)
        ds = self.datasets.get(source_id)
        if ds is not None and channel not in ds.channels:
            raise KeyError(f"Source {source_id!r} has no channel {channel!r}.")
        if display.is_default():
            src.display.pop(channel, None)
        else:
            src.display[channel] = display.model_copy(deep=True)
        self._track_cache.pop((source_id, channel), None)

    def _forget(self, source_id: str) -> None:
        """Drop cached master times and tracks of a source (after its data or clock changed)."""
        self._master_cache.pop(source_id, None)
        for key in [k for k in self._track_cache if k[0] == source_id]:
            del self._track_cache[key]
        for key in [k for k in self._coverage_cache if k[0] == source_id]:
            del self._coverage_cache[key]

    # ------------------------------------------------------------ master time
    def master_time(self, source_id: str) -> np.ndarray | None:
        """Master epoch seconds of every row of a source (None if it has no absolute time).

        Cached per source while its clock model and the campaign zone stay the same; the array
        is read-only.
        """
        ds = self.datasets.get(source_id)
        if ds is None:
            return None
        clock, zone = self.campaign.source(source_id).clock, self.campaign.timezone
        cached = self._master_cache.get(source_id)
        if cached is not None and cached[0] == clock and cached[1] == zone:
            return cached[2]
        t = epoch_seconds(ds, zone)
        if t is not None:
            t = clock.apply(t, zone)
            t.setflags(write=False)
        self._master_cache[source_id] = (clock.model_copy(deep=True), zone, t)
        return t

    def span(self, source_id: str) -> tuple[float, float] | None:
        """First and last master time of a source."""
        t = self.master_time(source_id)
        if t is None or not np.isfinite(t).any():
            return None
        return float(np.nanmin(t)), float(np.nanmax(t))

    def track(self, source_id: str, channel: str) -> Track:
        """One channel of a source on master time.

        The same Track object is returned until the source or its clock changes, so a plot can
        tell which curves it does not need to redraw.
        """
        ds = self.datasets[source_id]
        x = self.master_time(source_id)
        cached = self._track_cache.get((source_id, channel))
        if cached is not None and cached.x is x:
            return cached
        if x is None:
            raise ValueError(f"Source {source_id!r} has no absolute time; set TimeConfig.origin.")
        y = ds.df[channel].cast(float).to_numpy().astype(float)
        gap = np.flatnonzero(ds.flag_mask(["gap"]).to_numpy())
        src = self.campaign.source(source_id)
        y.setflags(write=False)
        display = src.channel_display(channel)
        name = display.alias or channel
        tr = Track(
            source_id, channel, f"{src.label or src.id}: {name}", ds.units.get(channel, ""), x, y, gap,
            name=name, style=src.plot_style(channel), y_scale=display.y_scale,
        )
        self._track_cache[(source_id, channel)] = tr
        return tr

    def coverage(self, source_id: str, channels: Sequence[str] | None = None) -> SourceCoverage:
        """When a source has values on master time and where it has gaps (see `campaign.coverage`).

        `channels`: count only these channels (default: a row counts if any channel has a value).
        A source without absolute time (or a missing file) has no span and no segments.
        """
        src = self.campaign.source(source_id)
        label = src.label or src.id
        ds = self.datasets.get(source_id)
        t = self.master_time(source_id)
        key = (source_id, None if channels is None else tuple(channels))
        cached = self._coverage_cache.get(key)
        if cached is not None and cached[0] is t and cached[1].label == label:
            return cached[1]
        if ds is None or t is None:
            cov = SourceCoverage(source_id, label, None)
        else:
            gap = np.flatnonzero(ds.flag_mask(["gap"]).to_numpy())
            segments, gaps = coverage_intervals(t, value_rows(ds, channels), gap)
            cov = SourceCoverage(source_id, label, self.span(source_id), segments, gaps)
        self._coverage_cache[key] = (t, cov)
        return cov

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
