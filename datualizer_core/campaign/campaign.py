"""Measurement campaign: several sources on one master time axis (Etappe 2b, step 3).

`Campaign` is the serializable description (`campaign.json`): which files, read how (profile
and resolved ingestion spec), and how each clock maps to master time. `CampaignSession` holds
the loaded datasets and computes master times. The JSON is the seed of the recipe AST (Etappe 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from datualizer_core.campaign.clock import ClockAnchor, ClockCells, ClockModel, epoch_seconds, to_epoch_s
from datualizer_core.campaign.coverage import SourceCoverage, coverage_intervals, value_rows
from datualizer_core.campaign.display import ChannelDisplay, YScale
from datualizer_core.campaign.evaluation import (
    DEFAULT_WINDOW,
    STEP_COLUMNS,
    Evaluation,
    StepClassification,
    StepWindow,
    classify_steps,
    measure_value,
    override_key,
    parameter_columns,
)
from datualizer_core.campaign.intervals import (
    Interval,
    IntervalConfigError,
    IntervalMapping,
    IntervalSet,
    ResolvedInterval,
    WindowOverride,
    WindowRule,
    derive_intervals,
    master_datetime,
    resolve_manual,
    suggest_mapping,
)
from datualizer_core.campaign.sync import XCorrResult, clock_from_cells, cross_correlate
from datualizer_core.dataset import DualModeDataset
from datualizer_core.ingestion.config import IngestionConfig
from datualizer_core.ingestion.profiles import SourceProfile, detect_profile
from datualizer_core.ingestion.source_loader import load_source
from datualizer_core.schema import PlotStyle
from datualizer_core.stats.window import Check, StatsConfig, WindowResult, WindowSpec, window_stats
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
    time_reference: datetime | None = Field(
        None, description="Reference t0 of the relative time axis (with UTC offset); None = start of the data shown."
    )
    sources: list[CampaignSource] = Field(default_factory=list)
    interval_mappings: list[IntervalMapping] = Field(
        default_factory=list, description="Sources that list measurement steps, with their column roles."
    )
    intervals: list[Interval] = Field(default_factory=list, description="Measurement steps set by hand.")
    window_rule: WindowRule = Field(default_factory=WindowRule, description="Evaluated part of each step.")
    window_overrides: dict[str, WindowOverride] = Field(
        default_factory=dict,
        description="Evaluated window set by hand, per step id (replaces the window rule there); "
        "'<step id>@<window>' for a named window.",
    )
    stats: StatsConfig = Field(default_factory=StatsConfig, description="Settings of the window statistics.")
    evaluation: Evaluation = Field(
        default_factory=Evaluation, description="Step classes, named windows and measures (the step table)."
    )
    notes: str = ""

    @model_validator(mode="after")
    def _override_keys(self) -> "Campaign":
        for key, ov in self.window_overrides.items():
            if ov.window is not None and not key.endswith(f"@{ov.window}"):
                raise ValueError(f"Window override {key!r} is for window {ov.window!r}; its key must end in '@{ov.window}'.")
        return self

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
    color: str | None = None  # manual colour '#rrggbb'

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.channel)


def _table_fields():
    """Fields of WindowResult that are table columns (not the arrays kept for drawing)."""
    return [f for f in fields(WindowResult) if f.metadata.get("table", True)]


def _result_columns() -> list[str]:
    out = []
    for f in _table_fields():
        if f.name == "quantiles":
            out += ["q05", "q95"]
        elif f.name == "slope_ts_ci":
            out += ["ts_ci_low", "ts_ci_high"]
        else:
            out.append(f.name)
    return out


def _result_row(r: WindowResult) -> dict:
    row = {}
    for f in _table_fields():
        v = getattr(r, f.name)
        if f.name in ("quantiles", "slope_ts_ci"):
            lo, hi = ("q05", "q95") if f.name == "quantiles" else ("ts_ci_low", "ts_ci_high")
            row[lo], row[hi] = (None, None) if v is None else v
        elif f.name in ("checks", "assumptions"):
            row[f.name] = "; ".join(v)
        else:
            row[f.name] = v
    return row


@dataclass(frozen=True)
class _StatsInput:
    """One channel ready for `window_stats`; `data` is (x, y, gap_rows, excluded) or the reason there is none."""

    source: str
    channel: str
    unit: str
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | str
    reading_u: float
    clock_u: float | None
    equivalence: float | None

    def run(self, window: WindowSpec, cfg: StatsConfig) -> WindowResult:
        x, y, gap, flags = self.data  # type: ignore[misc]
        return window_stats(
            x, y, gap, window, flags, cfg,
            reading_u=self.reading_u, clock_u_s=self.clock_u, equivalence=self.equivalence,
        )


def _to_datetimes(df: pl.DataFrame, columns: Sequence[str], zone: str) -> pl.DataFrame:
    """Epoch-second columns as date-times in the campaign zone."""
    return df.with_columns(
        (pl.col(c) * 1e6).round().cast(pl.Int64).cast(pl.Datetime("us", "UTC")).dt.convert_time_zone(zone)
        for c in columns
    )


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
        self._interval_cache: tuple[tuple, IntervalSet] | None = None
        self._class_cache: tuple[tuple, StepClassification] | None = None
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
        self.campaign.interval_mappings = [
            m for m in self.campaign.interval_mappings if source_id not in (m.source_id, m.clock_source)
        ]
        self.datasets.pop(source_id, None)
        self._forget(source_id)

    def set_clock(self, source_id: str, clock: ClockModel) -> None:
        self.campaign.source(source_id).clock = clock
        self._forget(source_id)

    def set_timezone(self, zone: str) -> None:
        """Change the master zone (raises TimeConfigError for an unknown zone).

        The axis labels follow it. Sources whose times carry no zone are read in it, so their
        master times move; sources with their own zone (from profile or spec) keep their instants.
        The data itself is not changed.
        """
        check_zone(zone)
        self.campaign.timezone = zone

    @property
    def time_reference_s(self) -> float | None:
        """t0 of the relative time axis in epoch seconds (None = not set)."""
        ref = self.campaign.time_reference
        if ref is None:
            return None
        return to_epoch_s(ref, self.campaign.timezone)

    def set_time_reference(self, epoch_s: float | None) -> None:
        """Set t0 of the relative time axis (None = start of the data shown); stored in UTC."""
        self.campaign.time_reference = (
            None if epoch_s is None else datetime.fromtimestamp(epoch_s, tz=timezone.utc)
        )

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
            name=name, style=src.plot_style(channel), y_scale=display.y_scale, color=display.color,
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

    # ---------------------------------------------------------------- intervals
    def suggest_mapping(self, source_id: str) -> IntervalMapping:
        """Column roles for steps of a source: the profile's suggestion if it fits, else from the data shape."""
        src = self.campaign.source(source_id)
        ds = self.datasets.get(source_id)
        if ds is None:
            raise IntervalConfigError(f"Source {source_id!r} is not loaded.")
        return suggest_mapping(ds, source_id, src.profile.intervals if src.profile else None)

    def set_mapping(self, mapping: IntervalMapping) -> None:
        """Add or replace the step mapping of `mapping.source_id` (checked against the data first)."""
        self.campaign.source(mapping.source_id)
        if mapping.clock_source is not None:
            self.campaign.source(mapping.clock_source)
        self.preview_steps(mapping)  # unknown columns fail loudly here, not later in the plot
        others = [m for m in self.campaign.interval_mappings if m.source_id != mapping.source_id]
        self.campaign.interval_mappings = others + [mapping]

    def remove_mapping(self, source_id: str) -> None:
        self.campaign.interval_mappings = [m for m in self.campaign.interval_mappings if m.source_id != source_id]

    def set_window_rule(self, rule: WindowRule) -> None:
        self.campaign.window_rule = rule

    def add_interval(self, start: datetime, end: datetime, label: str = "", **kwargs) -> Interval:
        """Add a step set by hand (master wall-clock times in the campaign zone); start and end may be swapped."""
        if end < start:
            start, end = end, start
        ids = {iv.id for iv in self.campaign.intervals}
        n = len(ids) + 1
        while f"m{n}" in ids:
            n += 1
        iv = Interval(id=f"m{n}", label=label or f"Step {n}", start=start, end=end, **kwargs)
        self.campaign.intervals.append(iv)
        return iv

    def update_interval(self, interval_id: str, **changes) -> Interval:
        for i, iv in enumerate(self.campaign.intervals):
            if iv.id == interval_id:
                new = Interval.model_validate({**iv.model_dump(), **changes})
                self.campaign.intervals[i] = new
                return new
        raise KeyError(f"No manual step {interval_id!r}.")

    def remove_interval(self, interval_id: str) -> None:
        self.campaign.intervals = [iv for iv in self.campaign.intervals if iv.id != interval_id]

    def intervals(self) -> IntervalSet:
        """All measurement steps on master time: derived from the mappings, then the manual ones.

        Derived steps are computed anew whenever data, clocks, mappings or the window rule change.
        """
        c = self.campaign
        key = (
            c.timezone,
            c.model_dump_json(include={"interval_mappings", "intervals", "window_rule", "window_overrides", "evaluation"}),
            tuple((s.id, s.clock.model_dump_json(), id(self.datasets.get(s.id))) for s in c.sources),
        )
        if self._interval_cache is not None and self._interval_cache[0] == key:
            return self._interval_cache[1]
        out, problems = [], []
        for m in c.interval_mappings:
            try:
                ivs, probs = self.preview_steps(m)
            except (IntervalConfigError, KeyError) as exc:
                ivs, probs = [], [f"{m.source_id}: {exc}"]
            out += ivs
            problems += probs
        out += [resolve_manual(iv, c.timezone, c.window_rule) for iv in c.intervals if iv.enabled]
        out, problems = self._apply_overrides(out, problems)
        result = IntervalSet(sorted(out, key=lambda iv: iv.start), problems)
        self._interval_cache = (key, result)
        return result

    def _apply_overrides(
        self, steps: list[ResolvedInterval], problems: list[str]
    ) -> tuple[list[ResolvedInterval], list[str]]:
        """Replace the rule's window by the one set by hand; report overrides that no longer fit.

        Overrides of named windows are applied in `step_window`; here they are only checked.
        """
        overrides = self.campaign.window_overrides
        if not overrides:
            return steps, problems
        zone = self.campaign.timezone
        windows = self.campaign.evaluation.window_names()
        problems = list(problems)
        by_id = {iv.id: iv for iv in steps}
        out = []
        for iv in steps:
            ov = overrides.get(iv.id)
            if ov is None or ov.window is not None:
                out.append(iv)
                continue
            out.append(replace(iv, window=(to_epoch_s(ov.start, zone), to_epoch_s(ov.end, zone)), window_override=True))
        for key in sorted(overrides):
            ov = overrides[key]
            sid = key if ov.window is None else key[: -len(ov.window) - 1]
            what = "window set by hand" + ("" if ov.window is None else f" ('{ov.window}')")
            iv = by_id.get(sid)
            if iv is None:
                problems.append(f"{sid}: {what}, but there is no such step (kept in the campaign).")
            elif ov.window is not None and ov.window not in windows:
                problems.append(f"{sid}: {what}, but there is no such window (kept in the campaign).")
            elif ov.label and ov.label != iv.label:
                problems.append(f"{sid}: {what} on step '{ov.label}', the step is now '{iv.label}'.")
        return out, problems

    def set_window_override(self, step_id: str, start: float, end: float, window: str = DEFAULT_WINDOW) -> WindowOverride:
        """Evaluated window of one step set by hand (master epoch seconds; may be swapped).

        `window`: which named window it replaces (default: the window rule). KeyError for an
        unknown step or window.
        """
        step = self.intervals().get(step_id)
        self.campaign.evaluation.window(window)
        start, end = sorted((start, end))
        zone = self.campaign.timezone
        ov = WindowOverride(
            start=master_datetime(start, zone), end=master_datetime(end, zone), label=step.label, window=window
        )
        self.campaign.window_overrides[override_key(step_id, window)] = ov
        return ov

    def clear_window_override(self, step_id: str, window: str = DEFAULT_WINDOW) -> None:
        """Back to the window's rule for this step."""
        self.campaign.window_overrides.pop(override_key(step_id, window), None)

    # ---------------------------------------------------- classes and named windows
    def set_evaluation(self, evaluation: Evaluation) -> None:
        """Replace step classes, named windows and measures (checked by the model)."""
        self.campaign.evaluation = Evaluation.model_validate(evaluation.model_dump())

    def step_classes(self) -> StepClassification:
        """Class of every step (`Evaluation.step_classes`); raises ExprError for a formula that does not fit."""
        ivs = self.intervals()
        rules = self.campaign.evaluation.model_dump_json(include={"step_classes"})
        cached = self._class_cache
        if cached is not None and cached[0][0] is ivs and cached[0][1] == rules:
            return cached[1]
        result = classify_steps(ivs.intervals, self.campaign.evaluation.step_classes)
        self._class_cache = ((ivs, rules), result)
        return result

    def step_window(
        self, step: str | ResolvedInterval, window: str = DEFAULT_WINDOW, klasse: str | None | bool = False,
        evaluation: Evaluation | None = None,
    ) -> StepWindow:
        """Window `window` of one step on master time (KeyError for an unknown step or window).

        `default` is the step's window from the window rule (or set by hand). A named window
        applies its rule to the step, unless it is only for other classes (then `applies` is
        False). `klasse`: the step's class if already known (False = look it up).
        `evaluation`: where the named windows are defined (default the campaign's).
        """
        iv = self.intervals().get(step) if isinstance(step, str) else step
        nw = (evaluation or self.campaign.evaluation).window(window)
        if nw is None:
            return StepWindow(DEFAULT_WINDOW, iv.window, True, iv.window_override)
        if nw.classes:
            k = self.step_classes().of(iv.id) if klasse is False else klasse
            if k not in nw.classes:
                why = "the step has no class" if k is None else f"the step is '{k}'"
                return StepWindow(window, None, False, note=f"window '{window}' is for {', '.join(nw.classes)}; {why}")
        ov = self.campaign.window_overrides.get(override_key(iv.id, window))
        if ov is not None:
            zone = self.campaign.timezone
            return StepWindow(window, (to_epoch_s(ov.start, zone), to_epoch_s(ov.end, zone)), True, True)
        w = nw.rule.apply(iv.start, iv.end)
        return StepWindow(window, w, True, note="" if w is not None else f"window rule '{window}' leaves nothing")

    def preview_steps(self, m: IntervalMapping):
        """Steps a mapping would give (without storing it); unknown columns raise IntervalConfigError."""
        ds = self.datasets.get(m.source_id)
        if ds is None:
            return [], [f"{m.source_id}: source not loaded, no steps."]
        src = self.campaign.source(m.source_id)
        clock = self.campaign.source(m.clock_source or m.source_id).clock
        formats = src.spec.time.datetime_formats if src.spec is not None else ()
        return derive_intervals(ds, m, self.campaign.timezone, clock, self.campaign.window_rule, formats)

    # ---------------------------------------------------------- window statistics
    def set_stats_config(self, config: StatsConfig) -> None:
        self.campaign.stats = config.model_copy(deep=True)

    def window_stats(
        self, step: str | ResolvedInterval, source_id: str, channel: str, config: StatsConfig | None = None,
        window: str = DEFAULT_WINDOW,
    ) -> WindowResult:
        """Statistics of one channel in an evaluated window of one step (what `window_table` gives per row).

        `window`: a named window (`Evaluation.windows`); default the window rule. A step whose
        window rule leaves nothing is evaluated over the whole step; a finding says so.
        Raises ValueError if the source is not loaded or has no absolute time, or if the named
        window does not apply to the step's class; KeyError for an unknown step, window, source or channel.
        """
        cfg = config if config is not None else self.campaign.stats
        iv = self.intervals().get(step) if isinstance(step, str) else step
        sw = self.step_window(iv, window)
        if not sw.applies:
            raise ValueError(f"{iv.id}: {sw.note}")
        inp = self._stats_input(source_id, channel, cfg)
        if isinstance(inp.data, str):
            raise ValueError(f"{source_id}: {inp.data}")
        lo, hi = sw.window if sw.window is not None else (iv.start, iv.end)
        r = inp.run(WindowSpec(lo, hi), cfg)
        if sw.window is None:
            r.checks.append(Check("window rule leaves nothing: whole step evaluated"))
        return r

    def window_table(
        self, channels: Sequence[tuple[str, str]], config: StatsConfig | None = None
    ) -> pl.DataFrame:
        """Statistics of every step's window for every channel: one row per step × channel.

        `channels`: (source id, channel) pairs. `config`: default the campaign's `stats`.
        Columns: step id, label, start/end and window on master time (campaign zone), the step's
        parameters, source, channel, unit, then the fields of `WindowResult` (`quantiles` as
        `q05`/`q95`, `slope_ts_ci` as `ts_ci_low`/`ts_ci_high`, `checks` and `assumptions` joined
        with "; "). A step without window or a source without data gets a row with the reason in
        `checks`. An unknown source or channel raises KeyError.
        """
        cfg = config if config is not None else self.campaign.stats
        zone = self.campaign.timezone
        steps = self.intervals().intervals
        inputs = [self._stats_input(sid, ch, cfg) for sid, ch in channels]

        fixed = ["step_id", "step_label", "step_start", "step_end", "window_start", "window_end"]
        result_cols = _result_columns()
        taken = set(fixed) | {"window_override", "source", "channel", "unit"} | set(result_cols)
        param_names = parameter_columns(steps, tuple(taken))

        rows = []
        for iv in steps:
            for inp in inputs:
                row = {
                    "step_id": iv.id, "step_label": iv.label, "step_start": iv.start, "step_end": iv.end,
                    "window_start": iv.window[0] if iv.window else None,
                    "window_end": iv.window[1] if iv.window else None,
                    "window_override": iv.window_override,
                    **{param_names[k]: iv.params.get(k) for k in param_names},
                    "source": inp.source, "channel": inp.channel, "unit": inp.unit,
                    **dict.fromkeys(result_cols),
                }
                if iv.window is None:
                    row["checks"] = "window rule leaves nothing"
                elif isinstance(inp.data, str):
                    row["checks"] = inp.data
                else:
                    row.update(_result_row(inp.run(WindowSpec(*iv.window), cfg)))
                rows.append(row)
        if not rows:
            return pl.DataFrame()
        df = pl.DataFrame(rows, infer_schema_length=None)
        return _to_datetimes(df, fixed[2:], zone)

    def step_table(self, spec: Evaluation | None = None, config: StatsConfig | None = None) -> pl.DataFrame:
        """One row per step: parameters, class and every measure with its standard uncertainty.

        `spec`: classes, named windows and measures (default the campaign's `evaluation`).
        Columns: step_id, step_label, step_start, step_end (campaign zone), dauer_s, the step
        parameters, klasse, klasse_hinweis (no class matches, or several did), then per measure
        `<alias>`, `u_<alias>` (see `evaluation.MEASURE_U`; null where none is defined) and
        `checks_<alias>` (the window's findings joined with "; ").

        A window that is not for the step's class leaves its measures empty (value, u and checks
        null). A window the rule leaves nothing of, or a source without data, gives a null value
        with the reason in checks. Unknown sources, channels or windows raise KeyError, a class
        formula that does not fit the steps ExprError, a measure column equal to a parameter ValueError.
        """
        ev = spec if spec is not None else self.campaign.evaluation
        cfg = config if config is not None else self.campaign.stats
        steps = self.intervals().intervals
        params = parameter_columns(steps)
        measure_cols = [col for m in ev.measures for col in m.columns()]
        clash = sorted(set(params.values()) & set(measure_cols))
        if clash:
            raise ValueError(f"Measure column(s) {', '.join(clash)} are also step parameters; choose another alias.")
        for m in ev.measures:
            ev.window(m.window)
        inputs = {(m.source, m.channel): self._stats_input(m.source, m.channel, cfg) for m in ev.measures}
        classes = self.step_classes() if spec is None else classify_steps(steps, ev.step_classes)

        rows = []
        for iv in steps:
            klasse = classes.of(iv.id)
            row = {
                "step_id": iv.id, "step_label": iv.label, "step_start": iv.start, "step_end": iv.end,
                "dauer_s": iv.duration_s,
                **{col: iv.params.get(k) for k, col in params.items()},
                "klasse": klasse, "klasse_hinweis": classes.notes.get(iv.id),
            }
            results: dict[tuple, WindowResult] = {}  # measures of one channel and window share the statistics
            for m in ev.measures:
                col, u_col, checks_col = m.columns()
                row[col] = row[u_col] = row[checks_col] = None
                sw = self.step_window(iv, m.window, klasse, ev)
                if not sw.applies:
                    continue
                inp = inputs[(m.source, m.channel)]
                if sw.window is None:
                    row[checks_col] = sw.note or "window rule leaves nothing"
                elif isinstance(inp.data, str):
                    row[checks_col] = inp.data
                else:
                    key = (m.source, m.channel, sw.window)
                    if key not in results:
                        results[key] = inp.run(WindowSpec(*sw.window), cfg)
                    r = results[key]
                    row[col], row[u_col] = measure_value(r, m.stat, inp.reading_u)
                    row[checks_col] = "; ".join(r.checks)
            rows.append(row)

        types: dict[str, pl.DataType] = {
            "step_id": pl.String, "step_label": pl.String, "step_start": pl.Float64, "step_end": pl.Float64,
            "dauer_s": pl.Float64, "klasse": pl.String, "klasse_hinweis": pl.String,
        }
        for m in ev.measures:
            col, u_col, checks_col = m.columns()
            types.update({col: pl.Float64, u_col: pl.Float64, checks_col: pl.String})
        if rows:
            df = pl.DataFrame(rows, infer_schema_length=None)
            df = df.with_columns(pl.col(c).cast(t) for c, t in types.items())
        else:
            df = pl.DataFrame(schema={c: types.get(c, pl.Null) for c in [*STEP_COLUMNS, *measure_cols]})
        return _to_datetimes(df, ["step_start", "step_end"], self.campaign.timezone)

    def _stats_input(self, source_id: str, channel: str, cfg: StatsConfig) -> "_StatsInput":
        """What `window_stats` needs of one channel, or the reason why it has nothing (`data` is text)."""
        src = self.campaign.source(source_id)
        key = f"{source_id}/{channel}"
        reading_u = cfg.reading_u.get(key, 0.0)
        equivalence = cfg.equivalence.get(key)
        ds = self.datasets.get(source_id)
        if ds is None:
            return _StatsInput(source_id, channel, "", "source not loaded", reading_u, None, equivalence)
        if channel not in ds.channels:
            raise KeyError(f"Source {source_id!r} has no channel {channel!r}.")
        unit = ds.units.get(channel, "")
        if self.master_time(source_id) is None:
            return _StatsInput(source_id, channel, unit, "source has no absolute time", reading_u, None, equivalence)
        tr = self.track(source_id, channel)
        excluded = np.zeros(len(tr.x), dtype=bool)
        if cfg.exclude_flags:
            ev = ds.quality_flags.filter(
                pl.col("flag").is_in(cfg.exclude_flags)
                & (pl.col("channel").is_null() | (pl.col("channel") == channel))
            )
            excluded[ev["row"].unique().to_numpy()] = True
        data = (tr.x, tr.y, tr.gap_rows, excluded)
        return _StatsInput(source_id, channel, unit, data, reading_u, src.clock.uncertainty_s, equivalence)

    # ---------------------------------------------------------- clock alignment
    def clock_from_cells(self, cells: ClockCells) -> tuple[ClockModel, list[ClockAnchor]]:
        """Offset suggested by cell pairs in the file of source `cells.source_id` (not applied).

        Apply it to the source whose clock the pairs describe with `set_clock`.
        """
        src = self.campaign.source(cells.source_id)
        return clock_from_cells(self.resolve(src), cells, self.campaign.timezone)

    def cross_correlate(
        self,
        reference: tuple[str, str],
        moving: tuple[str, str],
        start: float,
        end: float,
        lag_min: float,
        lag_max: float,
        *,
        derivative: bool = False,
        step_s: float | None = None,
    ) -> XCorrResult:
        """Shift of the moving track (source, channel) onto the reference track, on master time.

        The lag is relative to the moving source's current clock: apply it with
        `clock.shifted(result.lag_s)`.
        """
        if reference[0] == moving[0]:
            raise ValueError("Both tracks belong to the same source; its clock cannot be aligned to itself.")
        ref, mov = self.track(*reference), self.track(*moving)
        return cross_correlate(
            ref.x, ref.y, mov.x, mov.y, start, end, lag_min, lag_max,
            step_s=step_s, derivative=derivative, ref_gaps=ref.gap_rows, mov_gaps=mov.gap_rows,
        )

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
