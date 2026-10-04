"""Quality flags (Datenkatalog P12–P17): mark suspicious samples, never change or drop them.

The detectors only read an already loaded table and return flag events; the data stays as it
is. Which thresholds apply is decided at ingestion (`QualityConfig`, resolved into the
`ingestion_spec`); this module has no imports from `ingestion/`.

Flags (one event per row, channel and flag; `channel` is null for row-level flags):

- `gap` (row level, P12): the time step before this row is more than `gap_factor` times the
  median step of its run. Plots must not draw a line across it.
- `missing` / `missing_calculated` (P16): a null in a channel that has values in the same run.
  `missing_calculated` is used for calculated channels, where a null usually comes from the
  formula (e.g. division by a zero difference), not from a failed measurement.
- `stuck` (P15): the same value repeated for at least `stuck_min_duration_s`.
- `jump` (P14): a step larger than the channel's jump threshold that does not return
  (e.g. a refill).
- `dropout` (P13): a jump away and back to the previous level within `dropout_max_duration_s`;
  every sample in between is flagged (e.g. a balance reporting 0.0 for some seconds).
- `out_of_range` (P17): outside the user-given plausible range of the channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import polars as pl

FLAGS_SCHEMA = {"row": pl.Int64, "channel": pl.String, "flag": pl.String}

_SEG = "__q_segment"
_ROW = "__q_row"


@dataclass(frozen=True)
class QualitySettings:
    """Resolved detector settings with loaded channel names (see `ingestion.config.QualityConfig`)."""

    gap_factor: float = 5.0
    stuck_min_duration_s: float = 30.0
    stuck_min_samples: int = 3
    jump_thresholds: Mapping[str, float] = field(default_factory=dict)
    dropout_max_duration_s: float = 60.0
    ranges: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    calculated_channels: Sequence[str] = ()


def empty_flags() -> pl.DataFrame:
    return pl.DataFrame(schema=FLAGS_SCHEMA)


def suggest_jump_threshold(
    df: pl.DataFrame, channel: str, run_columns: Sequence[str], factor: float
) -> float:
    """Return `factor` x the 99th percentile of the absolute sample-to-sample change.

    Changes are taken inside each run only (a new run may start at another level). Returns 0.0
    (jump detection off) for channels that never change.
    """
    prev = pl.col(channel).forward_fill().shift(1)
    step = pl.col(channel) - prev
    if run_columns:
        step = step.over(list(run_columns))
    steps = df.select(step.abs().alias("d"))["d"]
    steps = steps.filter(steps > 0)
    if steps.is_empty():
        return 0.0
    return float(factor * steps.quantile(0.99))


def detect_flags(
    df: pl.DataFrame,
    time_col: str,
    run_columns: Sequence[str],
    channels: Sequence[str],
    settings: QualitySettings,
) -> pl.DataFrame:
    """Return all flag events of the table, sorted by row, channel and flag."""
    if df.is_empty() or time_col not in df.columns:
        return empty_flags()
    keys = list(run_columns)
    gaps = _gap_rows(df, time_col, keys, settings.gap_factor)
    is_gap = pl.Series(_ROW, range(len(df))).is_in(gaps)
    frame = df.with_row_index(_ROW).with_columns(is_gap.cum_sum().alias(_SEG))
    parts = [_events(gaps, None, "gap")]
    calculated = set(settings.calculated_channels)
    for ch in channels:
        if ch not in df.columns:
            continue
        missing = "missing_calculated" if ch in calculated else "missing"
        parts.append(_events(_missing_rows(frame, ch, keys), ch, missing))
        parts.append(_events(_stuck_rows(frame, ch, time_col, settings), ch, "stuck"))
        threshold = settings.jump_thresholds.get(ch, 0.0)
        if threshold > 0:
            jumps, dropouts = _jump_rows(frame, ch, time_col, keys, threshold, settings)
            parts.append(_events(jumps, ch, "jump"))
            parts.append(_events(dropouts, ch, "dropout"))
        if ch in settings.ranges:
            parts.append(_events(_range_rows(frame, ch, *settings.ranges[ch]), ch, "out_of_range"))
    return pl.concat(parts).sort("row", "channel", "flag", nulls_last=False)


def flag_columns(df: pl.DataFrame, flags: pl.DataFrame, columns: Sequence[str], suffix: str) -> pl.DataFrame:
    """Return `df` plus one String column per given column with its comma-joined flags (or null)."""
    out = df
    for col, channel in columns:
        sub = flags.filter(
            pl.col("channel").is_null() if channel is None else pl.col("channel") == channel
        )
        joined = sub.group_by("row").agg(pl.col("flag").sort().str.join(","))
        values = (
            pl.DataFrame({"row": range(len(df))}, schema={"row": pl.Int64})
            .join(joined, on="row", how="left", maintain_order="left")["flag"]
        )
        out = out.with_columns(values.alias(f"{col}{suffix}"))
    return out


def _events(rows: Sequence[int] | pl.Series, channel: str | None, flag: str) -> pl.DataFrame:
    rows = pl.Series("row", list(rows), dtype=pl.Int64)
    return pl.DataFrame({
        "row": rows,
        "channel": pl.Series([channel] * len(rows), dtype=pl.String),
        "flag": pl.Series([flag] * len(rows), dtype=pl.String),
    })


def _gap_rows(df: pl.DataFrame, time_col: str, keys: list[str], factor: float) -> list[int]:
    """P12: rows whose time step is far larger than the typical step of their run (P11-aware)."""
    if factor <= 0:
        return []
    t = pl.col(time_col)
    dt = t - t.shift(1)
    # The typical step is taken inside each run, so steps across run boundaries do not bias it.
    typical = (t - t.shift(1)).over(keys).median().over(keys) if keys else dt.median()
    res = df.select(((dt > factor * typical) & (typical > 0)).fill_null(False).alias("g"))["g"]
    return res.arg_true().to_list()


def _missing_rows(frame: pl.DataFrame, ch: str, keys: list[str]) -> pl.Series:
    """P16: nulls inside runs where the channel is recorded at all (not the sparse matrix of P10)."""
    present = pl.col(ch).is_not_null().any()
    present = present.over(keys) if keys else present
    return frame.filter(pl.col(ch).is_null() & present)[_ROW]


def _stuck_rows(frame: pl.DataFrame, ch: str, time_col: str, s: QualitySettings) -> pl.Series:
    """P15: runs of identical values; a gap or a null ends the run. Run boundaries do not."""
    if s.stuck_min_duration_s <= 0:
        return pl.Series(_ROW, [], dtype=pl.UInt32)
    v = pl.col(ch)
    new_block = (v != v.shift(1)).fill_null(True) | (pl.col(_SEG) != pl.col(_SEG).shift(1)).fill_null(True)
    blocks = frame.select(_ROW, time_col, ch, new_block.cum_sum().alias("b")).filter(v.is_not_null())
    long = blocks.group_by("b").agg(
        pl.len().alias("n"), (pl.col(time_col).max() - pl.col(time_col).min()).alias("dur")
    ).filter((pl.col("n") >= s.stuck_min_samples) & (pl.col("dur") >= s.stuck_min_duration_s))
    return blocks.filter(pl.col("b").is_in(long["b"].implode()))[_ROW]


def _jump_rows(
    frame: pl.DataFrame,
    ch: str,
    time_col: str,
    keys: list[str],
    threshold: float,
    s: QualitySettings,
) -> tuple[list[int], list[int]]:
    """P13/P14: steps above the threshold; an excursion that returns to its level is a dropout."""
    part = [*keys, _SEG]
    # Compare each value with the previous value of the same run and segment; nulls are skipped
    changed = pl.any_horizontal((pl.col(c) != pl.col(c).shift(1)).fill_null(True) for c in part)
    valid = frame.filter(pl.col(ch).is_not_null()).with_columns(
        pl.col(ch).shift(1).over(part).alias("lvl"),
        changed.cum_sum().alias("p"),
    )
    steps = valid.filter((pl.col(ch) - pl.col("lvl")).abs() > threshold)
    rows = steps[_ROW].to_list()
    values = steps[ch].to_list()
    levels = steps["lvl"].to_list()
    times = steps[time_col].to_list()
    parts = steps["p"].to_list()
    valid_rows = valid[_ROW].to_list()
    position = {r: i for i, r in enumerate(valid_rows)}

    jumps: list[int] = []
    dropouts: list[int] = []
    i = 0
    while i < len(rows):
        back = None
        for k in range(i + 1, len(rows)):
            if parts[k] != parts[i] or times[k] - times[i] > s.dropout_max_duration_s:
                break
            if abs(values[k] - levels[i]) <= threshold:
                back = k
                break
        if back is None:
            jumps.append(rows[i])
            i += 1
        else:
            dropouts.extend(valid_rows[position[rows[i]]:position[rows[back]]])
            i = back + 1
    return jumps, dropouts


def _range_rows(frame: pl.DataFrame, ch: str, low: float | None, high: float | None) -> pl.Series:
    """P17: values outside the plausible band; they are marked, not filtered."""
    cond = pl.lit(False)
    if low is not None:
        cond = cond | (pl.col(ch) < low)
    if high is not None:
        cond = cond | (pl.col(ch) > high)
    return frame.filter(cond.fill_null(False))[_ROW]
