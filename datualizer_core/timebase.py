"""Absolute time axis: date-time parsing, time zones and daylight saving (Etappe 2b).

A dataset keeps relative `time_seconds` plus the absolute time of `time_seconds == 0`
(`time_origin`). Absolute times are handled as integer epoch microseconds:
- with a zone: true UTC instants, so elapsed seconds stay correct across DST changes;
- without a zone: plain wall-clock arithmetic, independent of the machine's local zone.

Time zones are resolved by Polars, which ships its own zone database (no `tzdata` needed on Windows).
This module has no imports from `ingestion/` so `dataset.py` can use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import re
from typing import Sequence

import polars as pl

_EPOCH_NAIVE = datetime(1970, 1, 1)
_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


class TimeConfigError(ValueError):
    """A time setting does not fit the data (forced format, unknown zone, origin for date-times)."""


@dataclass(frozen=True)
class LocalizedTimes:
    """Epoch microseconds of naive date-times read in a zone, with the rows that needed a decision."""

    epoch_us: list[int | None]
    ambiguous: list[int]     # occurs twice when clocks go back; the earlier instant is used
    nonexistent: list[int]   # skipped when clocks go forward; becomes null


def parse_datetime(text: str, formats: Sequence[str]) -> datetime | None:
    """Parse ISO 8601 (with optional offset or `Z`) or one of the strptime `formats`."""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


_DATE_PATTERNS = [
    (re.compile(r"(?<!\d)(\d{4})[-_./](\d{1,2})[-_./](\d{1,2})(?!\d)"), (1, 2, 3)),  # 2026-10-12
    (re.compile(r"(?<!\d)(\d{1,2})[-_./](\d{1,2})[-_./](\d{4})(?!\d)"), (3, 2, 1)),  # 12.10.2026
]


def date_in_text(text: str) -> date | None:
    """First calendar date written in a text (sheet or file name), e.g. '12.10.2026' or '2026-10-12'.

    Only a suggestion for `TimeConfig.clock_date`; the user confirms it. Day-first for dotted dates.
    """
    for pattern, (y, m, d) in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                return date(int(match.group(y)), int(match.group(m)), int(match.group(d)))
            except ValueError:
                continue
    return None


def check_zone(zone: str) -> None:
    """Raise TimeConfigError if Polars does not know the IANA zone name."""
    try:
        pl.Series([_EPOCH_NAIVE], dtype=pl.Datetime("us")).dt.replace_time_zone(zone)
    except pl.exceptions.PolarsError as exc:
        raise TimeConfigError(f"Unknown time zone {zone!r}.") from exc


def localize(values: Sequence[datetime | None], zone: str | None) -> LocalizedTimes:
    """Convert naive date-times to epoch microseconds, read in `zone` (None = wall clock)."""
    series = pl.Series("t", list(values), dtype=pl.Datetime("us"))
    if zone is None:
        return LocalizedTimes(series.dt.epoch("us").to_list(), [], [])
    check_zone(zone)
    early = series.dt.replace_time_zone(zone, ambiguous="earliest", non_existent="null").dt.epoch("us")
    late = series.dt.replace_time_zone(zone, ambiguous="latest", non_existent="null").dt.epoch("us")
    early_l, late_l = early.to_list(), late.to_list()
    nonexistent = [i for i, (v, e) in enumerate(zip(values, early_l)) if v is not None and e is None]
    ambiguous = [i for i, (e, la) in enumerate(zip(early_l, late_l)) if e is not None and e != la]
    return LocalizedTimes(early_l, ambiguous, nonexistent)


def aware_to_epoch_us(value: datetime) -> int:
    """Epoch microseconds of a date-time with offset."""
    return (value - _EPOCH_UTC) // timedelta(microseconds=1)


def origin_epoch_us(origin: datetime, zone: str | None) -> tuple[int, str | None]:
    """Epoch microseconds and display zone of a user-given origin."""
    if origin.tzinfo is not None:
        return aware_to_epoch_us(origin), zone or "UTC"
    loc = localize([origin], zone)
    if loc.epoch_us[0] is None:
        raise TimeConfigError(f"Origin {origin} does not exist in time zone {zone!r}.")
    return loc.epoch_us[0], zone


def epoch_us_to_datetime(epoch_us: int, zone: str | None) -> datetime:
    """Origin as a datetime: UTC-aware if a zone is known, otherwise naive wall-clock."""
    if zone is None:
        return _EPOCH_NAIVE + timedelta(microseconds=epoch_us)
    return _EPOCH_UTC + timedelta(microseconds=epoch_us)


def absolute_series(
    seconds: pl.Series, origin: datetime, zone: str | None, name: str = "time_abs"
) -> pl.Series:
    """Absolute times `origin + seconds` as a Datetime series, in `zone` if one is known."""
    origin_us = aware_to_epoch_us(origin) if origin.tzinfo is not None else (
        (origin - _EPOCH_NAIVE) // timedelta(microseconds=1)
    )
    us = (seconds.cast(pl.Float64) * 1e6).round().cast(pl.Int64) + origin_us
    out = us.cast(pl.Datetime("us")).alias(name)
    if zone is not None:
        out = out.dt.replace_time_zone("UTC").dt.convert_time_zone(zone)
    return out
