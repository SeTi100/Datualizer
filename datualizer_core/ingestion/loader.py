"""Loader module for parsing CSV into Polars DataFrame with error harvesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import io
from pathlib import Path
import re
from typing import Sequence, TextIO
import polars as pl

from datualizer_core.dataset import SOURCES_SCHEMA, AuditEntry, AuditLog, DualModeDataset
from datualizer_core.ingestion.config import IngestionConfig, LongFormatConfig, LongFormatMode, TimeFormat
from datualizer_core.ingestion.long_format import detect_long_format, pivot_long_to_wide
from datualizer_core.ingestion.pre_scanner import PreScanResult, pre_scan
from datualizer_core.ingestion.quality_rules import resolve_quality, to_settings
from datualizer_core.ingestion.roles import RoleConfigError, assign_parameter_roles, resolve_run_columns
from datualizer_core.ingestion.sources import read_source
from datualizer_core.ingestion.units import resolve_units, split_header_unit
from datualizer_core.ingestion.type_inference import (
    ColumnKind,
    infer_column_kind,
    is_integer_text,
    parse_number,
)
from datualizer_core.pipeline.operators import clean_column_name, clean_names, drop_footer
from datualizer_core.timebase import (
    TimeConfigError,
    aware_to_epoch_us,
    check_zone,
    epoch_us_to_datetime,
    localize,
    origin_epoch_us,
    parse_datetime,
)

_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2}(?:[.,]\d+)?)$")


@dataclass(frozen=True)
class _TimeAxis:
    """Relative seconds plus where they start in absolute time."""

    seconds: pl.Series
    origin: datetime | None
    zone: str | None
    format: TimeFormat


def _parse_clock(text: str) -> float | None:
    """Seconds of an `H:MM:SS[.fff]` value (comma or dot fraction)."""
    m = _CLOCK_RE.match(text)
    if not m:
        return None
    return float(m.group(1)) * 3600.0 + float(m.group(2)) * 60.0 + float(m.group(3).replace(",", "."))


def _parse_number_seconds(text: str) -> float | None:
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def read_raw_table(pre_scan_result: PreScanResult) -> pl.DataFrame:
    """Read the pre-scanned text as an all-String table (empty frame for empty input)."""
    if not pre_scan_result.cleaned_text.strip():
        return pl.DataFrame()
    return pl.read_csv(
        io.StringIO(pre_scan_result.cleaned_text),
        separator=pre_scan_result.delimiter,
        infer_schema_length=0,
        has_header=True,
    )


class CSVLoader:
    """Configurable loader for measurement and sensor CSV data.

    All behaviour is driven by an `IngestionConfig`. The keyword shortcuts override the
    corresponding config fields. The returned dataset carries the fully resolved config
    in `ingestion_spec`, which reproduces the same result when passed back in.
    """

    def __init__(
        self,
        config: IngestionConfig | None = None,
        *,
        time_column: str | None = None,
        clean_column_names: bool | None = None,
        sentinels: Sequence[str] | None = None,
        pivot_long: bool | None = None,
    ) -> None:
        cfg = (config or IngestionConfig()).model_copy(deep=True)
        if time_column is not None:
            cfg.time_column = time_column
        if clean_column_names is not None:
            cfg.clean_column_names = clean_column_names
        if sentinels is not None:
            cfg.sentinels = list(sentinels)
        if pivot_long is not None:
            cfg.long_format.pivot = pivot_long
        self.config = cfg
        self.time_column = cfg.time_column
        self.clean_column_names = cfg.clean_column_names
        self.sentinels = {s.strip() for s in cfg.sentinels}
        self.upper_sentinels = {s.upper() for s in self.sentinels}

    def load(
        self,
        source: str | Path | TextIO,
        pre_scan_result: PreScanResult | None = None,
    ) -> DualModeDataset:
        """Load CSV data into a DualModeDataset with non-strict casting and error harvesting."""
        if pre_scan_result is None:
            pre_scan_result = pre_scan(source)
        df_raw = read_raw_table(pre_scan_result)
        return self.load_frame(df_raw, pre_scan_result.decimal_separator)

    def column_name_map(self, columns: Sequence[str]) -> dict[str, str]:
        """Return the loaded (cleaned, de-duplicated) name for every raw header."""
        time_col = self._resolve_time_column(list(columns))
        clean_col_names: list[str] = []
        seen: dict[str, int] = {}
        for i, col in enumerate(columns):
            if col == time_col:
                clean_name = "time_seconds" if self.clean_column_names else col
            elif self.clean_column_names:
                # The unit is metadata (P20): drop the part a unit pattern matched from the name
                units = self.config.units
                base = split_header_unit(col, units.header_patterns)[0] if units.from_headers else col
                clean_name = clean_column_name(base or col, index=i)
            else:
                clean_name = col

            count = seen.get(clean_name, 0)
            if count > 0:
                clean_col_names.append(f"{clean_name}_{count}")
                seen[clean_name] = count + 1
            else:
                seen[clean_name] = 1
                clean_col_names.append(clean_name)
        return dict(zip(columns, clean_col_names))

    def load_frame(
        self,
        df_raw: pl.DataFrame,
        decimal_separator: str = ".",
        audit_log: AuditLog | None = None,
        sources: pl.DataFrame | None = None,
    ) -> DualModeDataset:
        """Build a DualModeDataset from a raw all-String table (one column per header).

        `audit_log` may carry entries recorded before loading (e.g. merge conflicts); their row
        indices refer to rows of `df_raw`, like every entry the loader adds.
        """
        if df_raw.width == 0 or len(df_raw) == 0:
            empty_df = pl.DataFrame(schema={"time_seconds": pl.Float64})
            return DualModeDataset(
                df=empty_df, audit_log=audit_log or AuditLog(), time_col="time_seconds", sources=sources
            )

        audit_log = audit_log if audit_log is not None else AuditLog()

        # Identify time column and build column name mapping (raw -> cleaned)
        detected_time_col = self._resolve_time_column(df_raw.columns)
        raw_to_clean = self.column_name_map(df_raw.columns)

        # 1. Parse and convert time column to relative seconds (plus their absolute origin)
        time_axis = self._convert_time_column(
            df_raw[detected_time_col],
            col_name=raw_to_clean[detected_time_col],
            raw_col_name=detected_time_col,
            audit_log=audit_log,
        )
        time_seconds = time_axis.seconds

        # 2. Infer column kinds, then cast only numeric channels with error harvesting
        processed_series: list[pl.Series] = [time_seconds]
        column_kinds: dict[str, ColumnKind] = {time_seconds.name: ColumnKind.TIME}
        resolved_kinds: dict[str, ColumnKind] = {}
        explicit: set[str] = set()
        channel_names: list[str] = [c for c in df_raw.columns if c != detected_time_col]

        for col_name in channel_names:
            series = df_raw[col_name]
            clean_name = raw_to_clean[col_name]
            kind = self.config.kind_override(col_name, clean_name)
            if kind is not None and kind is not ColumnKind.TIME:
                explicit.add(clean_name)
            if kind is None or kind is ColumnKind.TIME:
                kind = infer_column_kind(
                    col_name,
                    series.to_list(),
                    decimal_sep=decimal_separator,
                    upper_sentinels=self.upper_sentinels,
                    vocabulary=self.config.vocabulary,
                    numeric_ratio_threshold=self.config.numeric_ratio_threshold,
                )
            column_kinds[clean_name] = kind
            resolved_kinds[col_name] = kind
            if kind in (ColumnKind.NUMERIC, ColumnKind.PARAMETER):
                processed_series.append(self._harvest_and_cast_channel(
                    series,
                    col_name=clean_name,
                    raw_col_name=col_name,
                    decimal_sep=decimal_separator,
                    audit_log=audit_log,
                ))
            else:
                processed_series.append(
                    self._clean_text_column(series, clean_name, col_name, kind, audit_log)
                )

        df = pl.DataFrame(processed_series)

        # 3. Safely apply drop_footer for trailing empty rows that are not audit errors
        error_rows = {e.row_index for e in audit_log.entries}
        time_col = "time_seconds" if "time_seconds" in df.columns else df.columns[0]
        channel_cols = [c for c in df.columns if c != time_col]

        if channel_cols and len(df) > 0:
            trailing_drop = 0
            for r_idx in range(len(df) - 1, -1, -1):
                if r_idx in error_rows:
                    break
                row_vals = [df[c][r_idx] for c in channel_cols]
                if all(v is None for v in row_vals):
                    trailing_drop += 1
                else:
                    break
            if trailing_drop > 0:
                df = df.slice(0, len(df) - trailing_drop)

        # 4. Long-format sources (time, variable, value) are pivoted to the wide primary mode
        lf_cfg = self._long_config_with_clean_names(raw_to_clean)
        spec = detect_long_format(df, column_kinds, time_col, lf_cfg, self.config.vocabulary)

        channel_attrs: dict[str, dict] = {}
        raw_variables: dict[str, str] = {}
        source_format = "wide" if spec is None else "long"
        if spec is not None and lf_cfg.pivot:
            pivoted = pivot_long_to_wide(df, spec, column_kinds, time_col, audit_log)
            raw_variables = {
                clean_column_name(v): v for v in df[spec.variable_col].drop_nulls().unique().to_list()
            }
            df, channel_attrs = pivoted.df, pivoted.channel_attrs
            column_kinds = pivoted.column_kinds
            for ch in channel_attrs:
                kind = self.config.kind_override(raw_variables.get(ch, ch), ch)
                if kind is None:
                    continue
                if kind not in (ColumnKind.NUMERIC, ColumnKind.PARAMETER):
                    raise RoleConfigError(
                        f"Long-format channel '{ch}' can only be NUMERIC or PARAMETER, not {kind.value}."
                    )
                column_kinds[ch] = kind
                explicit.add(ch)

        # 5. Measurement vs. parameter roles (constant per run -> PARAMETER)
        run_columns = resolve_run_columns(
            df, column_kinds, self.config.roles, self.config.vocabulary, raw_to_clean
        )
        column_kinds = assign_parameter_roles(
            df, column_kinds, run_columns, explicit, self.config.roles, audit_log
        )

        # 6. Quality flag settings (flags themselves are computed lazily by the dataset)
        channels = [c for c, k in column_kinds.items() if k is ColumnKind.NUMERIC]
        names = {**raw_to_clean, **{raw: ch for ch, raw in raw_variables.items()}}
        quality = resolve_quality(
            self.config.quality, df, channels, run_columns, channel_attrs, names, self.config.vocabulary
        )

        # 7. Units from headers or long-table attributes (P20)
        numeric = [c for c, k in column_kinds.items() if k in (ColumnKind.NUMERIC, ColumnKind.PARAMETER)]
        units = resolve_units(
            self.config.units, numeric, raw_to_clean, names, channel_attrs, self.config.vocabulary
        )

        # Freeze every decision into a replayable spec
        resolved = self.config.model_copy(deep=True)
        resolved.quality = quality
        resolved.units.units = units
        resolved.time_column = detected_time_col
        resolved.time.format = time_axis.format
        resolved.column_kinds = {
            raw: column_kinds.get(raw_to_clean[raw], kind) for raw, kind in resolved_kinds.items()
        }
        resolved.column_kinds.update({ch: column_kinds[ch] for ch in channel_attrs})
        resolved.roles.run_columns = run_columns
        if spec is None:
            resolved.long_format = LongFormatConfig(mode=LongFormatMode.OFF, pivot=lf_cfg.pivot)
        else:
            resolved.long_format = LongFormatConfig(
                mode=LongFormatMode.FORCE,
                pivot=lf_cfg.pivot,
                variable_col=spec.variable_col,
                value_col=spec.value_col,
                channel_attr_cols=list(spec.channel_attr_cols),
                row_meta_cols=list(spec.row_meta_cols),
            )

        return DualModeDataset(
            df=df,
            audit_log=audit_log,
            time_col=time_col,
            column_kinds=column_kinds,
            channel_attrs=channel_attrs,
            source_format=source_format,
            ingestion_spec=resolved,
            run_columns=run_columns,
            aborted_run_fraction=self.config.runs.aborted_fraction,
            sources=sources,
            quality=to_settings(quality),
            units=units,
            time_origin=time_axis.origin,
            time_zone=time_axis.zone,
        )

    def _long_config_with_clean_names(self, raw_to_clean: dict[str, str]) -> LongFormatConfig:
        """Translate user-given raw header names in the long-format config to loaded column names."""
        lf = self.config.long_format

        def name(c: str) -> str:
            return raw_to_clean.get(c, c)

        return lf.model_copy(update={
            "variable_col": name(lf.variable_col) if lf.variable_col else None,
            "value_col": name(lf.value_col) if lf.value_col else None,
            "channel_attr_cols": (
                [name(c) for c in lf.channel_attr_cols] if lf.channel_attr_cols is not None else None
            ),
            "row_meta_cols": (
                [name(c) for c in lf.row_meta_cols] if lf.row_meta_cols is not None else None
            ),
        })

    def _resolve_time_column(self, columns: list[str]) -> str:
        explicit = self.time_column or next(
            (c for c, k in self.config.column_kinds.items() if k is ColumnKind.TIME), None
        )
        if explicit:
            for c in columns:
                if c == explicit or c.lower() == explicit.lower():
                    return c

        candidates = [n.lower() for n in self.config.vocabulary.time_names]
        for c in columns:
            if c.lower() in candidates:
                return c

        return columns[0]

    def _convert_time_column(
        self,
        series: pl.Series,
        col_name: str,
        raw_col_name: str,
        audit_log: AuditLog,
    ) -> _TimeAxis:
        """Convert the time column into Float64 seconds since the first valid sample.

        Also returns the absolute time of that sample (`origin`, if determinable), the zone
        absolute times are shown in, and the resolved format.
        """
        tcfg = self.config.time
        vals = series.to_list()
        sample = [
            str(v).strip()
            for v in vals
            if v is not None and str(v).strip() != "" and str(v).strip() not in self.sentinels
        ][:50]
        if tcfg.timezone is not None:
            check_zone(tcfg.timezone)

        if not sample:
            for idx, val in enumerate(vals):
                if val is not None and str(val).strip() != "":
                    audit_log.add(idx, col_name, val, "invalid_timestamp", raw_column=raw_col_name)
            return _TimeAxis(pl.Series(col_name, [None] * len(series), dtype=pl.Float64), None, None, tcfg.format)

        parsers = {
            TimeFormat.CLOCK: _parse_clock,
            TimeFormat.DATETIME: lambda v: parse_datetime(v, tcfg.datetime_formats),
            TimeFormat.NUMERIC: _parse_number_seconds,
        }
        fmt = tcfg.format
        if fmt is TimeFormat.AUTO:
            fmt = next(
                (f for f in (TimeFormat.CLOCK, TimeFormat.DATETIME)
                 if sum(parsers[f](v) is not None for v in sample) >= len(sample) / 2),
                TimeFormat.NUMERIC,
            )
        elif sum(parsers[fmt](v) is not None for v in sample) < len(sample) / 2:
            raise TimeConfigError(
                f"Time column '{raw_col_name}' does not match the forced format '{fmt.value}'."
            )
        if fmt is TimeFormat.DATETIME and tcfg.origin is not None:
            raise TimeConfigError(
                f"Time column '{raw_col_name}' holds date-times; an origin is only allowed for "
                "clock or numeric time."
            )

        parsed: list = []
        for idx, raw in enumerate(vals):
            if raw is None or str(raw).strip() == "":
                parsed.append(None)
                continue
            v_str = str(raw).strip()
            if v_str in self.sentinels or v_str.upper() in self.upper_sentinels:
                audit_log.add(idx, col_name, raw, "sentinel_value", raw_column=raw_col_name)
                parsed.append(None)
                continue
            value = parsers[fmt](v_str)
            if value is None:
                audit_log.add(idx, col_name, raw, "invalid_timestamp", raw_column=raw_col_name)
            parsed.append(value)

        if fmt is TimeFormat.DATETIME:
            return self._datetime_axis(parsed, vals, col_name, raw_col_name, audit_log)

        seconds = parsed
        if fmt is TimeFormat.CLOCK:
            # Time of day without date: a drop of more than 12 h is a midnight rollover
            seconds, day_offset, prev = [], 0.0, None
            for value in parsed:
                if value is None:
                    seconds.append(None)
                    continue
                if prev is not None and value + day_offset < prev - 43200.0:
                    day_offset += 86400.0
                prev = value + day_offset
                seconds.append(prev)
        base = next((s for s in seconds if s is not None), None)
        rel = [None if s is None else s - base for s in seconds]

        origin = zone = None
        if tcfg.origin is not None:
            origin_us, zone = origin_epoch_us(tcfg.origin, tcfg.timezone)
            origin = epoch_us_to_datetime(origin_us, zone)
        return _TimeAxis(pl.Series(col_name, rel, dtype=pl.Float64), origin, zone, fmt)

    def _datetime_axis(
        self,
        parsed: list[datetime | None],
        vals: list,
        col_name: str,
        raw_col_name: str,
        audit_log: AuditLog,
    ) -> _TimeAxis:
        """Epoch-based seconds for date-times; with offset -> UTC instants, else read in the zone."""
        tz = self.config.time.timezone
        n_aware = sum(d is not None and d.tzinfo is not None for d in parsed)
        n_naive = sum(d is not None and d.tzinfo is None for d in parsed)
        aware = n_aware >= n_naive and n_aware > 0

        values: list[datetime | None] = []
        for idx, d in enumerate(parsed):
            if d is not None and (d.tzinfo is not None) != aware:
                audit_log.add(idx, col_name, vals[idx], "mixed_timezone", raw_column=raw_col_name)
                d = None
            values.append(d)

        if aware:
            epoch = [None if d is None else aware_to_epoch_us(d) for d in values]
            zone = tz or "UTC"
        else:
            loc = localize(values, tz)
            for idx in loc.ambiguous:
                audit_log.add(idx, col_name, vals[idx], "ambiguous_local_time", raw_column=raw_col_name)
            for idx in loc.nonexistent:
                audit_log.add(idx, col_name, vals[idx], "nonexistent_local_time", raw_column=raw_col_name)
            epoch, zone = loc.epoch_us, tz

        base = next((e for e in epoch if e is not None), None)
        rel = [None if e is None else (e - base) / 1e6 for e in epoch]
        origin = None if base is None else epoch_us_to_datetime(base, zone)
        return _TimeAxis(pl.Series(col_name, rel, dtype=pl.Float64), origin, zone, TimeFormat.DATETIME)

    def _harvest_and_cast_channel(
        self,
        series: pl.Series,
        col_name: str,
        raw_col_name: str,
        decimal_sep: str,
        audit_log: AuditLog,
    ) -> pl.Series:
        """Perform non-strict casting to Float64 while harvesting errors into the AuditLog."""
        vals = series.to_list()
        f64_vals: list[float | None] = []

        for idx, raw in enumerate(vals):
            if raw is None:
                f64_vals.append(None)
                continue
            v_str = str(raw).strip()
            if not v_str:
                f64_vals.append(None)
                continue

            if v_str in self.sentinels or v_str.upper() in self.upper_sentinels:
                audit_log.add(idx, col_name, raw, "sentinel_value", raw_column=raw_col_name)
                f64_vals.append(None)
                continue

            val_f64 = parse_number(v_str, decimal_sep)
            if val_f64 is None:
                audit_log.add(idx, col_name, raw, "non_convertible_float", raw_column=raw_col_name)
            f64_vals.append(val_f64)

        return pl.Series(col_name, f64_vals, dtype=pl.Float64)

    def _clean_text_column(
        self,
        series: pl.Series,
        col_name: str,
        raw_col_name: str,
        kind: ColumnKind,
        audit_log: AuditLog,
    ) -> pl.Series:
        """Strip text cells and map empty/sentinel cells to null; identifiers become Int64."""
        vals: list[str | None] = []
        for raw in series.to_list():
            v_str = str(raw).strip() if raw is not None else ""
            if not v_str or v_str.upper() in self.upper_sentinels:
                vals.append(None)
            else:
                vals.append(v_str)

        if kind is ColumnKind.IDENTIFIER:
            ints: list[int | None] = []
            for idx, v in enumerate(vals):
                if v is not None and not is_integer_text(v):
                    # Only reachable when the user forces IDENTIFIER on a non-integer column
                    audit_log.add(idx, col_name, v, "non_convertible_int", raw_column=raw_col_name)
                    v = None
                ints.append(int(v) if v is not None else None)
            return pl.Series(col_name, ints, dtype=pl.Int64)
        return pl.Series(col_name, vals, dtype=pl.String)


def load_csv(
    source: str | Path | TextIO,
    time_column: str | None = None,
    clean_column_names: bool | None = None,
    sentinels: Sequence[str] | None = None,
    pre_scan_result: PreScanResult | None = None,
    pivot_long: bool | None = None,
    config: IngestionConfig | None = None,
) -> DualModeDataset:
    """Convenience function to load a CSV into DualModeDataset.

    The result records the file and its SHA-256 in `sources`. Several files: see `load_csvs`.
    """
    loader = CSVLoader(
        config,
        time_column=time_column,
        clean_column_names=clean_column_names,
        sentinels=sentinels,
        pivot_long=pivot_long,
    )
    info = read_source(source)
    if pre_scan_result is None:
        pre_scan_result = pre_scan(info.scan_input)
    df_raw = read_raw_table(pre_scan_result)
    sources = pl.DataFrame(
        [{
            "source": info.label,
            "sha256": info.sha256,
            "n_rows": len(df_raw),
            "n_kept": len(df_raw),
            "n_replaced": 0,
            "n_conflicts": 0,
            "duplicate_of": None,
        }],
        schema=SOURCES_SCHEMA,
    )
    return loader.load_frame(df_raw, pre_scan_result.decimal_separator, sources=sources)
