"""Loader module for parsing CSV into Polars DataFrame with error harvesting."""

from __future__ import annotations

from datetime import datetime
import io
from pathlib import Path
import re
from typing import Sequence, TextIO
import polars as pl

from datualizer_core.dataset import AuditEntry, AuditLog, DualModeDataset
from datualizer_core.ingestion.long_format import detect_long_format, pivot_long_to_wide
from datualizer_core.ingestion.pre_scanner import PreScanResult, pre_scan
from datualizer_core.ingestion.type_inference import ColumnKind, infer_column_kind, parse_number
from datualizer_core.pipeline.operators import clean_column_name, clean_names, drop_footer


class CSVLoader:
    """Configurable loader for measurement and sensor CSV data."""

    def __init__(
        self,
        time_column: str | None = None,
        clean_column_names: bool = True,
        sentinels: Sequence[str] | None = None,
        pivot_long: bool = True,
    ) -> None:
        self.time_column = time_column
        self.clean_column_names = clean_column_names
        self.pivot_long = pivot_long
        raw_sentinels = (
            sentinels
            if sentinels is not None
            else [
                "", "NA", "N/A", "null", "NULL", "NaN", "None", "-999", "ERR", "#N/A", "#VALUE!",
                "nan", "NAN", "error", "ERROR", "undef", "UNDEF", "overflow", "OVERFLOW"
            ]
        )
        self.sentinels = {s.strip() for s in raw_sentinels}
        self.upper_sentinels = {s.upper() for s in self.sentinels}

    def load(
        self,
        source: str | Path | TextIO,
        pre_scan_result: PreScanResult | None = None,
    ) -> DualModeDataset:
        """Load CSV data into a DualModeDataset with non-strict casting and error harvesting."""
        if pre_scan_result is None:
            pre_scan_result = pre_scan(source)

        if not pre_scan_result.cleaned_text.strip():
            empty_df = pl.DataFrame(schema={"time_seconds": pl.Float64})
            return DualModeDataset(df=empty_df, audit_log=AuditLog(), time_col="time_seconds")

        df_raw = pl.read_csv(
            io.StringIO(pre_scan_result.cleaned_text),
            separator=pre_scan_result.delimiter,
            infer_schema_length=0,
            has_header=True,
        )

        if len(df_raw) == 0:
            empty_df = pl.DataFrame(schema={"time_seconds": pl.Float64})
            return DualModeDataset(df=empty_df, audit_log=AuditLog(), time_col="time_seconds")

        audit_log = AuditLog()

        # Identify time column
        detected_time_col = self._resolve_time_column(df_raw.columns)

        # Build column name mapping (raw -> cleaned)
        clean_col_names: list[str] = []
        seen: dict[str, int] = {}
        for i, col in enumerate(df_raw.columns):
            if col == detected_time_col:
                clean_name = "time_seconds" if self.clean_column_names else col
            elif self.clean_column_names:
                clean_name = clean_column_name(col, index=i)
            else:
                clean_name = col

            count = seen.get(clean_name, 0)
            if count > 0:
                unique_name = f"{clean_name}_{count}"
                seen[clean_name] = count + 1
                clean_col_names.append(unique_name)
            else:
                seen[clean_name] = 1
                clean_col_names.append(clean_name)

        raw_to_clean = dict(zip(df_raw.columns, clean_col_names))

        # 1. Parse and convert time column to relative seconds
        time_seconds = self._convert_time_column(
            df_raw[detected_time_col],
            col_name=raw_to_clean[detected_time_col],
            raw_col_name=detected_time_col,
            audit_log=audit_log,
        )

        # 2. Infer column kinds, then cast only numeric channels with error harvesting
        processed_series: list[pl.Series] = [time_seconds]
        column_kinds: dict[str, ColumnKind] = {time_seconds.name: ColumnKind.TIME}
        channel_names: list[str] = [c for c in df_raw.columns if c != detected_time_col]

        for col_name in channel_names:
            series = df_raw[col_name]
            clean_name = raw_to_clean[col_name]
            kind = infer_column_kind(
                col_name,
                series.to_list(),
                decimal_sep=pre_scan_result.decimal_separator,
                upper_sentinels=self.upper_sentinels,
            )
            column_kinds[clean_name] = kind
            if kind is ColumnKind.NUMERIC:
                processed_series.append(self._harvest_and_cast_channel(
                    series,
                    col_name=clean_name,
                    raw_col_name=col_name,
                    decimal_sep=pre_scan_result.decimal_separator,
                    audit_log=audit_log,
                ))
            else:
                processed_series.append(self._clean_text_column(series, clean_name, kind))

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
        spec = detect_long_format(df, column_kinds, time_col)
        if spec is not None and not self.pivot_long:
            return DualModeDataset(
                df=df, audit_log=audit_log, time_col=time_col,
                column_kinds=column_kinds, source_format="long",
            )
        if spec is not None:
            pivoted = pivot_long_to_wide(df, spec, column_kinds, time_col, audit_log)
            return DualModeDataset(
                df=pivoted.df,
                audit_log=audit_log,
                time_col=time_col,
                column_kinds=pivoted.column_kinds,
                channel_attrs=pivoted.channel_attrs,
                source_format="long",
            )

        return DualModeDataset(
            df=df, audit_log=audit_log, time_col=time_col, column_kinds=column_kinds
        )

    def _resolve_time_column(self, columns: list[str]) -> str:
        if self.time_column:
            for c in columns:
                if c == self.time_column or c.lower() == self.time_column.lower():
                    return c

        candidates = ["time", "time_seconds", "zeit", "timestamp", "datetime", "date"]
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
    ) -> pl.Series:
        """Convert time column into Float64 relative seconds from start of measurement."""
        if len(series) == 0:
            return pl.Series(col_name, [], dtype=pl.Float64)

        vals = series.to_list()
        sample = [
            str(v).strip()
            for v in vals
            if v is not None and str(v).strip() != "" and str(v).strip() not in self.sentinels
        ][:50]

        if not sample:
            for idx, val in enumerate(vals):
                if val is not None and str(val).strip() != "":
                    audit_log.add(idx, col_name, val, "invalid_timestamp", raw_column=raw_col_name)
            return pl.Series(col_name, [None] * len(series), dtype=pl.Float64)

        # Detect format
        is_hms = sum(
            bool(re.match(r"^\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?$", v)) for v in sample
        ) >= len(sample) / 2

        is_dt = False
        if not is_hms:
            dt_matches = sum(bool(self._parse_datetime(v)) for v in sample)
            if dt_matches >= len(sample) / 2:
                is_dt = True

        rel_seconds_list: list[float | None] = []
        base_seconds: float | None = None

        if is_hms:
            day_offset = 0.0
            prev_abs: float | None = None
            for idx, raw in enumerate(vals):
                if raw is None or str(raw).strip() == "":
                    rel_seconds_list.append(None)
                    continue
                v_str = str(raw).strip()
                if v_str in self.sentinels or v_str.upper() in self.upper_sentinels:
                    audit_log.add(idx, col_name, raw, "sentinel_value", raw_column=raw_col_name)
                    rel_seconds_list.append(None)
                    continue
                m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2}(?:[.,]\d+)?)$", v_str)
                if not m:
                    audit_log.add(idx, col_name, raw, "invalid_timestamp", raw_column=raw_col_name)
                    rel_seconds_list.append(None)
                    continue
                h = float(m.group(1))
                mins = float(m.group(2))
                secs = float(m.group(3).replace(",", "."))
                abs_s = h * 3600.0 + mins * 60.0 + secs

                if prev_abs is not None and (abs_s + day_offset) < prev_abs - 43200.0:
                    day_offset += 86400.0

                current_total = abs_s + day_offset
                if base_seconds is None:
                    base_seconds = current_total
                rel_seconds_list.append(current_total - base_seconds)
                prev_abs = current_total

        elif is_dt:
            for idx, raw in enumerate(vals):
                if raw is None or str(raw).strip() == "":
                    rel_seconds_list.append(None)
                    continue
                v_str = str(raw).strip()
                if v_str in self.sentinels or v_str.upper() in self.upper_sentinels:
                    audit_log.add(idx, col_name, raw, "sentinel_value", raw_column=raw_col_name)
                    rel_seconds_list.append(None)
                    continue
                dt = self._parse_datetime(v_str)
                if dt is None:
                    audit_log.add(idx, col_name, raw, "invalid_timestamp", raw_column=raw_col_name)
                    rel_seconds_list.append(None)
                    continue
                abs_s = dt.timestamp()
                if base_seconds is None:
                    base_seconds = abs_s
                rel_seconds_list.append(abs_s - base_seconds)

        else:
            for idx, raw in enumerate(vals):
                if raw is None or str(raw).strip() == "":
                    rel_seconds_list.append(None)
                    continue
                v_str = str(raw).strip()
                if v_str in self.sentinels or v_str.upper() in self.upper_sentinels:
                    audit_log.add(idx, col_name, raw, "sentinel_value", raw_column=raw_col_name)
                    rel_seconds_list.append(None)
                    continue
                try:
                    num = float(v_str.replace(",", "."))
                except ValueError:
                    audit_log.add(idx, col_name, raw, "invalid_timestamp", raw_column=raw_col_name)
                    rel_seconds_list.append(None)
                    continue
                if base_seconds is None:
                    base_seconds = num
                rel_seconds_list.append(num - base_seconds)

        return pl.Series(col_name, rel_seconds_list, dtype=pl.Float64)

    @staticmethod
    def _parse_datetime(s: str) -> datetime | None:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            pass
        for fmt in (
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y %H:%M:%S.%f",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M:%S.%f",
        ):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                pass
        return None

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

    def _clean_text_column(self, series: pl.Series, col_name: str, kind: ColumnKind) -> pl.Series:
        """Strip text cells and map empty/sentinel cells to null; identifiers become Int64."""
        vals: list[str | None] = []
        for raw in series.to_list():
            v_str = str(raw).strip() if raw is not None else ""
            if not v_str or v_str.upper() in self.upper_sentinels:
                vals.append(None)
            else:
                vals.append(v_str)

        if kind is ColumnKind.IDENTIFIER:
            return pl.Series(col_name, [int(v) if v is not None else None for v in vals], dtype=pl.Int64)
        return pl.Series(col_name, vals, dtype=pl.String)


def load_csv(
    source: str | Path | TextIO,
    time_column: str | None = None,
    clean_column_names: bool = True,
    sentinels: Sequence[str] | None = None,
    pre_scan_result: PreScanResult | None = None,
    pivot_long: bool = True,
) -> DualModeDataset:
    """Convenience function to load a CSV into DualModeDataset."""
    loader = CSVLoader(
        time_column=time_column,
        clean_column_names=clean_column_names,
        sentinels=sentinels,
        pivot_long=pivot_long,
    )
    return loader.load(source, pre_scan_result=pre_scan_result)
