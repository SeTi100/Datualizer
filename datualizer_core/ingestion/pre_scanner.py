"""Pre-scanner module for raw CSV inspection and dialect detection.

Provides a two-stage pre-scanner:
1. Structural Scan: header detection, footer detection, delimiter recognition,
   and header cleaning.
2. Dialect & Content Scan: decimal separator detection (German comma vs dot)
   and sentinel value detection.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO


DEFAULT_SENTINELS = {"", "NA", "N/A", "null", "NULL", "NaN", "None", "-999", "ERR", "#N/A", "#VALUE!"}


@dataclass
class PreScanResult:
    """Results from two-stage pre-scanning of raw CSV data."""

    delimiter: str
    decimal_separator: str
    header_line_idx: int
    skip_rows: int
    raw_headers: list[str]
    cleaned_headers: list[str]
    footer_lines_count: int
    data_lines_count: int
    cleaned_text: str
    sentinels: set[str] = field(default_factory=set)


class PreScanner:
    """Two-stage pre-scanner for tabular sensor and measurement data."""

    def __init__(
        self,
        candidate_delimiters: list[str] | None = None,
        candidate_sentinels: set[str] | None = None,
    ) -> None:
        self.candidate_delimiters = candidate_delimiters or [",", ";", "\t", "|"]
        self.candidate_sentinels = candidate_sentinels or set(DEFAULT_SENTINELS)

    def scan(self, source: str | Path | TextIO) -> PreScanResult:
        """Scan raw input text, detect dialect, headers, footers, and produce cleaned text."""
        raw_lines = self._read_lines(source)
        if not raw_lines:
            return PreScanResult(
                delimiter=",",
                decimal_separator=".",
                header_line_idx=0,
                skip_rows=0,
                raw_headers=[],
                cleaned_headers=[],
                footer_lines_count=0,
                data_lines_count=0,
                cleaned_text="",
                sentinels=set(),
            )

        # Stage 1: Structural Scan
        # 1.1 Strip trailing blank lines
        while raw_lines and not raw_lines[-1].strip():
            raw_lines.pop()

        if not raw_lines:
            return PreScanResult(
                delimiter=",",
                decimal_separator=".",
                header_line_idx=0,
                skip_rows=0,
                raw_headers=[],
                cleaned_headers=[],
                footer_lines_count=0,
                data_lines_count=0,
                cleaned_text="",
                sentinels=set(),
            )

        # 1.2 Delimiter detection
        delimiter = self._detect_delimiter(raw_lines)

        # 1.3 Header row detection
        header_line_idx, raw_headers = self._detect_header_row(raw_lines, delimiter)
        cleaned_headers = self._clean_headers(raw_headers)

        # 1.4 Footer detection
        data_start_idx = header_line_idx + 1
        footer_lines_count = self._detect_footer_lines(
            raw_lines[data_start_idx:], delimiter
        )

        valid_data_end_idx = len(raw_lines) - footer_lines_count
        if valid_data_end_idx < data_start_idx:
            valid_data_end_idx = data_start_idx

        data_lines = raw_lines[data_start_idx:valid_data_end_idx]
        data_lines_count = len(data_lines)

        # Stage 2: Dialect & Content Scan
        decimal_separator = self._detect_decimal_separator(data_lines, delimiter)
        found_sentinels = self._detect_sentinels(data_lines, delimiter)

        # Build cleaned text with clean headers and footers removed
        cleaned_text = self._build_cleaned_text(
            cleaned_headers, delimiter, data_lines
        )

        return PreScanResult(
            delimiter=delimiter,
            decimal_separator=decimal_separator,
            header_line_idx=header_line_idx,
            skip_rows=header_line_idx,
            raw_headers=raw_headers,
            cleaned_headers=cleaned_headers,
            footer_lines_count=footer_lines_count,
            data_lines_count=data_lines_count,
            cleaned_text=cleaned_text,
            sentinels=found_sentinels,
        )

    def _read_lines(self, source: str | Path | TextIO) -> list[str]:
        if isinstance(source, Path) or (
            isinstance(source, str) and ("\n" not in source and "\r" not in source and Path(source).is_file())
        ):
            path = Path(source)
            try:
                with open(path, "r", encoding="utf-8-sig") as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(path, "r", encoding="latin-1") as f:
                    lines = f.readlines()
        elif isinstance(source, str):
            lines = source.lstrip("\ufeff").splitlines(keepends=True)
        elif hasattr(source, "readlines"):
            lines = source.readlines()
        elif hasattr(source, "read"):
            lines = source.read().splitlines(keepends=True)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        if lines and lines[0].startswith("\ufeff"):
            lines[0] = lines[0].lstrip("\ufeff")
        return lines

    def _detect_delimiter(self, lines: list[str]) -> str:
        sample = lines[: min(len(lines), 100)]
        best_delim = ","
        best_score = -1.0

        for delim in self.candidate_delimiters:
            try:
                reader = csv.reader(sample, delimiter=delim)
                rows = [row for row in reader if row]
                if not rows:
                    continue
                row_lengths = [len(row) for row in rows]
                most_common_len = max(set(row_lengths), key=row_lengths.count)
                if most_common_len > 1:
                    consistency = row_lengths.count(most_common_len) / len(row_lengths)
                    # Check if tokens contain other structural delimiters (; or \t)
                    other_delims = [d for d in [";", "\t", "|"] if d != delim]
                    embedded_other = sum(
                        1 for r in rows for cell in r for od in other_delims if od in cell
                    )
                    score = consistency * 1000 + most_common_len - (embedded_other * 50)
                    if score > best_score:
                        best_score = score
                        best_delim = delim
            except Exception:
                continue

        return best_delim

    def _detect_header_row(
        self, lines: list[str], delimiter: str
    ) -> tuple[int, list[str]]:
        for idx, line in enumerate(lines[: min(len(lines), 50)]):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = next(csv.reader([line], delimiter=delimiter))
                if len(row) > 1:
                    # Valid header line found
                    return idx, row
            except Exception:
                continue
        # Fallback to row 0
        try:
            row = next(csv.reader([lines[0]], delimiter=delimiter))
            return 0, row
        except Exception:
            return 0, []

    def _clean_headers(self, raw_headers: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: dict[str, int] = {}

        for i, h in enumerate(raw_headers):
            name = h.lstrip("\ufeff").strip()
            if not name:
                # Requirement: Spalte 0 ist leerer String -> automatisch z. B. 'time' benennen
                name = "time" if i == 0 else f"col_{i}"

            # Deduplicate if needed
            count = seen.get(name, 0)
            if count > 0:
                unique_name = f"{name}_{count}"
                seen[name] = count + 1
                name = unique_name
            else:
                seen[name] = 1

            cleaned.append(name)
        return cleaned

    def _detect_footer_lines(self, data_lines: list[str], delimiter: str) -> int:
        footer_count = 0
        for line in reversed(data_lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//", "*")):
                footer_count += 1
                continue
            try:
                row = next(csv.reader([line], delimiter=delimiter))
                if len(row) <= 1:
                    footer_count += 1
                    continue
                # Measurement channels are columns 1..end
                channels = row[1:]
                # Footer lines with purely empty measurement values (empty string or whitespace only)
                if all(not c.strip() for c in channels):
                    footer_count += 1
                else:
                    # Found first valid data row from bottom
                    break
            except Exception:
                footer_count += 1
        return footer_count

    def _detect_decimal_separator(
        self, data_lines: list[str], delimiter: str
    ) -> str:
        sample = data_lines[: min(len(data_lines), 200)]
        comma_count = 0
        dot_count = 0

        for line in sample:
            try:
                row = next(csv.reader([line], delimiter=delimiter))
                # Check data columns (from index 1 onward or all)
                for val in row[1:]:
                    val_str = val.strip()
                    if "," in val_str and "." in val_str:
                        if val_str.rfind(",") > val_str.rfind("."):
                            comma_count += 1
                        else:
                            dot_count += 1
                    elif re.search(r"\d,\d+", val_str):
                        comma_count += 1
                    elif re.search(r"\d\.\d+", val_str):
                        dot_count += 1
            except Exception:
                continue

        return "," if comma_count > dot_count else "."

    def _detect_sentinels(
        self, data_lines: list[str], delimiter: str
    ) -> set[str]:
        found: set[str] = set()
        sample = data_lines[: min(len(data_lines), 200)]
        for line in sample:
            try:
                row = next(csv.reader([line], delimiter=delimiter))
                for val in row:
                    val_str = val.strip()
                    if val_str in self.candidate_sentinels:
                        found.add(val_str)
            except Exception:
                continue
        return found

    def _build_cleaned_text(
        self, cleaned_headers: list[str], delimiter: str, data_lines: list[str]
    ) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
        writer.writerow(cleaned_headers)
        buf.write("".join(data_lines))
        return buf.getvalue()


def pre_scan(
    source: str | Path | TextIO,
    candidate_delimiters: list[str] | None = None,
    candidate_sentinels: set[str] | None = None,
) -> PreScanResult:
    """Convenience functional wrapper around PreScanner."""
    scanner = PreScanner(
        candidate_delimiters=candidate_delimiters,
        candidate_sentinels=candidate_sentinels,
    )
    return scanner.scan(source)
