"""Unit tests for Datualizer ingestion components (pre-scanner, loader, operators, dataset)."""

from __future__ import annotations

import io
import numpy as np
import polars as pl
import pytest

from datualizer_core import (
    AuditEntry,
    AuditLog,
    DualModeDataset,
    clean_names,
    drop_footer,
    load_csv,
    pre_scan,
    unpivot,
)


class TestPreScanner:
    """Tests for two-stage pre-scanner."""

    def test_delimiter_detection(self) -> None:
        comma_csv = "time,ch1,ch2\n00:00:00,1.1,2.2\n00:00:01,1.2,2.3\n"
        semi_csv = "time;ch1;ch2\n00:00:00;1,1;2,2\n00:00:01;1,2;2,3\n"
        tab_csv = "time\tch1\tch2\n00:00:00\t1.1\t2.2\n00:00:01\t1.2\t2.3\n"

        res_comma = pre_scan(comma_csv)
        assert res_comma.delimiter == ","

        res_semi = pre_scan(semi_csv)
        assert res_semi.delimiter == ";"

        res_tab = pre_scan(tab_csv)
        assert res_tab.delimiter == "\t"

    def test_decimal_separator_detection(self) -> None:
        german_csv = 'time;ch1\n00:00:00;24,109\n00:00:01;24,095\n'
        std_csv = 'time,ch1\n00:00:00,24.109\n00:00:01,24.095\n'

        assert pre_scan(german_csv).decimal_separator == ","
        assert pre_scan(std_csv).decimal_separator == "."

    def test_empty_header_column_0(self) -> None:
        # Header with empty column 0 -> should become 'time'
        raw = '"","Kanal 1","Kanal 2"\n"00:00:00","10.0","20.0"\n'
        res = pre_scan(raw)
        assert res.cleaned_headers[0] == "time"
        assert res.cleaned_headers[1] == "Kanal 1"
        assert res.cleaned_headers[2] == "Kanal 2"

    def test_empty_intermediate_header(self) -> None:
        raw = '"time","","Kanal 2"\n"00:00:00","10.0","20.0"\n'
        res = pre_scan(raw)
        assert res.cleaned_headers[0] == "time"
        assert res.cleaned_headers[1] == "col_1"
        assert res.cleaned_headers[2] == "Kanal 2"

    def test_duplicate_header_deduplication(self) -> None:
        raw = "time,temp,temp\n00:00:00,10.0,20.0\n"
        res = pre_scan(raw)
        assert res.cleaned_headers == ["time", "temp", "temp_1"]

    def test_metadata_skip_rows(self) -> None:
        raw = "# Test Bench Run 42\n# Sensor Model: PT100\ntime;ch1\n00:00:00;10,5\n00:00:01;11,0\n"
        res = pre_scan(raw)
        assert res.header_line_idx == 2
        assert res.skip_rows == 2
        assert res.cleaned_headers == ["time", "ch1"]
        assert res.data_lines_count == 2

    def test_footer_detection_with_empty_channels(self) -> None:
        raw = (
            'time;ch1;ch2\n'
            '00:00:00;10,0;20,0\n'
            '00:00:01;11,0;21,0\n'
            '00:00:02;;\n'  # footer: time present, channels empty
            ';;\n'          # footer: all empty
            '\n'            # trailing empty line
        )
        res = pre_scan(raw)
        assert res.footer_lines_count == 2
        assert res.data_lines_count == 2

    def test_semicolon_delim_with_german_decimal_without_header(self) -> None:
        csv_text = (
            "1;1,5;2,5;3,5;4,5;5,5\n"
            "2;1,6;2,6;3,6;4,6;5,6\n"
            "3;1,7;2,7;3,7;4,7;5,7\n"
        )
        res = pre_scan(csv_text)
        assert res.delimiter == ";"
        assert res.decimal_separator == ","

    def test_utf8_bom_handling(self) -> None:
        csv_text = "\ufefftime,ch1,ch2\n00:00:00,1.0,2.0\n"
        res = pre_scan(csv_text)
        assert res.cleaned_headers[0] == "time"
        assert "\ufeff" not in res.cleaned_headers[0]



class TestLoaderAndErrorHarvesting:
    """Tests for CSVLoader, float casting, error harvesting, and timestamps."""

    def test_timestamp_conversion_to_relative_seconds(self) -> None:
        csv_data = "time;ch1\n01:00:00;10,0\n01:00:05;11,0\n01:01:00;12,0\n"
        ds = load_csv(csv_data)
        assert "time_seconds" in ds.columns
        assert ds["time_seconds"].dtype == pl.Float64
        # 01:00:00 -> 0.0s, 01:00:05 -> 5.0s, 01:01:00 -> 60.0s
        assert ds["time_seconds"].to_list() == [0.0, 5.0, 60.0]

    def test_timestamp_midnight_rollover(self) -> None:
        csv_data = "time;ch1\n23:59:58;1,0\n23:59:59;2,0\n00:00:00;3,0\n00:00:01;4,0\n"
        ds = load_csv(csv_data)
        assert ds["time_seconds"].to_list() == [0.0, 1.0, 2.0, 3.0]

    def test_numeric_timestamp_column(self) -> None:
        csv_data = "time,ch1\n100.0,1.0\n100.5,2.0\n101.0,3.0\n"
        ds = load_csv(csv_data)
        assert ds["time_seconds"].to_list() == [0.0, 0.5, 1.0]

    def test_german_decimal_comma_casting(self) -> None:
        csv_data = "time;ch1;ch2\n00:00:00;24,109;100,5\n00:00:01;24,095;101,25\n"
        ds = load_csv(csv_data)
        assert ds["ch1"].dtype == pl.Float64
        assert ds["ch2"].dtype == pl.Float64
        assert ds["ch1"].to_list() == [24.109, 24.095]
        assert ds["ch2"].to_list() == [100.5, 101.25]

    def test_non_strict_float_casting_with_error_harvesting(self) -> None:
        csv_data = (
            "time;ch1\n"
            "00:00:00;23,4\n"
            "00:00:01;ERR\n"        # error code
            "00:00:02;\n"           # empty string -> null without error
            "00:00:03;OVERFLOW\n"   # invalid float
            "00:00:04;12,5\n"
        )
        ds = load_csv(csv_data)
        assert len(ds) == 5
        # Values should be floats or nulls
        expected = [23.4, None, None, None, 12.5]
        assert ds["ch1"].to_list() == expected

        # Check AuditLog
        audit = ds.audit_log
        assert audit.has_errors()
        assert len(audit) == 2

        # Verify entry 1: ERR at row 1
        entry0 = audit.entries[0]
        assert entry0.row_index == 1
        assert entry0.column == "ch1"
        assert entry0.raw_value == "ERR"

        # Verify entry 2: OVERFLOW at row 3
        entry1 = audit.entries[1]
        assert entry1.row_index == 3
        assert entry1.column == "ch1"
        assert entry1.raw_value == "OVERFLOW"

        # Check audit log to_dataframe
        audit_df = audit.to_dataframe()
        assert isinstance(audit_df, pl.DataFrame)
        assert audit_df.shape == (2, 4)
        assert "row_index" in audit_df.columns
        assert "column" in audit_df.columns
        assert "raw_value" in audit_df.columns
        assert "reason" in audit_df.columns

    def test_timestamp_iso_datetime(self) -> None:
        csv_data = (
            "time,val\n"
            "2026-03-01 10:00:00,24.1\n"
            "2026-03-01 10:00:05,24.2\n"
            "2026-03-01 10:01:00,24.3\n"
        )
        ds = load_csv(csv_data)
        assert ds["time_seconds"].to_list() == [0.0, 5.0, 60.0]
        assert not ds.audit_log.has_errors()

    def test_timestamp_german_datetime(self) -> None:
        csv_data = (
            "time;val\n"
            "01.03.2026 10:00:00;24,1\n"
            "01.03.2026 10:00:10;24,2\n"
        )
        ds = load_csv(csv_data)
        assert ds["time_seconds"].to_list() == [0.0, 10.0]

    def test_timestamp_german_comma_milliseconds(self) -> None:
        csv_data = (
            "time;val\n"
            "00:00:00,000;24,1\n"
            "00:00:00,500;24,2\n"
            "00:00:01,250;24,3\n"
        )
        ds = load_csv(csv_data)
        assert ds["time_seconds"].to_list() == [0.0, 0.5, 1.25]

    def test_timestamp_empty_or_sentinel_first_row(self) -> None:
        csv_data = (
            'time;val\n'
            '"";24,1\n'
            '"00:00:01";24,2\n'
            '"00:00:02";24,3\n'
        )
        ds = load_csv(csv_data)
        assert ds["time_seconds"].to_list() == [None, 0.0, 1.0]

    def test_error_harvesting_preserves_trailing_fault_rows(self) -> None:
        csv_data = (
            "time;Kanal 1 Letzte (C)\n"
            "00:00:00;24,1\n"
            "00:00:01;ERR\n"
        )
        ds = load_csv(csv_data)
        assert len(ds) == 2
        assert ds["kanal_1"].to_list() == [24.1, None]
        assert ds.audit_log.has_errors()
        assert len(ds.audit_log) == 1
        entry = ds.audit_log[0]
        assert entry.row_index == 1
        assert entry.column == "kanal_1"
        assert entry.raw_column == "Kanal 1 Letzte (C)"
        assert entry.raw_value == "ERR"

    def test_german_thousand_separator(self) -> None:
        csv_data = "time;druck\n00:00:00;1.013,25\n00:00:01;1.014,50\n"
        ds = load_csv(csv_data)
        assert ds["druck"].to_list() == [1013.25, 1014.5]



class TestPipelineOperators:
    """Tests for clean_names, drop_footer, and unpivot."""

    def test_clean_names_removes_units_and_qualifiers(self) -> None:
        df = pl.DataFrame({
            "": [0.0],
            "Kanal 1 Letzte (C)": [24.1],
            "Kanal 2 Letzte (C)": [23.7],
            "Temperatur [°C]": [21.0],
            "Druck (bar)": [1.013],
            "Sensor #5 [mA]": [12.0],
        })
        cleaned = clean_names(df)
        assert cleaned.columns == [
            "time",
            "kanal_1",
            "kanal_2",
            "temperatur",
            "druck",
            "sensor_5",
        ]

    def test_clean_names_preserves_indexed_channels(self) -> None:
        df = pl.DataFrame({
            "Kanal (1) [°C]": [1.0],
            "Kanal (2) [°C]": [2.0],
        })
        cleaned = clean_names(df)
        assert cleaned.columns == ["kanal_1", "kanal_2"]

    def test_clean_names_german_umlauts(self) -> None:
        df = pl.DataFrame({
            "Gehäuse (C)": [1.0],
            "Kühlwasser [°C]": [2.0],
            "Öldruck": [3.0],
        })
        cleaned = clean_names(df)
        assert cleaned.columns == ["gehaeuse", "kuehlwasser", "oeldruck"]

    def test_drop_footer_trailing_only(self) -> None:
        df = pl.DataFrame({
            "time_seconds": [0.0, 1.0, 2.0, 3.0, 4.0],
            "k1": [10.0, None, 12.0, None, None],
            "k2": [20.0, None, 22.0, None, None],
        })
        # Trailing rows 3 and 4 should be dropped, row 1 in the middle should be kept
        result = drop_footer(df, trailing_only=True)
        assert len(result) == 3
        assert result["time_seconds"].to_list() == [0.0, 1.0, 2.0]

    def test_drop_footer_all_nulls(self) -> None:
        df = pl.DataFrame({
            "time_seconds": [0.0, 1.0, 2.0, 3.0],
            "k1": [10.0, None, 12.0, None],
            "k2": [20.0, None, 22.0, None],
        })
        result = drop_footer(df, trailing_only=False)
        assert len(result) == 2
        assert result["time_seconds"].to_list() == [0.0, 2.0]

    def test_unpivot_wide_to_long(self) -> None:
        df = pl.DataFrame({
            "time_seconds": [0.0, 1.0],
            "kanal_1": [24.1, 24.2],
            "kanal_2": [23.7, 23.8],
        })
        long_df = unpivot(df)
        assert len(long_df) == 4  # 2 rows * 2 channels
        assert long_df.columns == ["time_seconds", "channel", "value"]
        assert set(long_df["channel"].to_list()) == {"kanal_1", "kanal_2"}


class TestDualModeDataset:
    """Tests for DualModeDataset zero-copy and reshaping capabilities."""

    def test_zero_copy_numpy_extraction(self) -> None:
        df = pl.DataFrame({
            "time_seconds": [0.0, 1.0, 2.0],
            "kanal_1": [24.1, 24.2, 24.3],
        })
        ds = DualModeDataset(df)
        arr = ds["kanal_1"].to_numpy()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 3
        assert np.isclose(arr[0], 24.1)

        # Access via .df and .to_numpy()
        assert np.array_equal(ds.to_numpy("kanal_1"), arr)
        assert np.array_equal(ds.df["kanal_1"].to_numpy(), arr)

    def test_to_long_on_demand_and_cached(self) -> None:
        df = pl.DataFrame({
            "time_seconds": [0.0, 1.0],
            "kanal_1": [10.0, 20.0],
            "kanal_2": [30.0, 40.0],
        })
        ds = DualModeDataset(df)
        long1 = ds.to_long()
        long2 = ds.to_long()
        assert long1 is long2  # cached
        assert len(long1) == 4

    def test_empty_csv_handling(self) -> None:
        ds = load_csv("")
        assert len(ds) == 0
        assert "time_seconds" in ds.columns

    def test_header_only_csv_handling(self) -> None:
        ds = load_csv("time;k1;k2\n")
        assert len(ds) == 0
        assert "time_seconds" in ds.columns

    def test_stringio_and_stream_sources(self) -> None:
        content = "time,temp\n00:00:00,21.5\n00:00:01,22.0\n"
        stream = io.StringIO(content)
        ds = load_csv(stream)
        assert len(ds) == 2
        assert ds["temp"].to_list() == [21.5, 22.0]

    def test_audit_log_empty(self) -> None:
        audit = AuditLog()
        assert not audit.has_errors()
        assert len(audit) == 0
        df = audit.to_dataframe()
        assert len(df) == 0
        assert "row_index" in df.columns

    def test_dual_mode_dataset_methods(self) -> None:
        df = pl.DataFrame({
            "time_seconds": [0.0, 1.0, 2.0],
            "val": [10.0, 20.0, 30.0],
        })
        ds = DualModeDataset(df)
        assert ds.shape == (3, 2)
        assert ds.columns == ["time_seconds", "val"]
        assert len(ds) == 3
        assert len(ds.head(2)) == 2
        assert len(ds.tail(2)) == 2
        assert "DualModeDataset" in repr(ds)
        arrow_table = ds.to_arrow()
        assert arrow_table.num_rows == 3

    def test_dual_mode_dataset_contains_and_iter(self) -> None:
        df = pl.DataFrame({
            "time_seconds": [0.0, 1.0],
            "kanal_1": [24.1, 24.2],
        })
        ds = DualModeDataset(df)
        assert "kanal_1" in ds
        assert "time" in ds
        assert "time_seconds" in ds
        assert "nonexistent" not in ds
        cols = list(ds)
        assert cols == ["time_seconds", "kanal_1"]

    def test_dual_mode_dataset_to_long_custom_columns(self) -> None:
        df = pl.DataFrame({
            "step": [1, 2],
            "temp": [100.0, 105.0],
            "pressure": [1.0, 1.1],
        })
        ds = DualModeDataset(df)  # no 'time_seconds' column
        long_df = ds.to_long(id_vars="step")
        assert len(long_df) == 4
        assert set(long_df.columns) == {"step", "channel", "value"}

    def test_audit_log_iterable_and_for_column(self) -> None:
        audit = AuditLog()
        audit.add(0, "kanal_1", "ERR", "sentinel_value", raw_column="Kanal 1 Letzte (C)")
        audit.add(1, "kanal_2", "BAD", "non_convertible_float")
        assert len(audit) == 2
        assert list(audit) == audit.entries
        assert audit[0].column == "kanal_1"
        assert len(audit.for_column("kanal_1")) == 1
        assert len(audit.for_column("Kanal 1 Letzte (C)")) == 1
        assert len(audit.for_column("kanal_2")) == 1
        assert len(audit.for_column("kanal_3")) == 0


