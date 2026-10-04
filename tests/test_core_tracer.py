"""Tracer bullet verification test suite on real SA_testmessung_1.csv data."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl
import pytest

from datualizer_core import AuditLog, DualModeDataset, load_csv, pre_scan


@pytest.fixture(scope="module")
def csv_path() -> Path:
    """Resolve path to the real SA_testmessung_1.csv file."""
    path = Path(__file__).resolve().parent.parent / "SA_testmessung_1.csv"
    assert path.exists(), f"Required test file not found: {path}"
    return path


def test_pre_scanner_tracer(csv_path: Path) -> None:
    """Verify pre-scanner on SA_testmessung_1.csv:

    - Recognizes delimiter as comma
    - Recognizes decimal separator as German comma (',')
    - Cleans empty header column 0 to 'time'
    - Identifies and trims trailing empty/footer line(s).
    """
    res = pre_scan(csv_path)

    assert res.delimiter == ","
    assert res.decimal_separator == ","
    assert res.header_line_idx == 0
    assert res.cleaned_headers[0] == "time"
    assert res.footer_lines_count >= 1
    # Mathematical data lines from 00:00:00 to 01:22:10 (0s to 4930s = 4931 lines)
    assert res.data_lines_count in (4931, 4932)


def test_core_tracer_loading(csv_path: Path) -> None:
    """Verify end-to-end ingestion and dataset properties for SA_testmessung_1.csv:

    - Genau 4931 gültige Datenzeilen (Zeile 4933/4934 sauber abgeschnitten;
      spec mentions 4932 via total 4934 lines - 2 footer lines).
    - time_seconds von 0.0 bis 4930.0s als Float64.
    - Alle 8 Kanäle ('kanal_1' bis 'kanal_8') als Float64.
    - Zero-Copy NumPy-Extraktion für PyQtGraph (df['kanal_1'].to_numpy()).
    - to_long() erzeugt len(dataset) * 8 Zeilen (39'448 bzw. 39'456).
    """
    dataset = load_csv(csv_path)

    # 1. Type and encapsulation
    assert isinstance(dataset, DualModeDataset)
    assert isinstance(dataset.df, pl.DataFrame)
    assert isinstance(dataset.audit_log, AuditLog)

    # 2. Row count verification (Zeile 4933/4934 cleanly truncated)
    # The real data contains 00:00:00 to 01:22:10 (4930s - 0s + 1 = 4931 data rows)
    assert len(dataset) in (4931, 4932)
    assert len(dataset) == 4931, (
        f"Expected 4931 data rows (00:00:00 to 01:22:10 inclusive = 4931s); got {len(dataset)}"
    )

    # 3. Time column verification
    assert "time_seconds" in dataset.columns
    assert dataset["time_seconds"].dtype == pl.Float64
    assert dataset["time_seconds"].min() == 0.0
    assert dataset["time_seconds"].max() == 4930.0

    # 4. Channel columns and data types
    channel_names = [f"kanal_{i}" for i in range(1, 9)]
    for ch in channel_names:
        assert ch in dataset.columns, f"Missing channel: {ch}"
        assert dataset[ch].dtype == pl.Float64, f"Channel {ch} is not Float64: {dataset[ch].dtype}"

    # Verify sensor values are reasonable temperatures (e.g. ~20°C - 100°C)
    assert dataset["kanal_1"][0] == 24.109
    assert dataset["kanal_8"][0] == 23.451

    # 5. Middle sensor pause/dropouts (00:20:21 to 00:25:34) are cleanly handled as nulls
    assert dataset["kanal_1"].null_count() == 314
    assert dataset["kanal_8"].null_count() == 314

    # 6. Zero-Copy NumPy extraction for PyQtGraph
    # For time_seconds (no nulls), zero-copy is strictly guaranteed
    arr_time = dataset.to_numpy("time_seconds", allow_copy=False)
    assert isinstance(arr_time, np.ndarray)
    assert arr_time.base is not None
    assert len(arr_time) == len(dataset)

    # For channels with nulls, allow_copy=True maps nulls to np.nan for PyQtGraph
    arr1 = dataset["kanal_1"].to_numpy()
    assert isinstance(arr1, np.ndarray)
    assert len(arr1) == len(dataset)
    assert arr1.dtype == np.float64
    assert np.isnan(arr1[1221])  # One of the middle dropout null rows

    # Direct access on .df as well
    arr_df = dataset.df["kanal_1"].to_numpy()
    assert isinstance(arr_df, np.ndarray)
    assert len(arr_df) == len(dataset)

    # Convenience access checks
    assert "kanal_1" in dataset
    assert "time" in dataset
    assert np.array_equal(dataset["time"].to_numpy(), arr_time)

    # 7. Reshaping into Tidy Long-Format
    long_df = dataset.to_long()
    assert isinstance(long_df, pl.DataFrame)
    assert len(long_df) == len(dataset) * 8
    assert len(long_df) in (4931 * 8, 4932 * 8)
    assert set(long_df.columns) == {"time_seconds", "channel", "value"}
    assert long_df["value"].dtype == pl.Float64
    assert set(long_df["channel"].unique().to_list()) == set(channel_names)
