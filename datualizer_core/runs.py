"""Run segmentation (Datenkatalog P10, P11, P18).

A run is the set of rows sharing the same values in the run columns (e.g. `run_id`). Which
columns define a run is decided at ingestion (`RoleConfig.run_columns`); this module only
summarizes an already loaded table and never changes it. No imports from `ingestion/`.
"""

from __future__ import annotations

from typing import Sequence

import polars as pl


def summarize_runs(
    df: pl.DataFrame,
    time_col: str,
    run_columns: Sequence[str],
    parameters: Sequence[str] = (),
    aborted_fraction: float = 0.1,
) -> pl.DataFrame:
    """Return one row per run (in order of appearance) with size, timing and parameter values.

    Columns: run columns, `n_samples`, `t_start`, `t_end`, `duration_s`, `median_dt_s`
    (median sampling interval inside the run, P11), one column per parameter (its first value
    in the run) and `aborted` (fewer than `aborted_fraction` x median samples of all runs, P18).
    """
    keys = list(run_columns)
    if not keys:
        return pl.DataFrame()
    t = pl.col(time_col)
    summary = df.group_by(keys, maintain_order=True).agg(
        pl.len().alias("n_samples"),
        t.min().alias("t_start"),
        t.max().alias("t_end"),
        t.sort().diff().median().alias("median_dt_s"),
        *[pl.col(p).drop_nulls().first().alias(p) for p in parameters],
    )
    threshold = aborted_fraction * summary["n_samples"].median()
    return summary.with_columns(
        (pl.col("t_end") - pl.col("t_start")).alias("duration_s"),
        (pl.col("n_samples") < threshold).alias("aborted"),
    ).select(*keys, "n_samples", "t_start", "t_end", "duration_s", "median_dt_s", *parameters, "aborted")


def channel_availability(
    df: pl.DataFrame,
    run_columns: Sequence[str],
    channels: Sequence[str],
) -> pl.DataFrame:
    """Return the fill ratio (0.0 to 1.0) of every channel inside every run (P10)."""
    keys = list(run_columns)
    if not keys:
        return pl.DataFrame()
    return df.group_by(keys, maintain_order=True).agg(
        *[(pl.col(ch).count() / pl.len()).cast(pl.Float64).alias(ch) for ch in channels]
    )


def run_time(df: pl.DataFrame, time_col: str, run_columns: Sequence[str]) -> pl.Series:
    """Return seconds since the start of each row's run (the global time stays `time_col`)."""
    t = pl.col(time_col)
    expr = t - t.min().over(list(run_columns)) if run_columns else t - t.min()
    return df.select(expr.alias(time_col)).to_series()
