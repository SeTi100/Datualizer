"""Multi-file merge with dedupe (Datenkatalog P2, P3).

Real exports overlap: the same file is saved twice under different names (P2), and a later
snapshot repeats all rows of an earlier one plus new ones (P3). `load_csvs` turns several such
files into one dataset:

1. Every file is fingerprinted (SHA-256). Byte-identical files are skipped and audited
   (`duplicate_file`), never loaded twice.
2. Each remaining file is loaded on its own to resolve its layout. All files must agree on
   decimal separator, time column and wide/long layout, otherwise `MergeError`.
3. The raw tables are stacked. Rows with the same key (time, plus the variable for long tables)
   are one sample: the file given later wins. A value of an earlier file that differs from the
   winner is audited (`merge_conflict`) or, with `MergeConflictMode.ERROR`, raises.
4. The merged raw table is loaded once, so types, roles and runs are decided on all data.

Keys are compared as raw text, before any parsing, so no float rounding is involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, TextIO

import polars as pl

from datualizer_core.dataset import SOURCES_SCHEMA, AuditLog, DualModeDataset
from datualizer_core.ingestion.config import IngestionConfig, MergeConflictMode
from datualizer_core.ingestion.loader import CSVLoader, read_raw_table
from datualizer_core.ingestion.pre_scanner import pre_scan
from datualizer_core.ingestion.sources import read_source

_SRC = "__merge_src"
_IDX = "__merge_idx"


class MergeError(ValueError):
    """Files cannot be merged without guessing (different layout, missing key, forbidden conflict)."""


def load_csvs(
    sources: Sequence[str | Path | TextIO],
    config: IngestionConfig | None = None,
) -> DualModeDataset:
    """Load several files into one dataset, dropping duplicate files and overlapping rows.

    Later sources win over earlier ones. The result's `sources` table reports per file what was
    kept, replaced or skipped; `ingestion_spec` replays the merge when passed back with the same
    files in the same order.
    """
    if not sources:
        raise MergeError("No sources given.")
    cfg = (config or IngestionConfig()).model_copy(deep=True)
    loader = CSVLoader(cfg)
    audit_log = AuditLog()

    # 1. Fingerprint, skip byte-identical files
    infos = [read_source(s, index=i) for i, s in enumerate(sources)]
    first_with_hash: dict[str, str] = {}
    duplicate_of: list[str | None] = []
    for info in infos:
        duplicate_of.append(first_with_hash.get(info.sha256))
        first_with_hash.setdefault(info.sha256, info.label)
    for info, dup in zip(infos, duplicate_of):
        if dup is not None:
            audit_log.add(-1, "", info.label, "duplicate_file", raw_column=dup)
    unique = [i for i, dup in enumerate(duplicate_of) if dup is None]

    # 2. Resolve each file's layout and check that all files agree
    scans = {i: pre_scan(infos[i].scan_input) for i in unique}
    raws = {i: read_raw_table(scans[i]) for i in unique}
    decimals = {scans[i].decimal_separator for i in unique}
    if len(decimals) > 1:
        raise MergeError(f"Files use different decimal separators {sorted(decimals)}.")
    decimal = decimals.pop()

    specs = {}
    for i in unique:
        spec = loader.load_frame(raws[i], decimal).ingestion_spec
        if spec is None:
            raise MergeError(f"{infos[i].label} contains no data.")
        specs[i] = spec
    layouts = {
        i: (s.time_column, s.long_format.mode, s.long_format.variable_col, s.long_format.value_col)
        for i, s in specs.items()
    }
    if len(set(layouts.values())) > 1:
        detail = "; ".join(
            f"{infos[i].label}: time={t!r}, layout={m.value}, variable={v!r}, value={val!r}"
            for i, (t, m, v, val) in layouts.items()
        )
        raise MergeError(
            "Files differ in time column or wide/long layout; set them explicitly in the "
            f"IngestionConfig. {detail}"
        )
    ref = specs[unique[0]]

    # The merged load uses the layout every file agreed on
    merged_cfg = cfg.model_copy(deep=True)
    merged_cfg.time_column = ref.time_column
    merged_cfg.long_format.mode = ref.long_format.mode
    merged_cfg.long_format.variable_col = ref.long_format.variable_col
    merged_cfg.long_format.value_col = ref.long_format.value_col
    key = cfg.merge.key_columns or _default_key(loader, raws[unique[0]].columns, ref)
    merged_cfg.merge.key_columns = list(key)
    merged_loader = CSVLoader(merged_cfg)

    for i in unique:
        missing = [k for k in key if k not in raws[i].columns]
        if missing:
            raise MergeError(f"{infos[i].label} lacks key column(s) {missing}.")

    # 3. Stack raw tables, keep the latest file's rows per key, audit differing values
    stacked = pl.concat(
        [raws[i].with_columns(pl.lit(i, dtype=pl.Int64).alias(_SRC)) for i in unique],
        how="diagonal",
    ).with_row_index(_IDX)
    value_cols = [c for c in stacked.columns if c not in (*key, _SRC, _IDX)]
    has_key = pl.all_horizontal(pl.col(k).is_not_null() for k in key)
    stacked = stacked.with_columns(
        pl.when(has_key).then(pl.col(_SRC).max().over(key)).otherwise(pl.col(_SRC)).alias("__win"),
        pl.when(has_key).then(pl.col(_IDX).min().over(key)).otherwise(pl.col(_IDX)).alias("__first"),
    )
    kept = stacked.filter(pl.col(_SRC) == pl.col("__win")).sort("__first", _IDX)
    losers = stacked.filter(pl.col(_SRC) != pl.col("__win"))

    merged_raw = kept.drop(_SRC, _IDX, "__win", "__first")
    kept = kept.with_row_index("__row")
    winners = kept.group_by(key, maintain_order=True).first()
    conflicts = _conflicts(losers, winners, key, value_cols)
    if len(conflicts) and cfg.merge.conflict is MergeConflictMode.ERROR:
        first = conflicts.row(0, named=True)
        raise MergeError(
            f"{len(conflicts)} value(s) differ between files for the same key, e.g. column "
            f"{first['column']!r}: {first['old']!r} in {infos[first[_SRC]].label} vs. {first['new']!r}."
        )
    raw_to_clean = merged_loader.column_name_map(merged_raw.columns)
    for c in conflicts.iter_rows(named=True):
        audit_log.add(
            c["__row"], raw_to_clean[c["column"]], c["old"], "merge_conflict", raw_column=c["column"]
        )

    # 4. Load the merged table once and report per source
    source_table = _source_table(infos, duplicate_of, raws, kept, losers, conflicts)
    return merged_loader.load_frame(merged_raw, decimal, audit_log=audit_log, sources=source_table)


def _default_key(loader: CSVLoader, columns: Sequence[str], spec: IngestionConfig) -> list[str]:
    """Time column, plus the raw variable column for long tables (spec holds loaded names)."""
    key = [spec.time_column]
    variable = spec.long_format.variable_col
    if variable is not None:
        raw = [r for r, c in loader.column_name_map(columns).items() if c == variable or r == variable]
        key.append(raw[0] if raw else variable)
    return key


def _conflicts(
    losers: pl.DataFrame, winners: pl.DataFrame, key: list[str], value_cols: list[str]
) -> pl.DataFrame:
    """One row per value of a replaced row that the winning row does not reproduce.

    A value the earlier file did not have (null) is an update, not a conflict.
    """
    schema = {"__row": pl.UInt32, _SRC: pl.Int64, "column": pl.String, "old": pl.String, "new": pl.String}
    if losers.is_empty() or not value_cols:
        return pl.DataFrame(schema=schema)
    joined = losers.select(*key, _SRC, _IDX, *value_cols).join(
        winners.select(*key, "__row", *value_cols), on=key, how="inner", suffix="__new"
    )
    parts = [
        joined.filter(pl.col(c).is_not_null() & pl.col(c).ne_missing(pl.col(f"{c}__new"))).select(
            "__row", _SRC, _IDX, pl.lit(c).alias("column"), pl.col(c).alias("old"), pl.col(f"{c}__new").alias("new")
        )
        for c in value_cols
    ]
    return pl.concat(parts).sort(_IDX, "column").select(*schema).cast(schema)


def _source_table(
    infos: list,
    duplicate_of: list[str | None],
    raws: dict[int, pl.DataFrame],
    kept: pl.DataFrame,
    losers: pl.DataFrame,
    conflicts: pl.DataFrame,
) -> pl.DataFrame:
    kept_n = dict(kept.group_by(_SRC).len().iter_rows())
    lost_n = dict(losers.group_by(_SRC).len().iter_rows()) if len(losers) else {}
    conf_n = dict(conflicts.group_by(_SRC).len().iter_rows()) if len(conflicts) else {}
    rows = []
    for i, (info, dup) in enumerate(zip(infos, duplicate_of)):
        loaded = dup is None
        rows.append({
            "source": info.label,
            "sha256": info.sha256,
            "n_rows": len(raws[i]) if loaded else None,
            "n_kept": kept_n.get(i, 0) if loaded else 0,
            "n_replaced": lost_n.get(i, 0) if loaded else None,
            "n_conflicts": conf_n.get(i, 0) if loaded else None,
            "duplicate_of": dup,
        })
    return pl.DataFrame(rows, schema=SOURCES_SCHEMA)
