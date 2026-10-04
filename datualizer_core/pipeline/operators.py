"""Pipeline operators for data cleaning, reshaping, and reshaping."""

from __future__ import annotations

import re
from typing import Sequence
import polars as pl


def clean_column_name(raw_name: str, index: int = 0) -> str:
    """Clean a single column name into a standardized snake_case identifier."""
    name = raw_name.lstrip("\ufeff").strip()
    if not name:
        return "time" if index == 0 else f"col_{index}"

    # Preserve indexed channel/sensor numbers in parentheses or brackets: Kanal (1) -> Kanal 1
    name = re.sub(r"[\(\[\{]\s*(\d+)\s*[\)\]\}]", r" \1 ", name)

    # Remove unit indicators in brackets or parentheses: (C), [°C], (bar), [V], etc.
    name = re.sub(r"[\(\[\{].*?[\)\]\}]", "", name).strip()

    # Transliterate German umlauts before snake_case conversion
    umlauts = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "ae", "Ö": "oe", "Ü": "ue"}
    for k, v in umlauts.items():
        name = name.replace(k, v)

    # Remove common export statistics/suffixes: Letzte, Last, Wert, Value, Aktuell
    name = re.sub(r"\b(letzte|last|wert|value|aktuell)\b", "", name, flags=re.IGNORECASE).strip()

    # Replace special characters and whitespace with underscore
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")

    return name.lower() or ("time" if index == 0 else f"col_{index}")


def clean_names(df: pl.DataFrame) -> pl.DataFrame:
    """Clean column names of a DataFrame:

    - Removes units in parentheses/brackets (e.g. '(C)')
    - Removes common export qualifiers like 'Letzte'
    - Converts to snake_case identifiers (e.g. 'kanal_1', ..., 'kanal_8')
    - Automatically handles empty column names (e.g. index 0 -> 'time')
    - Ensures unique column names.
    """
    new_names: list[str] = []
    seen: dict[str, int] = {}

    for i, col in enumerate(df.columns):
        cleaned = clean_column_name(col, index=i)
        count = seen.get(cleaned, 0)
        if count > 0:
            unique_name = f"{cleaned}_{count}"
            seen[cleaned] = count + 1
            new_names.append(unique_name)
        else:
            seen[cleaned] = 1
            new_names.append(cleaned)

    mapping = dict(zip(df.columns, new_names))
    return df.rename(mapping)


def drop_footer(
    df: pl.DataFrame,
    channel_cols: Sequence[str] | None = None,
    time_col: str | None = None,
    trailing_only: bool = True,
) -> pl.DataFrame:
    """Filter out rows where all measurement channels are null/empty/NaN.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    channel_cols : Sequence[str] | None
        Measurement channel column names. If None, all columns except time/index
        are treated as measurement channels.
    time_col : str | None
        Name of the time column to exclude if channel_cols is None.
    trailing_only : bool
        If True (default for drop_footer), only trailing rows at the end of the DataFrame
        where all channels are null are removed. If False, all rows with all-null channels
        are removed.
    """
    if len(df) == 0:
        return df

    if channel_cols is None:
        excluded = {time_col} if time_col else set()
        default_time_candidates = {"time", "time_seconds", "timestamp", "date", "index"}
        channel_cols = [
            c for c in df.columns
            if c not in excluded and c.lower() not in default_time_candidates
        ]

    if not channel_cols:
        return df

    # Check for null, NaN, or empty strings
    conditions: list[pl.Expr] = []
    for c in channel_cols:
        dtype = df.schema[c]
        if dtype in (pl.Float32, pl.Float64):
            conditions.append(pl.col(c).is_null() | pl.col(c).is_nan())
        elif dtype == pl.String:
            conditions.append(pl.col(c).is_null() | (pl.col(c).str.strip_chars() == ""))
        else:
            conditions.append(pl.col(c).is_null())

    all_null_expr = pl.all_horizontal(conditions)
    is_all_null = df.select(all_null_expr).to_series()

    if trailing_only:
        non_null_indices = (~is_all_null).arg_true()
        if len(non_null_indices) == 0:
            return df.slice(0, 0)
        last_valid_idx = non_null_indices[-1]
        return df.slice(0, last_valid_idx + 1)
    else:
        return df.filter(~is_all_null)


def unpivot(
    df: pl.DataFrame,
    id_vars: str | Sequence[str] | None = None,
    value_vars: str | Sequence[str] | None = None,
    variable_name: str = "channel",
    value_name: str = "value",
) -> pl.DataFrame:
    """Transform wide-format DataFrame into tidy long-format.

    Parameters
    ----------
    df : pl.DataFrame
        Input wide DataFrame.
    id_vars : str | Sequence[str] | None
        Identifier column(s) to keep fixed (e.g. 'time_seconds').
        Defaults to time column or first column.
    value_vars : str | Sequence[str] | None
        Measurement columns to unpivot. Defaults to all non-id columns.
    variable_name : str
        Name of the column containing unpivoted variable names (default 'channel').
    value_name : str
        Name of the column containing values (default 'value').
    """
    if id_vars is None:
        # Detect time or index column
        for candidate in ["time_seconds", "time", "timestamp"]:
            if candidate in df.columns:
                id_vars = [candidate]
                break
        if id_vars is None:
            id_vars = [df.columns[0]] if df.columns else []
    elif isinstance(id_vars, str):
        id_vars = [id_vars]
    else:
        id_vars = list(id_vars)

    if value_vars is None:
        value_vars = [c for c in df.columns if c not in id_vars]
    elif isinstance(value_vars, str):
        value_vars = [value_vars]
    else:
        value_vars = list(value_vars)

    return df.unpivot(
        index=id_vars,
        on=value_vars,
        variable_name=variable_name,
        value_name=value_name,
    )
