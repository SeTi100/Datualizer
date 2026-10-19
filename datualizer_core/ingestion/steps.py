"""Column roles that turn a table (e.g. a lab journal) into measurement steps (Etappe 2b, step 5).

Which column holds the start, the end, the name or the parameters of a step differs from lab to
lab, so it is never written in code: the user picks it, or a source profile suggests it as data
(`SourceProfile.intervals`). Kept here, free of campaign imports, so profiles can use it.

Columns are found by their header text only (never by position): case, spaces and a unit
suffix do not matter (`V_MFC` finds `V_MFC [Nl/min]`). Which rows of the table are steps is
`RowSelection`: a range, a stop rule (e.g. the start time jumps back: another day below),
a formula over the row values and rows left out by hand.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EndKind(str, Enum):
    COLUMN = "column"  # a column with the end time (clock time or date-time)
    DURATION = "duration_column"  # a column with the duration in seconds
    NEXT_START = "next_start"  # a step ends where the next one starts
    FIXED = "fixed_s"  # every step lasts `seconds`


class EndRule(BaseModel):
    """Where a step ends; `fallback` for rows where this rule gives no end (e.g. an empty end cell)."""

    model_config = ConfigDict(extra="forbid")

    kind: EndKind = EndKind.COLUMN
    column: str | None = Field(None, description="End time or duration column (header text).")
    seconds: float | None = Field(None, gt=0, description="Fixed step length for kind=fixed_s.")
    fallback: "EndRule | None" = Field(
        None, description="Used where this rule gives no end, e.g. next_start for an empty end cell (noted per row)."
    )

    @model_validator(mode="after")
    def _complete(self) -> "EndRule":
        if self.kind in (EndKind.COLUMN, EndKind.DURATION) and not self.column:
            raise ValueError(f"EndRule kind={self.kind.value} needs a column.")
        if self.kind is EndKind.FIXED and self.seconds is None:
            raise ValueError("EndRule kind=fixed_s needs seconds.")
        if self.fallback is not None and self.fallback.kind is self.kind and self.fallback.column == self.column:
            raise ValueError("EndRule fallback must differ from the rule.")
        return self

    def columns(self) -> list[str]:
        """Columns this rule and its fallbacks read."""
        own = [self.column] if self.column and self.kind in (EndKind.COLUMN, EndKind.DURATION) else []
        return own + (self.fallback.columns() if self.fallback is not None else [])


class StopRule(str, Enum):
    NONE = "none"  # read to the end of the table (or last_row)
    FIRST_EMPTY_START = "first_empty_start"  # stop at the first row without a start time
    TIME_JUMP = "time_jump"  # stop where the start time jumps back (another day or block below)


class RowSelection(BaseModel):
    """Which rows of the table are steps. Row numbers are the rows of the sheet (1-based, as in Excel).

    For a table that is not a sheet (CSV, SQLite) the header is row 1 and the data starts in row 2.
    Every row left out is reported with its reason.
    """

    model_config = ConfigDict(extra="forbid")

    header_row: int | None = Field(
        None, ge=1, description="Row with the headers. None = the first row that has all headers of the mapping."
    )
    first_row: int | None = Field(None, ge=1, description="First row read. None = the row below the header.")
    last_row: int | None = Field(None, ge=1, description="Last row read. None = the end of the table.")
    stop: StopRule = Field(StopRule.NONE, description="Where reading stops before last_row.")
    time_jump_min: float = Field(
        5.0, gt=0, description="stop=time_jump: stop where the start is more than this many minutes before the previous one."
    )
    where: str | None = Field(
        None,
        description="Formula over the row values (cleaned header names, numbers and text), e.g. \"index <= 20\". "
        "Rows where it is not true are left out.",
    )
    skip_rows: list[int] = Field(default_factory=list, description="Rows left out by hand.")

    @model_validator(mode="after")
    def _ordered(self) -> "RowSelection":
        if self.first_row is not None and self.last_row is not None and self.last_row < self.first_row:
            raise ValueError(f"last_row {self.last_row} is before first_row {self.first_row}.")
        if self.header_row is not None and self.first_row is not None and self.first_row <= self.header_row:
            raise ValueError(f"first_row {self.first_row} must be below the header row {self.header_row}.")
        return self


class StepColumns(BaseModel):
    """Roles of the columns of a step table. Names are header texts (raw or cleaned)."""

    model_config = ConfigDict(extra="forbid")

    start_column: str | None = Field(
        None, description="Start time column; None = the time column of the source."
    )
    end: EndRule = Field(default_factory=lambda: EndRule(kind=EndKind.NEXT_START))
    label_columns: list[str] = Field(default_factory=list, description="Joined into the step name.")
    fill_down: list[str] = Field(
        default_factory=list,
        description="Label columns written only in the first row of a block (filled into the rows below).",
    )
    parameter_columns: list[str] = Field(default_factory=list, description="Set points kept with each step.")
    parameter_defaults: dict[str, float | str] = Field(
        default_factory=dict,
        description="Value of a parameter where its column (or JSON key) is missing or the cell is empty, e.g. "
        "{\"luft\": 100}. Keys are cleaned names; a key that is no parameter column becomes one (noted).",
    )
    id_column: str | None = Field(
        None,
        description="Column with a stable step number (e.g. 'Index'): step id '<source>:<value>' instead of the row, "
        "so windows set by hand stay with their step when rows are inserted.",
    )
    rows: RowSelection = Field(default_factory=RowSelection)

    @model_validator(mode="after")
    def _fill_down_is_label(self) -> "StepColumns":
        extra = [c for c in self.fill_down if c not in self.label_columns]
        if extra:
            raise ValueError(f"fill_down {extra} must also be label_columns.")
        return self


CellDirection = Literal["right", "below"]


class CellRef(BaseModel):
    """A cell found by the text next to it (robust when columns or rows are inserted), or by address.

    `label`: text of the cell that names the value (case and spaces do not matter); the value is
    the `nth` non-empty cell to its right (or below). `cell`: an A1 address instead.
    """

    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    direction: CellDirection = "right"
    nth: int = Field(1, ge=1, description="The n-th non-empty cell in `direction` (empty cells are skipped).")
    cell: str | None = Field(None, description="A1 address (e.g. 'E5') instead of a label.")

    @model_validator(mode="after")
    def _one_way(self) -> "CellRef":
        if (self.label is None) == (self.cell is None):
            raise ValueError("A cell is given either by label or by address (cell), not both or neither.")
        return self

    def describe(self) -> str:
        if self.cell is not None:
            return self.cell
        where = "right of" if self.direction == "right" else "below"
        return f"{'' if self.nth == 1 else f'{self.nth}. value '}{where} '{self.label}'"
