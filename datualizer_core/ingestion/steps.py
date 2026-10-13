"""Column roles that turn a table (e.g. a lab journal) into measurement steps (Etappe 2b, step 5).

Which column holds the start, the end, the name or the parameters of a step differs from lab to
lab, so it is never written in code: the user picks it, or a source profile suggests it as data
(`SourceProfile.intervals`). Kept here, free of campaign imports, so profiles can use it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EndKind(str, Enum):
    COLUMN = "column"  # a column with the end time (clock time or date-time)
    DURATION = "duration_column"  # a column with the duration in seconds
    NEXT_START = "next_start"  # a step ends where the next one starts
    FIXED = "fixed_s"  # every step lasts `seconds`


class EndRule(BaseModel):
    """Where a step ends."""

    model_config = ConfigDict(extra="forbid")

    kind: EndKind = EndKind.COLUMN
    column: str | None = Field(None, description="End time or duration column (raw or cleaned name).")
    seconds: float | None = Field(None, gt=0, description="Fixed step length for kind=fixed_s.")

    @model_validator(mode="after")
    def _complete(self) -> "EndRule":
        if self.kind in (EndKind.COLUMN, EndKind.DURATION) and not self.column:
            raise ValueError(f"EndRule kind={self.kind.value} needs a column.")
        if self.kind is EndKind.FIXED and self.seconds is None:
            raise ValueError("EndRule kind=fixed_s needs seconds.")
        return self


class StepColumns(BaseModel):
    """Roles of the columns of a step table. Names may be raw headers or cleaned names."""

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

    @model_validator(mode="after")
    def _fill_down_is_label(self) -> "StepColumns":
        extra = [c for c in self.fill_down if c not in self.label_columns]
        if extra:
            raise ValueError(f"fill_down {extra} must also be label_columns.")
        return self
