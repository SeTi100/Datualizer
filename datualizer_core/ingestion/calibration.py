"""Linear device calibrations from source metadata (Etappe 2b, step 8, PR 2).

A device that reports `signal = sens · concentration + zero` (Testa iFID: pA = pA/ppm · ppm + pA)
stores sensitivity and zero in its settings. `suggest_linear_calibration` reads them from a
dataset's `source_meta` and suggests the scale of a calculated channel,
`y = factor · (signal − zero) / sens = a · signal + b` with `a = factor / sens`, `b = −zero · a`.
The factor (e.g. 1.616 mgC/ppm for propane) is not in the device file; the user gives it.

Which block and fields hold the values is device specific and comes from the source profile
(`SourceProfile.calibration`), never from code (principle 1). The result is a suggestion the
user confirms (principle 2).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field


class CalibrationFields(BaseModel):
    """Where a reader's metadata holds a linear device calibration (part of a source profile)."""

    model_config = ConfigDict(extra="forbid")

    block: str = Field(description="Metadata block of the calibration in use, e.g. 'calib_data'.")
    sens: str = Field(description="Field with the sensitivity (signal per concentration unit).")
    zero: str = Field(description="Field with the zero signal.")
    time: str | None = Field(None, description="Field with the time of the calibration (device clock).")
    signal_unit: str = ""
    concentration_unit: str = Field("", description="Concentration unit of the calibration, e.g. 'ppm'.")
    result_unit: str = Field("", description="Unit of the calculated channel after the factor, e.g. 'mgC'.")
    signal_channel: str | None = Field(None, description="Channel of the raw signal in the device's data export.")


@dataclass(frozen=True)
class LinearCalibration:
    """y = a · signal + b with a = factor / sens and b = −zero · a (a suggestion, not applied)."""

    a: float
    b: float
    sens: float
    zero: float
    factor: float
    block: str
    time: str | None  # time of the calibration as the device reported it (device clock)
    fields: CalibrationFields

    def formula(self, channel: str) -> str:
        """The formula with the device values, so its origin stays visible: factor · (signal − zero) / sens."""
        return f"{self.factor!r} * ({channel} - {self.zero!r}) / {self.sens!r}"

    def note(self, source: str = "") -> str:
        """Where the scale comes from, for the calculated channel's note."""
        f = self.fields
        per = "/".join(u for u in (f.result_unit, f.concentration_unit) if u)
        head = f"linear calibration{' from ' + source if source else ''} ({self.block}"
        head += f", {self.time})" if self.time else ")"
        return "; ".join([
            head,
            f"sens {self.sens!r} {'/'.join(u for u in (f.signal_unit, f.concentration_unit) if u)}".rstrip(),
            f"zero {self.zero!r} {f.signal_unit}".rstrip(),
            f"factor {self.factor!r} {per}".rstrip(),
            f"a = {self.a:.6g}, b = {self.b:.6g}",
        ])


def calibration_fields(meta: Mapping[str, Any]) -> CalibrationFields | None:
    """The calibration description stored with a source's metadata (from its profile), if any."""
    raw = meta.get("calibration_fields")
    return CalibrationFields.model_validate(raw) if raw else None


def suggest_linear_calibration(
    meta: Mapping[str, Any], mgc_per_ppm: float, fields: CalibrationFields | None = None
) -> LinearCalibration:
    """Scale of a calculated channel from a device calibration in `source_meta` (a suggestion).

    `meta`: a dataset's `source_meta` from a reader that keeps metadata blocks (`meta["xml"]`).
    `mgc_per_ppm`: result units per concentration unit (e.g. 1.616 mgC/ppm), given by the user.
    `fields`: where the values are; default the profile's `calibration` stored with the metadata.
    Raises ValueError if the block or a value is missing or the sensitivity is zero.
    """
    fields = fields if fields is not None else calibration_fields(meta)
    if fields is None:
        raise ValueError("The source has no calibration description (profile field 'calibration').")
    block = (meta.get("xml") or {}).get(fields.block)
    if not isinstance(block, Mapping):
        raise ValueError(f"The source has no calibration block {fields.block!r}.")
    values = {}
    for key in (fields.sens, fields.zero):
        v = block.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"Calibration block {fields.block!r}: {key!r} is missing or not a number ({v!r}).")
        values[key] = float(v)
    sens, zero = values[fields.sens], values[fields.zero]
    if sens == 0:
        raise ValueError(f"Calibration block {fields.block!r}: sensitivity {fields.sens!r} is 0.")
    if not math.isfinite(mgc_per_ppm) or mgc_per_ppm == 0:
        raise ValueError(f"The factor must be a non-zero number, got {mgc_per_ppm!r}.")
    a = float(mgc_per_ppm) / sens
    time = block.get(fields.time) if fields.time else None
    return LinearCalibration(
        a, -zero * a, sens, zero, float(mgc_per_ppm), fields.block, None if time is None else str(time), fields
    )
