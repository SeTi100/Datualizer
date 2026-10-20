"""Values with linear uncertainty propagation (Etappe 2b, step 8, PR 1).

A `UArray` is a value array plus the exact first derivative with respect to every uncertain input
it depends on (forward-mode dual numbers). The arithmetic and the functions of the formula language
(`calc/expr.py`) carry the derivatives along, so after evaluating a formula the sensitivity
∂f/∂xᵢ of every input is known without numerical differentiation.

From the sensitivities and the input uncertainties uᵢ follow (GUM, linear propagation):

- the contribution of one input, cᵢ = ∂f/∂xᵢ · uᵢ (signed), see `UArray.budget()`;
- `u_random()` = √(Σ cᵢ²) over the inputs of kind "zufall": random inputs are independent;
- `u_syst()` = √(Σ_g (Σ_{i∈g} cᵢ)² + Σ cⱼ²): systematic inputs with the same `group` are fully
  correlated (r = 1), their contributions add linearly within the group before squaring; systematic
  inputs without a group are independent of everything else;
- `u()` = √(u_random² + u_syst²): random and systematic parts are independent of each other.

The `group` of a random input has no effect. A full correlation lets contributions cancel: for
f = a − b with a, b in one group and equal uᵢ the systematic part is 0, as it should be when both
come from the same error source (e.g. two concentrations from one reference gas, two masses from one
balance calibration). Give a group only to inputs that share one physical error source. Different
quantities measured on the same day (air flow, set-point temperature, response factor) are
independent and get no common group: a shared group would let their errors cancel or add without
reason. "Same constant in every step of a day" is a correlation between rows, see below.

Within one `UArray` the correlation acts per element (one step, one row). Correlation between
different rows through the same input (the same constant in every step of a day) is the business
of later analyses (fits over many steps) and is not modelled here.

Linearity: the propagation is exact for the derivative, but the uncertainty is first order; for
strongly non-linear formulas with large uᵢ (log near 0, division by a small, uncertain value)
it is an approximation. Non-differentiable functions (`round`, comparisons) have derivative 0;
`abs`, `min`, `max` and `where` take the derivative of the branch that is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Kind = Literal["zufall", "syst"]
KINDS: tuple[str, ...] = ("zufall", "syst")


class Constant(BaseModel):
    """A named input of a calculation with its standard uncertainty; serializable (campaign.json)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field("", description="Identifier of the constant.")
    value: float
    u: float = Field(0.0, ge=0.0, description="Standard uncertainty (k = 1), same unit as value.")
    kind: Kind = Field("syst", description="'zufall' (random) or 'syst' (systematic).")
    group: str | None = Field(
        None, description="Systematic inputs with the same group share one error source and are fully correlated "
        "(e.g. one reference gas). Not for different quantities of the same day."
    )
    scope: str = Field("", description="Scope of a systematic constant across steps: a column of the result "
                       "table (e.g. 'tag'), 'campaign', or '' = the analysis' scope (evaluated in PR 6).")
    unit: str = ""
    note: str = ""

    def to_uarray(self) -> "UArray":
        """The constant as a `UArray`; an uncertain input when u > 0, else a plain value."""
        if self.u == 0:
            return UArray(self.value)
        if not self.name:
            raise ValueError("Constant must have a non-empty name to produce an uncertain input.")
        return UArray.input(self.name, self.value, self.u, kind=self.kind, group=self.group)


@dataclass(frozen=True, eq=False)
class InputInfo:
    """What an uncertain input contributes: its standard uncertainty (scalar or per element), kind and group."""

    name: str
    u: np.ndarray
    kind: str
    group: str | None

    def same_as(self, other: "InputInfo") -> bool:
        return (
            self.kind == other.kind
            and self.group == other.group
            and self.u.shape == other.u.shape
            and bool(np.array_equal(self.u, other.u, equal_nan=True))
        )


def _as_values(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == bool:
        return arr
    return np.asarray(arr, dtype=float)


class UArray:
    """Value array with derivatives with respect to its uncertain inputs (dual numbers).

    `value` is a float (or bool, after a comparison) ndarray, scalar shape () or one value per row.
    `grads[name]` is ∂value/∂name, broadcastable to `value`; `inputs[name]` holds the input's u,
    kind and group. Inputs are identified by name: two `UArray` that share an input name must agree
    on its u, kind and group, otherwise combining them raises ``ValueError``.

    Arithmetic: ``+ - * / **`` and unary minus with numbers, arrays and other `UArray`; the
    functions of the formula language are module-level functions (`log`, `exp`, `where`, …).
    Uncertainties: `u`, `u_random`, `u_syst`, `budget` (see the module docstring).
    """

    __slots__ = ("value", "grads", "inputs")
    __array_ufunc__ = None  # numpy defers to the reflected operators below
    __hash__ = None

    def __init__(
        self,
        value,
        grads: Mapping[str, np.ndarray] | None = None,
        inputs: Mapping[str, InputInfo] | None = None,
    ) -> None:
        self.value = _as_values(value)
        self.grads = {k: np.asarray(v, dtype=float) for k, v in (grads or {}).items()}
        self.inputs = dict(inputs or {})
        missing = set(self.grads) - set(self.inputs)
        if missing:
            raise ValueError(f"Derivatives without input information: {sorted(missing)}.")

    @classmethod
    def input(cls, name: str, value, u, kind: Kind = "zufall", group: str | None = None) -> "UArray":
        """An uncertain input `name` with standard uncertainty `u` (scalar or one per element)."""
        if kind not in KINDS:
            raise ValueError(f"Input {name!r}: kind must be one of {KINDS}, got {kind!r}.")
        u_arr = np.asarray(u, dtype=float)
        if np.any(u_arr < 0):
            raise ValueError(f"Input {name!r}: uncertainty must not be negative.")
        values = np.asarray(value, dtype=float)
        info = InputInfo(name, u_arr, kind, group)
        return cls(values, {name: np.ones_like(values)}, {name: info})

    @classmethod
    def of(cls, x) -> "UArray":
        """Coerce numbers, arrays, `Constant` or `UArray` (returned unchanged) into a `UArray`."""
        if isinstance(x, UArray):
            return x
        if isinstance(x, Constant):
            return x.to_uarray()
        if hasattr(x, "to_numpy") and not isinstance(x, np.ndarray):  # polars / pandas Series
            x = x.to_numpy()
        return cls(x)

    # --- uncertainty -------------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def budget(self) -> dict[str, np.ndarray]:
        """Signed contribution cᵢ = ∂f/∂xᵢ · uᵢ of every uncertain input, broadcast to the value shape.

        Where the result does not depend on an input (∂f/∂xᵢ = 0, e.g. the branch `where` did not
        take), its contribution is 0 even if its u is NaN there (a measure without data in that row).
        """
        out: dict[str, np.ndarray] = {}
        with np.errstate(invalid="ignore", over="ignore"):
            for name, grad in self.grads.items():
                c = np.where(grad == 0, 0.0, grad * self.inputs[name].u)
                out[name] = np.broadcast_to(c, self.value.shape).copy()
        return out

    def u_random(self) -> np.ndarray:
        """Random part: √(Σ cᵢ²) over the inputs of kind 'zufall'."""
        var = np.zeros(self.value.shape)
        for name, c in self.budget().items():
            if self.inputs[name].kind == "zufall":
                var = var + c**2
        return np.sqrt(var)

    def u_syst(self) -> np.ndarray:
        """Systematic part: contributions of one group add linearly, groups and ungrouped inputs in quadrature."""
        var = np.zeros(self.value.shape)
        grouped: dict[str, np.ndarray] = {}
        for name, c in self.budget().items():
            info = self.inputs[name]
            if info.kind != "syst":
                continue
            if info.group is None:
                var = var + c**2
            else:
                grouped[info.group] = grouped.get(info.group, 0.0) + c
        for c in grouped.values():
            var = var + c**2
        return np.sqrt(var)

    def u(self) -> np.ndarray:
        """Combined standard uncertainty √(u_random² + u_syst²)."""
        return np.sqrt(self.u_random() ** 2 + self.u_syst() ** 2)

    def __repr__(self) -> str:
        return f"UArray(value={self.value!r}, inputs={sorted(self.inputs)})"

    # --- arithmetic --------------------------------------------------------------------------

    def __neg__(self) -> "UArray":
        return _unary(self, -self.value, -1.0)

    def __pos__(self) -> "UArray":
        return self

    def __add__(self, other) -> "UArray":
        return add(self, other)

    def __radd__(self, other) -> "UArray":
        return add(other, self)

    def __sub__(self, other) -> "UArray":
        return sub(self, other)

    def __rsub__(self, other) -> "UArray":
        return sub(other, self)

    def __mul__(self, other) -> "UArray":
        return mul(self, other)

    def __rmul__(self, other) -> "UArray":
        return mul(other, self)

    def __truediv__(self, other) -> "UArray":
        return div(self, other)

    def __rtruediv__(self, other) -> "UArray":
        return div(other, self)

    def __pow__(self, other) -> "UArray":
        return power(self, other)

    def __rpow__(self, other) -> "UArray":
        return power(other, self)


# --- combination rules ---------------------------------------------------------------------------


def _merge_inputs(*arrays: UArray) -> dict[str, InputInfo]:
    merged: dict[str, InputInfo] = {}
    for arr in arrays:
        for name, info in arr.inputs.items():
            known = merged.get(name)
            if known is None:
                merged[name] = info
            elif not known.same_as(info):
                raise ValueError(f"Input {name!r} appears with different u, kind or group.")
    return merged


def _linear(value, *terms: tuple[UArray, object]) -> UArray:
    """Result whose derivative is Σ factor · grad of each operand (chain rule)."""
    grads: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for arr, factor in terms:
            for name, g in arr.grads.items():
                contrib = g * factor
                grads[name] = grads[name] + contrib if name in grads else contrib
    return UArray(value, grads, _merge_inputs(*(arr for arr, _ in terms)))


def _unary(x: UArray, value, factor) -> UArray:
    return _linear(value, (x, factor))


def _float(x: UArray) -> np.ndarray:
    return np.asarray(x.value, dtype=float)


def add(a, b) -> UArray:
    a, b = UArray.of(a), UArray.of(b)
    return _linear(_float(a) + _float(b), (a, 1.0), (b, 1.0))


def sub(a, b) -> UArray:
    a, b = UArray.of(a), UArray.of(b)
    return _linear(_float(a) - _float(b), (a, 1.0), (b, -1.0))


def mul(a, b) -> UArray:
    a, b = UArray.of(a), UArray.of(b)
    va, vb = _float(a), _float(b)
    return _linear(va * vb, (a, vb), (b, va))


def div(a, b) -> UArray:
    a, b = UArray.of(a), UArray.of(b)
    va, vb = _float(a), _float(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = va / vb
        return _linear(value, (a, 1.0 / vb), (b, -value / vb))


def power(a, b) -> UArray:
    """a ** b; the ∂/∂b term (a^b · ln a) only when b is uncertain, so a < 0 with a fixed exponent works."""
    a, b = UArray.of(a), UArray.of(b)
    va, vb = _float(a), _float(b)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value = va**vb
        terms: list[tuple[UArray, object]] = []
        if a.grads:
            terms.append((a, vb * va ** (vb - 1.0)))
        if b.grads:
            terms.append((b, value * np.log(va)))
    return _linear(value, *terms)


def neg(a) -> UArray:
    return -UArray.of(a)


def _func(x, f, df) -> UArray:
    x = UArray.of(x)
    v = _float(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value = f(v)
        return _unary(x, value, df(v, value) if x.grads else 0.0)


def log(x) -> UArray:
    """Natural logarithm."""
    return _func(x, np.log, lambda v, y: 1.0 / v)


def log10(x) -> UArray:
    return _func(x, np.log10, lambda v, y: 1.0 / (v * np.log(10.0)))


def exp(x) -> UArray:
    return _func(x, np.exp, lambda v, y: y)


def sqrt(x) -> UArray:
    return _func(x, np.sqrt, lambda v, y: 0.5 / y)


def absolute(x) -> UArray:
    """|x|; derivative sign(x), 0 at x = 0."""
    return _func(x, np.abs, lambda v, y: np.sign(v))


def round_(x, ndigits=0) -> UArray:
    """Round half to even (like Python and Polars); derivative 0, so the result carries no uncertainty."""
    x = UArray.of(x)
    n = int(np.asarray(UArray.of(ndigits).value))
    return UArray(np.round(_float(x), n))


def where(cond, a, b) -> UArray:
    """Element-wise `a` where `cond` is true, else `b`; derivatives from the branch that is taken."""
    c = np.asarray(UArray.of(cond).value).astype(bool)
    a, b = UArray.of(a), UArray.of(b)
    value = np.where(c, _float(a), _float(b))
    grads: dict[str, np.ndarray] = {}
    for name in set(a.grads) | set(b.grads):
        ga = a.grads.get(name, 0.0)
        gb = b.grads.get(name, 0.0)
        grads[name] = np.where(c, ga, gb)
    return UArray(value, grads, _merge_inputs(a, b))


def _extreme(args, pick_first) -> UArray:
    if len(args) < 2:
        raise ValueError("min/max need at least two arguments.")
    result = UArray.of(args[0])
    for other in args[1:]:
        other = UArray.of(other)
        va, vb = _float(result), _float(other)
        with np.errstate(invalid="ignore"):
            first = pick_first(va, vb)
        chosen = where(first, result, other)
        nan = np.isnan(va) | np.isnan(vb)  # NaN propagates, like any arithmetic
        if np.any(nan):
            chosen = UArray(np.where(nan, np.nan, chosen.value), chosen.grads, chosen.inputs)
        result = chosen
    return result


def minimum(*args) -> UArray:
    return _extreme(args, lambda a, b: a <= b)


def maximum(*args) -> UArray:
    return _extreme(args, lambda a, b: a >= b)


def isfinite(x) -> UArray:
    return UArray(np.isfinite(_float(UArray.of(x))))


# --- comparisons and logic: boolean values without derivatives -----------------------------------


def _compare(a, b, op) -> UArray:
    a, b = UArray.of(a), UArray.of(b)
    with np.errstate(invalid="ignore"):
        return UArray(op(a.value, b.value))


def _bool(x) -> np.ndarray:
    return np.asarray(UArray.of(x).value).astype(bool)


def logical_and(a, b) -> UArray:
    return UArray(np.logical_and(_bool(a), _bool(b)))


def logical_or(a, b) -> UArray:
    return UArray(np.logical_or(_bool(a), _bool(b)))


def logical_not(a) -> UArray:
    return UArray(np.logical_not(_bool(a)))


COMPARISONS = {
    "<": lambda a, b: _compare(a, b, np.less),
    "<=": lambda a, b: _compare(a, b, np.less_equal),
    ">": lambda a, b: _compare(a, b, np.greater),
    ">=": lambda a, b: _compare(a, b, np.greater_equal),
    "==": lambda a, b: _compare(a, b, np.equal),
    "!=": lambda a, b: _compare(a, b, np.not_equal),
}
