"""Uncertainty propagation with dual numbers (datualizer_core/calc/uncertain.py)."""

from __future__ import annotations

import json
import math

import numpy as np
import pydantic
import pytest

from datualizer_core.calc import Constant, ExprError, UArray, compile_expr, evaluate

CONVERSION = "1 - (S - B) * f / c_in"


def _numeric_grad(text: str, values: dict[str, float], name: str, rel: float = 1e-6) -> float:
    """Central difference of the formula with plain floats (no derivatives involved)."""
    expr = compile_expr(text)
    h = rel * max(abs(values[name]), 1.0)
    up = expr.evaluate({**values, name: values[name] + h}).value
    down = expr.evaluate({**values, name: values[name] - h}).value
    return float((up - down) / (2 * h))


# --- hand calculation: 17.10., step Z16 ---------------------------------------------------------


def test_conversion_z16_against_hand_calculation():
    """X = 1 − (S − B)·f/c_in with S 25.807 ± 0.10 and B 2.266 ± 1.0 (both random), f and c_in exact."""
    f, c_in = 2.701, 144.361
    env = {
        "S": UArray.input("S", 25.807, 0.10),
        "B": UArray.input("B", 2.266, 1.0),
        "f": Constant(name="f", value=f),
        "c_in": c_in,
    }
    x = evaluate(CONVERSION, env)
    assert float(x.value) == pytest.approx(0.560, abs=5e-4)

    budget = x.budget()
    assert set(budget) == {"S", "B"}  # exact inputs contribute nothing
    assert float(budget["B"]) == pytest.approx(f / c_in * 1.0, rel=1e-12)
    assert float(budget["B"]) == pytest.approx(0.0187, abs=5e-5)
    assert float(budget["S"]) == pytest.approx(-f / c_in * 0.10, rel=1e-12)
    expected_u = math.hypot(f / c_in * 0.10, f / c_in * 1.0)
    assert float(x.u()) == pytest.approx(expected_u, rel=1e-12)
    assert float(x.u_random()) == pytest.approx(expected_u, rel=1e-12)
    assert float(x.u_syst()) == 0.0


def test_conversion_z16_against_numerical_derivative():
    values = {"S": 25.807, "B": 2.266, "f": 2.701, "c_in": 144.361}
    u = {"S": 0.10, "B": 1.0, "f": 0.135, "c_in": 4.3}
    kinds = {"S": "zufall", "B": "zufall", "f": "syst", "c_in": "syst"}
    env = {n: UArray.input(n, values[n], u[n], kind=kinds[n]) for n in values}
    x = evaluate(CONVERSION, env)
    for name in values:
        assert float(x.grads[name]) == pytest.approx(_numeric_grad(CONVERSION, values, name), rel=1e-6)
        assert float(x.budget()[name]) == pytest.approx(_numeric_grad(CONVERSION, values, name) * u[name], rel=1e-6)
    b = x.budget()
    assert float(x.u_random()) == pytest.approx(math.hypot(b["S"], b["B"]))
    assert float(x.u_syst()) == pytest.approx(math.hypot(b["f"], b["c_in"]))
    assert float(x.u()) == pytest.approx(math.sqrt(sum(float(v) ** 2 for v in b.values())))


# --- every operation and function against the numerical derivative -----------------------------


@pytest.mark.parametrize(
    "text",
    [
        "a + b", "a - b", "a * b", "a / b", "-a / b", "a ** 2", "a ** b", "b ** a", "2 ** a", "a ** -0.5",
        "log(a)", "log10(a * b)", "exp(a / b)", "sqrt(a)", "abs(a - b)", "abs(b - a)",
        "min(a, b)", "max(a, b)", "min(a, b * 3, 2)", "where(a > b, a * b, a / b)",
        "a if a < b else b ** 2", "-log(1 - a / 10) * b",
        "1 - (a - b) * 2.701 / 144.361",
    ],
)
def test_derivatives_match_numerical(text):
    values = {"a": 2.3, "b": 1.7}
    env = {"a": UArray.input("a", values["a"], 0.1), "b": UArray.input("b", values["b"], 0.2)}
    r = evaluate(text, env)
    for name in values:
        assert float(r.grads.get(name, 0.0)) == pytest.approx(_numeric_grad(text, values, name), rel=1e-6, abs=1e-9)


def test_round_and_comparisons_carry_no_uncertainty():
    env = {"a": UArray.input("a", 2.34, 0.1)}
    assert not evaluate("round(a, 1)", env).inputs
    assert float(evaluate("round(a, 1) + a", env).u()) == pytest.approx(0.1)
    cmp = evaluate("a > 2", env)
    assert cmp.value.dtype == bool and not cmp.grads


def test_power_with_negative_base_and_fixed_exponent():
    r = evaluate("a ** 2", {"a": UArray.input("a", -3.0, 0.1)})
    assert float(r.value) == 9.0
    assert float(r.grads["a"]) == pytest.approx(-6.0)


def test_element_wise_values_and_uncertainties():
    s = UArray.input("S", [25.807, 10.0, 5.0], [0.10, 0.2, 0.3])
    b = UArray.input("B", 2.266, 1.0)  # one blank for all rows
    x = evaluate(CONVERSION, {"S": s, "B": b, "f": 2.701, "c_in": np.array([144.361, 100.0, 80.0])})
    assert x.shape == (3,)
    k = 2.701 / np.array([144.361, 100.0, 80.0])
    np.testing.assert_allclose(x.budget()["S"], -k * [0.10, 0.2, 0.3])
    np.testing.assert_allclose(x.budget()["B"], k * 1.0)
    np.testing.assert_allclose(x.u(), np.hypot(k * [0.10, 0.2, 0.3], k))


def test_where_takes_the_derivative_of_the_used_branch():
    a = UArray.input("a", [1.0, 3.0], 0.1)
    b = UArray.input("b", [2.0, 2.0], 0.5)
    r = evaluate("where(a > b, a, 2 * b)", {"a": a, "b": b})
    np.testing.assert_allclose(r.value, [4.0, 3.0])
    np.testing.assert_allclose(r.budget()["a"], [0.0, 0.1])
    np.testing.assert_allclose(r.budget()["b"], [1.0, 0.0])


def test_same_input_twice_is_one_input():
    r = evaluate("a - a + a * a", {"a": UArray.input("a", 3.0, 0.1)})
    assert float(r.grads["a"]) == pytest.approx(6.0)
    assert float(r.u()) == pytest.approx(0.6)


def test_conflicting_inputs_fail_loudly():
    with pytest.raises(ExprError, match="different u, kind or group"):
        evaluate("a + b", {"a": UArray.input("x", 1.0, 0.1), "b": UArray.input("x", 2.0, 0.2)})


# --- correlated systematic inputs --------------------------------------------------------------


def test_systematic_inputs_of_one_group_add_linearly():
    """f(x) = a·x + b with a, b systematic: one group → |x·u_a + u_b|, two groups → in quadrature."""
    x, u_a, u_b = 4.0, 0.05, 0.3
    text = "a * x + b"
    same = evaluate(text, {
        "a": Constant(name="a", value=2.0, u=u_a, kind="syst", group="day 1"),
        "b": Constant(name="b", value=1.0, u=u_b, kind="syst", group="day 1"),
        "x": x,
    })
    assert float(same.value) == 9.0
    assert float(same.u_syst()) == pytest.approx(x * u_a + u_b)
    assert float(same.u()) == pytest.approx(x * u_a + u_b)
    assert float(same.u_random()) == 0.0
    np.testing.assert_allclose([same.budget()["a"], same.budget()["b"]], [x * u_a, u_b])  # budget stays per input

    apart = evaluate(text, {
        "a": Constant(name="a", value=2.0, u=u_a, kind="syst", group="day 1"),
        "b": Constant(name="b", value=1.0, u=u_b, kind="syst", group="day 2"),
        "x": x,
    })
    assert float(apart.u()) == pytest.approx(math.hypot(x * u_a, u_b))
    ungrouped = evaluate(text, {
        "a": Constant(name="a", value=2.0, u=u_a, kind="syst"),
        "b": Constant(name="b", value=1.0, u=u_b, kind="syst"),
        "x": x,
    })
    assert float(ungrouped.u()) == pytest.approx(math.hypot(x * u_a, u_b))


def test_correlated_errors_cancel_in_a_difference():
    a = Constant(name="a", value=5.0, u=0.2, kind="syst", group="balance")
    b = Constant(name="b", value=3.0, u=0.2, kind="syst", group="balance")
    r = evaluate("a - b", {"a": a, "b": b})
    assert float(r.u()) == pytest.approx(0.0, abs=1e-15)


def test_group_has_no_effect_on_random_inputs():
    a = Constant(name="a", value=5.0, u=0.3, kind="zufall", group="g")
    b = Constant(name="b", value=3.0, u=0.4, kind="zufall", group="g")
    r = evaluate("a + b", {"a": a, "b": b})
    assert float(r.u_random()) == pytest.approx(0.5)
    assert float(r.u_syst()) == 0.0


def test_random_and_systematic_split():
    env = {
        "S": UArray.input("S", 25.807, 0.10),
        "f": Constant(name="f", value=2.701, u=0.135, kind="syst", group="rf"),
        "c": Constant(name="c", value=144.361, u=4.3, kind="syst", group="air"),
    }
    r = evaluate("S * f / c", env)
    b = r.budget()
    assert float(r.u_random()) == pytest.approx(abs(float(b["S"])))
    assert float(r.u_syst()) == pytest.approx(math.hypot(b["f"], b["c"]))
    assert float(r.u()) == pytest.approx(math.hypot(r.u_random(), r.u_syst()))


# --- Constant -----------------------------------------------------------------------------------


def test_constant_defaults_and_json_round_trip():
    c = Constant(name="f", value=2.701, u=0.135, note="mgC -> mg Ethanol (RF 0.71)")
    assert c.kind == "syst" and c.group is None and c.unit == ""
    again = Constant.model_validate(json.loads(c.model_dump_json()))
    assert again == c
    assert not Constant(name="V_kat", value=0.0135, unit="m3").to_uarray().inputs


def test_constant_validation():
    with pytest.raises(pydantic.ValidationError):
        Constant(name="f", value=1.0, u=-0.1)
    with pytest.raises(pydantic.ValidationError):
        Constant(name="f", value=1.0, kind="random")
    with pytest.raises(pydantic.ValidationError):
        Constant(name="f", value=1.0, unknown=1)
    with pytest.raises(ValueError, match="kind"):
        UArray.input("x", 1.0, 0.1, kind="other")
    with pytest.raises(ValueError, match="negative"):
        UArray.input("x", 1.0, -0.1)


def test_arithmetic_operators_on_uarray():
    a = UArray.input("a", 2.0, 0.1)
    r = 1 - (3 * a) / 2 + a**2 - np.float64(1.0)
    assert float(r.value) == pytest.approx(1 - 3 + 4 - 1)
    assert float(r.grads["a"]) == pytest.approx(-1.5 + 4.0)
    r2 = np.array([1.0, 2.0]) * a
    assert isinstance(r2, UArray) and r2.shape == (2,)
