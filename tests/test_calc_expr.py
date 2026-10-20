"""Formula parser (datualizer_core/calc/expr.py): positive list, error messages, Polars = numpy."""

from __future__ import annotations

import builtins
import math

import numpy as np
import polars as pl
import pydantic
import pytest

from datualizer_core.calc import FUNCTIONS, Constant, ExprError, UArray, compile_expr, evaluate


# --- rejected constructs ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, fragment, column",
    [
        ("__import__('os')", "unknown function '__import__'", 1),
        ("a.b()", "method calls", 1),
        ("x[0]", "indexing", 1),
        ("lambda: 1", "lambda is not allowed", 1),
        ("(lambda y: y)(a)", "only the listed functions", 1),
        ("[v for v in x]", "comprehensions are not allowed", 1),
        ("sum(v for v in x)", "unknown function 'sum'", 1),
        ("'text'", "text can only be compared", 1),
        ("a + 'text'", "text can only be compared", 5),
        ("b'raw' == a", "byte strings are not allowed", 1),
        ("(a + b) // 2", "operator '//' is not allowed", 9),
        ("a % b", "operator '%' is not allowed", 3),
        ("a ^ 2", "use '**' for powers", 3),
        ("~a", "use 'not'", 1),
        ("a in x", "comparison 'in' is not allowed", 3),
        ("a is b", "comparison 'is' is not allowed", 3),
        ("log(x=a)", "takes no keyword arguments", 1),
        ("log(a, b)", "log() takes 1 arguments, got 2", 1),
        ("min(a)", "min() takes at least 2 arguments", 1),
        ("where(a, b)", "where() takes 3 arguments", 1),
        ("round(a, b)", "n must be a whole number", 10),
        ("round(a, 1.5)", "n must be a whole number", 10),
        ("log + 1", "function 'log' needs arguments", 1),
        ("(y := a)", "assignments", 2),
        ("(a, b)", "tuples are not allowed", 1),
        ("x[0].mean", "attribute access is not allowed", 1),
        ("a.__class__.__bases__[0]", "indexing", 1),
        ("f'{a}'", "f-strings", 1),
        ("a +* b", "syntax error", 4),
        ("import os", "syntax error", 1),
    ],
)
def test_rejects_with_message_and_column(text, fragment, column):
    with pytest.raises(ExprError) as info:
        compile_expr(text)
    err = info.value
    assert fragment in err.message
    assert err.column == column
    assert text in str(err)  # the formula with a caret under the position
    assert str(err).splitlines()[-1].strip() == "^"


def test_unknown_name_lists_known_names_and_suggests():
    with pytest.raises(ExprError) as info:
        compile_expr("S_x - B", names=["S_b", "B", "f"])
    err = info.value
    assert "unknown name 'S_x'" in err.message
    assert "did you mean 'S_b'" in err.message
    assert err.known == ["B", "S_b", "f"]
    assert "Known names: B, S_b, f" in str(err)


def test_syntax_error_lists_known_names():
    with pytest.raises(ExprError) as info:
        compile_expr("S -", names=["S"])
    assert info.value.known == ["S"]


def test_empty_and_overlong_formulas():
    with pytest.raises(ExprError, match="empty"):
        compile_expr("   ")
    with pytest.raises(ExprError, match="longer than"):
        compile_expr("a+" * 6000 + "a")


def test_position_counts_characters_not_bytes():
    with pytest.raises(ExprError) as info:
        compile_expr("Δp·2 + x[0]")
    assert info.value.column == 8


def test_leading_blanks_keep_positions():
    with pytest.raises(ExprError) as info:
        compile_expr("   a // b")
    assert info.value.column == 6


def test_no_eval_or_exec(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("eval/exec must not be used")

    monkeypatch.setattr(builtins, "eval", forbidden)
    monkeypatch.setattr(builtins, "exec", forbidden)
    expr = compile_expr("where(x > 0, sqrt(x), -1) + max(x, 2) ** 2")
    assert expr.evaluate({"x": 4.0}).value == pytest.approx(2 + 16)


# --- names --------------------------------------------------------------------------------------


def test_dotted_names_are_one_name():
    expr = compile_expr("fid.mean - blank.fid.mean + fid.mean", names=["fid.mean", "blank.fid.mean"])
    assert expr.names == ("fid.mean", "blank.fid.mean")
    df = pl.DataFrame({"fid.mean": [3.0, 5.0], "blank.fid.mean": [1.0, 1.0]})
    assert df.select(expr.to_polars().alias("y"))["y"].to_list() == [5.0, 9.0]
    assert expr.evaluate({"fid.mean": 3.0, "blank.fid.mean": 1.0}).value == 5.0


def test_names_in_order_of_appearance_and_none_means_any():
    assert compile_expr("1 - (S - B) * f / c_in").names == ("S", "B", "f", "c_in")
    assert compile_expr("2 * 3").names == ()


def test_typographic_signs():
    assert compile_expr("a − b·c × 2").evaluate({"a": 10, "b": 2, "c": 1.5}).value == pytest.approx(4.0)


def test_evaluate_reports_missing_name():
    expr = compile_expr("a + bb")
    with pytest.raises(ExprError) as info:
        expr.evaluate({"a": 1.0})
    assert "no value for name 'bb'" in info.value.message
    assert info.value.column == 5
    assert info.value.known == ["a"]


def test_to_polars_columns_and_constants():
    expr = compile_expr("(S - B) * f")
    out = pl.DataFrame({"fid_mgC": [3.0, 4.0], "B": [1.0, 2.0]}).select(
        expr.to_polars(columns={"S": "fid_mgC"}, constants={"f": Constant(name="f", value=2.0, u=0.1)}).alias("y")
    )
    assert out["y"].to_list() == [4.0, 4.0]


def test_function_list_matches_the_specification():
    assert set(FUNCTIONS) == {"log", "log10", "exp", "sqrt", "abs", "min", "max", "where", "isfinite", "round"}


# --- the same result from Polars and numpy ------------------------------------------------------

FRAME = pl.DataFrame(
    {
        "a": [1.0, -2.5, 0.0, 3.0, math.nan, 7.25, 2.5, -0.5],
        "b": [2.0, 0.5, 0.0, -1.0, 1.0, math.nan, 0.5, 4.0],
        "n": [1, 2, 3, 0, -4, 5, 6, 7],  # integer column
    }
)

FORMULAS = [
    "a + b * 2 - n / 3",
    "-a ** 2",
    "(-a) ** 2",
    "a ** b",
    "2 ** -n",
    "n ** 2",
    "a / b",
    "1 - (a - b) * 2.701 / 144.361",
    "log(a) + log10(b)",
    "exp(-a) * sqrt(b)",
    "abs(a - b)",
    "min(a, b)",
    "max(a, b, n)",
    "min(a, 1, b)",
    "where(a > b, a, b)",
    "a if a >= 0 else -a",
    "isfinite(a / b)",
    "round(a)",
    "round(a * 10, 1)",
    "round(a, -1)",
    "a < b",
    "0 < a <= 3",
    "a == b or not (n > 2)",
    "a > 0 and b > 0 and n != 3",
    "where(n, a, b)",
    "where(a, 1, 0)",
    "not a",
    "(a > 1) * 2 + n",
]


def _same(x: np.ndarray, y: np.ndarray) -> bool:
    x = np.broadcast_to(np.asarray(x), y.shape)
    if y.dtype == bool or x.dtype == bool:
        return np.array_equal(x.astype(bool), y.astype(bool))
    return np.allclose(x.astype(float), y.astype(float), rtol=1e-12, atol=0, equal_nan=True)


@pytest.mark.parametrize("text", FORMULAS)
def test_polars_and_numpy_agree(text):
    expr = compile_expr(text, names=FRAME.columns)
    polars_result = FRAME.select(expr.to_polars().alias("y"))["y"].to_numpy()
    numpy_result = expr.evaluate({c: FRAME[c] for c in FRAME.columns}).value
    assert _same(numpy_result, polars_result), (text, numpy_result, polars_result)


def test_nan_propagates_through_min_and_max():
    r = evaluate("min(a, b) + max(b, a)", {"a": np.array([np.nan, 1.0]), "b": np.array([1.0, np.nan])})
    assert np.isnan(r.value).all()


def test_polars_null_propagates():
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, 1.0]})

    def run(text):
        return df.select(compile_expr(text).to_polars().alias("y"))["y"].to_list()

    assert run("min(a, b)") == [None, None, 1.0]
    assert run("max(a, b)") == [None, None, 3.0]
    assert run("a + b") == [None, None, 4.0]
    assert run("where(a > 1, 1, 0)") == [0.0, None, 1.0]
    assert run("1 if b > 1 else 0") == [None, 1.0, 0.0]


def test_evaluate_returns_uarray_and_accepts_inputs():
    r = evaluate("2 * x + c", {"x": UArray.input("x", [1.0, 2.0], 0.1), "c": Constant(name="c", value=1.0)})
    assert isinstance(r, UArray)
    np.testing.assert_allclose(r.value, [3.0, 5.0])
    np.testing.assert_allclose(r.u(), [0.2, 0.2])


# --- named numbers nan and inf (PR 10b) ---------------------------------------------------------


def test_nan_and_inf_are_numbers_in_both_targets():
    # OCR scale: invalid or implausible readings blanked out instead of dividing by zero
    text = "where(is_valid == 1 and mass_g > 9600, mass_g, nan)"
    df = pl.DataFrame({"is_valid": [1.0, 0.0, 1.0], "mass_g": [9650.0, 9651.0, 12.0]})
    e = compile_expr(text, names=df.columns)
    assert e.names == ("is_valid", "mass_g")  # nan is no name
    got = df.select(e.to_polars().alias("v"))["v"].to_list()
    assert got[0] == 9650.0 and math.isnan(got[1]) and math.isnan(got[2])
    np.testing.assert_array_equal(e.evaluate({c: df[c] for c in df.columns}).value, [9650.0, np.nan, np.nan])
    assert float(evaluate("-inf < x", {"x": 1.0}).value) == 1.0
    assert float(evaluate("min(x, inf)", {"x": 2.0}).value) == 2.0


def test_nan_and_inf_cannot_be_names():
    from datualizer_core.campaign.evaluation import Calculation

    with pytest.raises(pydantic.ValidationError, match="not nan or inf"):
        Calculation(name="nan", expr="1")
