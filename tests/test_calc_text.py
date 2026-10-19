"""Etappe 2b, Schritt 8, PR 3: Text in Formeln.

Text-Literale und Textspalten nur mit `==` und `!=`, alles andere ist ein `ExprError` mit
Position. Polars und numpy rechnen gleich, auch mit fehlendem Text (weder gleich noch ungleich).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from datualizer_core.calc import ExprError, UArray, compile_expr, evaluate, is_text, suggest_name

FRAME = pl.DataFrame(
    {
        "klasse": ["null", "dosierung", None, "null", "dosierung", ""],
        "probe": ["Eingang", "Ausgang", "Eingang", None, "Eingang", "Ausgang"],
        "V_MFC": [0.0, 10.0, 5.0, 0.0, None, 2.0],
    }
)


def _numpy_env() -> dict:
    return {
        "klasse": np.array(FRAME["klasse"].to_list(), dtype=object),
        "probe": FRAME["probe"].to_list(),
        "V_MFC": FRAME["V_MFC"].fill_null(np.nan).to_numpy(),
    }


def _polars(text: str) -> list:
    return FRAME.select(compile_expr(text, FRAME.columns).to_polars().alias("y"))["y"].to_list()


def test_literal_in_single_or_double_quotes() -> None:
    r = evaluate("klasse == 'null'", _numpy_env())
    assert r.value.tolist() == [True, False, False, True, False, False]
    assert evaluate('klasse == "null"', _numpy_env()).value.tolist() == r.value.tolist()
    assert r.grads == {}  # a truth value carries no derivative


def test_text_and_numbers_combine_with_and() -> None:
    text = 'klasse != "null" and V_MFC > 0'
    r = evaluate(text, _numpy_env())
    assert r.value.tolist() == [False, True, False, False, False, True]
    assert _polars(text) == [False, True, False, False, None, True]  # Polars: null V_MFC stays null


@pytest.mark.parametrize(
    "text",
    [
        "klasse == 'null'",
        "klasse != 'null'",
        "'null' == klasse",
        "klasse == probe",
        "klasse != probe",
        "klasse == 'null' or probe == 'Eingang'",
        "not (klasse == 'dosierung')",
        "klasse == ''",
        "'a' == 'a' != 'b'",
    ],
)
def test_polars_and_numpy_agree_also_with_null(text: str) -> None:
    numpy_result = np.broadcast_to(evaluate(text, _numpy_env()).value, (len(FRAME),)).tolist()
    expr = compile_expr(text).to_polars(text_names=["klasse", "probe"])  # two columns, no literal: say so
    polars_result = FRAME.select(expr.alias("y"))["y"]
    assert numpy_result == (polars_result.to_list() if len(polars_result) == len(FRAME) else [polars_result[0]] * len(FRAME))


def test_missing_text_is_neither_equal_nor_unequal() -> None:
    env = _numpy_env()
    eq = evaluate("klasse == 'null'", env).value
    ne = evaluate("klasse != 'null'", env).value
    assert not eq[2] and not ne[2]  # row 2: klasse is missing
    assert _polars("klasse == 'null'")[2] is False and _polars("klasse != 'null'")[2] is False
    # NaN in an object array and None in a list are missing as well
    r = evaluate("x != 'a'", {"x": np.array(["a", np.nan, None, "b"], dtype=object)})
    assert r.value.tolist() == [False, False, False, True]


@pytest.mark.parametrize(
    "text, fragment, column",
    [
        ("klasse + 1", "text can only be compared", 1),
        ("klasse < 'a'", "'<' needs numbers", 10),
        ("'a' < klasse", "'<' needs numbers", 1),
        ("probe <= klasse", "'<=' needs numbers", 10),
        ("sqrt(klasse)", "text can only be compared", 6),
        ("V_MFC > 0 and klasse", "text can only be compared", 15),
        ("klasse == 1", "compares text with a number", 11),
        ("V_MFC == 'null'", "compares text with a number", 1),
        ("klasse if V_MFC > 0 else V_MFC", "text can only be compared", 1),
        ("V_MFC if V_MFC > 0 else 'x'", "text can only be compared", 25),
        ("-'a' == 'b'", "text can only be compared", 2),
    ],
)
def test_other_uses_of_text_fail_with_position(text: str, fragment: str, column: int) -> None:
    with pytest.raises(ExprError) as info:
        compile_expr(text, FRAME.columns, text_names=["klasse", "probe"])
    assert fragment in info.value.message and info.value.column == column
    with pytest.raises(ExprError) as again:  # the same from the values, without text_names
        compile_expr(text).evaluate(_numpy_env())
    assert fragment in again.value.message and again.value.column == column


def test_without_text_names_a_name_compared_with_text_is_text() -> None:
    expr = compile_expr("klasse == 'null' and V_MFC > 0")
    assert expr.text_names == {"klasse"}
    with pytest.raises(ExprError, match="text can only be compared"):
        compile_expr("klasse == 'null' and klasse > 0")  # once text, text everywhere
    with pytest.raises(ExprError, match="compares text with a number"):  # the values say otherwise
        expr.evaluate({"klasse": np.array([1.0, 0.0]), "V_MFC": np.array([1.0, 1.0])})


def test_text_columns_in_polars_need_no_literal_when_named() -> None:
    expr = compile_expr("klasse == probe")
    assert expr.text_names == frozenset()  # nothing says they are text
    out = FRAME.select(expr.to_polars(text_names=["klasse", "probe"]).alias("y"))["y"].to_list()
    assert out == [False, False, False, False, False, False]
    with pytest.raises(ExprError, match="text can only be compared"):
        compile_expr("klasse + 1").to_polars(text_names=["klasse"])


def test_categorical_column_and_text_inputs() -> None:
    df = FRAME.with_columns(pl.col("klasse").cast(pl.Categorical))
    out = df.select(compile_expr("klasse == 'null'").to_polars().alias("y"))["y"].to_list()
    assert out == [True, False, False, True, False, False]
    assert is_text(df["klasse"]) and is_text("a") and is_text(np.array(["a"]))
    assert not is_text(np.array([None, np.nan], dtype=object)) and not is_text(UArray(1.0))
    r = evaluate("probe == 'Eingang'", {"probe": df["probe"]})
    assert r.value.tolist() == [True, False, True, False, True, False]


def test_numbers_with_uncertainty_stay_numbers_next_to_text() -> None:
    env = {"klasse": ["null", "dosierung"], "S": UArray.input("S", [1.0, 2.0], 0.1)}
    r = evaluate("where(klasse == 'null', 0, S * 2)", env)
    np.testing.assert_allclose(r.value, [0.0, 4.0])
    np.testing.assert_allclose(r.u(), [0.0, 0.2])


def test_suggestion_prefers_first_letter_and_case() -> None:
    with pytest.raises(ExprError, match="did you mean 'B'"):
        compile_expr("Bx + 1", names=["B", "x"])
    assert suggest_name("v_mfc", ["V_MFC", "v_mfd"]) == "V_MFC"
    assert suggest_name("S_x", ["S_b", "B", "f"]) == "S_b"
    assert suggest_name("zzz", ["B", "x"]) is None
