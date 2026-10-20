"""Tests for step relations (Etappe 2b, step 8, PR 4).

Covers nearest_reference, cumulative (with changed(col) and gap resets), bin, and Pydantic models.
Synthetic data mirrors real catalysis run conditions from 17.10.2026.
"""

from __future__ import annotations

import math
import numpy as np
import polars as pl
import pytest
from pydantic import TypeAdapter, ValidationError

from datualizer_core.calc.expr import ExprError
from datualizer_core.calc.relations import (
    BinRelation,
    CumulativeRelation,
    FallbackValue,
    NearestReferenceRelation,
    Relation,
    apply_relation,
    apply_relations,
    bin,
    cumulative,
    nearest_reference,
)


# ------------------------------------------------------------------ Synthetic catalysis fixture
@pytest.fixture()
def catalysis_steps() -> pl.DataFrame:
    """Synthetic step table reproducing 17.10. catalysis cases:

    - Z1, Z2: morning zero phases, flat (S_b = 2.25 and 2.29 -> mean 2.27 mgC, |dS*60| <= 0.1).
    - Z23: dosing step 1 (150 °C, 10 Nl/min, 12.544 g/h, 19.65 min = 1179 s).
    - Z24: dosing step 2 directly after Z23 (150 °C, 10 Nl/min, 17.291 g/h, 11.217 min = 673.02 s).
    - Z5: zero phase after dosing, still falling (S_b = 3.50, dS*60 = -0.6 mgC/min -> NOT usable).
    - Z6: evening zero phase, flat (S_b = 2.31, |dS*60| <= 0.1 -> usable).
    - Z25: dosing step at 175 °C (triggers temperature change reset).
    - Z30: dosing step at 220 °C without matching zero phase (triggers fallback).
    """
    rows = [
        # Z1: Morning zero phase 1 (flat, usable)
        {
            "step_id": "Z1", "step_label": "Null morgens 1", "step_start": 0.0, "step_end": 600.0,
            "dauer_s": 600.0, "klasse": "null", "T_soll": 150.0, "luft": 100.0,
            "S_b": 2.25, "dS": 0.0005,  # dS*60 = 0.03 mgC/min <= 0.1 (flach)
            "mdot": float("nan"), "T_bett": 149.0,
        },
        # Z2: Morning zero phase 2 (flat, usable; mean with Z1 is 2.27)
        {
            "step_id": "Z2", "step_label": "Null morgens 2", "step_start": 630.0, "step_end": 1230.0,
            "dauer_s": 600.0, "klasse": "null", "T_soll": 150.0, "luft": 100.0,
            "S_b": 2.29, "dS": 0.0008,  # dS*60 = 0.048 <= 0.1 (flach)
            "mdot": float("nan"), "T_bett": 149.2,
        },
        # Z23: Dosing step 1 at 150 °C: 12.544 g/h, 19.65 min (1179 s) -> 4.108 g
        {
            "step_id": "Z23", "step_label": "Z23 (150/10)", "step_start": 1300.0, "step_end": 2479.0,
            "dauer_s": 19.65 * 60.0, "klasse": "dosierung", "T_soll": 150.0, "luft": 100.0,
            "S_b": float("nan"), "dS": 0.0001,
            "mdot": -12.544 / 3600.0,  # balance slope in g/s
            "T_bett": 149.5,
        },
        # Z24: Dosing step 2 directly after Z23 (gap 6 s < 300 s): 17.291 g/h, 11.217 min (673.02 s) -> +3.233 g -> cum 7.340 g
        {
            "step_id": "Z24", "step_label": "Z24 (150/10)", "step_start": 2485.0, "step_end": 3158.02,
            "dauer_s": 11.217 * 60.0, "klasse": "dosierung", "T_soll": 150.0, "luft": 100.0,
            "S_b": float("nan"), "dS": 0.0001,
            "mdot": -17.291 / 3600.0,
            "T_bett": 150.8,
        },
        # Z5: Zero phase after dosing, still falling (NOT flat, not usable)
        {
            "step_id": "Z5", "step_label": "Null nach Dosierung", "step_start": 3200.0, "step_end": 3600.0,
            "dauer_s": 400.0, "klasse": "null", "T_soll": 150.0, "luft": 100.0,
            "S_b": 3.50, "dS": -0.01,  # dS*60 = -0.6 mgC/min > 0.1 (faellt noch, unbrauchbar)
            "mdot": float("nan"), "T_bett": 150.0,
        },
        # Z6: Evening zero phase (flat, usable: S_b = 2.31)
        {
            "step_id": "Z6", "step_label": "Null abends", "step_start": 3650.0, "step_end": 4250.0,
            "dauer_s": 600.0, "klasse": "null", "T_soll": 150.0, "luft": 100.0,
            "S_b": 2.31, "dS": 0.0002,  # dS*60 = 0.012 <= 0.1 (flach)
            "mdot": float("nan"), "T_bett": 150.0,
        },
        # Z25: Dosing step at 175 °C: temperature change from 150 °C triggers reset!
        {
            "step_id": "Z25", "step_label": "Z25 (175/10)", "step_start": 4300.0, "step_end": 4900.0,
            "dauer_s": 600.0, "klasse": "dosierung", "T_soll": 175.0, "luft": 100.0,
            "S_b": float("nan"), "dS": 0.0001,
            "mdot": -12.0 / 3600.0,
            "T_bett": 174.8,
        },
        # Z30: Dosing step at 220 °C without matching zero phase (triggers fallback)
        {
            "step_id": "Z30", "step_label": "Z30 (220/10)", "step_start": 5000.0, "step_end": 5600.0,
            "dauer_s": 600.0, "klasse": "dosierung", "T_soll": 220.0, "luft": 100.0,
            "S_b": float("nan"), "dS": 0.0001,
            "mdot": -15.0 / 3600.0,
            "T_bett": 221.3,
        },
    ]
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ Tests: nearest_reference
class TestNearestReference:
    def test_bracketing_combines_before_and_after(self, catalysis_steps: pl.DataFrame):
        """Z23 at 150 °C has Z2 before (2.29) and Z6 after (2.31; Z5 is rejected as falling)."""
        res = nearest_reference(
            catalysis_steps,
            name="B",
            value="S_b",
            where="klasse == 'null' and abs(dS * 60) <= 0.1",
            match=["T_soll", "luft"],
            u_min=0.3,
            u_one=1.0,
            fallback={"value": 2.09, "u": 2.0},
        )
        assert res.columns == ["B", "u_B", "B_quelle", "B_art", "B_vor", "B_nach"]
        z23_row = catalysis_steps["step_id"].to_list().index("Z23")

        # Mean of 2.29 and 2.31 is 2.30
        assert res["B"][z23_row] == pytest.approx(2.30, abs=1e-4)
        assert res["B_vor"][z23_row] == pytest.approx(2.29, abs=1e-4)
        assert res["B_nach"][z23_row] == pytest.approx(2.31, abs=1e-4)
        assert res["B_art"][z23_row] == "eingeklammert"
        assert res["B_quelle"][z23_row] == "Z2; Z6"

        # Uncertainty: |2.29 - 2.31| / (2 * sqrt(3)) = 0.02 / 3.464 = 0.00577 < u_min (0.3) -> 0.3
        assert res["u_B"][z23_row] == pytest.approx(0.3, abs=1e-4)

    def test_fallback_when_no_reference_matches(self, catalysis_steps: pl.DataFrame):
        """Z30 at 220 °C has no matching zero phase at 220 °C -> fallback used."""
        res = nearest_reference(
            catalysis_steps,
            name="B",
            value="S_b",
            where="klasse == 'null' and abs(dS * 60) <= 0.1",
            match=["T_soll", "luft"],
            fallback=FallbackValue(value=2.09, u=2.0),
        )
        z30_row = catalysis_steps["step_id"].to_list().index("Z30")
        assert res["B"][z30_row] == pytest.approx(2.09, abs=1e-4)
        assert res["u_B"][z30_row] == pytest.approx(2.0, abs=1e-4)
        assert res["B_art"][z30_row] == "keiner"
        assert res["B_quelle"][z30_row] == ""
        assert math.isnan(res["B_vor"][z30_row])
        assert math.isnan(res["B_nach"][z30_row])

    def test_fallback_from_a_parameter_per_day(self, catalysis_steps: pl.DataFrame):
        """PR 10e: room air baseline as step parameter (journal cell or mapping default) per measuring day."""
        table = catalysis_steps.with_columns(
            pl.Series("raumluft", [2.09] * 6 + [float("nan"), 1.7])  # Z25 without a day value, Z30 another day
        )
        rel = NearestReferenceRelation.model_validate({
            "name": "B", "value": "S_b", "where": "klasse == 'null' and abs(dS * 60) <= 0.1",
            "match": ["T_soll", "luft"], "fallback": {"value": "raumluft", "u": "max(2.0, raumluft)"},
        })
        res = apply_relations(table, [rel])
        z25, z30 = res.row(6, named=True), res.row(7, named=True)
        assert z30["B"] == pytest.approx(1.7) and z30["u_B"] == pytest.approx(2.0) and z30["B_art"] == "keiner"
        assert math.isnan(z25["B"]) and math.isnan(z25["u_B"]) and z25["B_art"] == "keiner"
        assert res.row(2, named=True)["B_art"] == "eingeklammert"  # references still win
        with pytest.raises(ExprError, match="fallback value of 'B'.*unknown name 'raum'"):
            nearest_reference(table, name="B", value="S_b", where="klasse == 'null'", fallback={"value": "raum"})
        with pytest.raises(ValidationError, match="syntax"):
            FallbackValue(value="raumluft +")

    def test_one_sided_when_only_before_or_after_exists(self, catalysis_steps: pl.DataFrame):
        """Z25 at 175 °C has no matching zero phase at 175 °C either.
        If we match without T_soll (only luft), Z25 has Z6 before and none after -> einseitig."""
        res = nearest_reference(
            catalysis_steps,
            name="B",
            value="S_b",
            where="klasse == 'null' and abs(dS * 60) <= 0.1",
            match=["luft"],
            u_one=1.0,
        )
        z25_row = catalysis_steps["step_id"].to_list().index("Z25")
        assert res["B_art"][z25_row] == "einseitig"
        assert res["B"][z25_row] == pytest.approx(2.31, abs=1e-4)
        assert res["u_B"][z25_row] == pytest.approx(1.0, abs=1e-4)
        assert res["B_quelle"][z25_row] == "Z6"
        assert res["B_vor"][z25_row] == pytest.approx(2.31, abs=1e-4)
        assert math.isnan(res["B_nach"][z25_row])

    def test_side_before_only(self, catalysis_steps: pl.DataFrame):
        """When side='before', even bracketed steps only take the one before."""
        res = nearest_reference(
            catalysis_steps,
            name="B",
            value="S_b",
            where="klasse == 'null' and abs(dS * 60) <= 0.1",
            match=["T_soll", "luft"],
            side="before",
            u_one=1.0,
        )
        z23_row = catalysis_steps["step_id"].to_list().index("Z23")
        assert res["B_art"][z23_row] == "einseitig"
        assert res["B"][z23_row] == pytest.approx(2.29, abs=1e-4)
        assert res["B_quelle"][z23_row] == "Z2"
        assert math.isnan(res["B_nach"][z23_row])

    def test_side_after_only(self, catalysis_steps: pl.DataFrame):
        """When side='after', only the reference after is looked up."""
        res = nearest_reference(
            catalysis_steps,
            name="B",
            value="S_b",
            where="klasse == 'null' and abs(dS * 60) <= 0.1",
            match=["T_soll", "luft"],
            side="after",
            u_one=1.0,
        )
        z23_row = catalysis_steps["step_id"].to_list().index("Z23")
        assert res["B_art"][z23_row] == "einseitig"
        assert res["B"][z23_row] == pytest.approx(2.31, abs=1e-4)
        assert res["B_quelle"][z23_row] == "Z6"
        assert math.isnan(res["B_vor"][z23_row])

    def test_step_cannot_be_its_own_reference(self, catalysis_steps: pl.DataFrame):
        """Z1 is flat, but cannot reference itself before or after."""
        res = nearest_reference(
            catalysis_steps,
            name="B",
            value="S_b",
            where="klasse == 'null' and abs(dS * 60) <= 0.1",
            match=["T_soll", "luft"],
            fallback=FallbackValue(value=2.09, u=2.0),
        )
        # Z1 has no step before it, but Z2 after it
        assert res["B_art"][0] == "einseitig"
        assert res["B_quelle"][0] == "Z2"
        assert math.isnan(res["B_vor"][0])
        assert res["B_nach"][0] == pytest.approx(2.29, abs=1e-4)

    def test_empty_table_returns_empty_dataframe(self):
        empty = pl.DataFrame(schema={"step_id": pl.String, "step_start": pl.Float64, "step_end": pl.Float64, "S_b": pl.Float64})
        res = nearest_reference(empty, name="B", value="S_b", where="True")
        assert len(res) == 0
        assert res.columns == ["B", "u_B", "B_quelle", "B_art", "B_vor", "B_nach"]

    def test_unknown_value_or_match_column_fails_loudly(self, catalysis_steps: pl.DataFrame):
        with pytest.raises(ValueError, match="Value column 'nonexistent' not found"):
            nearest_reference(catalysis_steps, name="B", value="nonexistent", where="True")

        with pytest.raises(ValueError, match="Match column 'bad_col' not found"):
            nearest_reference(catalysis_steps, name="B", value="S_b", where="True", match=["bad_col"])

    def test_formula_syntax_error_fails_loudly(self, catalysis_steps: pl.DataFrame):
        with pytest.raises(ExprError):
            nearest_reference(catalysis_steps, name="B", value="S_b", where="klasse == 'null' and ")


# ------------------------------------------------------------------ Tests: cumulative & changed
class TestCumulative:
    def test_catalysis_loading_z23_and_z24_with_temperature_reset(self, catalysis_steps: pl.DataFrame):
        """Verify exact loading numbers from 17.10. catalysis run:

        - Z23: 150 °C, 10 Nl/min, 12.544 g/h, 19.65 min -> 4.108 g.
        - Z24: directly after (17.291 g/h, 11.217 min) -> 7.340 g cumulative.
        - Z25: at 175 °C -> changed(T_soll) triggers reset!
        """
        load = cumulative(
            catalysis_steps,
            name="beladung",
            expr="-mdot * dauer_s",
            reset_when="(klasse == 'null' and dauer_s >= 300) or changed(T_soll)",
            gap_s=300.0,
        )
        assert isinstance(load, pl.Series)
        assert load.name == "beladung"

        z23_row = catalysis_steps["step_id"].to_list().index("Z23")
        z24_row = catalysis_steps["step_id"].to_list().index("Z24")
        z25_row = catalysis_steps["step_id"].to_list().index("Z25")

        # Z23: 12.544 * 19.65 / 60 = 4.10816 g -> 4.108 g
        assert load[z23_row] == pytest.approx(4.108, abs=1e-3)

        # Z24: 4.10816 + 17.291 * 11.217 / 60 = 7.34071 g -> 7.340 g
        assert load[z24_row] == pytest.approx(7.340, abs=1e-3)

        # Z25 at 175 °C: reset occurred because T_soll changed!
        # Z25 increment: 12.0 * 600 / 3600 = 2.000 g
        assert load[z25_row] == pytest.approx(2.000, abs=1e-3)

    def test_null_steps_keep_nan_when_increment_is_nan(self, catalysis_steps: pl.DataFrame):
        load = cumulative(
            catalysis_steps,
            name="beladung",
            expr="-mdot * dauer_s",
            reset_when="(klasse == 'null' and dauer_s >= 300) or changed(T_soll)",
            keep_nan=True,
        )
        # In null rows, mdot is NaN -> output is NaN
        assert math.isnan(load[0])
        assert math.isnan(load[1])
        # Dosing steps have finite loading
        assert not math.isnan(load[2])

    def test_time_gap_triggers_reset(self):
        """Steps with gap >= gap_s reset the cumulative sum."""
        df = pl.DataFrame({
            "step_id": ["S1", "S2", "S3"],
            "step_start": [0.0, 100.0, 600.0],  # gap S2 -> S3 is 600 - 200 = 400 s >= 300 s
            "step_end": [100.0, 200.0, 700.0],
            "x": [10.0, 20.0, 5.0],
        })
        res = cumulative(df, expr="x", gap_s=300.0)
        assert res.to_list() == [10.0, 30.0, 5.0]

    def test_changed_preserves_value_across_intervening_null_steps(self):
        """Intervening null steps do NOT update the tracked dosing value for changed(col)."""
        df = pl.DataFrame({
            "step_id": ["D1", "N1", "D2", "D3"],
            "step_start": [0.0, 100.0, 200.0, 300.0],
            "step_end": [100.0, 150.0, 250.0, 350.0],
            "dauer_s": [100.0, 50.0, 50.0, 50.0],
            "klasse": ["dosierung", "null", "dosierung", "dosierung"],
            "T_soll": [150.0, 180.0, 150.0, 200.0],  # N1 has 180, but D2 is still 150 -> NOT changed!
            "m": [1.0, float("nan"), 2.0, 3.0],
        })
        # N1 duration is 50 s < 300 s, so N1 does not reset by duration.
        # D2 has T_soll = 150 == D1's 150 -> changed(T_soll) is False!
        # D3 has T_soll = 200 != 150 -> changed(T_soll) is True!
        res = cumulative(
            df,
            expr="m",
            reset_when="(klasse == 'null' and dauer_s >= 300) or changed(T_soll)",
            track_when="klasse == 'dosierung'",
            gap_s=1000.0,
        )
        assert res[0] == pytest.approx(1.0)
        assert math.isnan(res[1])
        # D2 continues accumulating from D1: 1.0 + 2.0 = 3.0
        assert res[2] == pytest.approx(3.0)
        # D3 resets because T_soll changed from 150 to 200: 0 + 3.0 = 3.0
        assert res[3] == pytest.approx(3.0)

    def test_changed_tracks_every_row_without_track_when(self):
        """No class column is guessed: without track_when every row counts, also a zero phase."""
        df = pl.DataFrame({
            "step_start": [0.0, 100.0, 200.0], "step_end": [100.0, 150.0, 250.0],
            "klasse": ["dosierung", "null", "dosierung"], "T_soll": [150.0, 180.0, 150.0], "m": [1.0, 0.0, 2.0],
        })
        res = cumulative(df, expr="m", reset_when="changed(T_soll)", gap_s=1000.0)
        assert res.to_list() == [1.0, 0.0, 2.0]  # 180 and back to 150 are both changes


# ------------------------------------------------------------------ Tests: bin
class TestBin:
    def test_bin_single_number(self):
        assert bin(149.5, 25.0) == 150.0
        assert bin(163.0, 25.0) == 175.0
        assert bin(175.0, 25.0) == 175.0
        assert bin(-12.0, 10.0) == -10.0

    def test_bin_series(self):
        s = pl.Series("temp", [149.2, 163.1, 174.8, 221.3, float("nan")])
        res = bin(s, 25.0)
        assert isinstance(res, pl.Series)
        assert res.name == "temp"
        assert res[0] == 150.0
        assert res[1] == 175.0
        assert res[2] == 175.0
        assert res[3] == 225.0
        assert math.isnan(res[4])

    def test_bin_table_with_expr(self, catalysis_steps: pl.DataFrame):
        res = bin(catalysis_steps, "T_bett", 25.0, name="T_klasse")
        assert isinstance(res, pl.Series)
        assert res.name == "T_klasse"
        assert res[0] == 150.0  # 149.0 -> 150
        assert res[6] == 175.0  # 174.8 -> 175
        assert res[7] == 225.0  # 221.3 -> 225

    def test_bin_invalid_width_raises_value_error(self):
        with pytest.raises(ValueError, match="width must be positive"):
            bin(100.0, 0.0)
        with pytest.raises(ValueError, match="width must be positive"):
            bin(100.0, -5.0)


# ------------------------------------------------------------------ Tests: Pydantic models & apply_relations
class TestRelationModels:
    def test_discriminated_union_parsing(self):
        ta = TypeAdapter(Relation)

        r1 = ta.validate_python({
            "name": "B", "kind": "nearest_reference", "value": "S_b",
            "where": "klasse == 'null' and abs(dS*60) <= 0.1", "match": ["T_soll", "luft"],
            "u_min": 0.3, "u_one": 1.0, "fallback": {"value": 2.09, "u": 2.0},
        })
        assert isinstance(r1, NearestReferenceRelation)
        assert r1.columns() == ("B", "u_B", "B_quelle", "B_art", "B_vor", "B_nach")
        assert r1.fallback is not None and r1.fallback.value == 2.09

        r2 = ta.validate_python({
            "name": "beladung", "kind": "cumulative", "expr": "-mdot * dauer_s",
            "reset_when": "(klasse == 'null' and dauer_s >= 300) or changed(T_soll)",
            "gap_s": 300.0,
        })
        assert isinstance(r2, CumulativeRelation)
        assert r2.columns() == ("beladung", "beladung_lauf")

        r3 = ta.validate_python({
            "name": "T_klasse", "kind": "bin", "expr": "T_bett", "width": 25.0,
        })
        assert isinstance(r3, BinRelation)
        assert r3.columns() == ("T_klasse",)

    def test_forbid_extra_fields(self):
        ta = TypeAdapter(Relation)
        with pytest.raises(ValidationError):
            ta.validate_python({"name": "B", "kind": "bin", "expr": "T", "width": 10, "extra": 123})

    def test_apply_relations_pipeline(self, catalysis_steps: pl.DataFrame):
        relations = [
            NearestReferenceRelation(
                name="B",
                value="S_b",
                where="klasse == 'null' and abs(dS * 60) <= 0.1",
                match=["T_soll", "luft"],
                fallback=FallbackValue(value=2.09, u=2.0),
            ),
            CumulativeRelation(
                name="beladung",
                expr="-mdot * dauer_s",
                reset_when="(klasse == 'null' and dauer_s >= 300) or changed(T_soll)",
            ),
            BinRelation(name="T_klasse", expr="T_bett", width=25.0),
        ]
        out = apply_relations(catalysis_steps, relations)
        # Check that all generated columns are present in the resulting DataFrame
        expected_cols = [
            "B", "u_B", "B_quelle", "B_art", "B_vor", "B_nach",
            "beladung", "T_klasse",
        ]
        for col in expected_cols:
            assert col in out.columns

        z23_idx = out["step_id"].to_list().index("Z23")
        assert out["B"][z23_idx] == pytest.approx(2.30, abs=1e-4)
        assert out["beladung"][z23_idx] == pytest.approx(4.108, abs=1e-3)
        assert out["T_klasse"][z23_idx] == 150.0
