"""Safe formulas for calculated quantities (Etappe 2b, step 8, PR 1).

A formula is parsed with Python's `ast` and checked against a positive list; it is never handed
to `eval` or `exec`. Allowed are:

- numbers; names, also dotted (``fid.mean`` is one name, not an attribute access);
- ``+ - * / **``, unary ``-`` and ``+``, comparisons (also chained, ``0 < x <= 1``),
  ``and``, ``or``, ``not``, ``a if c else b``;
- the functions in `FUNCTIONS`: log (natural), log10, exp, sqrt, abs, min, max (two or more
  arguments), where(c, a, b), isfinite, round(x, n=0).

Everything else (attribute access, method calls, indexing, lambdas, comprehensions, strings,
keyword arguments, other operators) is rejected with an `ExprError` that names the position and
the known names. The typographic signs − (minus), · and × (times) are read as ``-`` and ``*``.

One compiled `Expr` serves two targets with the same meaning:

- `Expr.to_polars()` builds a `pl.Expr` on columns (calculated channels on raw data); it carries
  values only, constants become literals.
- `Expr.evaluate(env)` computes with numpy on `UArray` and propagates the uncertainty of every
  input exactly to first order (step table).

Semantics shared by both: numbers are floats, ``/`` is true division, a division by zero gives
±inf or NaN, NaN propagates through ``min``/``max`` and compares unequal to everything (IEEE,
also in Polars, which would otherwise order NaN above all numbers); conditions (``and``/``or``/``not``,
``where``, ``if``) treat non-zero numbers as true. Polars additionally knows null (missing): it
propagates through every operation, including conditions and ``min``/``max``.
"""

from __future__ import annotations

import ast
import difflib
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import polars as pl

from datualizer_core.calc import uncertain as U
from datualizer_core.calc.uncertain import Constant, UArray

MAX_LENGTH = 10_000  # characters; a formula is a line, not a program

# name -> (min args, max args, short description); the evaluators below implement exactly these
FUNCTIONS: dict[str, tuple[int, int | None, str]] = {
    "log": (1, 1, "natural logarithm"),
    "log10": (1, 1, "decimal logarithm"),
    "exp": (1, 1, "e to the power x"),
    "sqrt": (1, 1, "square root"),
    "abs": (1, 1, "absolute value"),
    "min": (2, None, "smallest argument, element-wise"),
    "max": (2, None, "largest argument, element-wise"),
    "where": (3, 3, "where(c, a, b): a where c is true, else b"),
    "isfinite": (1, 1, "true for finite numbers"),
    "round": (1, 2, "round(x, n=0), half to even; carries no uncertainty"),
}

_TYPOGRAPHIC = str.maketrans({"−": "-", "·": "*", "×": "*"})

_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
_CMPOPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
_REJECTED = {
    ast.FloorDiv: "'//'", ast.Mod: "'%'", ast.MatMult: "'@'", ast.LShift: "'<<'", ast.RShift: "'>>'",
    ast.BitAnd: "'&' (use 'and')", ast.BitOr: "'|' (use 'or')", ast.BitXor: "'^' (use '**' for powers)",
    ast.Invert: "'~' (use 'not')", ast.Is: "'is'", ast.IsNot: "'is not'", ast.In: "'in'", ast.NotIn: "'not in'",
}
_FORBIDDEN_NODES = {
    ast.Subscript: "indexing ('x[...]') is not allowed",
    ast.Lambda: "lambda is not allowed",
    ast.ListComp: "comprehensions are not allowed",
    ast.SetComp: "comprehensions are not allowed",
    ast.DictComp: "comprehensions are not allowed",
    ast.GeneratorExp: "comprehensions are not allowed",
    ast.NamedExpr: "assignments (':=') are not allowed",
    ast.List: "lists are not allowed",
    ast.Tuple: "tuples are not allowed",
    ast.Set: "sets are not allowed",
    ast.Dict: "dicts are not allowed",
    ast.JoinedStr: "text (f-strings) is not allowed",
    ast.Starred: "'*' unpacking is not allowed",
    ast.Await: "'await' is not allowed",
    ast.Yield: "'yield' is not allowed",
    ast.YieldFrom: "'yield' is not allowed",
    ast.Slice: "slices are not allowed",
}


class ExprError(ValueError):
    """A formula that cannot be compiled or evaluated.

    `pos` is the 0-based character offset in `text` (None if unknown), `column` the same 1-based;
    `known` lists the names that would have been accepted.
    """

    def __init__(self, message: str, text: str = "", pos: int | None = None, known: Iterable[str] = ()) -> None:
        self.message = message
        self.text = text
        self.pos = pos
        self.known = sorted(known)
        super().__init__(self._format())

    @property
    def column(self) -> int | None:
        return None if self.pos is None else self.pos + 1

    def _format(self) -> str:
        parts = [self.message]
        if self.pos is not None:
            start = self.text.rfind("\n", 0, self.pos) + 1
            end = self.text.find("\n", self.pos)
            line = self.text[start:] if end < 0 else self.text[start:end]
            parts[0] += f" (column {self.pos - start + 1})"
            parts.append(f"    {line}\n    {' ' * (self.pos - start)}^")
        elif self.text:
            parts.append(f"    {self.text}")
        if self.known:
            parts.append("Known names: " + ", ".join(self.known))
        return "\n".join(parts)


# --- tree ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Node:
    pos: int


@dataclass(frozen=True)
class _Num(_Node):
    value: float


@dataclass(frozen=True)
class _Var(_Node):
    name: str


@dataclass(frozen=True)
class _Unary(_Node):
    op: str  # "-", "+", "not"
    operand: _Node


@dataclass(frozen=True)
class _Bin(_Node):
    op: str
    left: _Node
    right: _Node


@dataclass(frozen=True)
class _Cmp(_Node):
    ops: tuple[str, ...]
    operands: tuple[_Node, ...]


@dataclass(frozen=True)
class _Bool(_Node):
    op: str  # "and", "or"
    values: tuple[_Node, ...]


@dataclass(frozen=True)
class _If(_Node):
    cond: _Node
    then: _Node
    other: _Node


@dataclass(frozen=True)
class _Call(_Node):
    func: str
    args: tuple[_Node, ...]


# --- compiling ----------------------------------------------------------------------------------


class _Compiler:
    def __init__(self, text: str, source: str, lead: int, names: frozenset[str] | None) -> None:
        self.text = text
        self.lead = lead  # leading blanks cut off before parsing (they shift line 1 only)
        self.lines = source.split("\n")
        self.offsets = [lead]
        for line in self.lines[:-1]:
            self.offsets.append(self.offsets[-1] + len(line) + 1)
        self.known = names
        self.seen: list[str] = []

    def pos(self, node: ast.AST) -> int:
        line = getattr(node, "lineno", 1) - 1
        col_bytes = getattr(node, "col_offset", 0)
        col = len(self.lines[line].encode("utf-8")[:col_bytes].decode("utf-8", errors="ignore"))
        return self.offsets[line] + col

    def after(self, node: ast.AST) -> int:
        """Position of the operator that follows `node` (first character after it but blanks and ')')."""
        line = node.end_lineno - 1
        col_bytes = node.end_col_offset
        col = len(self.lines[line].encode("utf-8")[:col_bytes].decode("utf-8", errors="ignore"))
        pos = self.offsets[line] + col
        while pos < len(self.text) and self.text[pos] in " \t\n)":
            pos += 1
        return min(pos, len(self.text) - 1)

    def fail(self, message: str, node: ast.AST, known: bool = False, pos: int | None = None) -> ExprError:
        where = self.pos(node) if pos is None else pos
        return ExprError(message, self.text, where, (self.known or ()) if known else ())

    def convert(self, node: ast.AST) -> _Node:
        p = self.pos(node)
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                return _Num(p, float(v))
            if isinstance(v, (int, float)):
                try:
                    return _Num(p, float(v))
                except OverflowError:
                    raise self.fail("number is too large", node) from None
            if isinstance(v, (str, bytes)):
                raise self.fail("text constants are not allowed, only numbers", node)
            raise self.fail(f"constant {v!r} is not allowed, only numbers", node)
        if isinstance(node, (ast.Name, ast.Attribute)):
            return self.name(node)
        if isinstance(node, ast.UnaryOp):
            op = {ast.USub: "-", ast.UAdd: "+", ast.Not: "not"}.get(type(node.op))
            if op is None:
                raise self.fail(f"operator {_REJECTED.get(type(node.op), type(node.op).__name__)} is not allowed", node)
            operand = self.convert(node.operand)
            if op in "+-" and isinstance(operand, _Num):  # a signed number stays a number (round(x, -1))
                return _Num(p, -operand.value if op == "-" else operand.value)
            return _Unary(p, op, operand)
        if isinstance(node, ast.BinOp):
            op = _BINOPS.get(type(node.op))
            if op is None:
                what = _REJECTED.get(type(node.op), type(node.op).__name__)
                raise self.fail(f"operator {what} is not allowed; allowed are + - * / **", node, pos=self.after(node.left))
            return _Bin(p, op, self.convert(node.left), self.convert(node.right))
        if isinstance(node, ast.Compare):
            ops = []
            for o, left in zip(node.ops, [node.left, *node.comparators]):
                if type(o) not in _CMPOPS:
                    what = _REJECTED.get(type(o), type(o).__name__)
                    raise self.fail(f"comparison {what} is not allowed", node, pos=self.after(left))
                ops.append(_CMPOPS[type(o)])
            operands = tuple(self.convert(x) for x in [node.left, *node.comparators])
            return _Cmp(p, tuple(ops), operands)
        if isinstance(node, ast.BoolOp):
            op = "and" if isinstance(node.op, ast.And) else "or"
            return _Bool(p, op, tuple(self.convert(v) for v in node.values))
        if isinstance(node, ast.IfExp):
            return _If(p, self.convert(node.test), self.convert(node.body), self.convert(node.orelse))
        if isinstance(node, ast.Call):
            return self.call(node)
        message = _FORBIDDEN_NODES.get(type(node), f"'{type(node).__name__}' is not allowed in a formula")
        raise self.fail(message, node)

    def name(self, node: ast.AST) -> _Var:
        parts: list[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            raise self.fail("attribute access is not allowed; a dotted name must consist of names only", node)
        parts.append(cur.id)
        name = ".".join(reversed(parts))
        if name in FUNCTIONS:
            raise self.fail(f"function {name!r} needs arguments, e.g. {name}(x)", node)
        if self.known is not None and name not in self.known:
            hint = difflib.get_close_matches(name, sorted(self.known), n=1)
            extra = f"; did you mean {hint[0]!r}?" if hint else ""
            raise self.fail(f"unknown name {name!r}{extra}", node, known=True)
        if name not in self.seen:
            self.seen.append(name)
        return _Var(self.pos(node), name)

    def call(self, node: ast.Call) -> _Call:
        func = node.func
        if isinstance(func, ast.Attribute):
            raise self.fail("method calls ('a.b(...)') are not allowed", node)
        if not isinstance(func, ast.Name):
            raise self.fail("only the listed functions can be called", node)
        allowed = ", ".join(FUNCTIONS)
        if func.id not in FUNCTIONS:
            raise self.fail(f"unknown function {func.id!r}; allowed are {allowed}", node)
        if node.keywords:
            raise self.fail(f"{func.id}() takes no keyword arguments", node)
        lo, hi, _ = FUNCTIONS[func.id]
        n = len(node.args)
        if n < lo or (hi is not None and n > hi):
            want = f"{lo}" if lo == hi else (f"at least {lo}" if hi is None else f"{lo} to {hi}")
            raise self.fail(f"{func.id}() takes {want} arguments, got {n}", node)
        args = tuple(self.convert(a) for a in node.args)
        if func.id == "round" and len(args) == 2:
            nd = args[1]
            if not isinstance(nd, _Num) or nd.value != int(nd.value) or abs(nd.value) > 15:
                raise self.fail("round(x, n): n must be a whole number from -15 to 15", node.args[1])
        return _Call(self.pos(node), func.id, args)


def compile_expr(text: str, names: Iterable[str] | None = None) -> "Expr":
    """Parse and check a formula. `names` are the accepted names (None = any name)."""
    if not isinstance(text, str):
        raise ExprError(f"a formula must be text, got {type(text).__name__}")
    known = None if names is None else frozenset(names)
    known_list = known or ()
    if not text.strip():
        raise ExprError("the formula is empty", text, known=known_list)
    if len(text) > MAX_LENGTH:
        raise ExprError(f"the formula is longer than {MAX_LENGTH} characters", known=known_list)
    source = text.translate(_TYPOGRAPHIC)
    lead = len(source) - len(source.lstrip(" \t"))  # eval mode rejects an indented first line
    source = source[lead:]
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        pos = None
        if e.lineno is not None and e.offset is not None:
            lines = source.split("\n")
            pos = lead + sum(len(line) + 1 for line in lines[: e.lineno - 1]) + max(e.offset - 1, 0)
            pos = min(pos, len(text))
        raise ExprError(f"syntax error: {e.msg}", text, pos, known_list) from None
    except (RecursionError, MemoryError):
        raise ExprError("the formula is nested too deeply", text, known=known_list) from None
    compiler = _Compiler(text, source, lead, known)
    try:
        node = compiler.convert(tree.body)
    except RecursionError:
        raise ExprError("the formula is nested too deeply", text, known=known_list) from None
    return Expr(text, node, tuple(compiler.seen))


# --- the compiled expression --------------------------------------------------------------------


@dataclass(frozen=True)
class Expr:
    """A checked formula. `names` lists the names it uses, in order of appearance."""

    text: str
    node: _Node
    names: tuple[str, ...]

    def __str__(self) -> str:
        return self.text

    def evaluate(self, env: Mapping[str, Any]) -> UArray:
        """Compute with numpy. `env` maps every name to a `UArray`, `Constant`, number, array or Series."""
        missing = [n for n in self.names if n not in env]
        if missing:
            node = _find_var(self.node, missing[0])
            raise ExprError(f"no value for name {missing[0]!r}", self.text, node.pos if node else None, env.keys())
        values = {n: UArray.of(env[n]) for n in self.names}
        try:
            return _NumpyEval(values).run(self.node)
        except ValueError as e:
            raise ExprError(str(e), self.text) from None

    def to_polars(
        self,
        columns: Mapping[str, str] | None = None,
        constants: Mapping[str, float | Constant] | None = None,
    ) -> pl.Expr:
        """The formula as a Polars expression.

        A name becomes ``pl.col(columns.get(name, name))``, or a float literal if it is in
        `constants` (a `Constant` contributes its value; Polars carries no uncertainty).
        """
        return _PolarsBuild(columns or {}, constants or {}).run(self.node)


def _find_var(node: _Node, name: str) -> _Var | None:
    if isinstance(node, _Var):
        return node if node.name == name else None
    for child in _children(node):
        found = _find_var(child, name)
        if found is not None:
            return found
    return None


def _children(node: _Node) -> tuple[_Node, ...]:
    if isinstance(node, _Unary):
        return (node.operand,)
    if isinstance(node, _Bin):
        return (node.left, node.right)
    if isinstance(node, _Cmp):
        return node.operands
    if isinstance(node, _Bool):
        return node.values
    if isinstance(node, _If):
        return (node.cond, node.then, node.other)
    if isinstance(node, _Call):
        return node.args
    return ()


# --- numpy with uncertainties -------------------------------------------------------------------

_NUMPY_BIN: dict[str, Callable[[Any, Any], UArray]] = {
    "+": U.add, "-": U.sub, "*": U.mul, "/": U.div, "**": U.power,
}
_NUMPY_FUNC: dict[str, Callable[..., UArray]] = {
    "log": U.log, "log10": U.log10, "exp": U.exp, "sqrt": U.sqrt, "abs": U.absolute,
    "min": U.minimum, "max": U.maximum, "where": U.where, "isfinite": U.isfinite, "round": U.round_,
}


class _NumpyEval:
    def __init__(self, values: Mapping[str, UArray]) -> None:
        self.values = values

    def run(self, node: _Node) -> UArray:
        if isinstance(node, _Num):
            return UArray(node.value)
        if isinstance(node, _Var):
            return self.values[node.name]
        if isinstance(node, _Unary):
            x = self.run(node.operand)
            return U.logical_not(x) if node.op == "not" else (U.neg(x) if node.op == "-" else x)
        if isinstance(node, _Bin):
            return _NUMPY_BIN[node.op](self.run(node.left), self.run(node.right))
        if isinstance(node, _Cmp):
            operands = [self.run(o) for o in node.operands]
            result = None
            for op, a, b in zip(node.ops, operands, operands[1:]):
                part = U.COMPARISONS[op](a, b)
                result = part if result is None else U.logical_and(result, part)
            return result
        if isinstance(node, _Bool):
            combine = U.logical_and if node.op == "and" else U.logical_or
            result = self.run(node.values[0])
            for v in node.values[1:]:
                result = combine(result, self.run(v))
            return result
        if isinstance(node, _If):
            return U.where(self.run(node.cond), self.run(node.then), self.run(node.other))
        if isinstance(node, _Call):
            args = [self.run(a) for a in node.args]
            if node.func == "round" and len(args) == 2:
                return U.round_(args[0], int(node.args[1].value))  # type: ignore[attr-defined]
            return _NUMPY_FUNC[node.func](*args)
        raise TypeError(f"unexpected node {node!r}")


# --- polars -------------------------------------------------------------------------------------


def _f64(e: pl.Expr) -> pl.Expr:
    return e.cast(pl.Float64)


def _truth(e: pl.Expr) -> pl.Expr:
    return e.cast(pl.Boolean)


def _pl_where(c: pl.Expr, a: pl.Expr, b: pl.Expr) -> pl.Expr:
    c = _truth(c)
    return pl.when(c.is_null()).then(pl.lit(None)).when(c).then(a).otherwise(b)


def _pl_extreme(args: list[pl.Expr], first: Callable[[pl.Expr, pl.Expr], pl.Expr]) -> pl.Expr:
    result = _f64(args[0])
    for other in args[1:]:
        other = _f64(other)
        result = (
            pl.when(result.is_null() | other.is_null()).then(pl.lit(None, dtype=pl.Float64))
            .when(result.is_nan() | other.is_nan()).then(pl.lit(math.nan))
            .when(first(result, other)).then(result)
            .otherwise(other)
        )
    return result


_PL_CMP: dict[str, Callable[[pl.Expr, pl.Expr], pl.Expr]] = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}


def _pl_compare(op: str, a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """IEEE comparison like numpy: NaN is unequal to everything (Polars orders NaN above all numbers)."""
    a, b = _f64(a), _f64(b)
    return pl.when(a.is_nan() | b.is_nan()).then(pl.lit(op == "!=")).otherwise(_PL_CMP[op](a, b))


class _PolarsBuild:
    def __init__(self, columns: Mapping[str, str], constants: Mapping[str, float | Constant]) -> None:
        self.columns = columns
        self.constants = constants

    def run(self, node: _Node) -> pl.Expr:
        if isinstance(node, _Num):
            return pl.lit(node.value, dtype=pl.Float64)
        if isinstance(node, _Var):
            if node.name in self.constants:
                c = self.constants[node.name]
                return pl.lit(float(c.value if isinstance(c, Constant) else c), dtype=pl.Float64)
            return pl.col(self.columns.get(node.name, node.name))
        if isinstance(node, _Unary):
            x = self.run(node.operand)
            return ~_truth(x) if node.op == "not" else (-x if node.op == "-" else x)
        if isinstance(node, _Bin):
            a, b = self.run(node.left), self.run(node.right)
            if node.op == "**":
                return _f64(a).pow(b)
            return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[node.op]
        if isinstance(node, _Cmp):
            operands = [self.run(o) for o in node.operands]
            parts = [_pl_compare(op, a, b) for op, a, b in zip(node.ops, operands, operands[1:])]
            result = parts[0]
            for part in parts[1:]:
                result = result & part
            return result
        if isinstance(node, _Bool):
            result = _truth(self.run(node.values[0]))
            for v in node.values[1:]:
                nxt = _truth(self.run(v))
                result = (result & nxt) if node.op == "and" else (result | nxt)
            return result
        if isinstance(node, _If):
            return _pl_where(self.run(node.cond), self.run(node.then), self.run(node.other))
        if isinstance(node, _Call):
            args = [self.run(a) for a in node.args]
            f = node.func
            if f == "where":
                return _pl_where(*args)
            if f == "min":
                return _pl_extreme(args, lambda a, b: a <= b)
            if f == "max":
                return _pl_extreme(args, lambda a, b: a >= b)
            x = _f64(args[0])
            if f == "round":
                n = int(node.args[1].value) if len(node.args) == 2 else 0  # type: ignore[attr-defined]
                if n >= 0:
                    return x.round(n)
                scale = 10.0 ** -n  # Polars rounds to decimals >= 0 only; numpy does the same scaling
                return (x / scale).round(0) * scale
            return {
                "log": lambda: x.log(),
                "log10": lambda: x.log10(),
                "exp": lambda: x.exp(),
                "sqrt": lambda: x.sqrt(),
                "abs": lambda: x.abs(),
                "isfinite": lambda: x.is_finite(),
            }[f]()
        raise TypeError(f"unexpected node {node!r}")


def evaluate(text: str, env: Mapping[str, Any]) -> UArray:
    """Shortcut: compile against the names in `env` and evaluate."""
    return compile_expr(text, env.keys()).evaluate(env)


__all__ = ["FUNCTIONS", "Expr", "ExprError", "compile_expr", "evaluate"]
