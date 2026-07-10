"""Tools the agent can call on the conversational branch (function calling demo).

The calculator evaluates arithmetic by walking the AST instead of ``eval()``:
only numeric literals and whitelisted operators are allowed, so there is no
path to arbitrary code execution no matter what the LLM passes in.
"""

import ast
import operator
from datetime import date
from typing import Any

from langchain_core.tools import tool

_OPERATORS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

_MAX_EXPONENT = 100  # keeps 9**9**9-style expressions from hanging the process


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent too large (max {_MAX_EXPONENT})")
        return _OPERATORS[type(node.op)](left, right)
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:40]}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a pure arithmetic expression, e.g. "(2 + 3) * 4 ** 2".

    Supports +, -, *, /, %, ** and parentheses over numbers. Anything else
    (names, calls, strings) is rejected.
    """
    try:
        return str(_evaluate(ast.parse(expression, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"Error: {exc}"


@tool
def current_date() -> str:
    """Return today's date in ISO format (YYYY-MM-DD)."""
    return date.today().isoformat()


TOOLS = [calculator, current_date]
