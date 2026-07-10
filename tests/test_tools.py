"""Tool tests: safe calculator behavior and current_date format."""

from datetime import date

from agentic_rag.agent.tools import calculator, current_date


def test_calculator_basic_arithmetic() -> None:
    assert calculator.invoke({"expression": "2 + 3 * 4"}) == "14"
    assert calculator.invoke({"expression": "(2 + 3) * 4 ** 2"}) == "80"
    assert calculator.invoke({"expression": "-7 % 3"}) == "2"


def test_calculator_division() -> None:
    assert calculator.invoke({"expression": "7 / 2"}) == "3.5"
    assert calculator.invoke({"expression": "1 / 0"}).startswith("Error:")


def test_calculator_rejects_code_execution() -> None:
    # Anything that is not pure arithmetic must come back as an error string,
    # never be evaluated.
    assert calculator.invoke({"expression": "__import__('os').getcwd()"}).startswith("Error:")
    assert calculator.invoke({"expression": "'a' * 3"}).startswith("Error:")


def test_calculator_rejects_huge_exponents() -> None:
    assert calculator.invoke({"expression": "9 ** 9 ** 9"}).startswith("Error:")


def test_calculator_rejects_malformed_input() -> None:
    assert calculator.invoke({"expression": "2 +"}).startswith("Error:")


def test_current_date_is_iso_formatted() -> None:
    result = current_date.invoke({})
    assert date.fromisoformat(result) == date.today()
