import pytest

from src.solver import MathSolver


def test_solver_evaluates_arithmetic_expression():
    solver = MathSolver()

    result = solver.solve("2 + 3 * 4")

    assert result == 14


def test_solver_solves_simple_equation():
    solver = MathSolver()

    result = solver.solve("x + 2 = 5")

    assert result == [3]


def test_solver_supports_implicit_multiplication_with_variable():
    solver = MathSolver()

    result = solver.solve("2x + 3 = 7")

    assert result == [2]


def test_solver_evaluates_numeric_equation_as_boolean():
    solver = MathSolver()

    assert solver.solve("13 + 57 = 70") is True
    assert solver.solve("13 + 57 = 71") is False


def test_solver_describes_numeric_equation_sides():
    solver = MathSolver()

    description = solver.describe_solution("13 + 57 = 70")

    assert description["kind"] == "numeric_equation"
    assert description["left_side"] == "13+57"
    assert str(description["left_value"]) == "70"
    assert description["right_side"] == "70"
    assert str(description["right_value"]) == "70"
    assert description["result"] is True


def test_solver_normalizes_unicode_operators():
    solver = MathSolver()

    result = solver.solve("2 \u00d7 3")

    assert result == 6


def test_solver_normalizes_uppercase_variable():
    solver = MathSolver()

    result = solver.solve("2X+3=7")

    assert result == [2]


def test_solver_rejects_invalid_input():
    solver = MathSolver()

    with pytest.raises(ValueError):
        solver.solve("")


def test_solver_rejects_multiple_equals():
    solver = MathSolver()

    with pytest.raises(ValueError):
        solver.solve("1=2=3")
