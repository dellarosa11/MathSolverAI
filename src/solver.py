from __future__ import annotations

from typing import Any, Dict, List, Union

from sympy import Eq, simplify, solve, symbols
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


ALLOWED_CHARACTERS = set("0123456789x+-*/=() ")
NORMALIZATION_MAP = {
    "\u00d7": "*",
    "\u00f7": "/",
    "X": "x",
}
PARSER_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


class MathSolver:
    """
    Classe responsavel por resolver expressoes matematicas e equacoes.
    """

    def __init__(self, variable_name: str = "x"):
        self.variable = symbols(variable_name)

    @staticmethod
    def _has_balanced_parentheses(expression: str) -> bool:
        depth = 0
        for character in expression:
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    def normalize_expression(self, expression: str) -> str:
        if not expression or not isinstance(expression, str):
            raise ValueError("A expressao deve ser uma string nao vazia.")

        normalized = expression.strip()
        for source, target in NORMALIZATION_MAP.items():
            normalized = normalized.replace(source, target)

        normalized = "".join(normalized.split())
        if not normalized:
            raise ValueError("A expressao nao pode ficar vazia apos a normalizacao.")

        invalid_characters = sorted({char for char in normalized if char not in ALLOWED_CHARACTERS})
        if invalid_characters:
            invalid_repr = ", ".join(invalid_characters)
            raise ValueError(f"Caracteres nao suportados na expressao: {invalid_repr}")

        if normalized.count("=") > 1:
            raise ValueError("A expressao contem mais de um sinal de igualdade.")

        if not self._has_balanced_parentheses(normalized):
            raise ValueError("A expressao contem parenteses desbalanceados.")

        if "=" in normalized:
            left_side, right_side = normalized.split("=", 1)
            if not left_side or not right_side:
                raise ValueError("A equacao precisa ter conteudo nos dois lados do '='.")

        return normalized

    @staticmethod
    def _parse(expression: str):
        return parse_expr(
            expression,
            transformations=PARSER_TRANSFORMATIONS,
            evaluate=True,
        )

    def solve(self, expression: str) -> Union[Any, List[Any]]:
        clean_expression = self.normalize_expression(expression)

        try:
            if "=" in clean_expression:
                left_side, right_side = clean_expression.split("=", 1)
                left_expr = self._parse(left_side)
                right_expr = self._parse(right_side)
                difference = simplify(left_expr - right_expr)

                if not getattr(difference, "free_symbols", set()):
                    return bool(difference == 0)

                equation = Eq(left_expr, right_expr)
                return solve(equation, self.variable)

            return simplify(self._parse(clean_expression))
        except Exception as exc:
            raise ValueError(f"Erro ao processar a expressao '{expression}': {exc}") from exc

    def describe_solution(self, expression: str) -> Dict[str, Any]:
        clean_expression = self.normalize_expression(expression)

        if "=" not in clean_expression:
            return {
                "kind": "expression",
                "normalized_expression": clean_expression,
                "result": self.solve(clean_expression),
            }

        left_side, right_side = clean_expression.split("=", 1)
        left_expr = self._parse(left_side)
        right_expr = self._parse(right_side)
        left_value = simplify(left_expr)
        right_value = simplify(right_expr)
        difference = simplify(left_value - right_value)

        if not getattr(difference, "free_symbols", set()):
            return {
                "kind": "numeric_equation",
                "normalized_expression": clean_expression,
                "left_side": left_side,
                "right_side": right_side,
                "left_value": left_value,
                "right_value": right_value,
                "result": bool(difference == 0),
            }

        return {
            "kind": "symbolic_equation",
            "normalized_expression": clean_expression,
            "left_side": left_side,
            "right_side": right_side,
            "result": self.solve(clean_expression),
        }


def resolver(expressao: str) -> Any:
    """
    Funcao utilitaria para manter compatibilidade com versoes anteriores.
    """
    solver = MathSolver()
    return solver.solve(expressao)


if __name__ == "__main__":
    solver = MathSolver()
    print(f"x + 2 = 5 -> {solver.solve('x + 2 = 5')}")
    print(f"2 + 3 * 4 -> {solver.solve('2 + 3 * 4')}")
