from sympy import symbols, Eq, solve
from sympy.parsing.sympy_parser import parse_expr
from typing import Any

def resolver(expressao: str) -> Any:
    """
    Resolve uma expressão matemática ou equação em formato de string.
    """
    x = symbols('x')

    try:
        if "=" in expressao:
            lado_esq, lado_dir = expressao.split("=")
            
            expr_esq = parse_expr(lado_esq.strip())
            expr_dir = parse_expr(lado_dir.strip())
            
            equacao = Eq(expr_esq, expr_dir)
            resultado = solve(equacao, x)
            return resultado
        else:
            expr = parse_expr(expressao.strip())
            return expr
            
    except Exception as e:
        raise ValueError(f"Erro ao processar a expressão '{expressao}': {e}")