from sympy import symbols, Eq, solve
from sympy.parsing.sympy_parser import parse_expr

def resolver(expressao):
    x = symbols('x')

    if "=" in expressao:
        lado_esq, lado_dir = expressao.split("=")

        expr_esq = parse_expr(lado_esq)
        expr_dir = parse_expr(lado_dir)

        equacao = Eq(expr_esq, expr_dir)
        resultado = solve(equacao, x)

        return resultado

    else:
        expr = parse_expr(expressao)
        return expr