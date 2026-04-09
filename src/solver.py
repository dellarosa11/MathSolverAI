from sympy import symbols, Eq, solve
from sympy.parsing.sympy_parser import parse_expr
from typing import Any, List, Union, Optional

class MathSolver:
    """
    Classe responsável por resolver expressões matemáticas e equações.
    
    Aplica o princípio de responsabilidade única (SRP) ao focar apenas na 
    resolução simbólica de equações usando SymPy.
    """
    def __init__(self, variable_name: str = 'x'):
        """
        Inicializa o resolvedor com a variável simbólica padrão.
        
        Args:
            variable_name (str): Nome da variável a ser resolvida (padrão 'x').
        """
        self.variable = symbols(variable_name)

    def solve(self, expression: str) -> Union[Any, List[Any]]:
        """
        Resolve uma expressão matemática ou equação em formato de string.
        
        Args:
            expression (str): A string contendo a expressão ou equação (ex: 'x + 2 = 5').
            
        Returns:
            Union[Any, List[Any]]: O resultado da expressão ou a lista de soluções da equação.
            
        Raises:
            ValueError: Se a expressão for inválida ou não puder ser resolvida.
        """
        if not expression or not isinstance(expression, str):
            raise ValueError("A expressão deve ser uma string não vazia.")

        try:
            # Normaliza a expressão (remove espaços extras)
            clean_expr = expression.strip()
            
            if "=" in clean_expr:
                # Caso seja uma equação (ex: x + 2 = 5)
                left_side, right_side = clean_expr.split("=")
                
                expr_left = parse_expr(left_side.strip())
                expr_right = parse_expr(right_side.strip())
                
                equation = Eq(expr_left, expr_right)
                solution = solve(equation, self.variable)
                return solution
            else:
                # Caso seja apenas uma expressão (ex: 2 + 3 * 4)
                expr = parse_expr(clean_expr)
                return expr
                
        except Exception as e:
            raise ValueError(f"Erro ao processar a expressão '{expression}': {e}")

def resolver(expressao: str) -> Any:
    """
    Função utilitária para manter compatibilidade com versões anteriores.
    """
    solver = MathSolver()
    return solver.solve(expressao)

if __name__ == "__main__":
    # Testes rápidos
    solver = MathSolver()
    print(f"x + 2 = 5 -> {solver.solve('x + 2 = 5')}")
    print(f"2 + 3 * 4 -> {solver.solve('2 + 3 * 4')}")