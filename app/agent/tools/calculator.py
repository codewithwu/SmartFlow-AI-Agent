import ast
import operator

from langchain_core.tools import tool

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST math expression safely."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))
    if isinstance(node, ast.BinOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return op_func(left, right)
    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, **, %, //.

    Args:
        expression: A mathematical expression to evaluate, e.g. "(123 + 456) * 2"
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        if result == int(result):
            result = int(result)
        return f"计算结果: {expression} = {result}"
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
        return f"计算错误: {e}"
