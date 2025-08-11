from langchain_core.tools import tool


@tool
def exact_query(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
