from langgraph.prebuilt import create_react_agent
from src.types.langgraph_state_types import OverallState
from langchain_core.messages import AIMessage
from langchain_core.tools import tool


def question_out_of_range(state: OverallState) -> OverallState:
    return {
        **state,
        "answer": "您的問題已超出我可回覆的範圍(財務報表相關資訊)，請重新提問。",
    }
