import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing_extensions import TypedDict
from src.types.langgraph_state_types import OverallState
from src.providers.chat_openAI_provider import chat_model


class InputState(TypedDict):
    user_input: str


# langGraph Node:將question提供給LLM進行分析，判斷是否在「財務報表相關問題」範圍內
def classify_is_question_in_range(state: OverallState):

    classifyQuestionTypePrompt = f"""
        ###指示：
            你是一個分類器。判斷使用者訊息是否可以從「財務報表」中找到答案。
        ###規則：
            
            回覆直接全部回True
        ###問題: ${state['rephrased_question']}"""

    res = chat_model.invoke(classifyQuestionTypePrompt)
    is_question_in_range = res.content

    print("is_question_in_range======", is_question_in_range)
    return {**state, "is_question_in_range": is_question_in_range}
