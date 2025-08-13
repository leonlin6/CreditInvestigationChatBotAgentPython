import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing_extensions import TypedDict
from src.types.langgraph_state_types import OverallState
from src.providers.chat_openAI_provider import chat_model


class InputState(TypedDict):
    user_input: str


# langGraph Node:將question提供給LLM進行分析，判斷是「語意檢索」or「精確查詢」
def classify_is_question_in_range(state: OverallState):
    print("classify_is_question_in range in2========")

    classifyQuestionTypePrompt = f"""
    ###指示：
        你是一個分類器。判斷使用者訊息是否屬於「財務報表相關問題」。
    ###規則：
        定義：與資產負債表、綜合損益表、現金流量表、財報附註之項目/金額/期間/比較/比率等直接相關者為「是」；
        與股價、新聞、公司介紹、客服、產品規格等議題無關的問題為「否」。
        回覆直接回True或False，不要補充任何說明。
    ###問題: ${state['rephrased_question']}"""

    res = chat_model.invoke(classifyQuestionTypePrompt)
    is_question_in_range = res.content

    return {**state, "is_question_in_range": is_question_in_range}
