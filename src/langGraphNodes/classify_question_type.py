import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing_extensions import TypedDict
from src.types.langgraph_state_types import OverallState
from src.providers.chat_openAI_provider import chat_model


class InputState(TypedDict):
    user_input: str


# langGraph Node:將question提供給LLM進行分析，判斷是「語意檢索」or「精確查詢」
# Todo：改為structured output，比較不會造成誤判
def classify_question_type(state: OverallState):
    print("classify_question_type in=======")
    classifyQuestionTypePrompt = f"""
    ###指示：
        你是一個分類器。請判斷以下問題的類型：
    ###規則：
        如果問題需要查詢特定欄位、數值或主鍵，請回答「精確查詢」。
        如果問題需要語意相似度匹配或比較，請回答「語意檢索」。
        不要解釋過程，只需回答「語意檢索」或「精確查詢」。
    ###問題: ${state['rephrased_question']}"""

    res = chat_model.invoke(classifyQuestionTypePrompt)
    question_type = res.content

    print("classify_question_type over-----", question_type)

    return {**state, "question_type": question_type}
