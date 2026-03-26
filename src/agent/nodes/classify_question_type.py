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
    # print("classify_question_type in=======")
    classifyQuestionTypePrompt = f"""
        You are an intent classifier for a financial statement AI system.

        Your task is to classify the user question.

        Classification rules:

        EXACT_QUERY
        - ask for specific financial value
        - contains company name or stock code
        - contains year or quarter
        - contains exact financial item
        - answer should be a number or single field

        SEMANTIC_SEARCH
        - ask for analysis or judgement
        - ask about stability, risk, performance
        - ask about whether company is good or bad
        - requires multiple financial data
        - requires explanation

        Examples:

        Q: 請給我台泥2024年Q1的收取之股利
        A: {{"query_type":"EXACT_QUERY"}}

        Q: 請給我台泥2024年Q1的投資活動之淨現金流入（流出）
        A: {{"query_type":"EXACT_QUERY"}}

        Q: 1503這家公司2024年現金水位是否充足
        A: {{"query_type":"SEMANTIC_SEARCH"}}

        Q: 1503這家公司2024年負債結構是否有風險
        A: {{"query_type":"SEMANTIC_SEARCH"}}

        Return JSON only.

        User question:
         ${state['rephrased_question']}。
    ###問題: ${state['rephrased_question']}"""

    res = chat_model.invoke(classifyQuestionTypePrompt)
    question_type = res.content

    print("classify_question_type over-----", question_type)

    return {**state, "question_type": question_type}
