import os

from src.mappings.company_stock_code_array import CompanyStockCodeArray
from src.types.langgraph_state_types import OverallState
from src.providers.chat_openAI_provider import chat_model


def rephrase_question(state: OverallState) -> OverallState:

    custom_prompt = f"""
        你是一個專門將使用者問題重寫成完整查詢的助理，用於文件檢索。
        ### 指示：
        1. 讀取使用者的問題與對話歷史。
        2. 如果問題不完整或依賴前文，請將其改寫為可以單獨理解的完整問題。
        3. 保留問題原本的意圖與語意。
        4. 僅輸出重寫後的問題，不要輸出多餘文字。

        ### 使用者問題：
        {state['user_input']}

        ### 改寫後的問題："""

    response = chat_model.invoke(custom_prompt)

    # print("rephrased_question response======", response)

    rephrased_question = response.content
    print("rephrased_question======", rephrased_question)

    return {**state, "rephrased_question": rephrased_question}
