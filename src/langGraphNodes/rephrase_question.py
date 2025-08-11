import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.mappings.company_stock_code_array import CompanyStockCodeArray


def rephrase_question(user_input: str) -> str:
    openAIApiKey = os.getenv("OPENAI_API_KEY")

    custom_prompt = PromptTemplate(
        template=f"""
        你是一個專門將使用者問題重寫成完整查詢的助理，用於文件檢索。
        ### 指示：
        1. 讀取使用者的問題與對話歷史。
        2. 如果問題不完整或依賴前文，請將其改寫為可以單獨理解的完整問題。
        3. 保留問題原本的意圖與語意。
        4. 僅輸出重寫後的問題，不要輸出多餘文字。

        ### 使用者問題：
        {user_input}

        ### 改寫後的問題："""
    )

    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openAIApiKey, temperature=0.6)

    rephrased_question = llm.invoke(custom_prompt)
    print("轉化後的問題:==========", rephrased_question.content)

    return rephrased_question
