import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.mappings.company_stock_code_array import CompanyStockCodeArray
from typing_extensions import TypedDict


class InputState(TypedDict):
    user_input: str


class OverallState(TypedDict):
    user_input: str
    statement_type: str
    answer: str


def classify_statement_type(state: InputState) -> OverallState:
    print("classify_statement_type in =======")
    openAIApiKey = os.getenv("OPENAI_API_KEY")
    chatModel = ChatOpenAI(model_name="gpt-4o", openai_api_key=openAIApiKey)

    question_with_system_prompt = f"""
        使用者會問你一些與財報相關的問題，請根據「使用者問題中提及的關鍵項目」判斷該項目最常出現在哪一種財務報表中。請從以下3種類別中選擇，可能複選。

        若問題與財務、信用徵審沒有相關聯，請回覆：
        「此問題超出我可回答的範圍，請洽詢專業人士。」

        僅輸出類別名稱（可多選），用逗號隔開，不要補充說明。

        ### 報表種類（請依照會計實務為準）：
        1. 資產負債表：資產、負債、權益的期末狀況。例如「現金」「應收帳款」「預付款項」。
        2. 綜合損益表：本期的收入、成本與費用。例如「營業收入」「稅後淨利」「手續費收入」「股利收入」。
        3. 現金流量表：現金流入與流出，如「營業活動之現金流入」「投資活動」「收取之股利」。


        ### 問題：
        ${state.user_input}"""

    # 4. 權益變動表：如「資本公積」「特別盈餘公積」「股利分派」「保留盈餘調整」。
    # 5. 會計師查核報告：會計師查核或核閱意見相關的報告，不屬於一般財務報表內容。

    response = chatModel.invoke(question_with_system_prompt)
    print("response====", response.content)

    return {**state, "statement_type": response.content}
    # if (response.content === "此問題超出我可回答的範圍，請洽詢專業人士。") {
    #     return {
    #     ...state,
    #     category: "此問題超出我可回答的範圍，請洽詢專業人士。",
    #     answer: "此問題超出我可回答的範圍，請洽詢專業人士。",
    #     isQuestionOutOfRange: true,
    #     };
    # } else {
    #     return {
    #     ...state,
    #     category: response.content,
    #     };
    # }
