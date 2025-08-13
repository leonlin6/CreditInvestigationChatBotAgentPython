import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from src.mappings.company_stock_code_array import CompanyStockCodeArray
from typing_extensions import TypedDict, NotRequired, Annotated
from src.types.langgraph_state_types import OverallState
from src.providers.chat_openAI_provider import chat_model


# langGraph Node:將question提供給LLM進行分析，判斷是
# 將question提供給LLM進行分析，判斷此問題是不是在財務報表的問題範圍內
# 財務報表類別：資產負債表、綜合損益表、現金流量表、權益變動表、會計師查核報告。
# Todo：判斷式不要直接用reponse.content的中文來作比對，要增加一個判斷對錯的類型比較好，事先定義好錯誤的類別，用英文
def classify_statement_type(state: OverallState) -> OverallState:

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
        ${state['rephrased_question']}"""

    # 4. 權益變動表：如「資本公積」「特別盈餘公積」「股利分派」「保留盈餘調整」。
    # 5. 會計師查核報告：會計師查核或核閱意見相關的報告，不屬於一般財務報表內容。

    response = chat_model.invoke(question_with_system_prompt)
    statement_type = response.content

    return {**state, "statement_type": statement_type}
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
