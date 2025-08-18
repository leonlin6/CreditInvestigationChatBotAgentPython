import os
import chromadb

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.mappings.company_stock_code_array import CompanyStockCodeArray
from src.types.langgraph_state_types import OverallState
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain_community.utilities import SQLDatabase
from langchain_chroma import Chroma
from src.providers.chat_openAI_provider import chat_model


def exact_query(state: OverallState) -> OverallState:
    # 定義撈資料的DB
    db = SQLDatabase.from_uri("sqlite:///FinancialStatements.db")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    client = chromadb.HttpClient(host="localhost", port=8000)

    # 建立 VectorStore
    vector_store = Chroma(
        client=client,
        collection_name="a-test-collection",
        embedding_function=embeddings,
    )

    class Period(BaseModel):
        year: int = Field(..., description="使用者提問中提到的年度")
        quarter: int = Field(
            ..., description="使用者提問中提到的季度，例如'Q2'，只紀錄數字"
        )
        range: Optional[str] = Field(
            None, description="季度期間的文字表示，例如 '2024年Q1到Q3'"
        )

    class RequestedField(BaseModel):
        field: str = Field(
            ..., description="使用者請求的會計項目名稱，例如 '現金及約當現金'"
        )
        category: Optional[str] = Field(
            None,
            description="""該項目所屬的報表類別，根據以下3個項目取1個
            1. 資產負債表：資產、負債、權益的期末狀況。例如「現金」「應收帳款」「預付款項」。
            2. 綜合損益表：本期的收入、成本與費用。例如「營業收入」「稅後淨利」「手續費收入」「股利收入」。
            3. 現金流量表：例如「營業活動現金流量」「投資活動現金流量」。
            """,
        )

    class QuestionSchema(BaseModel):
        companyName: str = Field(description="使用者提問中提到的公司全名")
        companyCode: str = Field(
            description="對應該公司之正式股票代碼，例如台積電為 2330"
        )
        shortName: str = Field(description="公司常用簡稱，如台積電、亞泥")
        englishName: str = Field(description="該公司對應的英文名稱簡寫，如 TSMC")
        period: Period
        requested_fields: List[RequestedField]

    # 初始化兩個字典，用來存公司代碼和公司名稱對應的資料
    code_to_company_map = {}
    name_to_company_map = {}

    # 建立查找字典
    for item in CompanyStockCodeArray:
        # 用 companyCode 取得公司資料
        code_to_company_map[item["companyCode"]] = item

        # 用不同名稱取得公司資料
        name_to_company_map[item["companyName"]] = item
        name_to_company_map[item["shortName"]] = item
        name_to_company_map[item["englishName"]] = item

    # 根據 companyCode 查找公司
    def get_company_by_code(code: str):
        return code_to_company_map.get(code, None)

    # 根據各種公司名稱查找公司
    def get_company_by_name(name: str):
        return name_to_company_map.get(name, None)

    parser = JsonOutputParser(pydantic_object=QuestionSchema)
    # Define your desired data structure.
    # 目的是要組出結構化的schema，在filter的時候可以放入指定的schema參數
    prompt = PromptTemplate(
        template="""盡可能回覆問題，並組成指定的格式，無法取得資訊的欄位，填入空字串作為其value。
            companyName一定要從問題中取出文字代入。
            問題：{question}\n
            格式：{format_instructions}""",
    )

    chain = prompt | chat_model | parser
    schema = chain.invoke(
        {
            "question": state["user_input"],
            "format_instructions": parser.get_format_instructions(),
        }
    )
    print("\nschemae ========\n", schema)

    # 判斷 companyCode、companyName、shortName、englishName 是否有值
    company_identifiers = [
        schema["companyCode"],
        schema["companyName"],
        schema["shortName"],
        schema["englishName"],
    ]

    found_company = None

    for index, identifier in enumerate(company_identifiers):
        if identifier:
            if index == 0:
                found_company = get_company_by_code(identifier)
            else:
                # 先用完整名稱或簡稱比對
                found_company = get_company_by_name(identifier)

            # 如果沒有找到，改用包含關鍵字模糊比對
            if found_company is None:
                matches = [
                    item
                    for item in CompanyStockCodeArray
                    if any(
                        isinstance(value, str) and identifier in value
                        for value in item.values()
                    )
                ]
                found_company = matches[0] if matches else None

        print("\nbefore  if found_company:======\n", found_company)

        # 如果找到公司，就更新 sqlschema 並停止迴圈
        if found_company:
            schema["companyName"] = found_company["companyName"]
            schema["companyCode"] = found_company["companyCode"]
            schema["shortName"] = found_company["shortName"]
            schema["englishName"] = found_company["englishName"]
            break

        schema_company_name = schema.get("companyName")
        schema_year = schema.get("period", {}).get("year")
        schema_quarter = schema.get("period", {}).get("quarter")

    field = schema.get("requested_fields", [])
    print("=====field", field)
    results = vector_store.similarity_search_with_score(
        query=f"請問{field}的代碼是多少？",
        k=5,
        # filter={
        #     "$and": [
        #         {"companyName": {"$eq": schema_company_name}},
        #         {"year": {"$eq": schema_year}},
        #         {"quarter": {"$in": [schema_quarter]}},
        #     ]
        # },
    )

    get_account_title_prompt = f"""
        你是一個專業的信用徵審團隊助手，並根據'參考答案'給出最接近問題的會計代碼(account title code)
        只要回答會計代碼(account title code)就好，不用說明太多
        ###問題：{state['user_input']}
        ###參考答案：{results}
    """

    response_get_account_title = chat_model.invoke(get_account_title_prompt)
    account_title_code = response_get_account_title.content
    print("\n schema====", schema)
    print("\n state====", state)
    print("\n account_title_code====", account_title_code)
    answerValue = db.run(
        f"SELECT value FROM {state['statement_type']} WHERE company_code='{schema['companyCode']}' AND year={schema['period']['year']} AND quarter={schema['period']['quarter']} AND account_title_code='{account_title_code}';"
    )
    answerUnit = db.run(
        f"SELECT unit FROM {state['statement_type']} WHERE company_code='{schema['companyCode']}' AND year={schema['period']['year']} AND quarter={schema['period']['quarter']} AND account_title_code='{account_title_code}';"
    )
    answerData = {"value": answerValue, "unit": answerUnit}
    print("\n Answer Data-----------", answerData)

    finalPrompt = f"""
        你是一個專業的信用徵審團隊助手，並根據'財務報表資料'回答問題
        若答案為數字，則根據千位加入標點符號，並不要更改其正負號，並且答案要加入貨幣單位
        ###貨幣單位：新台幣仟元
        ###問題：{state['user_input']}
        ###財務報表資料：{answerData}
    """

    res = chat_model.invoke(finalPrompt)
    final_answer = res.content

    return {**state, "answer": final_answer}
