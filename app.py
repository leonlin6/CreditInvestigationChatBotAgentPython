import os
import sys
import asyncio
import chromadb

from typing import List, Dict, Optional
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from src.services.save_document_into_vectordb_service import establish_vector_data
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from src.mappings.company_stock_code_array import CompanyStockCodeArray
from langchain.retrievers import RePhraseQueryRetriever
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain.schema.output_parser import StrOutputParser
from src.langGraphNodes.rephrase_question import rephrase_question
from src.langGraphNodes.classify_statement_type import classify_statement_type
from src.langGraphNodes.exact_query import exact_query


# Load environment variables
load_dotenv()

openAIApiKey = os.getenv("OPENAI_API_KEY")
print("openAIApiKey=====", openAIApiKey)
chatModel = ChatOpenAI(model_name="gpt-4o", openai_api_key=openAIApiKey)

app = FastAPI()

client = chromadb.HttpClient(host="localhost", port=8000)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# langGraph Node:將question提供給LLM進行分析，判斷是「語意檢索」or「精確查詢」
def classifyQuestionType(state: MessagesState):
    classifyQuestionTypePrompt = f"""
    ###指示：
        你是一個分類器。請判斷以下問題的類型：
    ###規則：
        如果問題需要查詢特定欄位、數值或主鍵，請回答「精確查詢」。
        如果問題需要語意相似度匹配或比較，請回答「語意檢索」。
        不要解釋過程，只需回答「語意檢索」或「精確查詢」。
    ###問題: ${state['messages']}"""

    res = chatModel.invoke(classifyQuestionTypePrompt)
    type = res.content

    return {**state, "type": type}


# langGraph Node:判斷是「語意檢索」，所以後續丟給Agent繼續執行回覆答案
def semantic_retrieval(state: MessagesState):
    classifyQuestionTypePrompt = f"""
    ###指示：
        你是一個分類器。請判斷以下問題的類型：
    ###規則：
        如果問題需要查詢特定欄位、數值或主鍵，請回答「精確查詢」。
        如果問題需要語意相似度匹配或比較，請回答「語意檢索」。
        不要解釋過程，只需回答「語意檢索」或「精確查詢」。
    ###問題: ${state['messages']}"""

    res = chatModel.invoke(classifyQuestionTypePrompt)
    type = res.content

    return {**state, "type": type}


# 宣告Graph Workflow
workflow = StateGraph(MessagesState)
# 宣告LangGraph Ndoe and Edge
workflow.add_node(classifyQuestionType)
workflow.add_node(rephrase_question)
workflow.add_node(classify_statement_type)
workflow.add_node(exact_query)
workflow.add_node(semantic_retrieval)

workflow.add_edge(START, "classify_statement_type")
workflow.add_edge("classify_statement_type", END)

# workflow.add_edge("classifyQuestionType", "rephrase_question")
# workflow.add_edge("rephrase_question", "classify_statement_type")
# workflow.add_edge("classify_statement_type", "exact_query")
# workflow.add_edge("exact_query", END)

graph = workflow.compile()
# 定義撈資料的DB
db = SQLDatabase.from_uri("sqlite:///FinancialStatements.db")


# 建立 VectorStore
vector_store = Chroma(
    client=client, collection_name="a-test-collection", embedding_function=embeddings
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
    companyCode: str = Field(description="對應該公司之正式股票代碼，例如台積電為 2330")
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


# Terminal chat mode
async def terminal_chat():
    print("AI Chatbot (type 'exit' or 'quit' to leave)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            graph.invoke({"user_input": user_input})
            # Define your desired data structure.
            # 目的是要組出結構化的schema，在filter的時候可以放入指定的schema參數
            prompt = PromptTemplate(
                template="""盡可能回覆問題，並組成指定的格式，無法取得資訊的欄位，填入空字串作為其value。
                    companyName一定要從問題中取出文字代入。
                    問題：{question}\n
                    格式：{format_instructions}""",
            )

            chain = prompt | chatModel | parser
            schema = chain.invoke(
                {
                    "question": user_input,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            print("schema ======", schema)

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

                # 如果找到公司，就更新 sqlschema 並停止迴圈
                if found_company:
                    schema["companyName"] = found_company["companyName"]
                    schema["companyCode"] = found_company["companyCode"]
                    schema["shortName"] = found_company["shortName"]
                    schema["englishName"] = found_company["englishName"]
                    break

            print("找到公司:", found_company)
            print("更新後的 schema:", schema)

            schema_company_name = schema.get("companyName")
            schema_year = schema.get("period", {}).get("year")
            schema_quarter = schema.get("period", {}).get("quarter")

            field = schema.get("requested_fields", [])

            print("更新後的 company_name:", schema_company_name)
            print("更新後的 schema_year:", schema_year)
            print("更新後的 schema_quarter:", schema_quarter)
            print("更新後的 field", field)

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
            print("results:======", results)

            mergeAnswerPrompt = f"""
                你是一個專業的信用徵審團隊助手，並根據'參考答案'給出最接近問題的會計代碼(account title)
                只要回答會計代碼(account title)就好，不用說明太多
                ###問題：{user_input}
                ###參考答案：{results}
            """

            answer = chatModel.invoke(mergeAnswerPrompt)
            print("answer:======", answer.content)

            answerValue = db.run(
                f"SELECT value FROM balance_sheet WHERE company_code={schema['companyCode']} AND year={schema['period']['year']} AND quarter={schema['period']['quarter']} AND account_title_code={answer.content};"
            )
            print("answerData:======", answerValue)
            answerUnit = db.run(
                f"SELECT unit FROM balance_sheet WHERE company_code={schema['companyCode']} AND year={schema['period']['year']} AND quarter={schema['period']['quarter']} AND account_title_code={answer.content};"
            )
            print("answerData:======", answerValue)
            answerData = {"value": answerValue, "unit": answerUnit}
            finalPrompt = f"""
                你是一個專業的信用徵審團隊助手，並根據'參考答案'回答問題
                ###問題：{user_input}
                ###參考答案：{answerData}
            """

            finalAnswer = chatModel.invoke(finalPrompt)
            print("\n")
            print("\n")

            print("finalAnswer:======", finalAnswer.content)

        except Exception as err:
            print("Error:", err, file=sys.stderr)


if __name__ == "__main__":
    # Run terminal chat mode
    asyncio.run(terminal_chat())
    # asyncio.run(establish_vector_data())
