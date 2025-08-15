import os
import sys
import asyncio
import chromadb

from typing import List, Dict, Optional
from fastapi import FastAPI
from dotenv import load_dotenv
from src.services.save_document_into_vectordb_service import establish_vector_data
from pydantic import BaseModel, Field
from src.mappings.company_stock_code_array import CompanyStockCodeArray

# import LangChain lib
from langchain.retrievers import RePhraseQueryRetriever
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

# import LangGraph lib
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from typing_extensions import TypedDict, NotRequired, Annotated

# import langGraph nodes
from src.langGraphNodes.rephrase_question import rephrase_question
from src.langGraphNodes.classify_is_question_in_range import (
    classify_is_question_in_range,
)
from src.langGraphNodes.classify_statement_type import classify_statement_type
from src.langGraphNodes.exact_query import exact_query
from src.langGraphNodes.semantic_retrieval import semantic_retrieval
from src.langGraphNodes.classify_question_type import classify_question_type

# import type
from src.types.langgraph_state_types import OverallState

# Load environment variables
load_dotenv()

app = FastAPI()

client = chromadb.HttpClient(host="localhost", port=8000)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def question_type_condition_edge(state: OverallState) -> str:
    match state["question_type"]:
        case "語意檢索":
            return "semantic_retrieval"
        case "精確查詢":
            return "classify_statement_type"
        case _:
            return "semantic_retrieval"


# 若問題超出範圍，則回END
# 若沒超出範圍，則進入下一個Node：classify_question_type
def is_question_in_range_edge(state: OverallState) -> str:
    try:
        print("is_question_in_range_edge in========", state["is_question_in_range"])
        match state["is_question_in_range"]:
            case "True":
                return "classify_question_type"
            case "False":
                return "question_out_of_range"
            case _:
                return "END"
    except (ValueError, TypeError) as e:
        print(f"發生錯誤: {e}")


def question_out_of_range(state: OverallState) -> OverallState:
    return {
        **state,
        "answer": "您的問題已超出我可回覆的範圍(財務報表相關資訊)，請重新提問。",
    }


# 宣告Graph Workflow
workflow = StateGraph(OverallState)
# 宣告LangGraph Ndoe
workflow.add_node(rephrase_question)
workflow.add_node(classify_is_question_in_range)
workflow.add_node(classify_question_type)
workflow.add_node(classify_statement_type)
workflow.add_node(exact_query)
workflow.add_node(semantic_retrieval)
workflow.add_node(question_out_of_range)

# 宣告LangGraph Edge
workflow.add_edge(START, "rephrase_question")
workflow.add_edge("rephrase_question", "classify_is_question_in_range")
workflow.add_conditional_edges(
    source="classify_is_question_in_range",  # 判定問題是否涵蓋在「財務報表」類型的問題
    path=is_question_in_range_edge,
    path_map={  # 路徑映射
        "classify_question_type": "classify_question_type",
        "question_out_of_range": "question_out_of_range",
    },
)
workflow.add_edge("question_out_of_range", END)

workflow.add_conditional_edges(
    source="classify_question_type",  # 判定問題是「語意檢索」or「精確查詢」
    path=question_type_condition_edge,  # 決定要走哪個路的函式
    path_map={  # 路徑映射
        "semantic_retrieval": "semantic_retrieval",
        "classify_statement_type": "classify_statement_type",
    },
)
workflow.add_edge("classify_statement_type", "exact_query")

workflow.add_edge("exact_query", END)
workflow.add_edge("semantic_retrieval", END)

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
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            graph_answer = graph.invoke({"user_input": user_input})
            print("The answer is :", graph_answer["answer"])
        except Exception as err:
            print("Error:", err, file=sys.stderr)


if __name__ == "__main__":
    # Run terminal chat mode
    asyncio.run(terminal_chat())
    # asyncio.run(establish_vector_data())
