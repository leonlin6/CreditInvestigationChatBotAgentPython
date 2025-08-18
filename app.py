import os
import sys
import asyncio
import chromadb
import uvicorn

from typing import List, Dict, Optional
from fastapi import FastAPI, APIRouter
from dotenv import load_dotenv
from src.services.save_document_into_vectordb_service import establish_vector_data
from pydantic import BaseModel, Field
from src.mappings.company_stock_code_array import CompanyStockCodeArray
from fastapi.middleware.cors import CORSMiddleware

# import LangChain lib
from langchain.retrievers import RePhraseQueryRetriever
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

# import type
from src.types.langgraph_state_types import OverallState

# import graph
from src.agent.graph import graph

from src.api.chatbot import chatbot_router


# Load environment variables
load_dotenv()

app = FastAPI()
api_router = APIRouter()  # 以api_router作為APIRouter實例，本次重點!
api_router.include_router(chatbot_router)  # 把router1檔案裡的路由結合進api_router

app.include_router(api_router)  # app實例將api_router的路由結合進去


origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",  # Next.js
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # 需要帶 cookie/認證時要開
    allow_methods=["*"],
    allow_headers=["*"],
)

client = chromadb.HttpClient(host="localhost", port=8000)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# 定義撈資料的DB
db = SQLDatabase.from_uri("sqlite:///FinancialStatements.db")

# 建立 VectorStore
# vector_store = Chroma(
#     client=client, collection_name="a-test-collection", embedding_function=embeddings
# )


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

    # 建立API SERVER
    uvicorn.run(app, host="localhost", port=3001)

    # 測試用：建立terminal ai chat bot
    # asyncio.run(terminal_chat())

    # 建立語意化的account title code到vector database
    # asyncio.run(establish_vector_data())
