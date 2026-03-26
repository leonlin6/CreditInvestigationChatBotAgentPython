import os
import sys
import asyncio

# import chromadb
import uvicorn

from fastapi import FastAPI, APIRouter
from dotenv import load_dotenv

# from src.services.save_document_into_vectordb_service import establish_vector_data
from src.mappings.company_stock_code_array import CompanyStockCodeArray
from fastapi.middleware.cors import CORSMiddleware

# import LangChain lib
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# import type
from src.types.langgraph_state_types import OverallState

# import graph
from src.agent.graph import graph

# import api routers
from src.api.chatbot import chatbot_router


from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# 要開啟python專案時，都要下這個指令開啟該專案的虛擬環境
# python3 -m venv venv
# source venv/bin/activate


# Load environment variables
load_dotenv()

app = FastAPI()
api_router = APIRouter()
api_router.include_router(chatbot_router)
app.include_router(api_router)


# print("✅ OPENAI_MODEL_NAME:", os.getenv("OPENAI_MODEL_NAME"))
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


# 建立 VectorStore
# client = chromadb.HttpClient(host="localhost", port=8000)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# vector_store = Chroma(
#     client=client, collection_name="a-test-collection", embedding_function=embeddings
# )


# Terminal chat mode
async def terminal_chat():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            # graph_answer = graph.invoke({"user_input": user_input})
            graph_answer = graph.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "user_input": user_input,
                },
                config={"configurable": {"thread_id": "1"}},
            )
            # print("graph_answer==========:", graph_answer)

            # print("The answer is :", graph_answer["answer"])
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
