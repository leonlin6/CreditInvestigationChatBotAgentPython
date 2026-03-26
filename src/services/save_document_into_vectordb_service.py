import uuid
import os
import chromadb
import json
from typing import List, Dict

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_community.document_loaders import BSHTMLLoader


load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
client = chromadb.HttpClient(host="localhost", port=8000)

# 建立 VectorStore
vector_store = Chroma(
    client=client, collection_name="a-test-collection", embedding_function=embeddings
)
# vector_store = Chroma(
#     embedding_function=embeddings,
#     collection_name="a-test-collection",
#     host="localhost",
#     port=8000,
# )


# 給HTML用的
def get_pdf_files() -> List[Dict]:
    # List of PDF files and their metadata
    return [
        {
            "title": "Q1",
            "path": "src/services/tifrs-fr1-m1-ci-cr-1101-2024Q1.html",
            "companyName": "臺灣水泥股份有限公司",
            "shortName": "台泥",
            "companyCode": "1101",
            "year": 2024,
            "quarter": "Q1",
        },
        # {
        #     "title": "Q2",
        #     "path": "src/services/tifrs-fr1-m1-ci-cr-1101-2024Q2.html",
        #     "companyName": "臺灣水泥股份有限公司",
        #     "shortName": "台泥",
        #     "companyCode": "1101",
        #     "year": 2024,
        #     "quarter": "Q2",
        # },
        # {
        #     "title": "Q3",
        #     "path": "src/services/tifrs-fr1-m1-ci-cr-1101-2024Q3.html",
        #     "companyName": "臺灣水泥股份有限公司",
        #     "shortName": "台泥",
        #     "companyCode": "1101",
        #     "year": 2024,
        #     "quarter": "Q3",
        # },
        # {
        #     "title": "Q4",
        #     "path": "src/services/tifrs-fr1-m1-ci-cr-1101-2024Q4.html",
        #     "companyName": "臺灣水泥股份有限公司",
        #     "shortName": "台泥",
        #     "companyCode": "1101",
        #     "year": 2024,
        #     "quarter": "Q4",
        # },
    ]


# 給PDF用的
# def get_pdf_files() -> List[Dict]:
#     # List of PDF files and their metadata
#     return [
#         {
#             "title": "Q1",
#             "path": "src/services/1101Q1-Financial-report.pdf",
#             "companyName": "臺灣水泥股份有限公司",
#             "shortName": "台泥",
#             "companyCode": "1101",
#             "year": 2024,
#             "quarter": "Q1",
#         },
#         {
#             "title": "Q2",
#             "path": "src/services/1101Q2-Financial-report.pdf",
#             "companyName": "臺灣水泥股份有限公司",
#             "shortName": "台泥",
#             "companyCode": "1101",
#             "year": 2024,
#             "quarter": "Q2",
#         },
#         {
#             "title": "Q3",
#             "path": "src/services/1101Q3-Financial-report.pdf",
#             "companyName": "臺灣水泥股份有限公司",
#             "shortName": "台泥",
#             "companyCode": "1101",
#             "year": 2024,
#             "quarter": "Q3",
#         },
#         {
#             "title": "Q4",
#             "path": "src/services/1101Q4-Financial-report.pdf",
#             "companyName": "臺灣水泥股份有限公司",
#             "shortName": "台泥",
#             "companyCode": "1101",
#             "year": 2024,
#             "quarter": "Q4",
#         },
#     ]


# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Create vector store
# vector_store = Chroma(
#     embedding_function=embeddings,
#     collection_name="a-test-collection",
#     url="http://localhost:8000",
#     collection_metadata={
#         "hnsw:space": "cosine",
#     },
# )


# 將含有metadata的向量資料，spliitter後，存到vector db中
async def establish_vector_data():
    vectorStrArray = []
    # pdf_files = get_pdf_files()

    # 讀取本地 JSON 檔案（假設檔名為 data.json）
    with open(
        "/Users/leonlin/Cursor/BackEnd/CreditInvestigationChatBotAgentPython/src/services/parsedData.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
        vectorStrArray = [
            Document(
                f"現金流量表中，會計項目「{item['zh_tw']}」對應的 XBRL Concept 為「{item['concept_name']}」，會計代碼為{item['code']}"
            )
            for item in data
        ]

    vector_store.add_documents(vectorStrArray)

    # for pdf in pdf_files:
    #     # Load PDF document
    #     print("before PyPDFLoader======")
    #     loader = UnstructuredHTMLLoader(pdf["path"])
    #     print("before PyPDFLoader======")

    #     docs = loader.load()
    #     print("before RecursiveCharacterTextSplitter======", docs)

    #     # Create text splitter
    #     splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=300,
    #         chunk_overlap=10,
    #     )
    #     # print("\n")
    #     # print("\n")
    #     # print("\n")
    #     # print("\n")

    #     print("before MemoryVectorStore======", splitter)

    #     pages = []
    #     async for page in loader.alazy_load():
    #         pages.append(page)

    #     print("pages======", pages)

    #     documents_langchain = []

    #     for doc in pages:
    #         print("doc======", doc)
    #         documents_langchain.append(doc.page_content)

    #     # # Split documents
    #     output = splitter.create_documents(documents_langchain)
    #     print("output======", output)

    #     # # Add metadata to chunks
    #     chunks_with_metadata = []
    #     for chunk in output:
    #         chunk_dict = {
    #             "page_content": chunk.page_content,
    #             "metadata": {
    #                 **chunk.metadata,  # 若 splitter 本身有產 metadata
    #                 **pdf,  # 加上你要自己補充的 metadata
    #             },
    #         }
    #         chunks_with_metadata.append(chunk_dict)

    #     print("before MemoryVectorStore======", chunks_with_metadata)

    #     # # Create document array with UUIDs
    #     document_array = []
    #     for item in chunks_with_metadata:
    #         document = Document(
    #             page_content=item["page_content"], metadata=item["metadata"]
    #         )
    #         document_array.append(document)

    #     # Add documents to vector store
    #     print("before vector_store.add_documents======")

    #     print("切割後 chunk 數量：", len(document_array))

    #     vector_store.add_documents(document_array)
