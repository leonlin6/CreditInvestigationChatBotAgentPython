from langchain_openai import ChatOpenAI
import os

openAI_api_key = os.getenv("OPENAI_API_KEY")
openAI_model_name = os.getenv("OPENAI_MODEL_NAME")
chat_model = ChatOpenAI(
    model_name=openAI_model_name, openai_api_key=openAI_api_key, temperature=0
)
