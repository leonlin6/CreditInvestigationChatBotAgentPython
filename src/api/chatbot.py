from fastapi import APIRouter

# import graph
from src.agent.graph import graph

chatbot_router = APIRouter(tags=["chatbot"])


@chatbot_router.get("/chatbot/{user_input}")
async def get_chatbot_answer(user_input: str):
    graph_answer = graph.invoke({"user_input": user_input})

    return {"answer": graph_answer["answer"]}
