import os
import chromadb
import getpass


from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from src.providers.chat_openAI_provider import chat_model
from src.types.langgraph_state_types import OverallState
from langchain_core.messages import AIMessage


def semantic_retrieval(state: OverallState) -> OverallState:
    print("semantic_retrieval in =======")

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        print("use the get weather tool=====================")
        return f"It's always sunny in {city}!"

    agent = create_react_agent(
        model="gpt-4o", tools=[get_weather], prompt="You are a helpful assistant"
    )

    # Run the agent
    res = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": f"""{state['rephrased_question']}"""}
            ]
        }
    )

    answer = "res.messages[0].content"
    final_ai_msgs = [m for m in res["messages"] if isinstance(m, AIMessage)]
    final_answer = final_ai_msgs[-1].content

    print("res-------", res["messages"])
    print("\n")
    print("final_answer-------", final_answer)

    return {**state, "answer": answer}
