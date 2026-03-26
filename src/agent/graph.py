# import LangGraph lib
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from typing_extensions import TypedDict, NotRequired, Annotated

# import langGraph nodes
from src.agent.nodes.rephrase_question import rephrase_question
from src.agent.nodes.classify_is_question_in_range import (
    classify_is_question_in_range,
)
from src.agent.nodes.classify_statement_type import classify_statement_type
from src.agent.nodes.exact_query import exact_query
from src.agent.nodes.semantic_retrieval import semantic_retrieval
from src.agent.nodes.classify_question_type import classify_question_type
from src.agent.nodes.question_out_of_range import question_out_of_range

# import type
from src.types.langgraph_state_types import OverallState


from langgraph.checkpoint.memory import MemorySaver


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
        # print("is_question_in_range_edge in========", state["is_question_in_range"])
        match state["is_question_in_range"]:
            case "True":
                return "classify_question_type"
            case "False":
                return "question_out_of_range"
            case _:
                return "END"
    except (ValueError, TypeError) as e:
        print(f"發生錯誤: {e}")


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

# Add simple in-memory checkpointer
# memory = MemorySaver()
# graph = workflow.compile(checkpointer=memory)
graph = workflow.compile()
