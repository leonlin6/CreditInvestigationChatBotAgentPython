from sqlalchemy import Boolean
from typing_extensions import TypedDict, NotRequired
from typing import Any
from langgraph.graph import MessagesState


class OverallState(TypedDict):
    messages: MessagesState
    is_question_in_range: str
    user_input: str
    rephrased_question: str
    question_type: str
    statement_type: str
    reference_data: any
    answer: str
