# ---------- IMPORT ----------
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
# ---------- IMPORT ----------

class UserFinanceState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history
    summary: str                             # running summary of older messages
    user_goals: list[str]                    # list of user defined goals