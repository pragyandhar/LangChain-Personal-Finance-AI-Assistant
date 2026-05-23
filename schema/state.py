# ---------- IMPORT ----------
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages
from schema.budget import MonthlyBudgetPlan
# ---------- IMPORT ----------

class UserFinanceState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history
    summary: str                             # running summary of older messages
    user_goals: List[str]                    # list of user defined goals
    budget_plan: Optional[MonthlyBudgetPlan] # structured budget plan