# ---------- IMPORT ----------
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Import tools
from tools.financial_rag_tool import financial_rag_tool
from tools.budget_calculator_tool import calculate_budget_metrics
from tools.web_search_tool import web_search
from tools.Python_REPL_Tool import python_repl_tool
# ---------- IMPORT ----------

from schema.state import UserFinanceState

# Initialize Model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define tools list and bind to model
tools = [financial_rag_tool, calculate_budget_metrics, web_search, python_repl_tool]
model_with_tools = model.bind_tools(tools)

# --- Node Implementation ---


def detect_goal_node(state: UserFinanceState):
    """Detects if the user's message contains a financial goal."""

    last_message = state["messages"][-1]
    
    # Cheap keyword check to avoid unnecessary LLM calls
    goal_keywords = ["save", "target", "goal", "want to", "plan to", "budget", "spend less", "reduce", "invest"]
    message_content = last_message.content.lower()
    
    if not any(keyword in message_content for keyword in goal_keywords):
        return {}

    prompt = f"""Analyze the following user message and extract any specific financial goals.
    Examples of goals: "save $5000", "reduce coffee spending", "invest in stocks".
    If no new goals are found, return "None".
    If multiple goals are found, separate them with semicolons.
    
    User message: {last_message.content}
    """
    
    response = model.invoke([SystemMessage(content="You are a goal detection assistant."), HumanMessage(content=prompt)])
    
    new_goals = []
    if response.content.strip().lower() != "none":
        new_goals = [g.strip() for g in response.content.split(";") if g.strip()]
    
    if not new_goals:
        return {}

    existing_goals = state.get("user_goals", [])
    if existing_goals is None: # Safety check if the state value is None
        existing_goals = []
        
    return {"user_goals": existing_goals + new_goals}


def agent_node(state: UserFinanceState):
    """ReAct agent: reasons, calls tools, generates response."""
    messages = state["messages"]
    
    # Master System Identity
    identity = (
        "You are a Personal Finance AI Assistant. Your goal is to help users manage their money, "
        "track spending, and achieve financial goals. Use the provided tools to fetch transaction data, "
        "calculate metrics, or search the web for market info. Be professional, data-driven, and concise."
    )
    
    # Construct context from goals and summary
    goals = state.get("user_goals", [])
    summary = state.get("summary", "")
    
    context_parts = [identity]
    if goals:
        context_parts.append(f"CURRENT USER GOALS: {', '.join(goals)}")
    if summary:
        context_parts.append(f"CONVERSATION SUMMARY: {summary}")
        
    system_msg = "\n\n".join(context_parts)
    
    # Prepend the system identity and context to the message list
    messages = [SystemMessage(content=system_msg)] + messages
        
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def summarize_node(state: UserFinanceState):
    """Summarizes older messages if the history is too long."""
    messages = state["messages"]
    
    # Only summarize if we have more than 6 messages
    if len(messages) <= 6:
        return {}
    
    prompt = "Summarize the following conversation history into a concise summary, retaining key financial details and goals discussed."
    summary_response = model.invoke([SystemMessage(content=prompt)] + messages[:-2])
    
    return {"summary": summary_response.content}

# --- Graph Construction ---

def create_graph(checkpointer):
    workflow = StateGraph(UserFinanceState)

    # Add Nodes
    workflow.add_node("detect_goal", detect_goal_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("summarize", summarize_node)

    # Build Edges
    workflow.add_edge(START, "detect_goal")
    workflow.add_edge("detect_goal", "agent")

    # Conditional edge for tool calling
    def should_continue(state: UserFinanceState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "summarize"

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "summarize": "summarize"
        }
    )

    workflow.add_edge("tools", "agent")
    workflow.add_edge("summarize", END)

    # Compile the graph
    return workflow.compile(checkpointer=checkpointer)

# Note: Use with memory.checkpoint.get_sqlite_saver() context manager
# Example:
# with get_sqlite_saver() as saver:
#     app = create_graph(saver)
#     app.invoke(...)

