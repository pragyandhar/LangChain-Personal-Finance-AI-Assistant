import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from memory.checkpoint import get_sqlite_saver
from memory.graph import create_graph
from rag.loader import TransactionLoader

# Page configuration
st.set_page_config(page_title="Personal Finance AI Assistant", page_icon="💰", layout="wide")

# Load environment variables
load_dotenv()

# --- Initialize State & Data ---

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_user_1"

def sync_data():
    csv_file = "data/transactions.csv"
    db_path = "data/transactions.db"
    if os.path.exists(csv_file):
        loader = TransactionLoader(db_path=db_path)
        transactions = loader.load_from_csv(csv_file)
        loader.sync_to_db(transactions)
        return True
    return False

# Sidebar
with st.sidebar:
    st.title("Settings & Status")
    if st.button("🔄 Sync Transactions"):
        if sync_data():
            st.success("Data synced successfully!")
        else:
            st.error("transactions.csv not found.")
    
    st.divider()
    st.subheader("Conversation Context")
    context_container = st.empty()

# --- Chat Interface ---

st.title("💰 Personal Finance AI Assistant")
st.markdown("Ask me about your spending, set financial goals, or generate a budget plan.")

# Display chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with LangGraph
    with st.spinner("Thinking..."):
        with get_sqlite_saver() as saver:
            app = create_graph(saver)
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Run the graph
            result = app.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
            
            # Extract result
            last_msg = result["messages"][-1]
            assistant_response = last_msg.content
            
            # Update sidebar context
            goals = result.get("user_goals", [])
            summary = result.get("summary", "")
            with context_container.container():
                if goals:
                    st.write("**Goals:**")
                    for goal in goals:
                        st.write(f"- {goal}")
                if summary:
                    st.write("**Recent Summary:**")
                    st.info(summary)

            # Display assistant response
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
                
                # If a budget plan was generated, show it nicely
                if "budget_plan" in result and result["budget_plan"]:
                    plan = result["budget_plan"]
                    with st.expander("📊 View Structured Budget Plan", expanded=True):
                        st.subheader(f"Plan for {plan.month}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Income", f"${plan.total_income:,.2f}")
                        col2.metric("Fixed Costs", f"${plan.fixed_costs:,.2f}")
                        col3.metric("Savings Target", f"${plan.savings_target:,.2f}")
                        
                        st.write("### Variable Budgets")
                        for cat in plan.variable_budgets:
                            st.write(f"- **{cat.category}**: ${cat.budgeted_amount:,.2f}")
                            
                        st.write("### Recommendations")
                        for rec in plan.recommendations:
                            st.info(rec)
