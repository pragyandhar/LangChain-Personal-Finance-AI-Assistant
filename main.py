import os
from dotenv import load_dotenv

# Load API keys at the very beginning
load_dotenv()

from rag.loader import TransactionLoader
from memory.checkpoint import get_sqlite_saver
from memory.graph import create_graph
from langchain_core.messages import HumanMessage

def sync_data():
    """Syncs CSV transactions to the local SQLite database."""
    csv_file = "data/transactions.csv"
    db_path = "data/transactions.db"
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. RAG and Budget tools may have limited data.")
        return
        
    loader = TransactionLoader(db_path=db_path)
    print(f"Syncing transactions from {csv_file}...")
    transactions = loader.load_from_csv(csv_file)
    loader.sync_to_db(transactions)
    print("Sync complete.")

def start_chat():
    """Starts the ReAct Agent chat loop."""
    print("\n--- Personal Finance AI Assistant ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    # Use the context manager to ensure the checkpointer connection is closed properly
    with get_sqlite_saver() as saver:
        # Compile the graph with the checkpointer
        app = create_graph(saver)
        
        # We use a fixed thread_id for this session to maintain memory
        config = {"configurable": {"thread_id": "user_123"}}
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            # Invoke the graph
            # We pass the input as a list of messages
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            # The last message in the state is the agent's response
            assistant_response = result["messages"][-1].content
            print(f"Assistant: {assistant_response}\n")

if __name__ == "__main__":
    # 1. Sync data first
    sync_data()
    
    # 2. Start the AI Assistant
    start_chat()
