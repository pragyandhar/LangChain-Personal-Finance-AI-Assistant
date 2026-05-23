import sqlite3
import os
from contextlib import contextmanager
from langgraph.checkpoint.sqlite import SqliteSaver

@contextmanager
def get_sqlite_saver():
    """
    Creates a SQLite connection and yields a SqliteSaver instance.
    Ensures the connection is closed after use.
    """
    # Define the database path
    db_path = os.path.join("data", "checkpoints.db")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create the SQLite connection
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()
