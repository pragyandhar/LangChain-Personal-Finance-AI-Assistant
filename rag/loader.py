import os
import sqlite3
import pandas as pd
from typing import List
from langchain_community.document_loaders import CSVLoader
from schema.transactions import Transaction

class TransactionLoader:
    def __init__(self, db_path: str = "data/transactions.db"):
        self.db_path = db_path
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def load_from_csv(self, file_path: str) -> List[Transaction]:
        """
        Loads transactions from a CSV file, validates them using Pydantic,
        and returns a list of Transaction objects.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        if not file_path.endswith(".csv"):
            raise ValueError("Only CSV files are allowed")

        # Load CSV using LangChain
        loader = CSVLoader(file_path=file_path)
        docs = loader.load()

        validated_transactions = []
        
        for doc in docs:
            row_data = {}
            lines = doc.page_content.split("\n")

            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    row_data[key.strip().lower()] = value.strip()

            try:
                # Validate using Pydantic
                transaction = Transaction(**row_data)
                validated_transactions.append(transaction)
            except Exception as e:
                print(f"Validation failed for a row: {e}")

        return validated_transactions

    def sync_to_db(self, transactions: List[Transaction]):
        """
        Inserts a list of validated Transaction objects into the SQLite database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            amount REAL,
            category TEXT,
            description TEXT,
            source TEXT
        )
        """)

        # Insert Validated Rows
        for tx in transactions:
            cursor.execute("""
                INSERT INTO transactions (
                    date,
                    amount,
                    category,
                    description,
                    source
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                str(tx.date),
                tx.amount,
                tx.category,
                tx.description,
                tx.source
            ))

        conn.commit()
        conn.close()
        print(f"Successfully synced {len(transactions)} transactions to {self.db_path}")

def main():
    # Example usage:
    # Assuming transactions.csv is in the data/ directory
    csv_file = "data/transactions.csv"
    
    # Create a dummy CSV if it doesn't exist for demonstration/initialization
    if not os.path.exists(csv_file):
        os.makedirs("data", exist_ok=True)
        df_dummy = pd.DataFrame([
            {"date": "2023-10-01", "amount": 50.0, "category": "Food", "description": "Lunch", "source": "Credit Card"},
            {"date": "2023-10-02", "amount": 20.0, "category": "Transport", "description": "Bus", "source": "Cash"}
        ])
        df_dummy.to_csv(csv_file, index=False)
        print(f"Created dummy CSV at {csv_file}")

    loader = TransactionLoader()
    try:
        transactions = loader.load_from_csv(csv_file)
        loader.sync_to_db(transactions)
        return transactions
    except Exception as e:
        print(f"Error during loading: {e}")
        return []

if __name__ == "__main__":
    main()
