import sqlite3
import pandas as pd

from langchain_community.document_loaders import CSVLoader

from schema.transactions import Transaction

# -----------------------------
# CSV File Path
# -----------------------------
file_path = "transactions.csv"

# -----------------------------
# Validate Extension
# -----------------------------
if not file_path.endswith(".csv"):
    raise ValueError("Only CSV files are allowed")

# -----------------------------
# Load CSV using LangChain
# -----------------------------
loader = CSVLoader(file_path=file_path)
docs = loader.load()

# -----------------------------
# Convert Documents → DataFrame
# -----------------------------
rows = []

for doc in docs:
    row_data = {}
    lines = doc.page_content.split("\n")

    for line in lines:
        key, value = line.split(":", 1)
        row_data[key.strip()] = value.strip()

    rows.append(row_data)

df = pd.DataFrame(rows)

# -----------------------------
# Validate using Pydantic
# -----------------------------
validated_rows = []

for index, row in df.iterrows():
    try:
        validated_data = Transaction(
            date=row["date"],
            amount=row["amount"],
            category=row["category"],
            description=row["description"],
            source=row["source"]
        )
        validated_rows.append(validated_data.model_dump())
    
    except Exception as e:
        print(f"Validation failed at row {index}: {e}")

# -----------------------------
# Create Validated DataFrame
# -----------------------------
validated_df = pd.DataFrame(validated_rows)

# -----------------------------
# Connect SQLite Database
# -----------------------------
conn = sqlite3.connect("transactions.db")

cursor = conn.cursor()

# -----------------------------
# Create Table
# -----------------------------
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

# -----------------------------
# Insert Validated Rows
# -----------------------------
for _, row in validated_df.iterrows():

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

        str(row["date"]),
        float(row["amount"]),
        row["category"],
        row["description"],
        row["source"]
    ))

# -----------------------------
# Commit Changes
# -----------------------------
conn.commit()

# -----------------------------
# Close Connection
# -----------------------------
conn.close()