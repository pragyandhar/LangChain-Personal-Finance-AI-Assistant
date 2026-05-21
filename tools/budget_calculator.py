# ------------ IMPORTS ------------
from langchain.tools import tool

import pandas as pd
import sqlite3
# ------------ IMPORTS ------------

def fetch_data_from_DB(db_path: str = "data/transactions.db"):
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM transactions", conn)
        conn.close()
    except:
        raise ValueError("Error: database not found")
    
    return df

@tool
def calculate_budget_metrics(query: str) -> dict:
    """
    Use this tool to calculate savings rate, burn rate, and category 
    breakdowns from transaction data. Input is a plain english question.
    """
    df = fetch_data_from_DB()  # return a DataFrame, not CSV string
    
    income = df[df["category"] == "Income"]["amount"].sum()
    expenses = df[df["category"] != "Income"]["amount"].sum()
    savings_rate = ((income - expenses) / income) * 100
    burn_rate = expenses / pd.to_datetime(df["date"]).dt.to_period("M").nunique()  # per month
    category_breakdown = df.groupby("category")["amount"].sum().to_dict()

    return {
        "income": income,
        "expenses": expenses,
        "savings_rate": round(savings_rate, 2),
        "burn_rate": round(burn_rate, 2),
        "category_breakdown": category_breakdown
    }
