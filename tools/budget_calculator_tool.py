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
    try:
        df = fetch_data_from_DB()
        
        if df.empty:
            return {"error": "No transaction data found in the database."}
        
        income = df[df["category"].str.lower() == "income"]["amount"].sum()
        expenses = df[df["category"].str.lower() != "income"]["amount"].sum()
        
        # Calculate savings rate, handle zero income to avoid division by zero
        savings_rate = 0
        if income > 0:
            savings_rate = ((income - expenses) / income) * 100
            
        # Calculate burn rate
        unique_months = pd.to_datetime(df["date"]).dt.to_period("M").nunique()
        burn_rate = expenses / unique_months if unique_months > 0 else expenses
        
        category_breakdown = df.groupby("category")["amount"].sum().to_dict()

        return {
            "income": income,
            "expenses": expenses,
            "savings_rate": round(savings_rate, 2),
            "burn_rate": round(burn_rate, 2),
            "category_breakdown": category_breakdown,
            "data_points": len(df)
        }
    except Exception as e:
        return {"error": f"Failed to calculate metrics: {str(e)}"}
