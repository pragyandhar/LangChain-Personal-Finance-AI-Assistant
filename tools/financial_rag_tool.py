# ---------- IMPORT ----------
from langchain_core.tools import tool
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import pandas as pd
import os
# ---------- IMPORT ----------

def finance_summarizer():
    persist_directory = "finance_chroma"
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Check if the vectorstore already exists
    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Load the data
    df = pd.read_csv("data/transactions.csv")

    # Preprocess dates
    df['date'] = pd.to_datetime(df['date'])

    # Create a period for sorting and grouping
    df['month_period'] = df['date'].dt.to_period('M')

    # Group by month_period and category, then sum the amount
    grouped = df.groupby(['month_period', 'category'])['amount'].sum().reset_index()

    # Sort chronologically by the period
    grouped = grouped.sort_values(['month_period', 'category'])

    # Generate f-string summaries
    summaries = []
    for _, row in grouped.iterrows():
        month_name = row['month_period'].strftime('%B %Y')
        summaries.append(f"In {month_name}, the category '{row['category']}' had a total transaction amount of {row['amount']:.2f}.")
    
    vectorstore = Chroma.from_texts(
        texts=summaries,                 # List[str]
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vectorstore

def finance_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

try:
    VECTOR_STORE = finance_summarizer()
except Exception as e:
    VECTOR_STORE = None
    print(f"RAG init failed: {e}")

@tool
def financial_rag_tool(query: str):
    """Query personal financial history. Use for questions about spending by category, month, or time period."""
    retriever = finance_retriever(VECTOR_STORE)

    docs = retriever.invoke(query)

    return "\n\n".join([doc.page_content for doc in docs])
