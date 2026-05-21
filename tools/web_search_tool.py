# ---------- IMPORT ----------
from langchain_core.tools import tool

from tavily import TavilyClient
from tavily.errors import (
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)

import os
from dotenv import load_dotenv
# ---------- IMPORT ----------

load_dotenv()
@tool
def web_search(query: str):
    """Search the web for current financial data including inflation rates, interest rates, and market news."""
    try:
        client = TavilyClient(os.getenv("TAVILY_API_KEY"))

        response = client.search(
            query=query,
            search_depth="advanced"
        )

        results = response.get("results", [])
        formatted_results = []
        for res in results[:3]:
            title = res.get("title", "No Title")
            content = res.get("content", "")
            formatted_results.append(f"Title: {title}\nContent: {content}")

        return "\n\n".join(formatted_results)
    except (InvalidAPIKeyError, ForbiddenError, MissingAPIKeyError, UsageLimitExceededError):
        return "Search failed: API key is invalid or expired."
    except Exception as e:
        return f"Search failed: {e}"
