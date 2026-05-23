# Error Resolution & Debugging Log

This document tracks the major technical hurdles encountered during the development of the Personal Finance AI Assistant and the logic used to resolve them.

---

## 1. The "Orphaned Tool Call" Protocol Violation (OpenAI 400)

### The Error
`openai.BadRequestError: Error code: 400 - {'error': {'message': "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'..."}}`

### The Thought Process
1.  **Identification**: The error occurred inside `summarize_node`.
2.  **The Cause**: The code used a "Blind Slice" (`messages[:-2]`) to truncate history for summarization. If the assistant made a tool call at index `-3` and the response was at index `-2`, the slice would include the *call* but exclude the *response*.
3.  **The Rule**: OpenAI/Azure protocol mandates that every `AIMessage` with `tool_calls` must be followed immediately by its corresponding `ToolMessage`s.
4.  **The Fix**: Implement a "Safety Valve" that backtracks the slice until it finds a safe stopping point.

### Code Implementation (`memory/graph.py`)
```python
def summarize_node(state: UserFinanceState):
    messages = state["messages"]
    to_summarize = messages[:-2]
    
    # Backtrack if we end on an orphaned tool call
    while to_summarize and hasattr(to_summarize[-1], "tool_calls") and to_summarize[-1].tool_calls:
        to_summarize.pop()
    
    # ... call LLM with safe_to_summarize ...
```

### Dry Run Logic
*   **Initial State**: `[UserMsg, AIMsg(tool_calls), ToolMsg(result), UserMsg(next), AIMsg(tool_calls)]` (Length 5)
*   **Blind Slice (`[:-2]`)**: `[UserMsg, AIMsg(tool_calls), ToolMsg(result)]` -> **SAFE** (ToolMsg is present).
*   **Worst Case Scenario**: `[UserMsg, AIMsg(tool_calls), ToolMsg(result), AIMsg(tool_calls), UserMsg(current)]`
*   **Blind Slice (`[:-2]`)**: `[UserMsg, AIMsg(tool_calls), ToolMsg(result), AIMsg(tool_calls)]` -> **CRASH** (Last AIMsg has tool_calls but no ToolMsg follows).
*   **With Safety Loop**: The logic sees the last `AIMsg` has `tool_calls`, calls `.pop()`, and summarizes only up to the `ToolMsg`.

---

## 2. Azure OpenAI Endpoint 404 (Path Duplication)

### The Error
`Error code: 404 - {'error': {'code': '404', 'message': 'Resource not found'}}`

### The Thought Process
1.  **Identification**: 404 in Azure usually means a deployment name is wrong or the URL path is malformed.
2.  **Observation**: The user provided an endpoint like `.../openai/v1/`.
3.  **The Mechanism**: LangChain's `AzureChatOpenAI` and `AzureOpenAIEmbeddings` classes automatically append `/openai/deployments/...` to the base endpoint.
4.  **The Result**: The final URL became `.../openai/v1/openai/deployments/...`, which is a dead link.
5.  **The Fix**: Programmatically strip the `/openai/v1/` suffix from the environment variable before passing it to the model.

### Code Implementation (`memory/graph.py`)
```python
raw_endpoint = os.getenv("foundry_endpoint")
# Strip suffix to provide the BASE resource URL
endpoint = raw_endpoint.replace("/openai/v1/", "").replace("/openai/v1", "").rstrip("/") 

model = AzureChatOpenAI(azure_endpoint=endpoint, ...)
```

---

## 3. SQLite Resource Leak & Connection Locking

### The Problem
The database connection was opened during graph initialization but never closed. In a high-concurrency or long-running environment, this leads to memory leaks and "Database is locked" errors.

### The Thought Process
1.  **The Solution**: Standard Python best practice for resource management is the **Context Manager** (`with` statement).
2.  **The Constraint**: LangGraph's `SqliteSaver` needs an active connection throughout the graph's execution.
3.  **The Fix**: Wrap the connection logic in a `@contextmanager` that yields the saver and explicitly closes the connection in a `finally` block.

### Code Implementation (`memory/checkpoint.py`)
```python
@contextmanager
def get_sqlite_saver():
    db_path = "data/checkpoints.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close() # Connection is guaranteed to close
```

### Dry Run Logic
1.  App starts: `with get_sqlite_saver() as saver:` -> `sqlite3.connect()` is called.
2.  Graph runs: `app.invoke(...)` -> Uses the active `saver`.
3.  User exits or error occurs: The `finally` block triggers -> `conn.close()` is called immediately.

---

## 4. State Schema Redundancy

### The Problem
The project had `AgentState` defined inside `graph.py` and `UserFinanceState` in `schema/state.py`. This created a "Dual Source of Truth" where updates to one wouldn't reflect in the other.

### The Thought Process
1.  **Refactor**: Centralize state definitions in the `schema/` directory.
2.  **Unification**: Delete the local `AgentState` and import `UserFinanceState`. 
3.  **Benefit**: We ensured the `messages` field used `add_messages` (which deduplicates IDs) instead of a simple list append, making the graph's memory more robust.
