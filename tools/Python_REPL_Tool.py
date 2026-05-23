# ------------ IMPORTS ------------
from pathlib import Path
from langchain_core.tools import tool

import os
# ------------ IMPORTS ------------

@tool
def python_repl_tool(code: str):
    """Execute python code in a local sandbox environment. Useful for generating budget charts and saving figures to outputs/ directory."""
    path = "./outputs"
    folder = Path(path)
    folder.mkdir(exist_ok=True)
    
    before = set(os.listdir(folder))

    try:
        sandbox = {}
        exec(code, {}, sandbox)
    except Exception as e:
        return f"Error: {e}"
    
    after = set(os.listdir(folder))
    
    new_files = after - before
    file_paths = [str(folder / file) for file in new_files]
    
    if new_files:
        return f"Charts saved: {', '.join(file_paths)}"
    else:
        return "Error: no file saved — ensure plt.savefig() is called"
