from pydantic import BaseModel, Field, List

class Transaction(BaseModel):
    date: int,
    amount: int,
    category: List[str], 
    description: str,
    source: str