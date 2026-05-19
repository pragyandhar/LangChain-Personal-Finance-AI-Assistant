from pydantic import BaseModel
from datetime import date

class Transaction(BaseModel):
    date: date
    amount: float
    category: str
    description: str
    source: str