from pydantic import BaseModel, Field
from typing import List, Dict

class CategoryBudget(BaseModel):
    category: str
    budgeted_amount: float
    current_spending: float = 0.0

class MonthlyBudgetPlan(BaseModel):
    month: str = Field(..., description="The month this plan applies to, e.g., 'January 2024'")
    total_income: float = Field(..., gt=0, description="Total expected monthly income after tax")
    fixed_costs: float = Field(..., description="Total amount for recurring fixed expenses (rent, insurance, etc.)")
    variable_budgets: List[CategoryBudget] = Field(default_factory=list, description="List of budgets for variable categories")
    savings_target: float = Field(..., description="Target amount to save/invest this month")
    recommendations: List[str] = Field(default_factory=list, description="AI-generated recommendations for sticking to the budget")

    @property
    def remaining_budget(self) -> float:
        """Calculates total remaining unallocated income."""
        total_budgeted = self.fixed_costs + sum(b.budgeted_amount for b in self.variable_budgets) + self.savings_target
        return self.total_income - total_budgeted
