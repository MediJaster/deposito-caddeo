from typing import Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

SupportedOperations = Literal["addition", "subtraction", "multiplication", "division"]


class CalculatorToolInput(BaseModel):
    """Input schema for CalculatorTool."""

    calculation: SupportedOperations = Field(
        ..., description="The type of calculation to perform."
    )

    a: float = Field(..., description="The first number.")
    b: float = Field(..., description="The second number.")


class CalculatorTool(BaseTool):
    name: str = "CalculatorTool"
    description: str = "Use this tool to perform various calculations."
    args_schema: Type[BaseModel] = CalculatorToolInput

    def _run(
        self,
        calculation: SupportedOperations,
        a: float,
        b: float,
    ) -> float:
        """Use the tool synchronously."""

        match calculation:
            case "addition":
                return a + b
            case "subtraction":
                return a - b
            case "multiplication":
                return a * b
            case "division":
                if b == 0:
                    raise ValueError("Division by zero is not allowed.")
                return a / b
            case _:
                raise ValueError(f"Unsupported operation: {calculation}")
