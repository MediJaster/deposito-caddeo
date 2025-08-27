from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class AdditionToolInput(BaseModel):
    """Input schema for AdditionTool."""

    a: int = Field(..., description="First number to add.")
    b: int = Field(..., description="Second number to add.")


class AdditionTool(BaseTool):
    name: str = "Addition Tool"
    description: str = "Use this tool to add two numbers together."
    args_schema: Type[BaseModel] = AdditionToolInput

    def _run(self, a: int, b: int) -> int:
        return a + b
