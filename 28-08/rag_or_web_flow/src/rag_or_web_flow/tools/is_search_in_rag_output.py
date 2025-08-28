from pydantic import BaseModel, Field
from typing import Literal


class IsSearchInRagOutput(BaseModel):
    query: str = Field(..., description="The user's query.")
    method: Literal["rag", "web", "calculator"] = Field(
        ...,
        description="The method used to answer the query.",
    )
