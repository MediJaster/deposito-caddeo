from pydantic import BaseModel, Field


class ToolExplanationOutput(BaseModel):
    tool_name: str | None = Field(
        default=None, description="The name of the tool being explained."
    )
    explanation: str = Field(
        ...,
        description="A detailed explanation of the tool's functionalities and usage.",
    )
