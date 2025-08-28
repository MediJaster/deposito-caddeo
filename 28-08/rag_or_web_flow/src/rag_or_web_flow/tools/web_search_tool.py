from typing import Type

from ddgs import DDGS

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class WebSearchToolInput(BaseModel):
    """Input schema for WebSearchTool."""

    search_query: str = Field(..., description="Description of the search_query.")


class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "Use this tool to perform a web search with a given query."
    args_schema: Type[BaseModel] = WebSearchToolInput

    def _run(self, search_query: str) -> str:
        """Use the tool synchronously."""

        with DDGS(verify=False) as ddgs:
            results = ddgs.text(
                search_query, max_results=3, region="it-it", safesearch="off"
            )
            search_results = "\n".join(
                [f"- {result['title']}: {result['href']}" for result in results]
            )

            return search_results
