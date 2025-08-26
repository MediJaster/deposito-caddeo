from typing import Type

from crewai_tools import ScrapeWebsiteTool

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class WebSearchToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    search_query: str = Field(..., description="Description of the search_query.")


class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "Use this tool to perform a web search with a given query."
    args_schema: Type[BaseModel] = WebSearchToolInput

    def _run(self, search_query: str) -> str:
        """Use the tool synchronously."""

        website_url = f"https://duckduckgo.com/?q={search_query}"

        tool = ScrapeWebsiteTool(website_url=website_url)
        search_results = tool.run()

        return search_results
