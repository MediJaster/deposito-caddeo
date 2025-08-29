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
        """
        Esegue una ricerca web con DuckDuckGo e restituisce un elenco formattato di risultati.

        Parameters
        ----------
        search_query : str
            Query di ricerca. Unità: adimensionale. Range: stringa non vuota.

        Returns
        -------
        str
            Stringa con 0–3 risultati, uno per riga, nel formato "- {title}: {href}".
            Unità: adimensionale.

        Raises
        ------
        ValueError
            Se `search_query` è vuota o composta solo da spazi.
        RuntimeError
            Eventuali errori di rete o della libreria sottostante possono propagare.

        Notes
        -----
        Complessità temporale: O(n) nel numero di risultati richiesti (qui n ≤ 3).
        Complessità spaziale: O(n).

        Examples
        --------
        >>> from rag_or_web_flow.tools.web_search_tool import WebSearchTool
        >>> out = WebSearchTool()._run("example domain")
        >>> isinstance(out, str)
        True
        >>> out.count("\\n") <= 2  # al massimo 3 risultati -> al massimo 2 newline
        True
        """
        if not isinstance(search_query, str) or not search_query.strip():
            raise ValueError("search_query must be a non-empty string.")

        with DDGS(verify=False) as ddgs:
            results = ddgs.text(
                search_query, max_results=3, region="it-it", safesearch="off"
            )
            search_results = "\n".join(
                [f"- {result['title']}: {result['href']}" for result in results]
            )

            return search_results
