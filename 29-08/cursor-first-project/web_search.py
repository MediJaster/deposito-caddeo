"""Web search helper built on top of DDGS (DuckDuckGo Search)."""

from typing import Dict, List, Optional

from ddgs import DDGS


def search_web(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safesearch: str = "moderate",
    timelimit: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Search the web using DuckDuckGo (DDGS) and return normalized results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
        region: Region code for localization (e.g., "us-en", "it-it", "wt-wt").
        safesearch: Safe search level. Common values: "off", "moderate", "strict".
        timelimit: Optional time filter (e.g., "d" for day, "w" for week, "m" for month).

    Returns:
        A list of dictionaries, each containing:
        - "title": Result title
        - "url": Result URL
        - "snippet": Short description/snippet

    Raises:
        ValueError: If the query is empty or only whitespace.

    Notes:
        This function wraps DDGS text search and normalizes the output keys.
        DDGS may return fields like "title", "href", and "body" which are
        mapped here to "title", "url", and "snippet" for convenience.
    """

    if not query or query.strip() == "":
        raise ValueError("query must be a non-empty string")

    normalized_results: List[Dict[str, str]] = []

    with DDGS() as ddgs:
        results_iter = ddgs.text(
            query,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            max_results=max_results,
        )

        for item in results_iter:
            title = str(item.get("title", "")).strip()
            url = str(item.get("href", item.get("url", ""))).strip()
            snippet = str(item.get("body", item.get("snippet", ""))).strip()

            if not url:
                # Skip entries without a URL
                continue

            normalized_results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )

            if len(normalized_results) >= max_results:
                break

    return normalized_results


__all__ = ["search_web"]
