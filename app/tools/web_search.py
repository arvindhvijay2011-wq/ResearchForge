"""Web search tool for ResearchForge."""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse

from langchain.tools import tool
from tavily import TavilyClient


def _get_tavily_client() -> TavilyClient:
    """Create a Tavily client using the configured API key."""

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured."
        )

    return TavilyClient(
        api_key=api_key,
    )


def _classify_source(url: str) -> str:
    """Classify a source into a broad category."""

    hostname = urlparse(url).hostname or ""
    hostname = hostname.lower()

    if (
        hostname.endswith(".gov")
        or hostname.endswith(".gov.uk")
        or hostname.endswith(".gov.in")
        or hostname == "who.int"
        or hostname.endswith(".who.int")
        or hostname == "un.org"
        or hostname.endswith(".un.org")
    ):
        return "official"

    if (
        hostname.endswith(".edu")
        or hostname.endswith(".ac.uk")
        or hostname.endswith(".ac.in")
        or "scholar" in hostname
        or "pubmed" in hostname
        or "nih.gov" in hostname
        or "nature.com" in hostname
        or "sciencedirect.com" in hostname
    ):
        return "academic"

    if (
        "wikipedia.org" in hostname
        or "britannica.com" in hostname
    ):
        return "reference"

    if (
        "reuters.com" in hostname
        or "apnews.com" in hostname
        or "bbc.com" in hostname
        or "nytimes.com" in hostname
        or "theguardian.com" in hostname
    ):
        return "news"

    if (
        hostname == "reddit.com"
        or hostname.endswith(".reddit.com")
        or "facebook.com" in hostname
        or "youtube.com" in hostname
    ):
        return "community"

    return "commercial_or_other"


@tool
def web_search(
    query: str,
    max_results: int = 5,
) -> str:
    """Search the web for evidence relevant to a research task.

    Args:
        query: Focused research query.
        max_results: Maximum number of search results.
    """

    if not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    if not 1 <= max_results <= 10:
        raise ValueError(
            "max_results must be between 1 and 10."
        )

    client = _get_tavily_client()

    response = client.search(
        query=query,
        search_depth="basic",
        topic="general",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )

    results = response.get(
        "results",
        [],
    )

    formatted_results = []

    for result in results:
        url = result.get(
            "url",
            "",
        )

        formatted_results.append(
            {
                "title": result.get(
                    "title",
                    "",
                ),
                "url": url,
                "content": result.get(
                    "content",
                    "",
                ),
                "score": result.get(
                    "score",
                    0.0,
                ),
                "source_type": _classify_source(
                    url
                ),
            }
        )

    return json.dumps(
        {
            "query": query,
            "results": formatted_results,
        },
        ensure_ascii=False,
    )
