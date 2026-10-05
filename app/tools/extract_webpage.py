"""Webpage extraction tool for ResearchForge."""

from __future__ import annotations

import json
import os

from langchain.tools import tool
from tavily import TavilyClient


def _get_tavily_client() -> TavilyClient:
    """Create a Tavily client using the configured API key."""

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured."
        )

    return TavilyClient(api_key=api_key)


@tool
def extract_webpage(
    url: str,
    query: str = "",
) -> str:
    """Extract detailed readable content from a webpage.

    Args:
        url: URL of the webpage to inspect.
        query: Optional description of the relevant information.
    """

    if not url.strip():
        raise ValueError(
            "URL cannot be empty."
        )

    client = _get_tavily_client()

    response = client.extract(
        urls=[url],
        extract_depth="basic",
        format="markdown",
        query=query or None,
    )

    results = response.get(
        "results",
        [],
    )

    failed_results = response.get(
        "failed_results",
        [],
    )

    formatted_results = [
        {
            "url": result.get(
                "url",
                "",
            ),
            "content": result.get(
                "raw_content",
                "",
            ),
        }
        for result in results
    ]

    return json.dumps(
        {
            "results": formatted_results,
            "failed_results": failed_results,
        },
        ensure_ascii=False,
    )
