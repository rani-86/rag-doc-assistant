
import os

import httpx
from langchain_core.documents import Document

TAVILY_URL = "https://api.tavily.com/search"


def web_search_enabled() -> bool:
    return bool(os.getenv("TAVILY_API_KEY"))


def search_web(query: str, max_results: int = 5) -> list[Document]:
    """Search the web with Tavily and return results as Documents."""

    response = httpx.post(
        TAVILY_URL,
        headers={"Authorization": f"Bearer {os.getenv('TAVILY_API_KEY')}"},
        json={
            "query": query,
            "max_results": max_results,
            "search_depth": "basic"
        },
        timeout=20
    )
    response.raise_for_status()

    return [
        Document(
            page_content=result.get("content", ""),
            metadata={
                "source": result["url"],
                "filename": result.get("title") or result["url"],
                "section": "web search",
                "score": result.get("score")
            }
        )
        for result in response.json().get("results", [])
        if result.get("content")
    ]
