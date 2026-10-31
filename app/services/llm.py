
import os
from functools import lru_cache

from langchain_groq import ChatGroq

from app.config import GROQ_FAST_MODEL, GROQ_MODEL


@lru_cache(maxsize=4)
def get_llm(fast: bool = False, temperature: float = 0) -> ChatGroq:
    """Create (once) and return a Groq chat model.

    fast=True returns the smaller model used for grading-style decisions.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from .env")

    model = GROQ_FAST_MODEL if fast else GROQ_MODEL

    extra = {}
    if "gpt-oss" in model:
        # Reasoning tokens count against the rate limit; these tasks
        # don't need long chains of thought.
        extra["reasoning_effort"] = "low"

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
        # Retries 429s, waiting for the time the API asks for.
        max_retries=6,
        **extra
    )
