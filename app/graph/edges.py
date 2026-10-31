
from app.config import MAX_GENERATIONS, MAX_RETRIES
from app.graph.state import GraphState
from app.services.web_search import web_search_enabled


def route_after_analysis(state: GraphState) -> str:
    """Greetings and small talk skip retrieval entirely."""

    return "chitchat" if state.get("query_type") == "chitchat" else "retrieve"


def route_after_grading(state: GraphState) -> str:
    """Conditional edge after document grading.

    - some relevant chunks          -> generate
    - none relevant, retries left   -> rewrite_query (then retrieve again)
    - none relevant, out of retries -> web_search if TAVILY_API_KEY is set,
                                       otherwise fallback ("I don't know")
    """

    if state.get("relevant_documents"):
        return "generate"

    max_retries = state.get("max_retries", MAX_RETRIES)

    if state.get("retry_count", 0) < max_retries:
        return "rewrite_query"

    if web_search_enabled() and not state.get("web_search_used"):
        return "web_search"

    return "fallback"


def route_after_web_search(state: GraphState) -> str:
    """Generate from relevant web results, otherwise admit we don't know."""

    return "generate" if state.get("relevant_documents") else "fallback"


def route_after_hallucination_check(state: GraphState) -> str:
    """Regenerate an ungrounded answer once, then stop either way.

    If the answer is still ungrounded after the last attempt it is returned
    with status 'ungrounded' so the client can show a warning.
    """

    if state.get("grounded"):
        return "end"

    max_generations = state.get("max_generations", MAX_GENERATIONS)

    if state.get("generation_count", 0) < max_generations:
        return "generate"

    return "end"
