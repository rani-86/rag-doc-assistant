
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph.edges import (
    route_after_analysis,
    route_after_grading,
    route_after_hallucination_check,
    route_after_web_search
)
from app.graph.nodes import (
    analyze_query,
    check_hallucination,
    fallback,
    generate,
    grade_documents,
    respond_chitchat,
    retrieve,
    rewrite_query,
    web_search
)
from app.graph.state import GraphState


def build_workflow():
    """
    START -> analyze_query --chitchat--> respond_chitchat -> END
    analyze_query --question--> retrieve -> grade_documents
        grade_documents --relevant--------------> generate
        grade_documents --none, retries left----> rewrite_query -> retrieve
        grade_documents --none, retries used----> web_search (if TAVILY_API_KEY)
                                                   or fallback -> END
        web_search --relevant results--> generate, otherwise fallback
        generate -> check_hallucination
        check_hallucination --grounded / out of attempts--> END
        check_hallucination --ungrounded, attempts left---> generate
    """

    graph = StateGraph(GraphState)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("web_search", web_search)
    graph.add_node("fallback", fallback)
    graph.add_node("respond_chitchat", respond_chitchat)

    graph.add_edge(START, "analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "retrieve": "retrieve",
            "chitchat": "respond_chitchat"
        }
    )
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
            "fallback": "fallback"
        }
    )

    graph.add_conditional_edges(
        "web_search",
        route_after_web_search,
        {
            "generate": "generate",
            "fallback": "fallback"
        }
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "check_hallucination")

    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {
            "generate": "generate",
            "end": END
        }
    )

    graph.add_edge("fallback", END)
    graph.add_edge("respond_chitchat", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_workflow():
    return build_workflow()


def run_query(
    question: str,
    chat_history: list | None = None,
    max_retries: int | None = None
) -> GraphState:
    """Run the full self-corrective RAG workflow for one question."""

    initial_state: GraphState = {
        "question": question,
        "chat_history": chat_history or []
    }

    if max_retries is not None:
        initial_state["max_retries"] = max_retries

    return get_workflow().invoke(initial_state)
