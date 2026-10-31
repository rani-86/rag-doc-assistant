
from app.graph.edges import (
    route_after_grading,
    route_after_hallucination_check
)
from tests.conftest import make_doc


def test_relevant_documents_go_to_generation():
    state = {"relevant_documents": [make_doc("x")], "retry_count": 0}
    assert route_after_grading(state) == "generate"


def test_no_relevant_documents_rewrites_while_retries_left():
    state = {"relevant_documents": [], "retry_count": 1, "max_retries": 2}
    assert route_after_grading(state) == "rewrite_query"


def test_no_relevant_documents_falls_back_when_retries_exhausted():
    state = {"relevant_documents": [], "retry_count": 2, "max_retries": 2}
    assert route_after_grading(state) == "fallback"


def test_zero_retries_allowed_goes_straight_to_fallback():
    state = {"relevant_documents": [], "retry_count": 0, "max_retries": 0}
    assert route_after_grading(state) == "fallback"


def test_grounded_answer_ends():
    assert route_after_hallucination_check({"grounded": True}) == "end"


def test_ungrounded_answer_regenerates_once_then_ends():
    state = {"grounded": False, "generation_count": 1, "max_generations": 2}
    assert route_after_hallucination_check(state) == "generate"

    state["generation_count"] = 2
    assert route_after_hallucination_check(state) == "end"


def test_exhausted_retries_use_web_search_when_enabled(monkeypatch):
    from app.graph import edges

    monkeypatch.setattr(edges, "web_search_enabled", lambda: True)
    state = {"relevant_documents": [], "retry_count": 2, "max_retries": 2}

    assert route_after_grading(state) == "web_search"

    state["web_search_used"] = True
    assert route_after_grading(state) == "fallback"


def test_web_search_disabled_by_default(monkeypatch):
    from app.graph import edges

    monkeypatch.setattr(edges, "web_search_enabled", lambda: False)
    state = {"relevant_documents": [], "retry_count": 2, "max_retries": 2}

    assert route_after_grading(state) == "fallback"
