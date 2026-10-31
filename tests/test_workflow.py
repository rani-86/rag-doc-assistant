
from app.graph.nodes import (
    FALLBACK_ANSWER,
    GradeDocument,
    GroundednessCheck,
    QueryAnalysis,
    RewrittenQuery
)
from app.graph.workflow import run_query
from tests.conftest import FakeLLM, make_doc


def analysis(_):
    return QueryAnalysis(
        standalone_question="How do I validate an email?",
        query_type="how-to",
        search_query="email validation"
    )


def rewrite(_):
    return RewrittenQuery(search_query="another query")


def grade_by_keyword(messages):
    text = messages[-1].content
    return GradeDocument(
        relevance="relevant" if "isEmail" in text else "irrelevant"
    )


def grounded(_):
    return GroundednessCheck(grounded=True, reason="ok")


def test_happy_path_filters_irrelevant_chunks(patch_graph):
    llm = patch_graph(
        FakeLLM({
            "QueryAnalysis": analysis,
            "GradeDocument": grade_by_keyword,
            "GroundednessCheck": grounded
        }, answer="Use body('email').isEmail() [1]."),
        [make_doc("body('email').isEmail()"), make_doc("Routing basics")]
    )

    result = run_query("how to validate email?")

    assert result["status"] == "answered"
    assert result["retry_count"] == 0
    assert len(result["relevant_documents"]) == 1
    assert [s["id"] for s in result["sources"]] == [1]
    assert "RewrittenQuery" not in llm.calls


def test_retries_then_falls_back_and_terminates(patch_graph):
    llm = patch_graph(
        FakeLLM({
            "QueryAnalysis": analysis,
            "RewrittenQuery": rewrite,
            "GradeDocument": lambda _: GradeDocument(relevance="irrelevant")
        }),
        [make_doc("unrelated")]
    )

    result = run_query("train a CNN", max_retries=2)

    assert result["status"] == "no_relevant_documents"
    assert result["answer"] == FALLBACK_ANSWER
    assert result["sources"] == []
    assert result["retry_count"] == 2
    assert llm.calls.count("RewrittenQuery") == 2
    # initial query + one per retry
    assert result["search_queries"] == [
        "email validation", "another query", "another query"
    ]
    assert "generate" not in llm.calls


def test_rewrite_recovers_when_second_retrieval_is_relevant(patch_graph, monkeypatch):
    from app.graph import nodes

    llm = patch_graph(
        FakeLLM({
            "QueryAnalysis": analysis,
            "RewrittenQuery": rewrite,
            "GradeDocument": grade_by_keyword,
            "GroundednessCheck": grounded
        }),
        []
    )

    # Only the rewritten query ("another query") finds the relevant chunk.
    monkeypatch.setattr(
        nodes, "retrieve_documents",
        lambda query, k: [make_doc("isEmail()")] if query == "another query"
        else [make_doc("unrelated")]
    )

    result = run_query("validate email")

    assert result["status"] == "answered"
    assert result["retry_count"] == 1
    assert llm.calls.count("RewrittenQuery") == 1


def test_ungrounded_answer_is_regenerated_then_flagged(patch_graph):
    llm = patch_graph(
        FakeLLM({
            "QueryAnalysis": analysis,
            "GradeDocument": lambda _: GradeDocument(relevance="relevant"),
            "GroundednessCheck": lambda _: GroundednessCheck(
                grounded=False, reason="invented a library"
            )
        }),
        [make_doc("isEmail()")]
    )

    result = run_query("validate email")

    assert llm.calls.count("generate") == 2
    assert result["status"] == "ungrounded"
    assert result["grounded"] is False


def test_full_width_citations_are_normalised(patch_graph):
    patch_graph(
        FakeLLM({
            "QueryAnalysis": analysis,
            "GradeDocument": lambda _: GradeDocument(relevance="relevant"),
            "GroundednessCheck": grounded
        }, answer="See 【2】."),
        [make_doc("a"), make_doc("b")]
    )

    result = run_query("validate email")

    assert result["answer"] == "See [2]."
    assert [s["id"] for s in result["sources"]] == [2]


def test_web_search_answers_when_corpus_has_nothing(patch_graph, monkeypatch):
    from app.graph import edges, nodes

    monkeypatch.setattr(edges, "web_search_enabled", lambda: True)
    monkeypatch.setattr(
        nodes, "search_web",
        lambda query: [make_doc("isEmail() from the web", "Web page")]
    )
    patch_graph(
        FakeLLM({
            "QueryAnalysis": analysis,
            "RewrittenQuery": rewrite,
            "GradeDocument": grade_by_keyword,
            "GroundednessCheck": grounded
        }),
        [make_doc("unrelated")]
    )

    result = run_query("validate email", max_retries=1)

    assert result["web_search_used"] is True
    assert result["status"] == "answered"
    assert result["sources"][0]["filename"] == "Web page"


def test_failed_web_search_still_falls_back(patch_graph, monkeypatch):
    from app.graph import edges, nodes

    def broken(query):
        raise RuntimeError("network down")

    monkeypatch.setattr(edges, "web_search_enabled", lambda: True)
    monkeypatch.setattr(nodes, "search_web", broken)
    patch_graph(
        FakeLLM({
            "QueryAnalysis": analysis,
            "RewrittenQuery": rewrite,
            "GradeDocument": lambda _: GradeDocument(relevance="irrelevant")
        }),
        [make_doc("unrelated")]
    )

    result = run_query("train a CNN", max_retries=0)

    assert result["status"] == "no_relevant_documents"
    assert any("web_search: failed" in step for step in result["steps"])


def test_chitchat_skips_retrieval_and_sees_file_list(patch_graph, monkeypatch):
    from app.graph import nodes

    monkeypatch.setattr(nodes, "list_indexed_documents", lambda: [
        {"filename": "routing.md", "chunks": 5},
        {"filename": "error_handling.md", "chunks": 4}
    ])
    llm = patch_graph(
        FakeLLM({
            "QueryAnalysis": lambda _: QueryAnalysis(
                standalone_question="which files do you have",
                query_type="chitchat",
                search_query="files"
            )
        }, answer="I have routing.md and error_handling.md."),
        []
    )

    prompts = []
    original_invoke = llm.invoke
    llm.invoke = lambda messages: prompts.append(messages) or original_invoke(messages)

    result = run_query("which files do you have")

    assert result["status"] == "chitchat"
    assert result["answer"] == "I have routing.md and error_handling.md."
    # Analysis + one reply call; no retrieval, grading or rewriting.
    assert llm.calls == ["QueryAnalysis", "generate"]
    assert "documents" not in result
    # The real file list is given to the LLM.
    assert "routing.md (5 chunks)" in prompts[0][0].content


def test_first_retrieval_also_searches_the_original_question(patch_graph, monkeypatch):
    from app.graph import nodes

    patch_graph(
        FakeLLM({
            "QueryAnalysis": analysis,
            "GradeDocument": grade_by_keyword,
            "GroundednessCheck": grounded
        }),
        []
    )

    searched = []

    def fake_retrieve(query, k):
        searched.append(query)
        # The expanded query misses; the plain question finds the chunk.
        if query == "How do I validate an email?":
            return [make_doc("isEmail()"), make_doc("shared")]
        return [make_doc("unrelated"), make_doc("shared")]

    monkeypatch.setattr(nodes, "retrieve_documents", fake_retrieve)

    result = run_query("how to validate email?")

    assert searched == ["email validation", "How do I validate an email?"]
    # Merged and de-duplicated: unrelated, shared, isEmail()
    assert len(result["documents"]) == 3
    assert result["status"] == "answered"
    assert result["retry_count"] == 0


def test_retries_search_only_the_rewritten_query(patch_graph, monkeypatch):
    from app.graph import nodes

    patch_graph(
        FakeLLM({
            "QueryAnalysis": analysis,
            "RewrittenQuery": rewrite,
            "GradeDocument": lambda _: GradeDocument(relevance="irrelevant")
        }),
        []
    )

    searched = []
    monkeypatch.setattr(
        nodes, "retrieve_documents",
        lambda query, k: searched.append(query) or [make_doc("unrelated")]
    )

    run_query("train a CNN", max_retries=1)

    assert searched == [
        "email validation",             # round 1: expanded query
        "How do I validate an email?",  # round 1: original question
        "another query"                 # retry: rewritten query only
    ]
