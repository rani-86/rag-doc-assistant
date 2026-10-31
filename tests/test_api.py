
import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "count_chunks", lambda: 10)
    monkeypatch.setattr(routes, "DOCUMENTS_DIR", tmp_path)
    monkeypatch.setattr(
        "app.services.feedback.FEEDBACK_FILE", tmp_path / "feedback.jsonl"
    )
    monkeypatch.setattr("app.main.get_vector_store", lambda: None)
    monkeypatch.setattr("app.main.count_chunks", lambda: 10)

    with TestClient(app) as test_client:
        yield test_client


def fake_run_query(question, chat_history=None, max_retries=None):
    return {
        "answer": f"answer to {question} [1]",
        "sources": [{
            "id": 1, "source": "data/documents/validation.md",
            "filename": "validation.md", "section": "Validation",
            "score": 0.7, "snippet": "..."
        }],
        "status": "answered",
        "grounded": True,
        "query_type": "how-to",
        "search_queries": ["q"],
        "retry_count": 0,
        "steps": ["analyze_query"],
        "history_seen": chat_history
    }


def test_query_returns_answer_and_sources(client, monkeypatch):
    monkeypatch.setattr(routes, "run_query", fake_run_query)

    response = client.post("/query", json={"question": "How to validate?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["sources"][0]["filename"] == "validation.md"
    assert body["query_id"]


def test_query_rejects_blank_question(client):
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422


def test_query_without_index_returns_409(client, monkeypatch):
    monkeypatch.setattr(routes, "count_chunks", lambda: 0)

    response = client.post("/query", json={"question": "How to validate?"})

    assert response.status_code == 409


def test_llm_failure_returns_502(client, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(routes, "run_query", boom)

    response = client.post("/query", json={"question": "How to validate?"})

    assert response.status_code == 502


def test_session_history_is_passed_to_follow_ups(client, monkeypatch):
    seen = []

    def recording_run_query(question, chat_history=None, max_retries=None):
        seen.append(list(chat_history))
        return fake_run_query(question, chat_history)

    monkeypatch.setattr(routes, "run_query", recording_run_query)

    client.post("/query", json={"question": "First question", "session_id": "s1"})
    client.post("/query", json={"question": "And then?", "session_id": "s1"})

    assert seen[0] == []
    assert seen[1][0]["question"] == "First question"


def test_feedback_flow(client, monkeypatch):
    monkeypatch.setattr(routes, "run_query", fake_run_query)
    query_id = client.post(
        "/query", json={"question": "How to validate?"}
    ).json()["query_id"]

    ok = client.post("/feedback", json={
        "query_id": query_id, "rating": "up", "comment": "helpful"
    })
    unknown = client.post("/feedback", json={
        "query_id": "nope", "rating": "up"
    })
    invalid = client.post("/feedback", json={
        "query_id": query_id, "rating": "meh"
    })

    assert ok.status_code == 201
    assert unknown.status_code == 404
    assert invalid.status_code == 422


def test_ingest_requires_input(client):
    assert client.post("/ingest").status_code == 400


def test_ingest_rejects_only_unsupported_files(client):
    response = client.post(
        "/ingest",
        files=[("files", ("report.pdf", b"%PDF", "application/pdf"))]
    )

    assert response.status_code == 422
    assert "Unsupported" in response.json()["detail"]["failed"][0]["error"]


def test_ingest_indexes_markdown_upload(client, monkeypatch):
    indexed = []
    monkeypatch.setattr(routes, "index_documents", indexed.extend)

    response = client.post(
        "/ingest",
        files=[
            ("files", ("guide.md", b"# Guide\n\nHello.", "text/markdown")),
            ("files", ("bad.exe", b"x", "application/octet-stream"))
        ]
    )

    assert response.status_code == 201
    body = response.json()
    assert body["ingested"][0]["filename"] == "guide.md"
    assert body["failed"][0]["source"] == "bad.exe"
    assert indexed and indexed[0].metadata["section"] == "Guide"


def test_documents_lists_corpus(client, monkeypatch):
    monkeypatch.setattr(routes, "list_indexed_documents", lambda: [
        {"source": "a.md", "filename": "a.md", "file_type": ".md", "chunks": 3},
        {"source": "b.md", "filename": "b.md", "file_type": ".md", "chunks": 2}
    ])

    body = client.get("/documents").json()

    assert body["total_documents"] == 2
    assert body["total_chunks"] == 5


def test_delete_document(client, monkeypatch, tmp_path):
    (tmp_path / "old.md").write_text("x")
    monkeypatch.setattr(
        routes, "delete_source",
        lambda source: 3 if source == "data/documents/old.md" else 0
    )

    ok = client.delete("/documents", params={"source": "data/documents/old.md"})
    missing = client.delete("/documents", params={"source": "nope.md"})

    assert ok.status_code == 200 and ok.json()["chunks_removed"] == 3
    assert not (tmp_path / "old.md").exists()
    assert missing.status_code == 404


def test_short_greeting_is_accepted(client, monkeypatch):
    monkeypatch.setattr(routes, "run_query", fake_run_query)

    response = client.post("/query", json={"question": "hi"})

    assert response.status_code == 200
