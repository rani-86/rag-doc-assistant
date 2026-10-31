
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph import nodes  # noqa: E402


class FakeStructured:
    def __init__(self, responder):
        self.responder = responder

    def invoke(self, messages):
        return self.responder(messages)

    def batch(self, inputs, config=None):
        return [self.responder(messages) for messages in inputs]


class FakeLLM:
    """Stands in for ChatGroq. Each schema maps to a function that
    receives the prompt messages and returns an instance of the schema."""

    def __init__(self, structured: dict, answer: str = "Answer [1]."):
        self.structured = structured
        self.answer = answer
        self.calls: list[str] = []

    def with_structured_output(self, schema):
        def responder(messages):
            self.calls.append(schema.__name__)
            return self.structured[schema.__name__](messages)

        return FakeStructured(responder)

    def invoke(self, messages):
        self.calls.append("generate")
        return AIMessage(content=self.answer)


def make_doc(text: str, filename: str = "validation.md") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": f"data/documents/{filename}",
            "filename": filename,
            "section": "Section",
            "score": 0.8
        }
    )


@pytest.fixture
def patch_graph(monkeypatch):
    """Install a fake LLM and a fake retriever into the graph nodes."""

    def install(llm: FakeLLM, documents: list[Document]):
        monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
        monkeypatch.setattr(
            nodes, "retrieve_documents", lambda query, k: list(documents)
        )
        return llm

    return install
