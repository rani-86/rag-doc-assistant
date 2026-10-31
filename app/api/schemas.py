
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Pass the same id on follow-up questions to use "
                    "conversation memory."
    )
    max_retries: Optional[int] = Field(None, ge=0, le=5)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        # Short messages like "hi" are valid (handled as chitchat);
        # only empty / whitespace-only input is rejected.
        if not value:
            raise ValueError("question must not be empty")
        return value


class SourceOut(BaseModel):
    id: int
    source: str
    filename: str
    section: str
    score: Optional[float]
    snippet: str


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    sources: list[SourceOut]
    status: Literal[
        "answered", "ungrounded", "no_relevant_documents", "chitchat"
    ]
    grounded: Optional[bool]
    query_type: Optional[str]
    search_queries: list[str]
    retries: int
    used_web_search: bool
    steps: list[str]
    session_id: Optional[str]


class IngestedDocument(BaseModel):
    source: str
    filename: str
    chunks: int


class IngestError(BaseModel):
    source: str
    error: str


class IngestResponse(BaseModel):
    ingested: list[IngestedDocument]
    failed: list[IngestError]
    total_chunks: int


class DocumentOut(BaseModel):
    source: str
    filename: Optional[str]
    file_type: Optional[str]
    chunks: int


class DocumentsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: list[DocumentOut]


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(None, max_length=2000)


class FeedbackResponse(BaseModel):
    message: str
    query_id: str
    rating: Literal["up", "down"]
