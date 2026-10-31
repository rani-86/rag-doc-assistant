
import logging
from collections import Counter, OrderedDict
from pathlib import Path
from threading import Lock
from typing import Optional
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.api.schemas import (
    DocumentsResponse,
    FeedbackRequest,
    FeedbackResponse,
    IngestedDocument,
    IngestError,
    IngestResponse,
    QueryRequest,
    QueryResponse
)
from app.config import (
    DOCUMENTS_DIR,
    MAX_UPLOAD_BYTES,
    SUPPORTED_EXTENSIONS
)
from app.graph.workflow import run_query
from app.services import memory
from app.services.document_loader import (
    fetch_url,
    make_document,
    split_documents
)
from app.services.feedback import save_feedback
from app.services.vector_store import (
    count_chunks,
    delete_source,
    index_documents,
    list_indexed_documents
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Recent answers, kept so /feedback can validate query ids and store the
# question/answer next to the rating.
_recent_queries: "OrderedDict[str, dict]" = OrderedDict()
_recent_lock = Lock()
MAX_RECENT_QUERIES = 1000


def remember_query(query_id: str, record: dict) -> None:
    with _recent_lock:
        _recent_queries[query_id] = record
        while len(_recent_queries) > MAX_RECENT_QUERIES:
            _recent_queries.popitem(last=False)


# --------------------------------------------------------------------------
# POST /query
# --------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question with the self-corrective RAG workflow."""

    if count_chunks() == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No documents are indexed yet. Call POST /ingest first."
        )

    try:
        result = run_query(
            question=request.question,
            chat_history=memory.get_history(request.session_id),
            max_retries=request.max_retries
        )
    except ValueError as error:
        # Raised by get_llm() when the API key is missing.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error)
        )
    except Exception as error:
        logger.exception("Workflow failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The language model call failed: {type(error).__name__}"
        )

    memory.add_turn(request.session_id, request.question, result["answer"])

    query_id = str(uuid4())
    remember_query(query_id, {
        "question": request.question,
        "answer": result["answer"],
        "status": result.get("status")
    })

    return QueryResponse(
        query_id=query_id,
        question=request.question,
        answer=result["answer"],
        sources=result.get("sources", []),
        status=result.get("status", "answered"),
        grounded=result.get("grounded"),
        query_type=result.get("query_type"),
        search_queries=result.get("search_queries", []),
        retries=result.get("retry_count", 0),
        used_web_search=bool(result.get("web_search_used")),
        steps=result.get("steps", []),
        session_id=request.session_id
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["RAG"]
)
def clear_session(session_id: str) -> None:
    """Forget the conversation history of a session."""

    if not memory.clear_history(session_id):
        raise HTTPException(status_code=404, detail="Session not found")


# --------------------------------------------------------------------------
# POST /ingest
# --------------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"]
)
def ingest(
    files: Optional[list[UploadFile]] = File(
        None, description=".md, .txt or .html files"
    ),
    urls: Optional[list[str]] = Form(
        None, description="http(s) URLs of documentation pages"
    )
) -> IngestResponse:
    """Ingest uploaded files and/or URLs into the vector store.

    Uploaded files are also saved to data/documents so they are part of
    the corpus when ingest.py is run again.
    """

    files = [file for file in files or [] if file.filename]
    urls = [url.strip() for url in urls or [] if url.strip()]

    if not files and not urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one file or URL."
        )

    documents = []
    failed: list[IngestError] = []

    for upload in files:
        filename = Path(upload.filename).name
        suffix = Path(filename).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            failed.append(IngestError(
                source=filename,
                error=f"Unsupported file type '{suffix}'. "
                      f"Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
            ))
            continue

        raw = upload.file.read(MAX_UPLOAD_BYTES + 1)

        if len(raw) > MAX_UPLOAD_BYTES:
            failed.append(IngestError(
                source=filename,
                error=f"File is larger than {MAX_UPLOAD_BYTES} bytes."
            ))
            continue

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            failed.append(IngestError(
                source=filename, error="File is not valid UTF-8 text."
            ))
            continue

        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        saved_path = DOCUMENTS_DIR / filename
        saved_path.write_text(content, encoding="utf-8")

        documents.append(make_document(
            content=content,
            source=f"data/documents/{filename}",
            filename=filename
        ))

    for url in urls:
        if urlparse(url).scheme not in {"http", "https"}:
            failed.append(IngestError(
                source=url, error="Only http(s) URLs are supported."
            ))
            continue

        try:
            documents.append(fetch_url(url))
        except httpx.HTTPError as error:
            failed.append(IngestError(
                source=url, error=f"Could not fetch URL: {error}"
            ))

    empty = [doc for doc in documents if not doc.page_content]
    for doc in empty:
        failed.append(IngestError(
            source=doc.metadata["source"], error="Document has no text."
        ))
    documents = [doc for doc in documents if doc.page_content]

    if not documents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "None of the inputs could be ingested.",
                "failed": [item.model_dump() for item in failed]
            }
        )

    chunks = split_documents(documents)
    index_documents(chunks)

    per_source = Counter(chunk.metadata["source"] for chunk in chunks)

    return IngestResponse(
        ingested=[
            IngestedDocument(
                source=doc.metadata["source"],
                filename=doc.metadata["filename"],
                chunks=per_source[doc.metadata["source"]]
            )
            for doc in documents
        ],
        failed=failed,
        total_chunks=len(chunks)
    )


# --------------------------------------------------------------------------
# GET /documents
# --------------------------------------------------------------------------

@router.get("/documents", response_model=DocumentsResponse, tags=["Documents"])
def documents() -> DocumentsResponse:
    """List the documents currently in the vector store."""

    indexed = list_indexed_documents()

    return DocumentsResponse(
        total_documents=len(indexed),
        total_chunks=sum(item["chunks"] for item in indexed),
        documents=indexed
    )


@router.delete("/documents", tags=["Documents"])
def delete_document(source: str) -> dict:
    """Remove a document (by its `source` from GET /documents) from the
    index, and delete the file if it lives in data/documents."""

    removed = delete_source(source)

    if removed == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No indexed document with source '{source}'."
        )

    if source.startswith("data/documents/"):
        (DOCUMENTS_DIR / Path(source).name).unlink(missing_ok=True)

    return {"source": source, "chunks_removed": removed}


# --------------------------------------------------------------------------
# POST /feedback
# --------------------------------------------------------------------------

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Feedback"]
)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record a thumbs up/down (and optional comment) for an answer."""

    with _recent_lock:
        original = _recent_queries.get(request.query_id)

    if original is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown query_id. Use the query_id returned by POST /query."
        )

    save_feedback({**request.model_dump(), **original})

    return FeedbackResponse(
        message="Feedback recorded.",
        query_id=request.query_id,
        rating=request.rating
    )
