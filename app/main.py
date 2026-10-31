
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.services.vector_store import count_chunks, get_vector_store

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model and open Chroma before the first request.
    get_vector_store()
    yield


app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description=(
        "Answers questions about technical documentation using a "
        "self-corrective LangGraph workflow: query analysis -> retrieval -> "
        "LLM document grading -> (query rewrite and retry) -> generation "
        "with citations -> hallucination check."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)


@app.get("/health", tags=["Health"])
def health() -> dict:
    return {"status": "ok", "indexed_chunks": count_chunks()}
