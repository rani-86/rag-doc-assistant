
from collections import Counter
from functools import lru_cache
from uuid import NAMESPACE_URL, uuid5

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PERSIST_DIRECTORY,
    TOP_K
)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local embedding model once per process."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8}
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Connect to the persistent ChromaDB collection."""

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(PERSIST_DIRECTORY)
    )


def chunk_id(chunk: Document) -> str:
    unique_text = (
        f"{chunk.metadata['source']}:"
        f"{chunk.metadata['chunk_index']}:"
        f"{chunk.page_content}"
    )

    return str(uuid5(NAMESPACE_URL, unique_text))


def index_documents(chunks: list[Document]) -> Chroma:
    """Embed and index chunks, replacing older versions of each source."""

    if not chunks:
        raise ValueError("No document chunks provided.")

    vector_store = get_vector_store()
    ids = [chunk_id(chunk) for chunk in chunks]

    # Upsert first: if embedding fails, the previous version stays intact.
    vector_store.add_documents(documents=chunks, ids=ids)

    # Then drop chunks from older versions of the same sources, so
    # re-ingesting an edited file does not leave stale chunks behind.
    new_ids = set(ids)

    for source in {chunk.metadata["source"] for chunk in chunks}:
        existing = vector_store.get(where={"source": source}, include=[])
        stale = [id_ for id_ in existing["ids"] if id_ not in new_ids]

        if stale:
            vector_store.delete(ids=stale)

    print(f"Indexed {len(chunks)} chunks in ChromaDB.")

    return vector_store


def retrieve_documents(query: str, k: int = TOP_K) -> list[Document]:
    """Retrieve the top-k chunks and record their cosine similarity."""

    # Chroma returns squared L2 distance. Embeddings are normalised, so
    # cosine similarity = 1 - distance / 2 exactly (range -1..1). This avoids
    # LangChain's relevance-score conversion, which warns (and logs every
    # chunk) whenever an unrelated query gives a score below 0.
    results = get_vector_store().similarity_search_with_score(
        query=query,
        k=k
    )

    documents = []

    for document, distance in results:
        document.metadata["score"] = round(1 - float(distance) / 2, 4)
        documents.append(document)

    return documents


def delete_source(source: str) -> int:
    """Remove every chunk of one source document; returns chunks removed."""

    vector_store = get_vector_store()
    ids = vector_store.get(where={"source": source}, include=[])["ids"]

    if ids:
        vector_store.delete(ids=ids)

    return len(ids)


def count_chunks() -> int:
    return get_vector_store()._collection.count()


def list_indexed_documents() -> list[dict]:
    """Summarise the corpus: one entry per source with its chunk count."""

    records = get_vector_store().get(include=["metadatas"])
    metadatas = records.get("metadatas") or []

    counts = Counter(metadata["source"] for metadata in metadatas)
    details = {}

    for metadata in metadatas:
        details.setdefault(metadata["source"], metadata)

    return [
        {
            "source": source,
            "filename": details[source].get("filename"),
            "file_type": details[source].get("file_type"),
            "chunks": counts[source]
        }
        for source in sorted(counts)
    ]
