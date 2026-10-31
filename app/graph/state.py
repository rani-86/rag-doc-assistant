
import operator
from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document


class Turn(TypedDict):
    question: str
    answer: str


class Source(TypedDict):
    id: int
    source: str
    filename: str
    section: str
    score: Optional[float]
    snippet: str


class GraphState(TypedDict, total=False):

    # --- Input (set by the caller) ---
    question: str                 # raw user question, never modified
    chat_history: List[Turn]      # previous turns in this session

    # --- Query analysis ---
    standalone_question: str      # question with follow-up references resolved;
                                  # documents are graded and answered against this
    query_type: str               # conceptual | how-to | troubleshooting | api-reference | chitchat
    rewritten_query: str          # current search query used by retrieval

    # Every search query tried so far. The reducer appends instead of
    # overwriting, so the rewrite node can see what already failed.
    search_queries: Annotated[List[str], operator.add]

    # --- Retrieval and grading ---
    documents: List[Document]           # raw top-k from the vector store
    relevant_documents: List[Document]  # survivors of LLM grading

    # Self-correction counters. retry_count is set to 0 once by
    # analyze_query and only incremented by rewrite_query, so the
    # retrieve -> grade -> rewrite loop is guaranteed to terminate.
    retry_count: int
    max_retries: int

    # Set by the optional web_search node so it runs at most once.
    web_search_used: bool

    # --- Generation and hallucination check ---
    answer: str
    sources: List[Source]
    generation_count: int
    max_generations: int
    grounded: Optional[bool]
    grounding_feedback: str       # why the last answer failed the check

    # --- Observability ---
    status: str                   # answered | ungrounded | no_relevant_documents | chitchat
    steps: Annotated[List[str], operator.add]  # execution trace
