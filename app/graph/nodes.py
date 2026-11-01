
import re
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import CORPUS_DESCRIPTION, TOP_K
from app.graph.state import GraphState, Source
from app.services.llm import get_llm
from app.services.vector_store import (
    list_indexed_documents,
    retrieve_documents
)
from app.services.web_search import search_web


QueryType = Literal[
    "conceptual", "how-to", "troubleshooting", "api-reference", "chitchat"
]

FALLBACK_ANSWER = (
    "I don't know. I couldn't find information about this in the indexed "
    "documentation, so I'd rather not guess. Try rephrasing the question "
    "or ingest documentation that covers this topic."
)

ANSWER_STYLE = {
    "conceptual": "Explain the concept clearly, then mention why it matters.",
    "how-to": "Give step-by-step instructions and include code from the "
              "context when it helps.",
    "troubleshooting": "Identify the likely cause first, then the fix.",
    "api-reference": "Be precise about names, parameters, status codes "
                     "and return values."
}


# --------------------------------------------------------------------------
# Structured LLM outputs
# --------------------------------------------------------------------------

class QueryAnalysis(BaseModel):
    """Analysis of a user question before retrieval."""

    standalone_question: str = Field(
        description="The user's question rewritten so it is understandable "
                    "without the chat history. Same as the original if it "
                    "is not a follow-up."
    )
    query_type: QueryType = Field(
        description="The type of question. Use 'chitchat' ONLY when the "
                    "message is clearly a greeting (hi, hello), thanks or "
                    "goodbye, or a question about the assistant itself or "
                    "which documents/files/topics it has. Everything else "
                    "gets one of the other types and is searched, including "
                    "technical questions that seem unrelated to the corpus "
                    "and unclear or single-word messages (e.g. 'apple'), so "
                    "the system can say honestly that it is not covered."
    )
    search_query: str = Field(
        description="A search query optimised for semantic retrieval over "
                    "technical documentation: expand abbreviations, add "
                    "relevant synonyms and technical terms."
    )


class GradeDocument(BaseModel):
    """Relevance grade for one retrieved chunk."""

    relevance: Literal["relevant", "irrelevant"] = Field(
        description="'relevant' if the chunk contains information that helps "
                    "answer the question, otherwise 'irrelevant'."
    )


class RewrittenQuery(BaseModel):
    """A new search query after retrieval failed."""

    search_query: str = Field(
        description="A different search query that is more likely to "
                    "retrieve relevant documentation."
    )


class GroundednessCheck(BaseModel):
    """Whether an answer is supported by its context."""

    grounded: bool = Field(
        description="True if every factual claim in the answer is "
                    "supported by the context."
    )
    reason: str = Field(
        description="Short explanation. List unsupported claims if any."
    )


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

ANALYZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You prepare user questions for retrieval over a technical "
     f"documentation corpus: {CORPUS_DESCRIPTION}. Only clear greetings, "
     "thanks, goodbyes and questions about the assistant or its documents "
     "are 'chitchat'. When in doubt, do not use 'chitchat': unclear, "
     "single-word or off-topic messages must still be searched."),
    ("human",
     "Chat history (may be empty):\n{history}\n\n"
     "User question: {question}\n\n"
     "Resolve any references to the chat history, classify the question, "
     "and write a search query for it.")
])

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You grade whether a retrieved documentation chunk is relevant to a "
     "user question. A chunk is relevant if it contains keywords, concepts "
     "or examples that help answer the question, even partially. It does "
     "not need to answer the whole question. Be strict about chunks that "
     "are on a different topic."),
    ("human", "Question: {question}\n\nChunk:\n{document}")
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Retrieval over a technical documentation corpus returned no relevant "
     "chunks. Write a new search query for the same question. Do not repeat "
     "the previous queries: try different terminology, broader or more "
     "specific concepts, or the underlying technical term."),
    ("human",
     "Question: {question}\n\nQueries already tried:\n{previous}")
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical documentation assistant. Answer the question "
     "using ONLY the numbered context passages. Rules:\n"
     "- Cite every claim with the passage number in square brackets, "
     "e.g. [1] or [1][3].\n"
     "- If the context only partly answers the question, answer that part "
     "and say what is not covered.\n"
     "- Never add facts that are not in the context.\n"
     "- Style: {style}"),
    ("human",
     "Context:\n{context}\n\n"
     "Question: {question}\n"
     "{feedback}")
])

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check whether an answer is grounded in the provided context. "
     "Statements that the context does not cover, or that say information "
     "is missing, are fine. Mark the answer as not grounded only if it "
     "makes factual claims that the context does not support."),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
])

CHITCHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical documentation assistant. You answer questions by "
     "searching the indexed documents listed below and citing them. Right "
     "now the user sent a greeting, small talk, or a question about you or "
     "the corpus. Reply briefly and friendly to what they actually said "
     "(only greet back if they greeted you). If they ask which files or "
     "topics you have, list the documents below. Do not answer technical "
     "questions from your own knowledge; invite the user to ask them so you "
     "can search the documents.\n\n"
     "Indexed documents:\n{corpus}"),
    ("human",
     "Chat history (may be empty):\n{history}\n\n"
     "User message: {question}")
])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def format_history(history: list) -> str:
    if not history:
        return "(none)"

    return "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}"
        for turn in history
    )


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{index}] (source: {doc.metadata.get('filename')}"
        f"{' > ' + doc.metadata['section'] if doc.metadata.get('section') else ''})\n"
        f"{doc.page_content}"
        for index, doc in enumerate(documents, start=1)
    )


def build_sources(documents: list[Document], answer: str) -> list[Source]:
    """Return the passages the answer cites (all of them if it cites none)."""

    cited = {int(number) for number in re.findall(r"\[(\d+)\]", answer)}

    sources = [
        Source(
            id=index,
            source=doc.metadata.get("source", ""),
            filename=doc.metadata.get("filename", ""),
            section=doc.metadata.get("section", ""),
            score=doc.metadata.get("score"),
            snippet=doc.page_content[:300]
        )
        for index, doc in enumerate(documents, start=1)
    ]

    return [s for s in sources if s["id"] in cited] or sources


def chunk_key(doc: Document) -> tuple:
    """Identity of a chunk, used to de-duplicate merged search results."""

    return (
        doc.metadata.get("source"),
        doc.metadata.get("chunk_index"),
        doc.page_content
    )


def filter_relevant(question: str, documents: list[Document]) -> list[Document]:
    """LLM-grade each document (concurrently) and keep the relevant ones."""

    if not documents:
        return []

    grader = get_llm(fast=True).with_structured_output(GradeDocument)

    grades: list[GradeDocument] = grader.batch([
        GRADE_PROMPT.format_messages(
            question=question,
            document=doc.page_content
        )
        for doc in documents
    ])

    return [
        doc for doc, grade in zip(documents, grades)
        if grade.relevance == "relevant"
    ]


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def analyze_query(state: GraphState) -> dict:
    """Node 1: resolve follow-ups, classify, and rewrite for retrieval."""

    analyzer = get_llm().with_structured_output(QueryAnalysis)

    result: QueryAnalysis = analyzer.invoke(
        ANALYZE_PROMPT.format_messages(
            history=format_history(state.get("chat_history", [])),
            question=state["question"]
        )
    )

    return {
        "standalone_question": result.standalone_question,
        "query_type": result.query_type,
        "rewritten_query": result.search_query,
        "search_queries": [result.search_query],
        "retry_count": 0,
        "generation_count": 0,
        "steps": [f"analyze_query: type={result.query_type}, "
                  f"query='{result.search_query}'"]
    }


def retrieve(state: GraphState) -> dict:
    """Node 2: top-k similarity search for the current search query.

    On the first attempt the standalone question is searched as well and
    the results are merged. Query expansion usually helps, but it can also
    push the best chunk out of the top-k (e.g. extra words like
    "constructor arguments" matched near-duplicate sections). Searching
    both means a bad expansion cannot hide what the plain question finds.
    Retries search only the rewritten query: the question's chunks were
    already graded in the first round.
    """

    query = state.get("rewritten_query") or state["question"]
    documents = retrieve_documents(query=query, k=TOP_K)
    step = f"retrieve: {len(documents)} chunks for '{query}'"

    question = state.get("standalone_question") or state["question"]

    if state.get("retry_count", 0) == 0 and question != query:
        seen = {chunk_key(doc) for doc in documents}
        added = 0

        for doc in retrieve_documents(query=question, k=TOP_K):
            if chunk_key(doc) not in seen:
                seen.add(chunk_key(doc))
                documents.append(doc)
                added += 1

        step += f" + {added} new for the original question"

    return {
        "documents": documents,
        "steps": [step]
    }


def grade_documents(state: GraphState) -> dict:
    """Node 3: LLM-grade each chunk and keep only the relevant ones.

    Chunks are graded against the standalone question, not the search
    query, so a poor rewrite cannot make irrelevant chunks look relevant.
    """

    documents = state.get("documents", [])
    question = state.get("standalone_question") or state["question"]

    relevant = filter_relevant(question, documents)

    return {
        "relevant_documents": relevant,
        "steps": [f"grade_documents: {len(relevant)}/{len(documents)} relevant"]
    }


def rewrite_query(state: GraphState) -> dict:
    """Fallback path: write a new search query and count the retry."""

    rewriter = get_llm(fast=True).with_structured_output(RewrittenQuery)
    previous = state.get("search_queries", [])

    result: RewrittenQuery = rewriter.invoke(
        REWRITE_PROMPT.format_messages(
            question=state.get("standalone_question") or state["question"],
            previous="\n".join(f"- {query}" for query in previous)
        )
    )

    retry_count = state.get("retry_count", 0) + 1

    return {
        "rewritten_query": result.search_query,
        "search_queries": [result.search_query],
        "retry_count": retry_count,
        "steps": [f"rewrite_query (retry {retry_count}): "
                  f"'{result.search_query}'"]
    }


def generate(state: GraphState) -> dict:
    """Node 4: answer from the relevant chunks, with numbered citations."""

    documents = state["relevant_documents"]
    query_type = state.get("query_type", "conceptual")

    feedback = ""
    if state.get("grounded") is False and state.get("grounding_feedback"):
        feedback = (
            "\nYour previous answer contained claims not supported by the "
            f"context: {state['grounding_feedback']}\n"
            "Answer again, sticking strictly to the context."
        )

    response = get_llm().invoke(
        GENERATE_PROMPT.format_messages(
            style=ANSWER_STYLE.get(query_type, ANSWER_STYLE["conceptual"]),
            context=format_context(documents),
            question=state.get("standalone_question") or state["question"],
            feedback=feedback
        )
    )

    # gpt-oss sometimes cites in its own format, with full-width brackets
    # and optional line ranges (【1】, 【1†L1-L4】); normalise both to [1].
    answer = re.sub(r"【(\d+)(?:†[^】]*)?】", r"[\1]", response.content.strip())
    generation_count = state.get("generation_count", 0) + 1

    return {
        "answer": answer,
        "sources": build_sources(documents, answer),
        "generation_count": generation_count,
        "steps": [f"generate: attempt {generation_count}"]
    }


def check_hallucination(state: GraphState) -> dict:
    """Bonus (Self-RAG): verify the answer is supported by its context."""

    checker = get_llm(fast=True).with_structured_output(GroundednessCheck)

    result: GroundednessCheck = checker.invoke(
        GROUNDEDNESS_PROMPT.format_messages(
            context=format_context(state["relevant_documents"]),
            answer=state["answer"]
        )
    )

    return {
        "grounded": result.grounded,
        "grounding_feedback": "" if result.grounded else result.reason,
        "status": "answered" if result.grounded else "ungrounded",
        "steps": [f"check_hallucination: grounded={result.grounded}"]
    }


def web_search(state: GraphState) -> dict:
    """Bonus: the corpus has nothing relevant, so try the web (Tavily).

    Web results go through the same LLM grading as corpus chunks. If none
    survive, the graph still ends in the "I don't know" fallback.
    """

    question = state.get("standalone_question") or state["question"]

    try:
        results = search_web(question)
    except Exception as error:
        return {
            "relevant_documents": [],
            "web_search_used": True,
            "steps": [f"web_search: failed ({type(error).__name__})"]
        }

    relevant = filter_relevant(question, results)

    return {
        "relevant_documents": relevant,
        "web_search_used": True,
        "steps": [f"web_search: {len(relevant)}/{len(results)} results relevant"]
    }


def respond_chitchat(state: GraphState) -> dict:
    """Greetings and small talk: reply directly, no retrieval needed."""

    try:
        corpus = "\n".join(
            f"- {doc['filename']} ({doc['chunks']} chunks)"
            for doc in list_indexed_documents()
        ) or "(no documents indexed yet)"
    except Exception:
        corpus = "(document list unavailable)"

    # One small LLM call, so it can answer "hi", "thanks", "which files do
    # you have?" or "what can you do?" naturally, using the real file list.
    response = get_llm(fast=True).invoke(
        CHITCHAT_PROMPT.format_messages(
            corpus=corpus,
            history=format_history(state.get("chat_history", [])),
            question=state["question"]
        )
    )

    answer = response.content.strip()

    return {
        "answer": answer,
        "sources": [],
        "grounded": None,
        "status": "chitchat",
        "steps": ["respond_chitchat: no retrieval needed"]
    }


def fallback(state: GraphState) -> dict:
    """Retries exhausted with nothing relevant: say so instead of guessing."""

    return {
        "answer": FALLBACK_ANSWER,
        "sources": [],
        "grounded": None,
        "status": "no_relevant_documents",
        "steps": [f"fallback: no relevant documents after "
                  f"{state.get('retry_count', 0)} retries"]
    }
