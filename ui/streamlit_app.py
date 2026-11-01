
"""Minimal chat UI for the RAG API.

Run the API first (uvicorn app.main:app), then:
    streamlit run ui/streamlit_app.py
"""

import os
from uuid import uuid4

import httpx
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Docs Assistant", page_icon="📚")
st.title("📚 Technical Documentation Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
    st.session_state.messages = []


def send_feedback(query_id: str, rating: str) -> None:
    httpx.post(
        f"{API_URL}/feedback",
        json={"query_id": query_id, "rating": rating},
        timeout=10
    )
    st.toast("Thanks for the feedback!")


with st.sidebar:
    st.header("Corpus")

    # Shown once after a rerun (e.g. "Removed x.md").
    if notice := st.session_state.pop("notice", None):
        st.success(notice)

    corpus = {"documents": []}
    try:
        corpus = httpx.get(f"{API_URL}/documents", timeout=10).json()
        st.caption(
            f"{corpus['total_documents']} documents, "
            f"{corpus['total_chunks']} chunks"
        )
        for doc in corpus["documents"]:
            st.write(f"- {doc['filename']} ({doc['chunks']} chunks)")
    except httpx.HTTPError:
        st.error(f"API not reachable at {API_URL}")

    uploads = st.file_uploader(
        "Add documents", type=["md", "txt", "html"], accept_multiple_files=True
    )
    if uploads and st.button("Ingest"):
        response = httpx.post(
            f"{API_URL}/ingest",
            files=[("files", (f.name, f.getvalue())) for f in uploads],
            timeout=120
        )
        if response.is_success:
            st.success(f"Indexed {response.json()['total_chunks']} chunks")
        else:
            st.error(response.json().get("detail"))

    if corpus["documents"]:
        filenames = {d["source"]: d["filename"] for d in corpus["documents"]}
        to_remove = st.selectbox(
            "Remove a document",
            options=list(filenames),
            format_func=filenames.get,
            index=None,
            placeholder="Choose a document"
        )
        if to_remove and st.button("Remove"):
            response = httpx.delete(
                f"{API_URL}/documents", params={"source": to_remove}, timeout=30
            )
            if response.is_success:
                removed = response.json()["chunks_removed"]
                st.session_state.notice = (
                    f"Removed {filenames[to_remove]} ({removed} chunks)"
                )
                st.rerun()
            else:
                st.error(response.json().get("detail"))

    if st.button("New conversation"):
        st.session_state.session_id = str(uuid4())
        st.session_state.messages = []
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**[{source['id']}] {source['filename']}** "
                        f"— {source['section']}\n\n> {source['snippet']}"
                    )

        if message.get("trace"):
            with st.expander("Workflow trace"):
                st.code("\n".join(message["trace"]))

        if message.get("query_id"):
            up, down, _ = st.columns([1, 1, 8])
            up.button("👍", key=f"up-{message['query_id']}",
                      on_click=send_feedback, args=(message["query_id"], "up"))
            down.button("👎", key=f"down-{message['query_id']}",
                        on_click=send_feedback, args=(message["query_id"], "down"))


if question := st.chat_input("Ask about the documentation"):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Retrieving, grading and generating..."):
        try:
            response = httpx.post(
                f"{API_URL}/query",
                json={
                    "question": question,
                    "session_id": st.session_state.session_id
                },
                timeout=180
            )
            body = response.json()

            if response.is_success:
                content = body["answer"]
                if body.get("used_web_search") and body["status"] != "no_relevant_documents":
                    content = ("🌐 *Not found in the indexed documentation; "
                               "this answer is based on web search results.*"
                               "\n\n" + content)
                if body["status"] == "ungrounded":
                    content += ("\n\n⚠️ *The answer may contain claims not "
                                "supported by the documents.*")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": content,
                    "sources": body["sources"],
                    "trace": body["steps"],
                    "query_id": body["query_id"]
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error {response.status_code}: {body.get('detail')}"
                })
        except httpx.HTTPError as error:
            st.session_state.messages.append({
                "role": "assistant", "content": f"API error: {error}"
            })

    st.rerun()
