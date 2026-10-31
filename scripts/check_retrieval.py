"""Manual check of the retriever (no LLM calls).

    python -m scripts.check_retrieval "How do I validate an email?"
"""

import sys

from app.services.vector_store import retrieve_documents


def main():
    query = " ".join(sys.argv[1:]) or "How do I validate incoming request data?"

    print("\nQuery:", query)

    for i, document in enumerate(retrieve_documents(query), start=1):
        print(f"\nResult {i} (score {document.metadata['score']})")
        print("Source:", document.metadata.get("source"))
        print("Section:", document.metadata.get("section"))
        print(document.page_content[:400])


if __name__ == "__main__":
    main()
