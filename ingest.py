
from app.services.document_loader import load_documents, split_documents
from app.services.vector_store import count_chunks, index_documents


def main():
    print("\nStarting document ingestion...\n")

    # Step 1: Load documents from data/documents
    documents = load_documents()

    # Step 2: Split into section-aware chunks
    chunks = split_documents(documents)

    # Step 3: Embed and store in ChromaDB (replaces old versions of each file)
    index_documents(chunks)

    print("\nIngestion completed successfully!")
    print(f"Total indexed chunks: {count_chunks()}")


if __name__ == "__main__":
    main()
