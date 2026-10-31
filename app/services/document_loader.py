
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    BASE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS
)


# Split on #, ## and ### headings; deeper headings stay inside the section.
HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")


def html_to_markdown(html: str) -> str:
    """Convert HTML to plain text while keeping headings as Markdown
    headings, so HTML pages get the same section-aware chunking."""

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    for level in (1, 2, 3):
        for heading in soup.find_all(f"h{level}"):
            heading.string = f"\n{'#' * level} {heading.get_text(strip=True)}\n"

    for block in soup.find_all("pre"):
        block.string = f"\n```\n{block.get_text()}\n```\n"

    text = soup.get_text("\n")
    lines = [line.rstrip() for line in text.splitlines()]

    # Collapse runs of blank lines left behind by the HTML structure.
    cleaned = []
    for line in lines:
        if line or (cleaned and cleaned[-1]):
            cleaned.append(line)

    return "\n".join(cleaned).strip()


def make_document(content: str, source: str, filename: str) -> Document:
    """Normalise raw file content into a LangChain Document."""

    suffix = Path(filename).suffix.lower() or ".txt"

    if suffix in {".html", ".htm"}:
        content = html_to_markdown(content)

    return Document(
        page_content=content.strip(),
        metadata={
            "source": source,
            "filename": filename,
            "file_type": suffix
        }
    )


def load_documents(directory: Path = DOCUMENTS_DIR) -> list[Document]:
    """Load Markdown, text and HTML files from the documents directory."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {directory}"
        )

    documents = []

    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        document = load_file(file_path)

        if document.page_content:
            documents.append(document)

    if not documents:
        raise ValueError(
            f"No supported documents found in {directory}"
        )

    print(f"Loaded {len(documents)} documents.")

    return documents


def load_file(file_path: Path) -> Document:
    """Load a single file from disk."""

    try:
        source = file_path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        source = file_path.as_posix()

    return make_document(
        content=file_path.read_text(encoding="utf-8"),
        source=source,
        filename=file_path.name
    )


def fetch_url(url: str, timeout: float = 15.0) -> Document:
    """Fetch a page (HTML, Markdown or plain text) from a URL."""

    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()

    path_name = Path(urlparse(url).path).name or urlparse(url).netloc
    content_type = response.headers.get("content-type", "")

    if "html" in content_type:
        filename = path_name if path_name.endswith((".html", ".htm")) \
            else f"{path_name}.html"
    elif Path(path_name).suffix.lower() in {".md", ".txt"}:
        filename = path_name
    else:
        filename = f"{path_name}.txt"

    return make_document(
        content=response.text,
        source=url,
        filename=filename
    )


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading path, body) sections.

    Written instead of LangChain's MarkdownHeaderTextSplitter because that
    splitter strips whitespace from every line, which destroys the
    indentation of Python code examples. This version keeps lines verbatim
    and ignores '#' lines inside code blocks (Python comments).
    """

    sections = []
    path: list[str] = []
    body: list[str] = []
    in_code = False

    def flush():
        content = "\n".join(body).strip("\n")
        if content.strip():
            sections.append((" > ".join(path), content))
        body.clear()

    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code

        heading = None if in_code else HEADING.match(line)

        if heading:
            flush()
            level = len(heading.group(1))
            path[:] = path[:level - 1] + [heading.group(2)]
        else:
            body.append(line)

    flush()

    return sections


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    """Split documents into section-aware, overlapping chunks.

    1. Split on Markdown headings so a chunk never mixes two topics and
       each chunk knows which section it came from.
    2. Split long sections further by paragraph, then line, then sentence,
       so code examples and lists stay intact where possible.
    3. Prefix each chunk with its heading path, which gives the embedding
       model context that the chunk body alone often lacks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n```", "\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for document in documents:
        document_chunks = []

        for section_path, section_text in split_markdown_sections(
            document.page_content
        ):
            for piece in text_splitter.split_text(section_text):
                content = f"{section_path}\n\n{piece}" if section_path else piece

                document_chunks.append(
                    Document(
                        page_content=content,
                        metadata={
                            **document.metadata,
                            "section": section_path
                        }
                    )
                )

        # Index per document so IDs stay stable when other files change.
        for index, chunk in enumerate(document_chunks):
            chunk.metadata["chunk_index"] = index

        chunks.extend(document_chunks)

    print(f"Generated {len(chunks)} chunks.")

    return chunks
