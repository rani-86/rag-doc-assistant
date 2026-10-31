
from app.services.document_loader import (
    html_to_markdown,
    make_document,
    split_documents
)


MARKDOWN = """# Guide

Intro text.

## Install

Run pip install.

## Usage

Call the function.
"""


def test_chunks_follow_markdown_sections():
    doc = make_document(MARKDOWN, "data/documents/guide.md", "guide.md")
    chunks = split_documents([doc])

    sections = [chunk.metadata["section"] for chunk in chunks]
    assert sections == ["Guide", "Guide > Install", "Guide > Usage"]

    # The heading path is prepended so the embedding has context.
    assert chunks[1].page_content.startswith("Guide > Install")
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]


def test_long_sections_are_split_with_size_limit():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    doc = make_document(f"# Big\n\n{body}", "big.md", "big.md")

    chunks = split_documents([doc], chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 300 + len("Big\n\n") for c in chunks)


def test_html_headings_become_markdown_sections():
    html = (
        "<html><body><nav>menu</nav><h1>API</h1><p>Overview.</p>"
        "<h2>Errors</h2><p>Return 404.</p><script>x()</script></body></html>"
    )

    text = html_to_markdown(html)

    assert "# API" in text
    assert "## Errors" in text
    assert "menu" not in text and "x()" not in text


def test_code_indentation_and_comments_are_preserved():
    markdown = (
        "# API\n\n## Example\n\n```python\n"
        "# this is a Python comment, not a heading\n"
        "def read_item(q=None):\n"
        "    if q:\n"
        "        return {\"q\": q}\n"
        "```\n"
    )
    doc = make_document(markdown, "api.md", "api.md")

    chunks = split_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].metadata["section"] == "API > Example"
    assert "    if q:\n        return" in chunks[0].page_content
    assert "# this is a Python comment" in chunks[0].page_content


def test_deeper_headings_reset_correctly():
    markdown = "# A\n\n## B\n\ntext b\n\n### C\n\ntext c\n\n## D\n\ntext d\n"
    doc = make_document(markdown, "x.md", "x.md")

    sections = [c.metadata["section"] for c in split_documents([doc])]

    assert sections == ["A > B", "A > B > C", "A > D"]
