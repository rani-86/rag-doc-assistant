"""Download the corpus: five FastAPI tutorial pages from the official docs.

    python -m scripts.fetch_corpus

The FastAPI docs are Markdown files in the fastapi/fastapi GitHub repo, but
they use MkDocs-specific syntax. This script turns each page into plain,
self-contained Markdown:

- `{* ../../docs_src/... *}` include tags are replaced by the actual Python
  example code, so code examples are part of the corpus.
- `/// tip` ... `///` admonitions become "**Tip:**" paragraphs.
- Heading anchors (`{ #path-parameters }`) and inline HTML (<dfn>, <abbr>,
  <img>) are removed.

Then run `python ingest.py` to index the files.
"""

import posixpath
import re
from pathlib import Path

import httpx

from app.config import DOCUMENTS_DIR

# Pinned to a branch for simplicity; use a release tag for exact reproducibility.
REF = "master"
RAW_BASE = f"https://raw.githubusercontent.com/fastapi/fastapi/{REF}/"
DOCS_BASE = "docs/en/"            # include paths are relative to this
SITE_BASE = "https://fastapi.tiangolo.com/"

PAGES = {
    "fastapi_path_parameters.md": "tutorial/path-params",
    "fastapi_query_parameters.md": "tutorial/query-params",
    "fastapi_request_body.md": "tutorial/body",
    "fastapi_handling_errors.md": "tutorial/handling-errors",
    "fastapi_dependencies.md": "tutorial/dependencies/index",
}

# [ \t]* rather than \s*, so the regex never swallows the following newline.
INCLUDE = re.compile(r"^\{\*[ \t]*(\S+)[^*]*\*\}[ \t]*$", re.MULTILINE)
ADMONITION = re.compile(r"^/{3,}\s*(\w+)?(?:\s*\|\s*(.*))?\s*$")
HEADING_ANCHOR = re.compile(r"[ \t]*\{[ \t]*#[\w-]+[ \t]*\}[ \t]*$", re.MULTILINE)
INLINE_HTML = re.compile(r"</?(?:dfn|abbr|span|kbd)[^>]*>")
IMAGE = re.compile(r"<img[^>]*>")


def fetch(client: httpx.Client, path: str) -> str:
    response = client.get(RAW_BASE + path)
    response.raise_for_status()
    return response.text


def inline_code(client: httpx.Client, markdown: str) -> str:
    """Replace include tags with the referenced example file."""

    def replace(match: re.Match) -> str:
        source_path = posixpath.normpath(
            posixpath.join(DOCS_BASE, match.group(1))
        )
        code = fetch(client, source_path).strip()
        return f"```python\n{code}\n```"

    return INCLUDE.sub(replace, markdown)


def clean(markdown: str) -> str:
    """Convert MkDocs syntax to plain Markdown, leaving code blocks intact."""

    lines = []
    in_code = False

    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            lines.append(line)
            continue

        if in_code:
            lines.append(line)
            continue

        admonition = ADMONITION.match(line)
        if admonition:
            kind, title = admonition.groups()
            if kind and kind != "tab":
                label = kind.capitalize()
                lines.append(f"**{label}{' – ' + title if title else ''}:**")
            continue

        line = IMAGE.sub("", INLINE_HTML.sub("", line))
        lines.append(line)

    text = HEADING_ANCHOR.sub("", "\n".join(lines))

    # Collapse 3+ blank lines left behind by removed markup.
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def main():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=20, follow_redirects=True) as client:
        for filename, page in PAGES.items():
            markdown = fetch(client, f"{DOCS_BASE}docs/{page}.md")
            markdown = clean(inline_code(client, markdown))

            # Keep the public URL of the page right under the title.
            url = SITE_BASE + page.removesuffix("/index").rstrip("/") + "/"
            title, _, body = markdown.partition("\n")
            markdown = f"{title}\n\nSource: {url}\n{body}"

            (DOCUMENTS_DIR / filename).write_text(markdown, encoding="utf-8")
            print(f"Saved {filename} ({len(markdown)} chars) from {url}")


if __name__ == "__main__":
    main()
