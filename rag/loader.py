"""Load local Markdown documents for the workshop.

RAG starts with documents. In a real company those documents might live in
Google Drive, Notion, Confluence, PDFs, or a database. For a one-hour workshop,
plain Markdown files are easier to inspect and debug.
"""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Document:
    """A single source document.

    Attributes:
        filename: The short file name, such as "employee_handbook.md". We keep
            this because the model must cite where an answer came from.
        text: The full Markdown text.
    """

    filename: str
    text: str


def _strip_markdown_comments(text: str) -> str:
    """Remove HTML comments from Markdown before retrieval.

    The source files include teaching comments for humans reading the repo.
    Retrieval should search the actual company content, not those instructor
    notes, so we remove `<!-- ... -->` blocks when loading documents.
    """

    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()


def load_markdown_documents(data_dir: str | Path) -> list[Document]:
    """Read every Markdown file in a folder.

    Args:
        data_dir: Folder containing `.md` files.

    Returns:
        A list of Document objects sorted by filename, so workshop output is
        stable and easy to compare between learners.
    """

    folder = Path(data_dir)

    # Sorting makes demos deterministic: the same files appear in the same order
    # on every laptop.
    markdown_files = sorted(folder.glob("*.md"))

    documents: list[Document] = []
    for path in markdown_files:
        documents.append(
            Document(
                filename=path.name,
                # UTF-8 is the normal text encoding for Markdown files. We also
                # strip teaching-only comments so chunks contain company facts.
                text=_strip_markdown_comments(path.read_text(encoding="utf-8")),
            )
        )

    return documents
