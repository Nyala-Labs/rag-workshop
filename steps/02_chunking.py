"""Step 02: Load and chunk documents.

Teaching goal:
    Show that RAG begins before any model call. We first turn long documents
    into smaller overlapping chunks that can be searched later.

Requires:
    No API key. This step is pure local Python.
"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunker import chunk_documents  # noqa: E402
from rag.loader import load_markdown_documents  # noqa: E402


def main() -> None:
    """Print a few chunks so learners can inspect the sliding window."""

    documents = load_markdown_documents("data")
    chunks = chunk_documents(documents, chunk_size=70, overlap=20)

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} chunks with chunk_size=70 and overlap=20.\n")

    for chunk in chunks[:4]:
        print("=" * 78)
        print(f"{chunk.chunk_id} starts at word {chunk.start_word}")
        print("-" * 78)
        print(chunk.text)
        print()


if __name__ == "__main__":
    main()

