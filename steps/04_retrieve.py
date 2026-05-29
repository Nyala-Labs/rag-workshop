"""Step 04: Retrieve relevant chunks without generating an answer.

Teaching goal:
    Separate retrieval from generation. Before asking the LLM to write anything,
    we can inspect which chunks the vector search found.

Requires:
    OPENROUTER_API_KEY in `.env`.
"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunker import chunk_documents  # noqa: E402
from rag.embeddings import embed_text, embed_texts  # noqa: E402
from rag.loader import load_markdown_documents  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402


QUESTION = "How long does a Skylark 3 battery last?"


def main() -> None:
    """Build the store and print top matching chunks."""

    documents = load_markdown_documents("data")
    chunks = chunk_documents(documents, chunk_size=90, overlap=25)
    chunk_vectors = embed_texts([chunk.text for chunk in chunks])
    store = VectorStore(chunks, chunk_vectors)

    question_vector = embed_text(QUESTION)
    results = store.search(question_vector, top_k=3)

    print(f"Question: {QUESTION}\n")
    print("Top retrieved chunks, before generation:\n")

    for rank, result in enumerate(results, start=1):
        print("=" * 78)
        print(f"Rank {rank} | score={result.score:.3f} | source={result.chunk.source}")
        print("-" * 78)
        print(result.chunk.text)
        print()


if __name__ == "__main__":
    main()

