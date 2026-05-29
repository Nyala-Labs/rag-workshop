"""Split documents into overlapping chunks.

Why chunk?
    Embedding and chat models have limited context windows, and retrieval works
    better when each searchable unit is focused. A full handbook might contain
    vacation policy, payroll, security, and lunch information. A smaller chunk
    lets the vector search return the exact part that matches a question.

Why overlap?
    If we cut text into separate blocks with no overlap, an important sentence
    might be split across a boundary. Overlap repeats a few words from the
    previous chunk so related ideas stay together more often.
"""

from dataclasses import dataclass

from rag.loader import Document


@dataclass(frozen=True)
class Chunk:
    """A small piece of a source document.

    Attributes:
        text: The chunk text that will be embedded and later placed in a prompt.
        source: The original filename, used for citations.
        chunk_id: A stable label that makes debugging easier.
        start_word: The index of the first word in the original document.
    """

    text: str
    source: str
    chunk_id: str
    start_word: int


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 90,
    overlap: int = 25,
) -> list[Chunk]:
    """Create sliding-window word chunks from one document.

    The window moves through the document by `chunk_size - overlap` words.
    Example with chunk_size=10 and overlap=3:
        chunk 1 = words 0..9
        chunk 2 = words 7..16
        chunk 3 = words 14..23

    Args:
        document: The loaded Markdown document.
        chunk_size: Maximum number of words in each chunk.
        overlap: Number of words repeated from the previous chunk.

    Returns:
        A list of Chunk objects.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = document.text.split()
    step = chunk_size - overlap
    chunks: list[Chunk] = []

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]

        # Stop if there are no words. This can happen for an empty document.
        if not chunk_words:
            break

        chunk_number = len(chunks) + 1
        chunks.append(
            Chunk(
                text=" ".join(chunk_words),
                source=document.filename,
                chunk_id=f"{document.filename}::chunk-{chunk_number}",
                start_word=start,
            )
        )

        # If we reached the end, there is no need to create another shorter
        # overlapping chunk.
        if end >= len(words):
            break

    return chunks


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 90,
    overlap: int = 25,
) -> list[Chunk]:
    """Chunk many documents and return one flat list."""

    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(
            chunk_document(document, chunk_size=chunk_size, overlap=overlap)
        )
    return all_chunks

