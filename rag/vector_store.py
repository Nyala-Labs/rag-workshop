"""A tiny vector store implemented with plain NumPy.

This is the main math lesson of the workshop.

Cosine similarity normally means:
    dot(a, b) / (length(a) * length(b))

If we first normalize every vector to length 1, the denominator becomes 1, so:
    cosine_similarity = dot(normalized_a, normalized_b)

That is why this vector store normalizes vectors on insert and normalizes the
query vector before search. Then retrieval is just:
    scores = matrix dot query
    top_indexes = argsort(scores)
"""

from dataclasses import dataclass

import numpy as np

from rag.chunker import Chunk


@dataclass(frozen=True)
class SearchResult:
    """One retrieval result from the vector store."""

    chunk: Chunk
    score: float


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Return vectors scaled to unit length.

    A vector's length is also called its L2 norm. We divide each row by its own
    norm so every row points in the same direction but has length 1.
    """

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    # Avoid division by zero. Embedding models should not return all-zero
    # vectors, but this guard keeps the teaching code robust.
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


class VectorStore:
    """Store chunks and their embedding vectors in memory."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        self.chunks = chunks

        # Normalize once when we build the store. Search then becomes a fast dot
        # product between the matrix of chunk vectors and the query vector.
        self.vectors = normalize_vectors(vectors).astype(np.float32)

    def search(self, query_vector: np.ndarray, *, top_k: int = 3) -> list[SearchResult]:
        """Return the chunks whose embeddings are closest to the query."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if len(self.chunks) == 0:
            return []

        normalized_query = normalize_vectors(query_vector).reshape(-1)

        # Because both sides are normalized, dot product equals cosine
        # similarity. Higher scores mean the vectors point in more similar
        # directions, which usually means the texts are semantically related.
        scores = self.vectors @ normalized_query

        # argsort returns indexes from lowest score to highest score. We reverse
        # that order and keep only the top_k best matches.
        best_indexes = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in best_indexes
        ]

