"""Step 03: Compare phrases with embedding similarity.

Teaching goal:
    Embeddings capture meaning, not just exact words. We embed a few short
    phrases, normalize the vectors, and use dot product as cosine similarity.

Requires:
    OPENROUTER_API_KEY in `.env`.
"""

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.embeddings import embed_texts  # noqa: E402
from rag.vector_store import normalize_vectors  # noqa: E402


PHRASES = [
    "vacation days for employees",
    "paid time off policy",
    "robot battery charging dock",
    "warehouse robot charger",
]


def main() -> None:
    """Embed phrases and print pairwise cosine similarities."""

    vectors = normalize_vectors(embed_texts(PHRASES))

    print("Cosine similarity between phrases:")
    print("(Higher means more similar meaning.)\n")

    for i, left in enumerate(PHRASES):
        for j, right in enumerate(PHRASES):
            if j <= i:
                continue
            similarity = float(np.dot(vectors[i], vectors[j]))
            print(f"{similarity: .3f}  |  {left!r}  <->  {right!r}")


if __name__ == "__main__":
    main()

