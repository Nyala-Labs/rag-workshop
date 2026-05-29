"""Call OpenRouter for text embeddings.

An embedding is a list of numbers that represents the meaning of text. Similar
ideas should have vectors that point in similar directions. The vector store
uses those directions for cosine similarity search.

This file uses Python's standard library `urllib` instead of `requests` so the
workshop has only two installed dependencies: numpy and python-dotenv.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import request
from urllib.error import HTTPError

import numpy as np
from dotenv import load_dotenv


OPENROUTER_API_URL = "https://openrouter.ai/api/v1"


def _api_key() -> str:
    """Load the OpenRouter API key from `.env` or the shell environment."""

    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Copy .env.example to .env and add a key."
        )
    return key


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send one JSON POST request to OpenRouter and parse the JSON response."""

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{OPENROUTER_API_URL}{path}",
        data=body,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            # These headers are optional for OpenRouter, but they help identify
            # a teaching/demo app in provider dashboards.
            "HTTP-Referer": "https://example.com/rag-workshop",
            "X-Title": "RAG Workshop",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        # Beginners often mistype keys or model names. Including the response
        # body makes those errors much easier to diagnose.
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter request failed: {exc.code} {details}") from exc


def embedding_model_name() -> str:
    """Return the embedding model name.

    The workshop defaults to OpenRouter's OpenAI-compatible name for
    text-embedding-3-small. Keep it configurable because free model availability
    can change over time.
    """

    load_dotenv()
    return os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of strings and return a 2D NumPy array.

    Shape:
        If we embed 12 chunks and the model returns 1536 numbers per chunk, the
        returned array has shape `(12, 1536)`.
    """

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    data = _post_json(
        "/embeddings",
        {
            "model": embedding_model_name(),
            "input": texts,
        },
    )

    # OpenAI-compatible embedding APIs return a `data` list where every item has
    # an `embedding` field. We convert it to float32 to keep memory usage small.
    vectors = [item["embedding"] for item in data["data"]]
    return np.array(vectors, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    """Embed one string and return one vector."""

    return embed_texts([text])[0]

