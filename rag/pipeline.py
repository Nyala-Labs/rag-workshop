"""End-to-end RAG pipeline.

The pipeline glues together the three core RAG stages:

1. Retrieve: find document chunks that match the user's question.
2. Augment: place those chunks into the model prompt as context.
3. Generate: ask the chat model to answer using only that context.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import request
from urllib.error import HTTPError

from dotenv import load_dotenv

from rag.chunker import Chunk, chunk_documents
from rag.embeddings import OPENROUTER_API_URL, _api_key, embed_text, embed_texts
from rag.loader import load_markdown_documents
from rag.vector_store import SearchResult, VectorStore


def chat_model_name() -> str:
    """Return the chat model used for demos.

    OpenRouter free model availability can change. The default is configurable
    in `.env` so a facilitator can choose a current `:free` chat model without
    editing code.
    """

    load_dotenv()
    return os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-oss-20b:free")


def _post_chat(payload: dict[str, Any]) -> dict[str, Any]:
    """Send one OpenAI-compatible chat request through OpenRouter."""

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{OPENROUTER_API_URL}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://example.com/rag-workshop",
            "X-Title": "RAG Workshop",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter chat failed: {exc.code} {details}") from exc


def ask_plain_llm(question: str) -> str:
    """Ask the chat model directly, without private documents.

    This is the baseline. It demonstrates that the model cannot reliably answer
    questions about fictional private facts unless we provide the facts.
    """

    response = _post_chat(
        {
            "model": chat_model_name(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful assistant. If you do not know the "
                        "answer, say that you do not know."
                    ),
                },
                {"role": "user", "content": question},
            ],
            "temperature": 0.2,
        }
    )
    return response["choices"][0]["message"]["content"]


class Pipeline:
    """A minimal RAG pipeline over local Markdown files."""

    def __init__(self, chunks: list[Chunk], store: VectorStore) -> None:
        self.chunks = chunks
        self.store = store

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str = "data",
        *,
        chunk_size: int = 90,
        overlap: int = 25,
    ) -> "Pipeline":
        """Load documents, chunk them, embed chunks, and build a vector store."""

        documents = load_markdown_documents(data_dir)
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        vectors = embed_texts([chunk.text for chunk in chunks])
        store = VectorStore(chunks, vectors)
        return cls(chunks=chunks, store=store)

    def retrieve(self, question: str, *, top_k: int = 3) -> list[SearchResult]:
        """Embed the question and retrieve the most relevant chunks."""

        question_vector = embed_text(question)
        return self.store.search(question_vector, top_k=top_k)

    def ask(self, question: str, *, top_k: int = 3) -> str:
        """Answer a question with Retrieval-Augmented Generation."""

        # RAG STAGE 1: RETRIEVE
        # Turn the question into an embedding and use cosine similarity to find
        # the most relevant private document chunks.
        results = self.retrieve(question, top_k=top_k)

        # RAG STAGE 2: AUGMENT
        # Place the retrieved chunks into the prompt. Each chunk includes its
        # source filename so the model can cite where the answer came from.
        context_blocks = []
        for index, result in enumerate(results, start=1):
            context_blocks.append(
                f"[Context {index} | source: {result.chunk.source}]\n"
                f"{result.chunk.text}"
            )
        context = "\n\n".join(context_blocks)

        system_prompt = (
            "You answer questions about Nimbus Robotics using ONLY the provided "
            "context. If the answer is not present in the context, say: "
            "\"I don't know from the provided context.\" Cite the source "
            "filename for every factual answer."
        )

        # RAG STAGE 3: GENERATE
        # Ask the model to write the final answer, constrained by the retrieved
        # context. This is where generation happens, but the knowledge comes
        # from the retrieved chunks rather than from model memory.
        response = _post_chat(
            {
                "model": chat_model_name(),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context}\n\n"
                            f"Question: {question}\n\n"
                            "Answer with a concise explanation and source filename."
                        ),
                    },
                ],
                "temperature": 0.1,
            }
        )

        return response["choices"][0]["message"]["content"]

