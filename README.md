# RAG: Powering LLMs with Private Data

Retrieval-Augmented Generation, or RAG, lets an LLM answer with private documents by retrieving relevant text, adding it to the prompt, and then generating a grounded answer.

## Pipeline

```text
User question
     |
     v
[1] Embed the question
     |
     v
[2] Retrieve similar chunks from a NumPy vector store
     |
     v
[3] Augment the prompt with those chunks and source filenames
     |
     v
[4] Generate an answer that cites the source file
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and add your OpenRouter API key.

## Run Order

Run these in order during the workshop:

```bash
python steps/01_naive_llm.py
python steps/02_chunking.py
python steps/03_embeddings.py
python steps/04_retrieve.py
python steps/05_rag.py
python app.py
```

`steps/02_chunking.py` does not need an API key. The other steps call OpenRouter.

## What To Notice

- `rag/chunker.py` shows why documents are split with overlap.
- `rag/vector_store.py` shows the math: normalize vectors, dot product for cosine similarity, and `argsort` for top-k retrieval.
- `rag/pipeline.py` labels the three RAG stages: retrieve, augment, generate.
- The documents are fictional Nimbus Robotics policies, so the model should not know the facts unless retrieval provides them.

