# 60-Minute Facilitator Guide

<!--
This guide is written for a live beginner workshop. The goal is not to cover
every production concern. The goal is to make the core RAG loop visible.
-->

## 0-5 min: Frame the Problem

Explain the workshop title: **RAG: Powering LLMs with Private Data**.

Key message: an LLM can write fluent text, but it does not automatically know
your company's private handbook, product FAQ, or security policy.

Show the fictional `data/` folder and point out that Nimbus Robotics is invented.

## 5-12 min: Plain LLM Baseline

Run:

```bash
python steps/01_naive_llm.py
```

Ask the room what happened. The model may say it does not know, or it may guess.
Either outcome is useful: without private context, it cannot be trusted to answer
the Nimbus-specific question.

## 12-22 min: Loading and Chunking

Run:

```bash
python steps/02_chunking.py
```

Explain:

- We chunk because whole documents are too broad for precise retrieval.
- We overlap because important meaning can sit across a chunk boundary.
- This step needs no API key because it is just local text processing.

Open `rag/chunker.py` and highlight the sliding word window.

## 22-32 min: Embeddings

Run:

```bash
python steps/03_embeddings.py
```

Explain:

- An embedding is a vector, which is a list of numbers.
- Similar meanings point in similar directions.
- Cosine similarity compares direction, not exact wording.

Use the output to compare phrase pairs like "vacation days" and "paid time off".

## 32-42 min: Vector Store Retrieval

Run:

```bash
python steps/04_retrieve.py
```

Open `rag/vector_store.py` and highlight:

- `normalize_vectors`
- matrix dot query vector
- `np.argsort(scores)[::-1][:top_k]`

Emphasize that this is a tiny vector store built from NumPy, not a database.

## 42-52 min: Full RAG

Run:

```bash
python steps/05_rag.py
```

Point out the side-by-side structure:

- Same model.
- Same question.
- Plain LLM has no private source.
- RAG retrieves context and answers with `employee_handbook.md`.

Open `rag/pipeline.py` and show the comments labeled retrieve, augment, generate.

## 52-58 min: Interactive App

Run:

```bash
python app.py
```

Try questions:

- How long does the Skylark 3 battery last?
- What is Fogbound data?
- How long before laptops lock?
- What is Nimbus Robotics' stock price?

The last question should not be answered from context.

## 58-60 min: Wrap-Up

Summarize the core idea:

RAG does not train the model. It retrieves relevant private text and places it in
the prompt so the model can answer with grounding and citations.

Mention production topics learners can explore later: better document parsing,
chunk tuning, reranking, access control, evaluation, caching, and monitoring.

