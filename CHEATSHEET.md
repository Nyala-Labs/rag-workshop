# RAG Cheatsheet

<!--
This cheatsheet is plain English on purpose. It is meant to help beginners keep
the vocabulary straight while they inspect the code.
-->

## Glossary

**LLM**  
A large language model. It predicts useful text from a prompt.

**Private data**  
Information that is not generally available on the public internet, such as
company policies, internal product docs, support tickets, or customer records.

**RAG**  
Retrieval-Augmented Generation. Retrieve relevant information first, then give it
to the model as context for generation.

**Document**  
One source file or record, such as `employee_handbook.md`.

**Chunk**  
A smaller piece of a document. Chunks make search more precise.

**Overlap**  
Repeated words between neighboring chunks. Overlap reduces the chance that an
important idea gets split apart.

**Embedding**  
A list of numbers representing the meaning of text.

**Vector**  
Another word for a list of numbers. Embeddings are vectors.

**Cosine similarity**  
A score that compares the direction of two vectors. In this repo, vectors are
normalized first, so dot product equals cosine similarity.

**Vector store**  
A system that stores embeddings and searches for the nearest ones. In this repo,
the vector store is only NumPy arrays, dot product, and `argsort`.

**Top-k**  
The best `k` retrieval results. If `top_k=3`, the retriever returns three chunks.

**Context**  
The retrieved text inserted into the model prompt.

**Citation**  
The source filename included in the answer, such as `skylark3_faq.md`.

**Hallucination**  
An answer that sounds confident but is not supported by the provided facts.

## RAG vs Fine-Tuning

**RAG** is best when facts change often or must come from source documents. You
can add, remove, or edit documents without retraining a model.

**Fine-tuning** is best when you want to change behavior, format, style, or a
repeated task pattern. It is usually not the best way to inject fresh company
facts.

Short version:

- Use RAG to give the model knowledge.
- Use fine-tuning to teach the model a behavior.
- Many production systems use both.

## The RAG Loop In This Repo

1. Load Markdown documents from `data/`.
2. Split each document into overlapping chunks.
3. Embed every chunk.
4. Normalize vectors and store them in a NumPy matrix.
5. Embed the user's question.
6. Use dot product to score chunk similarity.
7. Use `argsort` to pick the top chunks.
8. Put those chunks into the prompt.
9. Ask the model to answer only from that context and cite filenames.

