"""Small teaching package for the RAG workshop.

The modules in this package are intentionally simple:

- loader.py reads plain Markdown files.
- chunker.py splits documents into overlapping word chunks.
- embeddings.py calls OpenRouter's embedding API.
- vector_store.py stores vectors in plain NumPy arrays.
- pipeline.py connects retrieval, prompt construction, and generation.

Nothing here tries to hide the RAG steps behind a framework. The goal is for a
beginner to be able to open each file and see the moving parts.
"""

