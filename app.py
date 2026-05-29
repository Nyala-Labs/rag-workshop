"""Interactive RAG chatbot over the Nimbus Robotics documents.

This file is intentionally small. It builds the same pipeline used in the
teaching steps, then loops over user questions in the terminal.

Requires:
    OPENROUTER_API_KEY in `.env`.
"""

from rag.pipeline import Pipeline


def main() -> None:
    """Start a simple terminal chatbot."""

    print("Building the Nimbus Robotics RAG index...")
    print("This embeds the document chunks once, then reuses them for each question.\n")
    pipeline = Pipeline.from_data_dir("data")

    print("Ask about Nimbus Robotics. Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        answer = pipeline.ask(question)
        print(f"\nRAG: {answer}\n")


if __name__ == "__main__":
    main()

