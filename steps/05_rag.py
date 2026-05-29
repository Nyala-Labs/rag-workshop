"""Step 05: Compare plain LLM vs full RAG.

Teaching goal:
    Same model, same question. The plain LLM lacks private context, while RAG
    retrieves the relevant Nimbus Robotics chunk and can answer with a source.

Requires:
    OPENROUTER_API_KEY in `.env`.
"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.pipeline import Pipeline, ask_plain_llm  # noqa: E402


QUESTION = "How many Nimbus Days of vacation do Nimbus Robotics employees get?"


def main() -> None:
    """Run the side-by-side demo."""

    print(f"Question: {QUESTION}\n")

    print("=" * 78)
    print("Plain LLM, no private context")
    print("-" * 78)
    print(ask_plain_llm(QUESTION))
    print()

    print("=" * 78)
    print("Full RAG, with retrieved Nimbus Robotics context")
    print("-" * 78)
    pipeline = Pipeline.from_data_dir("data")
    print(pipeline.ask(QUESTION))


if __name__ == "__main__":
    main()

