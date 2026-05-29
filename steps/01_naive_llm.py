"""Step 01: Ask the LLM without private context.

Teaching goal:
    A normal chat model has broad public knowledge, but it does not know the
    fictional private facts inside our Nimbus Robotics documents. This step
    should fail gracefully on a made-up company policy question.

Requires:
    OPENROUTER_API_KEY in `.env`.
"""

from pathlib import Path
import sys


# When learners run `python steps/01_naive_llm.py`, Python starts inside the
# steps/ folder. This line adds the project root so `import rag` works.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.pipeline import ask_plain_llm  # noqa: E402


QUESTION = "How many Nimbus Days of vacation do Nimbus Robotics employees get?"


def main() -> None:
    """Run the no-context baseline."""

    print("Question:")
    print(QUESTION)
    print("\nPlain LLM answer with no retrieved context:")
    print(ask_plain_llm(QUESTION))


if __name__ == "__main__":
    main()

