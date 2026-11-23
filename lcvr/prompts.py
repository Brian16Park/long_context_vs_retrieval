"""Prompt text shared by every system, so the only thing that differs is which documents they see.

Prompt layout (it must stay in this order for caching to work):
    [instructions: system prompt]  [documents]  [question]
Long context sends the whole corpus as the documents; RAG sends the retrieved paragraphs.
"""

INSTRUCTIONS = (
    "Answer the question using the documents below. Reply with only the answer: "
    "a short phrase such as a name, a date or a number, with no explanation. "
    "If the documents don't settle it, give your best guess."
)

# The closed-book baseline sees no documents: it measures what the model already knows.
CLOSED_BOOK_INSTRUCTIONS = (
    "Reply with only the answer to the question: a short phrase such as a name, "
    "a date or a number, with no explanation. If you're not sure, give your best guess."
)

MAX_ANSWER_TOKENS = 50      # answers are a few words; this just caps a rambling reply


def question_text(question):
    return f"Question: {question['question']}"