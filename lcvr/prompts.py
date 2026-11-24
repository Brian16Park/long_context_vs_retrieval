"""Prompt text shared by every system, so the only thing that differs is which documents they see.

Prompt layout (it must stay in this order for caching to work):
    [instructions: system prompt]  [documents]  [question]
Long context sends the whole corpus as the documents; RAG sends the retrieved paragraphs.

Two prompt styles. A run uses one style for every system, so they stay comparable:
    answer_only : reply with the answer alone (the pilot's style)
    reasoning   : up to three short sentences of reasoning, then a last line "Answer: ..."
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

# The same instructions with a little room to reason first. Only the middle sentence changes.
REASONING_INSTRUCTIONS = (
    "Answer the question using the documents below. First, in at most three short sentences, "
    "reason step by step and name the facts you use. Then, on a last line of its own, write "
    "'Answer:' followed by only the answer: a short phrase such as a name, a date or a number. "
    "If the documents don't settle it, give your best guess."
)

CLOSED_BOOK_REASONING_INSTRUCTIONS = (
    "Answer the question. First, in at most three short sentences, reason step by step and "
    "name the facts you use. Then, on a last line of its own, write 'Answer:' followed by only "
    "the answer: a short phrase such as a name, a date or a number. If you're not sure, give "
    "your best guess."
)

MAX_ANSWER_TOKENS = 50      # answers are a few words; this just caps a rambling reply
MAX_REASONING_TOKENS = 400  # replies average about 120 tokens; this only stops a runaway reply

STYLES = ("answer_only", "reasoning")
DEFAULT_STYLE = "answer_only"

# Expected reply length in tokens, for the cost estimate shown before a run
OUTPUT_ESTIMATE = {"answer_only": 10, "reasoning": 130}

SETTINGS = {   # (style, closed book?) -> (instructions, reply token cap)
    ("answer_only", False): (INSTRUCTIONS, MAX_ANSWER_TOKENS),
    ("answer_only", True): (CLOSED_BOOK_INSTRUCTIONS, MAX_ANSWER_TOKENS),
    ("reasoning", False): (REASONING_INSTRUCTIONS, MAX_REASONING_TOKENS),
    ("reasoning", True): (CLOSED_BOOK_REASONING_INSTRUCTIONS, MAX_REASONING_TOKENS),
}


def prompt_settings(style, closed_book=False):
    """(instructions, max_tokens) for a prompt style."""
    if style not in STYLES:
        raise ValueError(f"unknown prompt style {style!r}: expected one of {STYLES}")
    return SETTINGS[(style, closed_book)]


def style_label(style):
    """The label a call gets for its prompt style.

    answer_only gets none, so calls logged before there were styles (the smoke test and
    the pilot) keep their label keys and still count as done.
    """
    return {} if style == DEFAULT_STYLE else {"prompt": style}


def question_text(question):
    return f"Question: {question['question']}"
