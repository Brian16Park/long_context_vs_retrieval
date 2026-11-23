"""Corpora: groups of paragraphs that many questions share, and how they look in a prompt."""


def render_paragraph(paragraph):
    """How one paragraph appears in EVERY prompt, long-context and RAG alike: a title line, then the text."""
    return f"# {paragraph['title']}\n{paragraph['text']}"


def render_corpus(paragraphs):
    """Paragraphs in order, separated by a blank line."""
    return "\n\n".join(render_paragraph(p) for p in paragraphs)