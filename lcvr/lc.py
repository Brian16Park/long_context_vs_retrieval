"""The long-context system: the whole corpus in the prompt, cached, then one question at a time."""
from lcvr.corpus import render_corpus
from lcvr.llm import call_claude, label_key
from lcvr.prompts import DEFAULT_STYLE, prompt_settings, question_text, style_label


def lc_labels(corpus, qid, run, style=DEFAULT_STYLE):
    return {
        "run": run,
        "system": "lc",
        "dataset": corpus["dataset"],
        "seed": corpus["seed"],
        "corpus_tokens": corpus["target_tokens"],
        "corpus_id": corpus["corpus_id"],
        "question_id": qid,
        **style_label(style),
    }


def ask_corpus(corpus, questions_by_qid, paragraphs_by_pid, run, done, limit=None,
               style=DEFAULT_STYLE, stop=None):
    """Ask a corpus's first `limit` questions one after another, with the corpus cached.

    The first call writes the cache and every later call reads it, as long as each
    comes within 5 minutes of the one before. Questions this run already answered
    (`done`, from load_done_keys) are skipped, so a crashed run picks up where it
    stopped. `style` is the prompt style (lcvr/prompts.py). `stop` (a threading.Event)
    ends the corpus early, before its next question: when several corpora run at once
    and one fails for good, the others stop too. Returns how many questions were asked now.
    """
    instructions, max_tokens = prompt_settings(style)
    text = render_corpus(paragraphs_by_pid[p] for p in corpus["pids"])
    asked = 0
    for qid in corpus["qids"][:limit]:
        if stop is not None and stop.is_set():
            break
        labels = lc_labels(corpus, qid, run, style)
        if label_key(labels) in done:
            continue
        call_claude(question_text(questions_by_qid[qid]), labels, context=text,
                    cache_context=True, system=instructions, max_tokens=max_tokens)
        asked += 1
    return asked
