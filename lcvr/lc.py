"""The long-context system: the whole corpus in the prompt, cached, then one question at a time."""
from lcvr.corpus import render_corpus
from lcvr.llm import call_claude, label_key
from lcvr.prompts import INSTRUCTIONS, MAX_ANSWER_TOKENS, question_text


def lc_labels(corpus, qid, run):
    return {
        "run": run,
        "system": "lc",
        "dataset": corpus["dataset"],
        "seed": corpus["seed"],
        "corpus_tokens": corpus["target_tokens"],
        "corpus_id": corpus["corpus_id"],
        "question_id": qid,
    }


def ask_corpus(corpus, questions_by_qid, paragraphs_by_pid, run, done, limit=None):
    """Ask a corpus's first `limit` questions one after another, with the corpus cached.

    The first call writes the cache and every later call reads it, as long as each
    comes within 5 minutes of the one before. Questions this run already answered
    (`done`, from load_done_keys) are skipped, so a crashed run picks up where it
    stopped. Returns how many questions were asked now.
    """
    text = render_corpus(paragraphs_by_pid[p] for p in corpus["pids"])
    asked = 0
    for qid in corpus["qids"][:limit]:
        labels = lc_labels(corpus, qid, run)
        if label_key(labels) in done:
            continue
        call_claude(question_text(questions_by_qid[qid]), labels, context=text,
                    cache_context=True, system=INSTRUCTIONS, max_tokens=MAX_ANSWER_TOKENS)
        asked += 1
    return asked