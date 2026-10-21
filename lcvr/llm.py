"""One function for every model call in the project: call_claude()."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-5-5"
LOG_FILE = Path("logs") / "calls.jsonl"
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504, 529)

client = anthropic.Anthropic(max_retries=0)   # no silent retries, we do our own


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def save_to_log(record):
    LOG_FILE.parent.mkdir(exist_ok=True)      # create the logs/ folder if it's missing
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def is_retryable(error):
    if isinstance(error, anthropic.APIConnectionError):   # network problem or timeout
        return True
    if isinstance(error, anthropic.APIStatusError):        # the API answered with an error code
        return error.status_code in RETRYABLE_STATUS_CODES
    return False


def call_claude(question, labels, context=None, cache_context=False,
                system=None, max_tokens=100, max_attempts=8):
    """Ask Claude one question. Every attempt is logged with `labels`.

    question      : the question text (always goes last)
    labels        : dict saying what this call is for, e.g.
                    {"run": "pilot", "system": "lc", "corpus_tokens": 64000, "question_id": "q17"}
    context       : optional document text placed before the question
    cache_context : True = ask the API to cache everything up to the end of the context
    system        : optional instructions (sent before everything else)
    Returns (answer_text, log_record).
    """
    # Build the request: context first (optionally cached), question last
    content = []
    if context is not None:
        context_block = {"type": "text", "text": context}
        if cache_context:
            context_block["cache_control"] = {"type": "ephemeral"}
        content.append(context_block)
    content.append({"type": "text", "text": question})

    request = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": content}],
    }
    if system is not None:
        request["system"] = system

    # Send it: try, log every attempt, retry temporary failures
    for attempt in range(1, max_attempts + 1):
        start = time.perf_counter()
        try:
            response = client.messages.create(**request)
        except anthropic.APIError as error:
            latency = time.perf_counter() - start
            save_to_log({
                "time": now_utc(),
                **labels,
                "model": MODEL,
                "attempt": attempt,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "latency_s": round(latency, 3),
            })
            if not is_retryable(error) or attempt == max_attempts:
                raise
            wait = min(2 ** attempt, 60)
            print(f"  attempt {attempt} failed ({type(error).__name__}), retrying in {wait}s")
            time.sleep(wait)
            continue

        # Success: record what we were billed for, and the answer
        latency = time.perf_counter() - start
        u = response.usage
        answer = "".join(block.text for block in response.content if block.type == "text")
        record = {
            "time": now_utc(),
            **labels,
            "model": MODEL,
            "attempt": attempt,
            "ok": True,
            "question": question,
            "answer": answer,
            "input_tokens": u.input_tokens,
            "cache_write_tokens": u.cache_creation_input_tokens,
            "cache_read_tokens": u.cache_read_input_tokens,
            "output_tokens": u.output_tokens,
            "stop_reason": response.stop_reason,
            "latency_s": round(latency, 3),
        }
        save_to_log(record)
        return answer, record