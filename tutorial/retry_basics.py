import json
import time
from datetime import datetime, timezone

import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(max_retries=0) # no silent retries, we do our own

LOG_FILE = "calls.jsonl"

# --- Build the document ---
colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]
things = ["umbrella", "violin", "lantern", "teapot", "compass", "kite", "helmet", "globe"]

lines = []
for i in range(1, 401):
    color = colors[i % 8]
    thing = things[(i * 3) % 8]
    lines.append(f"Locker {i:03d} contains a {color} {thing}.")
document = "\n".join(lines)

# --- ask() is the same as before ---
def ask(question):
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=20,
        thinking={"type": "disabled"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": document, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": question},
                ],
            }
        ],
    )
    return response

def save_to_log(record):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def now_utc():
    return datetime.now(timezone.utc).isoformat()

# --- decide whether an error is worth retrying ---
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504, 529)

def is_retryable(error):
    if isinstance(error, anthropic.APIConnectionError):   # network problem or timeout
        return True
    if isinstance(error, anthropic.APIStatusError):        # the API answered with an error code
        return error.status_code in RETRYABLE_STATUS_CODES
    return False

# --- ask, and if it fails with a temporary error, wait and try again ---
def ask_with_retries(question, max_attempts=5):
    for attempt in range(1, max_attempts + 1):
        start = time.perf_counter()
        try:
            response = ask(question)
        except anthropic.APIError as error:
            latency = time.perf_counter() - start
            save_to_log({
                "time": now_utc(),
                "question": question,
                "attempt": attempt,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "latency_s": round(latency, 3),
            })
            if not is_retryable(error) or attempt == max_attempts:
                raise                                 # give up: re-raise the same error
            wait = min(2 ** attempt, 60)              # 2, 4, 8, 16... seconds, never over 60
            print(f"  attempt {attempt} failed ({type(error).__name__}), retrying in {wait}s")
            time.sleep(wait)
            continue                                  # go back to the top of the loop

        # if we get here, the call worked
        latency = time.perf_counter() - start
        u = response.usage
        answer = "".join(block.text for block in response.content if block.type == "text")
        record = {
            "time": now_utc(),
            "question": question,
            "attempt": attempt,
            "ok": True,
            "answer": answer,
            "input_tokens": u.input_tokens,
            "cache_write_tokens": u.cache_creation_input_tokens,
            "cache_read_tokens": u.cache_read_input_tokens,
            "output_tokens": u.output_tokens,
            "latency_s": round(latency, 3),
        }
        save_to_log(record)
        return answer, record

# --- much shorter now ---
for question in ["What is in locker 137?", "What is in locker 042?"]:
    answer, record = ask_with_retries(question)
    print(record)