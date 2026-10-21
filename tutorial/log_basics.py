import json                             
import time                               
from datetime import datetime, timezone

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

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

# --- Append one record to the log file ---
def save_to_log(record):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# --- ask, time, log ---
for question in ["What is in locker 137?", "What is in locker 042?"]:
    start = time.perf_counter()  #start the stopwatch
    response = ask(question)
    latency = time.perf_counter() - start  #stop it

    u = response.usage
    answer = "".join(block.text for block in response.content if block.type == "text")

    record = { #everything we want to keep
        "time": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "input_tokens": u.input_tokens,
        "cache_write_tokens": u.cache_creation_input_tokens,
        "cache_read_tokens": u.cache_read_input_tokens,
        "output_tokens": u.output_tokens,
        "latency_s": round(latency, 3),
    }
    save_to_log(record)
    print(record)   #print the whole record