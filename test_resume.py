import time

from lcvr.llm import call_claude, label_key, load_done_keys

# the same locker document as before
colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]
things = ["umbrella", "violin", "lantern", "teapot", "compass", "kite", "helmet", "globe"]
lines = []
for i in range(1, 401):
    lines.append(f"Locker {i:03d} contains a {colors[i % 8]} {things[(i * 3) % 8]}.")
document = "\n".join(lines)

questions = [
    ("locker_137", "What is in locker 137?"),
    ("locker_042", "What is in locker 042?"),
    ("locker_300", "What is in locker 300?"),
    ("locker_007", "What is in locker 007?"),
]

done = load_done_keys()                       # read the log ONCE, at the start
print(f"{len(done)} calls already done in the log")

for question_id, question in questions:
    labels = {"run": "resume_test", "system": "lc", "corpus": "lockers", "question_id": question_id}

    if label_key(labels) in done:
        print(f"skip {question_id} (already answered)")
        continue

    answer, record = call_claude(question, labels, context=document,
                                 cache_context=True, max_tokens=20)
    print(f"asked {question_id}: {answer}")
    time.sleep(5)                             # only for this test: gives time to press Ctrl+C