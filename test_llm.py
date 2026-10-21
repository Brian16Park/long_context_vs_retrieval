from lcvr.llm import call_claude

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
]

for question_id, question in questions:
    labels = {"run": "test", "system": "lc", "corpus": "lockers", "question_id": question_id}
    answer, record = call_claude(question, labels, context=document,
                                 cache_context=True, max_tokens=20)
    print(record)