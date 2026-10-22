import json

from lcvr.cost import call_cost

total = 0.0
count = 0
with open("logs/calls.jsonl", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        cost = call_cost(record)
        total += cost
        count += 1
        print(f"{record['run']:<12} {record['question_id']:<11} "
              f"write={record['cache_write_tokens']:>5}  read={record['cache_read_tokens']:>5}  "
              f"cost=${cost:.7f}")

print(f"\nTotal for {count} calls: ${total:.6f}")