"""Price every call in logs/calls.jsonl and print the total."""
import json

from lcvr.cost import call_cost

total = 0.0
count = 0            # successful calls
failed = 0           # failed attempts: no tokens were billed, so they cost $0
skipped = 0          # half-written lines left by a crash
with open("logs/calls.jsonl", encoding="utf-8") as f:
    for line in f:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:          # half-written line from a crash: skip it
            skipped += 1
            continue
        if not record.get("ok"):              # failed attempt: it has no token counts
            failed += 1
            continue
        cost = call_cost(record)
        total += cost
        count += 1
        print(f"{record.get('run', '-'):<12} {record.get('question_id', '-'):<11} "
              f"write={record['cache_write_tokens']:>7}  read={record['cache_read_tokens']:>7}  "
              f"cost=${cost:.7f}")

print(f"\nTotal for {count} calls: ${total:.6f}")
print(f"Failed attempts (cost $0): {failed}   Half-written lines skipped: {skipped}")