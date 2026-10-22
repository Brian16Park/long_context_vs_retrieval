"""Turn one logged call into dollars, using the frozen price sheet."""
from lcvr.prices import HAIKU_5_5


def call_cost(record, prices=HAIKU_5_5):
    """Dollar cost of one line of logs/calls.jsonl.

    Assumes all cache writes are 5-minute writes (the only kind we use).
    """
    if not record.get("ok"):
        return 0.0                            # failed attempt: no tokens were billed

    # Which tier? Decided by the WHOLE prompt length of this request.
    prompt_tokens = (record["input_tokens"]
                     + record["cache_write_tokens"]
                     + record["cache_read_tokens"])
    if prompt_tokens > prices["long_prompt_threshold"]:
        tier = "long"
    else:
        tier = "short"
    p = prices[tier]

    # Each kind of token x its price (prices are per million tokens).
    dollars = (record["input_tokens"] * p["input"]
               + record["cache_write_tokens"] * p["cache_write_5m"]
               + record["cache_read_tokens"] * p["cache_read"]
               + record["output_tokens"] * p["output"]) / 1_000_000
    return dollars