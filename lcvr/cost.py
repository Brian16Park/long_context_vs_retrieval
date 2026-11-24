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


def call_estimate(input_tokens, output_tokens, prices=HAIKU_5_5):
    """Rough cost of one uncached call, before running it."""
    tier = "long" if input_tokens > prices["long_prompt_threshold"] else "short"
    p = prices[tier]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def lc_estimate(corpus_tokens, n_questions, output_tokens=0, prices=HAIKU_5_5):
    """Rough cost of asking n questions against one cached corpus, before running it:
    one cache write, then n - 1 cache reads, plus `output_tokens` per reply.
    Leaves out the few tokens of each question."""
    tier = "long" if corpus_tokens > prices["long_prompt_threshold"] else "short"
    p = prices[tier]
    return (corpus_tokens * (p["cache_write_5m"] + (n_questions - 1) * p["cache_read"])
            + n_questions * output_tokens * p["output"]) / 1_000_000