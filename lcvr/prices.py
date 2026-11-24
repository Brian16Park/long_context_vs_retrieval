"""Price sheet for the research, frozen on one date.

Every dollar figure in the report is computed from logged token counts x these prices.
Do not edit the numbers after the sweep starts. If prices change, add a NEW sheet
(with its own date) instead of changing this one.

All prices are US dollars per 1,000,000 tokens.
"""

PRICE_SHEET_DATE = "2026-10-21"
SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"

HAIKU_5_5 = {
    "model": "claude-haiku-5-5",
    # A request whose prompt is OVER this many tokens pays the "long" prices for
    # everything in that request (input, cache, and output). Exactly 100,000 is still "short".
    # Prompt length = input + cache write + cache read tokens.
    "long_prompt_threshold": 100_000,
    "short": {
        "input": 0.10,
        "cache_write_5m": 0.125,
        "cache_write_1h": 0.20,
        "cache_read": 0.01,
        "output": 0.50,
    },
    "long": {
        "input": 0.50,
        "cache_write_5m": 0.625,
        "cache_write_1h": 1.00,
        "cache_read": 0.05,
        "output": 2.50,
    },
    # Batch API: 50% off. Applied later, only to calls we actually send through batch.
    "batch_multiplier": 0.50,
}


# Local GPU work for RAG (embedding the corpus, embedding each question, reranking) is priced
# as GPU time at an on-demand cloud rate. Our timings come from an RTX 3060 Ti; this is the
# common cloud instance with a comparable inference GPU (one NVIDIA T4). Added 2026-11-24.
GPU = {
    "instance": "AWS EC2 g4dn.xlarge (1x NVIDIA T4), on demand, us-east-1",
    "dollars_per_hour": 0.526,
    "date": "2026-11-24",
    "source": "https://instances.vantage.sh/aws/ec2/g4dn.xlarge",
}
