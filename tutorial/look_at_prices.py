from lcvr.prices import HAIKU_5_5, PRICE_SHEET_DATE

print("Price sheet date:", PRICE_SHEET_DATE)
print("Threshold:", HAIKU_5_5["long_prompt_threshold"], "tokens")
print()
print("Short-prompt cache read:", HAIKU_5_5["short"]["cache_read"], "$ per million tokens")
print("Long-prompt cache read: ", HAIKU_5_5["long"]["cache_read"], "$ per million tokens")
print()

# How much does ONE token cost?
per_token = HAIKU_5_5["short"]["input"] / 1_000_000
print("One short-prompt input token costs $", per_token)