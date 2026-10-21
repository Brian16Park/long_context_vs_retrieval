"""Step 1 check: can we call Claude Haiku 5.5 and see what the API bills us for?"""
import time

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()          # reads ANTHROPIC_API_KEY from the .env file in this folder
client = Anthropic()   # picks the key up automatically

MODEL = "claude-haiku-5-5"

start = time.perf_counter()
response = client.messages.create(
    model=MODEL,
    max_tokens=100,
    messages=[{"role": "user", "content": "In one sentence, what is retrieval-augmented generation?"}],
)
latency = time.perf_counter() - start

print("Answer:", response.content[0].text)
print()
print("What we were billed for (response.usage):")
u = response.usage
print("  input_tokens                :", u.input_tokens)
print("  cache_creation_input_tokens :", u.cache_creation_input_tokens)
print("  cache_read_input_tokens     :", u.cache_read_input_tokens)
print("  output_tokens               :", u.output_tokens)
print(f"  latency                     : {latency:.2f} s")