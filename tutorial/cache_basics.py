from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()          # load ANTHROPIC_API_KEY from .env
client = Anthropic()

# Build a long "document" (~4,000 tokens)
colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]
things = ["umbrella", "violin", "lantern", "teapot", "compass", "kite", "helmet", "globe"]

lines = []
for i in range(1, 401):
    color = colors[i % 8]
    thing = things[(i * 3) % 8]
    lines.append(f"Locker {i:03d} contains a {color} {thing}.")
document = "\n".join(lines)

print(lines[0])    # peek at the first two lines
print(lines[1])
print()

# --- Send the document + a question, asking the API to cache the document ---
def ask(question):
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=20,
        thinking={"type": "disabled"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": document,
                        "cache_control": {"type": "ephemeral"},  # cache everything up to here
                    },
                    {
                        "type": "text",
                        "text": question,                        # this part changes every time
                    },
                ],
            }
        ],
    )
    return response

# --- Ask two different questions about the SAME document ---
for question in ["What is in locker 137?", "What is in locker 042?"]:
    response = ask(question)
    u = response.usage
    answer = "".join(block.text for block in response.content if block.type == "text")
    print(question, "->", answer)
    print("   input:", u.input_tokens,
          "| cache write:", u.cache_creation_input_tokens,
          "| cache read:", u.cache_read_input_tokens,
          "| output:", u.output_tokens)