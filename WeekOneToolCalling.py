"""
WEEK 1 EXERCISE — Tool calling, the core primitive of every agent.
-------------------------------------------------------------------
Goal: ask the model a question it CAN'T answer on its own (sales numbers),
and watch it ask YOUR code to run a tool, then use the result to answer.

This is the whole foundation of agents. Read every comment.

SETUP (run these in your terminal first):
    pip install anthropic python-dotenv
    export ANTHROPIC_API_KEY="your-key-here"     # Mac/Linux
    # on Windows PowerShell:  $env:ANTHROPIC_API_KEY="your-key-here"

Then:  python week1_tool_calling.py
"""

from dotenv import load_dotenv
import anthropic
import os

load_dotenv()

# Debug: Check if API key is loaded
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")
print(f"[DEBUG] API key loaded: {api_key[:20]}...")

# The client reads your API key from the ANTHROPIC_API_KEY environment
# variable automatically. Never hardcode the key in the file.
client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# STEP 1: The ACTUAL tool — just normal Python. Real function, fake data.
# The model can NEVER run this itself. It can only ASK you to run it.
# ---------------------------------------------------------------------------
def get_book_sales(title: str) -> int:
    fake_sales_db = {
        "Dash Dragon": 142,
        "Menopause Wellness Journal": 88,
    }
    # .get() returns 0 if the title isn't in our fake DB
    return fake_sales_db.get(title, 0)


# ---------------------------------------------------------------------------
# STEP 2: The tool DEFINITION you send to the model.
# The model reads the `description` to decide WHEN to use it, and the
# `input_schema` to know WHAT arguments to provide. Write descriptions well —
# this is "prompt engineering for tools."
# ---------------------------------------------------------------------------
tools = [
    {
        "name": "get_book_sales",
        #"description": "Returns the number of copies sold for a given book title.",
        "description": "The Author of both the books is PixelNPlanner.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The exact title of the book, e.g. 'Dash Dragon'.",
                }
            },
            "required": ["title"],
        },
    }
]


def main():
    # The conversation starts with just the user's question.
    messages = [
        {"role": "user", "content": "how many pages are there in Menopause Wellness Journal book?"}
    ]

    # -----------------------------------------------------------------------
    # STEP 3: First API call. We send the question AND the tool definitions.
    # -----------------------------------------------------------------------
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    # ⭐ CHECKPOINT MOMENT ⭐
    # `stop_reason` tells you WHY the model stopped.
    #   - "tool_use"   -> it wants you to run a tool (it did NOT answer yet)
    #   - "end_turn"   -> it answered directly with text (no tool needed)
    # This is exactly what you'll explain back to me. Print it and look:
    print(f"[stop_reason] {response.stop_reason}\n")

    # If the model just answered directly, print and stop.
    if response.stop_reason != "tool_use":
        print(response.content[0].text)
        return

    # -----------------------------------------------------------------------
    # STEP 4: The model asked for a tool. The response `content` is a LIST of
    # blocks — some may be text ("Let me check..."), one is a `tool_use` block.
    # We must find the tool_use block and read what the model wants.
    # -----------------------------------------------------------------------
    # First, we append the model's whole turn to the conversation history,
    # because the model needs to see its own request when we call again.
    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"[model wants tool] {block.name}({block.input})\n")

            # Run the REAL function with the arguments the model chose.
            if block.name == "get_book_sales":
                result = get_book_sales(block.input["title"])

            # Package the result as a tool_result, linked by the SAME id
            # the model used in its request (block.id). The id is how the
            # model matches your answer to its question.
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result),
            })

    # The tool result goes back as a NEW user message.
    messages.append({"role": "user", "content": tool_results})

    # -----------------------------------------------------------------------
    # STEP 5: Second API call. Now the model has the data and writes the
    # final natural-language answer. (Same tools passed in case it wants
    # to call again — in a real agent this whole thing runs in a loop.)
    # -----------------------------------------------------------------------
    final = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    print("[final answer]")
    print(final.content[0].text)


if __name__ == "__main__":
    main()