# engines/ai_model.py
"""
Standard AI wrapper for all GPT/Codex calls.
Keeps the rest of the system clean and modular.
"""

import os
from openai import OpenAI

# Load API key from environment variable
# Make sure you: export OPENAI_API_KEY="yourkey"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_ai(prompt: str, model: str = "gpt-4.1") -> str:
    """
    Sends a prompt to the OpenAI API and returns the assistant's response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=2000,
    )

    return response.choices[0].message.content
