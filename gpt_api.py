"""
gpt_api.py — DeepSeek API Interface

Provides a minimal wrapper around the OpenAI-compatible DeepSeek Chat API
for C++ → Python code translation tasks.

Configuration:
    Set the environment variable DEEPSEEK_API_KEY before use.
    export DEEPSEEK_API_KEY="your_api_key"

Usage:
    from gpt_api import call_gpt
    response = call_gpt("Translate this C++ code to Python...")
"""

import os
from openai import OpenAI


# ---------------------------------------------------------------------------
# Client initialisation (module-level, lazy-fail on first call)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return the singleton OpenAI client pointed at the DeepSeek endpoint."""
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY environment variable is not set. "
                "Export it before running the translation system:\n"
                "  export DEEPSEEK_API_KEY=\"your_api_key\""
            )
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_gpt(prompt: str) -> str:
    """Send a prompt to DeepSeek-V4-Pro and return the model's response.

    Args:
        prompt: The full prompt text to send.

    Returns:
        The model's textual response (stripped of leading / trailing
        whitespace).

    Raises:
        RuntimeError: If DEEPSEEK_API_KEY is not set.
        openai.APIError: On upstream API failures (rate-limit,
            authentication, server errors, etc.).
    """
    client = _get_client()

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()
