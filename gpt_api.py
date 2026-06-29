"""
gpt_api.py — DeepSeek API Interface
====================================

Provides a minimal wrapper around the OpenAI-compatible DeepSeek Chat API
for C++ → Python code translation tasks.

Configuration:
    Set the environment variable ``DEEPSEEK_API_KEY`` before use::

        export DEEPSEEK_API_KEY="your_api_key"

Usage::

    from gpt_api import call_gpt
    response = call_gpt("Translate this C++ code to Python...")
"""

from __future__ import annotations

import os
import time

from openai import OpenAI

# ---------------------------------------------------------------------------
# Client initialisation (module-level, lazy-fail on first call)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None

# Retry settings for transient API failures
_MAX_RETRIES: int = 3
_RETRY_DELAY: float = 1.0  # seconds between retries
_REQUEST_TIMEOUT: float = 120.0  # seconds


def _get_client() -> OpenAI:
    """Return the singleton OpenAI client pointed at the DeepSeek endpoint.

    Returns:
        Configured :class:`OpenAI` client instance.

    Raises:
        RuntimeError: If ``DEEPSEEK_API_KEY`` is not set in the environment.
    """
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
            timeout=_REQUEST_TIMEOUT,
        )
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_gpt(prompt: str) -> str:
    """Send a prompt to DeepSeek-V4-Pro and return the model's response.

    Includes automatic retry with exponential backoff for transient
    API failures (rate limits, server errors, connection issues).

    Args:
        prompt: The full prompt text to send.

    Returns:
        The model's textual response (stripped of leading / trailing
        whitespace).

    Raises:
        RuntimeError: If ``DEEPSEEK_API_KEY`` is not set.
        openai.APIError: On persistent upstream API failures after all
            retries are exhausted.
    """
    client = _get_client()

    last_exception: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            last_exception = exc
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAY * (2 ** attempt)
                time.sleep(delay)
            # else: last attempt — fall through to re-raise

    raise last_exception  # type: ignore[misc]
