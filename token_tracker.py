"""
token_tracker.py — Thread-Safe Token Usage Tracker
====================================================

Per-program and session-level token counters for the experiment
pipeline.  Workers update their per-program counters; the main thread
reads aggregated totals after all workers complete.

Usage::

    from token_tracker import (
        reset_program_tokens, record_llm_call,
        get_program_tokens, get_session_snapshot,
    )

    reset_program_tokens()
    record_llm_call(prompt_tokens=1500, completion_tokens=800)
    record_llm_call(prompt_tokens=2000, completion_tokens=1200)
    ptok, ctok, ttok = get_program_tokens()
    # ptok=3500, ctok=2000, ttok=5500
"""

from __future__ import annotations

import threading


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_lock = threading.Lock()

# Per-program counters (reset for each program, accumulate across rounds)
_program_prompt: int = 0
_program_completion: int = 0
_program_total: int = 0
_program_calls: int = 0

# Session-level counters (accumulate across all programs)
_session_prompt: int = 0
_session_completion: int = 0
_session_total: int = 0
_session_calls: int = 0
_session_retries: int = 0

# Last-call token values (for per-round CSV rows)
_last_prompt: int = 0
_last_completion: int = 0
_last_total: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reset_program_tokens() -> None:
    """Reset per-program counters.  Call once at the start of each program."""
    global _program_prompt, _program_completion, _program_total, _program_calls
    global _last_prompt, _last_completion, _last_total
    with _lock:
        _program_prompt = 0
        _program_completion = 0
        _program_total = 0
        _program_calls = 0
        _last_prompt = 0
        _last_completion = 0
        _last_total = 0


def reset_session_stats() -> None:
    """Reset session-level counters.  Call at the start of an experiment."""
    global _session_prompt, _session_completion, _session_total
    global _session_calls, _session_retries
    with _lock:
        _session_prompt = 0
        _session_completion = 0
        _session_total = 0
        _session_calls = 0
        _session_retries = 0


def record_llm_call(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    retries: int = 0,
) -> None:
    """Record token usage from a single LLM API call.

    Updates both per-program and session-level counters atomically.
    Also updates the "last call" counters for per-round CSV logging.

    Args:
        prompt_tokens: Prompt tokens for this call.
        completion_tokens: Completion tokens for this call.
        total_tokens: Total tokens (prompt + completion).
        retries: Number of retries before success.
    """
    global _program_prompt, _program_completion, _program_total, _program_calls
    global _session_prompt, _session_completion, _session_total
    global _session_calls, _session_retries
    global _last_prompt, _last_completion, _last_total

    with _lock:
        _program_prompt += prompt_tokens
        _program_completion += completion_tokens
        _program_total += total_tokens
        _program_calls += 1

        _session_prompt += prompt_tokens
        _session_completion += completion_tokens
        _session_total += total_tokens
        _session_calls += 1
        _session_retries += retries

        _last_prompt = prompt_tokens
        _last_completion = completion_tokens
        _last_total = total_tokens


def get_program_tokens() -> tuple[int, int, int]:
    """Return ``(prompt, completion, total)`` for the current program."""
    with _lock:
        return (_program_prompt, _program_completion, _program_total)


def get_last_call_tokens() -> tuple[int, int, int]:
    """Return ``(prompt, completion, total)`` for the most recent call."""
    with _lock:
        return (_last_prompt, _last_completion, _last_total)


def get_session_snapshot() -> dict:
    """Return session-level token statistics as a dict."""
    with _lock:
        return {
            "session_calls": _session_calls,
            "session_retries": _session_retries,
            "total_prompt_tokens": _session_prompt,
            "total_completion_tokens": _session_completion,
            "total_tokens": _session_total,
        }


def estimate_cost(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    prompt_price_per_1m: float = 0.14,
    completion_price_per_1m: float = 0.28,
) -> float:
    """Estimate API cost from token counts.

    Default prices are for DeepSeek-V3 (USD per 1M tokens).

    Args:
        prompt_tokens: Total prompt tokens.
        completion_tokens: Total completion tokens.
        prompt_price_per_1m: Price per 1M prompt tokens.
        completion_price_per_1m: Price per 1M completion tokens.

    Returns:
        Estimated cost in USD.
    """
    prompt_cost = (prompt_tokens / 1_000_000) * prompt_price_per_1m
    completion_cost = (completion_tokens / 1_000_000) * completion_price_per_1m
    return round(prompt_cost + completion_cost, 6)
