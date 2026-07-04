"""
gpt_api.py — DeepSeek API Interface (v5.0)
============================================

Production-grade API client for the C++ → Python translation framework.

Features:
    - Session-level HTTP connection pooling (Keep-Alive)
    - Automatic retry with exponential backoff for transient failures
    - Token usage collection (prompt / completion / total)
    - Per-call timing
    - Thread-safe singleton client

Configuration:
    Set the environment variable ``DEEPSEEK_API_KEY`` before use::

        export DEEPSEEK_API_KEY="your_api_key"

Usage::

    from gpt_api import call_gpt, CallResult, get_session_stats

    result = call_gpt("Translate this C++ code to Python...")
    print(result.text)           # model response
    print(result.prompt_tokens)  # token count
    print(result.total_tokens)   # total tokens
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

from openai import OpenAI

from concurrency import get_semaphore
from token_tracker import record_llm_call

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    """Structured result from an LLM API call.

    Carries the response text, token usage, timing, and retry metadata.
    """

    text: str
    """The model's textual response (stripped)."""

    prompt_tokens: int = 0
    """Prompt tokens consumed for this call."""

    completion_tokens: int = 0
    """Completion tokens generated for this call."""

    total_tokens: int = 0
    """Sum of prompt + completion tokens."""

    elapsed_seconds: float = 0.0
    """Wall-clock time for the API call (including retries)."""

    retry_count: int = 0
    """Number of retry attempts that were made before success."""


# ---------------------------------------------------------------------------
# Session statistics (global, thread-safe accumulation)
# ---------------------------------------------------------------------------

@dataclass
class SessionStats:
    """Cumulative statistics across all API calls in this session."""

    total_calls: int = 0
    total_retries: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_elapsed: float = 0.0
    """Sum of wall-clock time for all calls."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, result: CallResult) -> None:
        """Atomically accumulate *result* into session totals."""
        with self._lock:
            self.total_calls += 1
            self.total_retries += result.retry_count
            self.total_prompt_tokens += result.prompt_tokens
            self.total_completion_tokens += result.completion_tokens
            self.total_tokens += result.total_tokens
            self.total_elapsed += result.elapsed_seconds


_session_stats = SessionStats()


def get_session_stats() -> SessionStats:
    """Return a snapshot of accumulated API session statistics."""
    return _session_stats


def reset_session_stats() -> None:
    """Reset all session statistics to zero."""
    global _session_stats
    _session_stats = SessionStats()


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

_client: OpenAI | None = None
_client_lock = threading.Lock()

# Retry settings
_MAX_RETRIES: int = 3
_BASE_DELAY: float = 1.0       # seconds — doubles each retry (1, 2, 4)
_REQUEST_TIMEOUT: float = 120.0  # seconds

# Transient error patterns to retry
_RETRYABLE_ERRORS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "connection",
    "connectionerror",
    "rate_limit",
    "429",
    "500",
    "502",
    "503",
    "server error",
    "service unavailable",
    "internal server error",
    "bad gateway",
    "gateway timeout",
)


def _is_retryable(exc: Exception) -> bool:
    """Determine whether an exception represents a transient failure.

    Args:
        exc: The exception raised by the API call.

    Returns:
        ``True`` if the call should be retried.
    """
    msg = str(exc).lower()
    for pattern in _RETRYABLE_ERRORS:
        if pattern in msg:
            return True
    # Also retry on generic connection-level exceptions
    exc_name = type(exc).__name__.lower()
    if "connection" in exc_name or "timeout" in exc_name:
        return True
    return False


def _get_client() -> OpenAI:
    """Return the singleton OpenAI client pointed at the DeepSeek endpoint.

    The client uses persistent HTTP connection pooling (httpx under the
    hood) so TCP connections are reused across calls.

    Returns:
        Configured :class:`OpenAI` client instance.

    Raises:
        RuntimeError: If ``DEEPSEEK_API_KEY`` is not set in the environment.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # double-checked locking
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
                    max_retries=0,  # we handle retries ourselves
                )
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_gpt(prompt: str) -> str:
    """Send a prompt to DeepSeek and return the response text.

    **Legacy compatibility wrapper.**  For new code, prefer
    :func:`call_gpt_structured` which returns a :class:`CallResult`.

    Args:
        prompt: The full prompt text to send.

    Returns:
        The model's textual response (stripped of whitespace).

    Raises:
        RuntimeError: If ``DEEPSEEK_API_KEY`` is not set.
        openai.APIError: On persistent upstream API failures.
    """
    return call_gpt_structured(prompt).text


def call_gpt_structured(prompt: str) -> CallResult:
    """Send a prompt to DeepSeek-V4-Pro and return structured results.

    Uses adaptive concurrency control — acquires a semaphore permit
    before each API call.  Sustained transient failures automatically
    reduce the effective concurrency; clean requests restore it.

    Args:
        prompt: The full prompt text to send.

    Returns:
        :class:`CallResult` with text, token counts, and timing.

    Raises:
        RuntimeError: If ``DEEPSEEK_API_KEY`` is not set.
    """
    client = _get_client()
    sem = get_semaphore()
    t0 = time.time()
    retries = 0

    # Acquire adaptive concurrency permit (blocks if too many
    # concurrent requests are failing)
    sem.acquire()

    try:
        for attempt in range(_MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                elapsed = time.time() - t0

                # Extract token usage from the API response
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                try:
                    usage = response.usage
                    if usage:
                        prompt_tokens = usage.prompt_tokens or 0
                        completion_tokens = usage.completion_tokens or 0
                        total_tokens = usage.total_tokens or 0
                except Exception:
                    pass  # token counts are best-effort

                # Auto-track tokens for per-program and session statistics
                record_llm_call(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    retries=retries,
                )

                result = CallResult(
                    text=(response.choices[0].message.content or "").strip(),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    elapsed_seconds=elapsed,
                    retry_count=retries,
                )

                _session_stats.record(result)
                # Signal success to adaptive controller
                sem.report_success()
                return result

            except Exception as exc:
                if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                    retries += 1
                    sem.report_failure(is_retryable=True)
                    delay = _BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                else:
                    # Last attempt or non-retryable error — re-raise
                    sem.report_failure(is_retryable=False)
                    raise
    finally:
        sem.release()
