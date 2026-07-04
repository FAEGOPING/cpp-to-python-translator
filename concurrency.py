"""
concurrency.py — Adaptive Concurrency Controller (Research Platform v5.1)
=========================================================================

Limits concurrent API requests and automatically reduces parallelism
when transient failures (HTTP 429, timeouts, connection resets) are
detected.  Concurrency is gradually restored as the connection
stabilises.

Thread-safe.  Does NOT change experiment behaviour — only the order
and timing of API requests are affected.

Usage::

    from concurrency import AdaptiveSemaphore

    sem = AdaptiveSemaphore(max_workers=4)
    with sem:
        result = call_gpt_structured(prompt)
    # On repeated failures, sem.max_workers drops toward 2.
    # On sustained success, it climbs back to 4.

Design
------
Every worker must acquire the semaphore before making an API call.
On retryable errors the semaphore's effective limit is reduced;
on each clean success it is nudged back up.
"""

from __future__ import annotations

import threading
import time


# ---------------------------------------------------------------------------
# Adaptive semaphore
# ---------------------------------------------------------------------------

class AdaptiveSemaphore:
    """A semaphore whose capacity adapts to observed failure rates.

    Starts at *max_workers* and never drops below *min_workers* (2).
    Thread-safe — all state mutations are protected by an internal lock.
    """

    def __init__(self, max_workers: int = 4, min_workers: int = 2) -> None:
        self._max = max(max_workers, 1)
        self._min = max(min_workers, 1)
        self._current = self._max
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(self._current)

        # Adaptive state
        self._consecutive_successes: int = 0
        self._consecutive_failures: int = 0
        self._last_failure_time: float = 0.0

        # Tuning constants
        self._FAILURE_REDUCTION_THRESHOLD: int = 2
        """Consecutive failures before reducing concurrency."""

        self._SUCCESS_RESTORE_THRESHOLD: int = 8
        """Consecutive successes before increasing concurrency."""

        self._COOLDOWN_SECONDS: float = 5.0
        """Minimum time between two reductions."""

        # Performance counters (readable after experiment ends)
        self.total_acquisitions: int = 0
        self.total_reductions: int = 0
        self.total_restorations: int = 0
        self.peak_waiters: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def max_workers(self) -> int:
        """Current effective concurrency limit."""
        with self._lock:
            return self._current

    @property
    def configured_max(self) -> int:
        """The original (configured) maximum."""
        return self._max

    def acquire(self) -> bool:
        """Acquire a permit, blocking if necessary.

        Returns:
            Always ``True`` (blocks until a permit is available).
        """
        self._sem.acquire()
        with self._lock:
            self.total_acquisitions += 1
            # Track peak waiters (approximate)
            waiting = self._max - self._current
            if waiting > self.peak_waiters:
                self.peak_waiters = waiting
        return True

    def release(self) -> None:
        """Release a previously acquired permit."""
        self._sem.release()

    def report_success(self) -> None:
        """Notify the controller that an API call succeeded.

        Gradual restoration: after *N* consecutive successes,
        increase the permit count by 1.
        """
        with self._lock:
            self._consecutive_failures = 0
            self._consecutive_successes += 1

            if (self._consecutive_successes >= self._SUCCESS_RESTORE_THRESHOLD
                    and self._current < self._max):
                self._increase()

    def report_failure(self, is_retryable: bool = True) -> None:
        """Notify the controller that an API call failed.

        If the failure is transient (retryable), consecutive failures
        are counted.  After a threshold, the effective concurrency is
        reduced.

        Non-retryable failures do not trigger reductions — they are
        likely permanent (e.g. bad API key, invalid request).
        """
        if not is_retryable:
            return

        now = time.time()
        with self._lock:
            self._consecutive_successes = 0
            self._consecutive_failures += 1

            if (self._consecutive_failures >= self._FAILURE_REDUCTION_THRESHOLD
                    and self._current > self._min
                    and (now - self._last_failure_time) > self._COOLDOWN_SECONDS):
                self._reduce()

    # ------------------------------------------------------------------
    # Context manager (preferred usage)
    # ------------------------------------------------------------------

    def __enter__(self) -> "AdaptiveSemaphore":
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reduce(self) -> None:
        """Reduce effective concurrency by 1."""
        # Drain one permit from the semaphore
        acquired = self._sem.acquire(blocking=False)
        if acquired:
            self._current -= 1
            self.total_reductions += 1
            self._last_failure_time = time.time()
            self._consecutive_failures = 0

    def _increase(self) -> None:
        """Increase effective concurrency by 1."""
        self._sem.release()
        self._current += 1
        self.total_restorations += 1
        self._consecutive_successes = 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return controller statistics as a dict."""
        with self._lock:
            return {
                "configured_max": self._max,
                "current_limit": self._current,
                "total_acquisitions": self.total_acquisitions,
                "total_reductions": self.total_reductions,
                "total_restorations": self.total_restorations,
                "peak_waiters": self.peak_waiters,
            }


# ---------------------------------------------------------------------------
# Module-level singleton (created and configured by experiment_runner)
# ---------------------------------------------------------------------------

_semaphore: AdaptiveSemaphore | None = None
_lock = threading.Lock()


def init_semaphore(max_workers: int) -> AdaptiveSemaphore:
    """Initialise the module-level adaptive semaphore.

    Must be called once before workers are started.  Idempotent.

    Args:
        max_workers: Initial concurrency limit (from ``--workers``).

    Returns:
        The configured :class:`AdaptiveSemaphore`.
    """
    global _semaphore
    with _lock:
        if _semaphore is None:
            _semaphore = AdaptiveSemaphore(max_workers=max_workers)
    return _semaphore


def get_semaphore() -> AdaptiveSemaphore:
    """Return the module-level semaphore, creating a default if needed."""
    global _semaphore
    if _semaphore is None:
        _semaphore = init_semaphore(4)
    return _semaphore
