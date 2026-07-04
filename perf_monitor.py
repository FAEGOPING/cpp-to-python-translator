"""
perf_monitor.py — Performance Monitoring (Research Platform v5.1)
==================================================================

Thread-safe timing accumulator for experiment performance analysis.
Each worker records per-phase timings; the main thread prints a
summary at the end of the experiment.

Usage::

    from perf_monitor import PerfMonitor

    mon = PerfMonitor()

    # In each worker, after each phase:
    mon.record_translation(seconds)
    mon.record_compile(seconds)
    mon.record_runtime_validation(seconds)
    mon.record_functional_validation(seconds)
    mon.record_api_wait(seconds)

    # At experiment end:
    mon.print_summary(total_programs, total_elapsed)
"""

from __future__ import annotations

import threading
import time


class PerfMonitor:
    """Thread-safe accumulator for experiment performance metrics.

    All state mutations are protected by a single lock.  Timing
    measurements are recorded in seconds and aggregated as sums
    and counts for later averaging.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Accumulated seconds per phase
        self._translation_time: float = 0.0
        self._compile_time: float = 0.0
        self._runtime_time: float = 0.0
        self._functional_time: float = 0.0
        self._api_wait_time: float = 0.0
        self._repair_time: float = 0.0

        # Counts (for averages)
        self._translation_count: int = 0
        self._compile_count: int = 0
        self._runtime_count: int = 0
        self._functional_count: int = 0
        self._api_call_count: int = 0
        self._repair_count: int = 0

        # Wall-clock start
        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # Recording API (lock-free call interface)
    # ------------------------------------------------------------------

    def record_translation(self, seconds: float) -> None:
        """Record time spent in LLM translation."""
        with self._lock:
            self._translation_time += seconds
            self._translation_count += 1

    def record_compile(self, seconds: float) -> None:
        """Record time spent in py_compile validation."""
        with self._lock:
            self._compile_time += seconds
            self._compile_count += 1

    def record_runtime(self, seconds: float) -> None:
        """Record time spent in Python subprocess execution."""
        with self._lock:
            self._runtime_time += seconds
            self._runtime_count += 1

    def record_functional(self, seconds: float) -> None:
        """Record time spent in differential / functional validation."""
        with self._lock:
            self._functional_time += seconds
            self._functional_count += 1

    def record_api_wait(self, seconds: float) -> None:
        """Record time spent waiting for API responses."""
        with self._lock:
            self._api_wait_time += seconds
            self._api_call_count += 1

    def record_repair(self, seconds: float) -> None:
        """Record time spent in repair LLM calls."""
        with self._lock:
            self._repair_time += seconds
            self._repair_count += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(
        self,
        total_programs: int,
        total_elapsed: float,
        workers: int = 1,
    ) -> None:
        """Print a human-readable performance breakdown.

        Args:
            total_programs: Number of programs processed.
            total_elapsed: Wall-clock duration of the experiment (seconds).
            workers: Number of parallel workers used.
        """
        with self._lock:
            t_trans = self._translation_time
            t_comp = self._compile_time
            t_run = self._runtime_time
            t_func = self._functional_time
            t_api = self._api_wait_time
            t_repair = self._repair_time

            n_trans = self._translation_count
            n_comp = self._compile_count
            n_run = self._runtime_count
            n_func = self._functional_count
            n_api = self._api_call_count
            n_repair = self._repair_count

        # Compute remaining (unaccounted) time
        accounted = t_trans + t_comp + t_run + t_func + t_api + t_repair
        other = max(total_elapsed - accounted, 0.0)

        pct = lambda v: (v / max(total_elapsed, 0.001)) * 100

        print(f"\n{'═' * 70}")
        print("PERFORMANCE SUMMARY")
        print(f"{'═' * 70}")
        print(f"  Total programs:       {total_programs}")
        print(f"  Workers:              {workers}")
        print(f"  Total wall time:      {total_elapsed:.1f}s")
        print(f"  Throughput:           {total_programs / max(total_elapsed, 0.001) * 60:.1f} programs/min")
        print(f"")
        print(f"  {'Phase':<28} {'Time':>8}  {'Calls':>6}  {'Avg':>8}  {'%Total':>7}")
        print(f"  {'─' * 28} {'─' * 8}  {'─' * 6}  {'─' * 8}  {'─' * 7}")
        _row("Translation (LLM)",       t_trans, n_trans, total_elapsed)
        _row("  └ API waiting",          t_api,   n_api,   total_elapsed)
        _row("Compile validation",       t_comp,  n_comp,  total_elapsed)
        _row("Runtime execution",        t_run,   n_run,   total_elapsed)
        _row("Functional validation",    t_func,  n_func,  total_elapsed)
        if t_repair > 0:
            _row("Repair (LLM)",         t_repair, n_repair, total_elapsed)
        _row("Other / overhead",         other,   0,       total_elapsed)
        print(f"  {'─' * 28} {'─' * 8}  {'─' * 6}  {'─' * 8}  {'─' * 7}")
        print(f"  {'TOTAL':<28} {total_elapsed:>8.1f}s")
        print(f"{'═' * 70}\n")


def _row(label: str, time_s: float, count: int, total: float) -> None:
    """Print one row of the performance summary table."""
    pct_val = (time_s / max(total, 0.001)) * 100
    avg = time_s / max(count, 1) if count > 0 else 0.0
    count_str = f"{count}" if count > 0 else "—"
    print(f"  {label:<28} {time_s:>7.1f}s  {count_str:>6}  {avg:>7.2f}s  {pct_val:>6.1f}%")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_monitor: PerfMonitor | None = None
_mon_lock = threading.Lock()


def get_monitor() -> PerfMonitor:
    """Return (and lazily create) the module-level performance monitor."""
    global _monitor
    if _monitor is None:
        with _mon_lock:
            if _monitor is None:
                _monitor = PerfMonitor()
    return _monitor


def reset_monitor() -> None:
    """Reset the performance monitor (call before each experiment)."""
    global _monitor
    with _mon_lock:
        _monitor = PerfMonitor()
