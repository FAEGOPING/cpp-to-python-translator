"""
dashboard.py — Real-Time Experiment Progress Dashboard
========================================================

Provides a live-updating progress display for the C++ → Python
translation experiment runner.

Designed for parallel execution (ThreadPoolExecutor).  Updates are
pushed by the main thread each time a worker future completes.

Usage::

    from dashboard import Dashboard

    dash = Dashboard(total=100, workers=4, experiment_id="my-exp")
    dash.start()

    for each completed program:
        dash.update(completed=42, program_time=30.5)

    dash.stop()
    dash.print_final()

Output (updated in-place)::

    ── Experiment Progress ─────────────────────────────────────
      Experiment:    2026-07-04_10-30-00
      Workers:       4
      Completed:     42 / 100  (42%)
      Elapsed:       2m 15s
      ETA:           3m 05s
      Throughput:    18.7 programs/min
      Avg time:      30.5s / program
      ── Tokens ────────────────────────────────────────────────
      Prompt:        1,250,000
      Completion:      840,000
      Total:         2,090,000
      Est. cost:     $0.41
      ──────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional


class Dashboard:
    """Real-time experiment progress display.

    Renders a multi-line status block that updates in-place.
    Thread-safe — updates are serialised via an internal lock.
    """

    def __init__(
        self,
        total: int,
        workers: int = 1,
        experiment_id: str = "",
        enabled: bool = True,
    ) -> None:
        """Create a dashboard.

        Args:
            total: Total number of programs to process.
            workers: Number of parallel workers.
            experiment_id: Experiment identifier for display.
            enabled: If ``False``, all methods are no-ops (no output).
        """
        self.total = total
        self.workers = workers
        self.experiment_id = experiment_id
        self.enabled = enabled and sys.stdout.isatty()

        # Shared state (protected by lock)
        self._lock = threading.Lock()
        self._completed: int = 0
        self._start_time: float = 0.0
        self._elapsed: float = 0.0
        self._program_times: list[float] = []  # rolling window
        self._current_program: str = ""
        self._running: bool = False

        # Pre-compute field widths
        self._prefix = "  "

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin timing and print the initial dashboard frame."""
        if not self.enabled:
            return
        with self._lock:
            self._start_time = time.time()
            self._running = True
        self._render()

    def update(
        self,
        completed: int,
        program_time: float = 0.0,
        current_program: str = "",
    ) -> None:
        """Record a completed program and refresh the display.

        Args:
            completed: Cumulative number of completed programs.
            program_time: Wall-clock time for the most recently
                completed program (seconds).
            current_program: Filename of the most recently completed
                program (for live display).
        """
        if not self.enabled:
            return
        with self._lock:
            self._completed = completed
            self._elapsed = time.time() - self._start_time
            self._current_program = current_program
            if program_time > 0:
                self._program_times.append(program_time)
                if len(self._program_times) > 20:
                    self._program_times = self._program_times[-20:]
        self._render()

    def stop(self) -> None:
        """Stop the dashboard and leave the cursor on a new line."""
        if not self.enabled:
            return
        with self._lock:
            self._running = False
        # Move past the dashboard block
        height = 12 if self.workers > 1 else 7
        print(f"\033[{height}B")

    def print_final(self) -> None:
        """Print a non-animated final summary block."""
        # Always print, even when not a tty
        self._render(final=True)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, final: bool = False) -> None:
        # Guard: only one thread renders at a time
        if not self.enabled and not final:
            return

        from token_tracker import get_session_snapshot, estimate_cost

        with self._lock:
            compl = self._completed
            elapsed = self._elapsed
            ptimes = list(self._program_times)
            cur_prog = self._current_program

        # Compute derived metrics
        pct = compl * 100 // max(self.total, 1)
        remaining = max(self.total - compl, 0)

        # ETA
        if compl > 0 and ptimes:
            avg_time = sum(ptimes) / len(ptimes)
            eta_sec = remaining * avg_time
        else:
            avg_time = 0.0
            eta_sec = 0.0

        # Throughput
        minutes = elapsed / 60.0 if elapsed > 0 else 0.001
        throughput = compl / max(minutes, 0.001)

        # Token stats
        snap = get_session_snapshot()
        prompt_tok = snap["total_prompt_tokens"]
        comp_tok = snap["total_completion_tokens"]
        total_tok = snap["total_tokens"]
        cost = estimate_cost(prompt_tok, comp_tok)

        lines: list[str] = []

        # Clear and reposition for in-place update
        if not final and self.enabled:
            lines.append("\033[2J\033[H")  # clear screen, home

        lines.append("")
        lines.append(f"{'─' * 60}")
        lines.append(f"  Experiment Progress")
        lines.append(f"{'─' * 60}")
        lines.append(f"  Experiment:    {self.experiment_id}")
        lines.append(f"  Workers:       {self.workers}")
        lines.append(f"  Completed:     {compl} / {self.total}  ({pct}%)")
        lines.append(f"  Remaining:     {remaining}")
        lines.append(f"  Elapsed:       {_fmt_duration(elapsed)}")

        if remaining > 0 and eta_sec > 0:
            lines.append(f"  ETA:           {_fmt_duration(eta_sec)}")

        lines.append(f"  Throughput:    {throughput:.1f} programs/min")
        if cur_prog:
            lines.append(f"  Last completed:{cur_prog}")
        if avg_time > 0:
            lines.append(f"  Avg time:      {avg_time:.1f}s / program")

        # Token section (only when active)
        if snap["session_calls"] > 0:
            lines.append(f"  {'─' * 50}")
            lines.append(f"  API Calls:     {snap['session_calls']}")
            lines.append(f"  Retries:       {snap['session_retries']}")
            lines.append(f"  Prompt tok:    {prompt_tok:,}")
            lines.append(f"  Completion:    {comp_tok:,}")
            lines.append(f"  Total tokens:  {total_tok:,}")
            lines.append(f"  Est. cost:     ${cost:.4f}")
            lines.append(f"  {'─' * 50}")

        lines.append(f"{'─' * 60}")
        lines.append("")

        output = "\n".join(lines)

        if final or not self.enabled:
            print(output, flush=True)
        else:
            sys.stdout.write(output)
            sys.stdout.flush()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string."""
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s:02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s"
