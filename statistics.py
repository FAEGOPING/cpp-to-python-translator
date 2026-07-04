"""
statistics.py — Unified Experiment Statistics Engine
======================================================

**Single source of truth** for all experiment metrics, computed
directly from ``experiment_results.csv`` and ``summary_results.csv``.

Every other module (figures, reports, analytics) reads data through
this module — never from raw CSV files directly.

Key features:
    - Auto-discovers the latest experiment output directory
    - Computes all standard metrics from CSV data
    - Gracefully handles missing optional columns
    - Provides both per-program and aggregate statistics
    - Used by :mod:`figures`, :mod:`report_generator`,
      :mod:`research_analytics`.

Usage::

    from statistics import ExperimentStats, load_stats

    stats = load_stats()                    # latest experiment
    stats = load_stats("experiment_results/2026-07-02_10-00-00")

    print(stats.compile_success_rate)       # 94.2
    print(stats.functional_success_rate)    # 87.5
    print(stats.avg_translation_time)       # 31.4
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dataset_manager.utils import read_csv, Logger


# ============================================================================
# Path resolution — auto-discover latest experiment
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.join(PROJECT_ROOT, "experiment_results")


def _find_latest_experiment() -> str | None:
    """Find the most recent experiment directory.

    Returns:
        Absolute path to the latest ``experiment_results/<timestamp>/``
        directory, or ``None`` if no experiments exist.
    """
    if not os.path.isdir(EXPERIMENT_ROOT):
        return None
    dirs = [
        d for d in os.listdir(EXPERIMENT_ROOT)
        if os.path.isdir(os.path.join(EXPERIMENT_ROOT, d))
    ]
    dirs.sort(reverse=True)
    for d in dirs:
        exp_dir = os.path.join(EXPERIMENT_ROOT, d)
        csv_dir = os.path.join(exp_dir, "csv")
        if os.path.isdir(csv_dir):
            return exp_dir
        # Also check flat layout
        if os.path.isfile(os.path.join(exp_dir, "experiment_results.csv")):
            return exp_dir
    return None


def _find_csv_dir(exp_dir: str) -> str:
    """Find the CSV directory within an experiment directory.

    Args:
        exp_dir: Experiment output directory.

    Returns:
        Path to the ``csv/`` subdirectory, or *exp_dir* itself if no
        ``csv/`` exists.
    """
    csv_sub = os.path.join(exp_dir, "csv")
    if os.path.isdir(csv_sub):
        return csv_sub
    return exp_dir


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ExperimentStats:
    """Complete experiment statistics computed from CSVs."""

    # ---- Source ----
    experiment_dir: str = ""
    """Path to the experiment output directory."""

    # ---- Program counts ----
    total_programs: int = 0
    """Total programs in the experiment."""

    # ---- Compile (Python py_compile) ----
    compile_pass: int = 0
    compile_fail: int = 0
    compile_success_rate: float = 0.0

    # ---- Runtime (Python execution) ----
    runtime_pass: int = 0
    runtime_fail: int = 0
    runtime_success_rate: float = 0.0

    # ---- Functional (output comparison) ----
    functional_pass: int = 0
    functional_fail: int = 0
    functional_success_rate: float = 0.0

    # ---- Final state (from summary) ----
    final_compile_pass: int = 0
    final_runtime_pass: int = 0
    final_functional_pass: int = 0
    overall_success: int = 0
    overall_success_rate: float = 0.0

    # ---- Timing (seconds) ----
    avg_translation_time: float = 0.0
    avg_compile_time: float = 0.0
    avg_runtime_time: float = 0.0
    avg_validation_time: float = 0.0
    avg_repair_time: float = 0.0
    avg_total_time: float = 0.0
    total_experiment_time: float = 0.0

    # ---- Repair ----
    programs_repaired: int = 0
    avg_repair_rounds: float = 0.0
    max_repair_rounds: int = 0
    repair_success_gain: int = 0
    """Programs that passed after repair but failed initially."""

    # ---- Test counts ----
    avg_generated_tests: float = 0.0
    avg_executed_tests: float = 0.0
    avg_passed_tests: float = 0.0
    avg_test_success_rate: float = 0.0

    # ---- Token statistics ----
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    avg_prompt_tokens_per_program: float = 0.0
    avg_completion_tokens_per_program: float = 0.0
    avg_tokens_per_program: float = 0.0

    # ---- Error distribution ----
    error_counts: Dict[str, int] = field(default_factory=dict)
    """ErrorType → count."""
    error_categories: Dict[str, int] = field(default_factory=dict)
    """Error category → count."""
    failure_reasons: List[str] = field(default_factory=list)
    """Short failure descriptions."""

    # ---- Per-program stats ----
    program_stats: List[Dict[str, Any]] = field(default_factory=list)
    """One dict per program with all metrics."""

    # ---- Experiment metadata ----
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Experiment configuration (compiler, git hash, etc.)."""

    # ---- Convenience checks ----
    @property
    def is_empty(self) -> bool:
        """``True`` when no data was loaded."""
        return self.total_programs == 0

    @property
    def has_repair_data(self) -> bool:
        """``True`` when repair was used (RepairRounds > 0 for some programs)."""
        return self.programs_repaired > 0 or self.max_repair_rounds > 0

    @property
    def has_runtime_data(self) -> bool:
        """``True`` when runtime data is available."""
        return self.runtime_pass + self.runtime_fail > 0

    @property
    def has_timing_data(self) -> bool:
        """``True`` when per-phase timing is available."""
        return self.avg_translation_time > 0 or self.avg_total_time > 0


# ============================================================================
# Core statistics computation
# ============================================================================

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Parse a float, returning *default* on error."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Parse an int, returning *default* on error."""
    if val is None or val == "":
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _safe_bool(val: Any) -> bool:
    """Parse a boolean from a CSV string."""
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in ("true", "1", "yes", "pass")


def compute_stats(
    detail_path: str,
    summary_path: str,
    exp_dir: str = "",
    logger: Logger | None = None,
) -> ExperimentStats:
    """Compute all experiment statistics from CSV files.

    Args:
        detail_path: Path to ``experiment_results.csv``.
        summary_path: Path to ``summary_results.csv``.
        exp_dir: Experiment directory for metadata lookup.
        logger: Optional :class:`Logger` instance.

    Returns:
        Fully populated :class:`ExperimentStats`.
    """
    detail_rows = read_csv(detail_path)
    summary_rows = read_csv(summary_path)

    if not detail_rows and not summary_rows:
        if logger:
            logger.warn(f"No CSV data found at {detail_path} or {summary_path}")
        return ExperimentStats()

    stats = ExperimentStats(experiment_dir=exp_dir)

    # Use summary rows for program-level stats (one row per program)
    source_rows = summary_rows if summary_rows else detail_rows
    # But also use detail rows for per-round error data
    use_detail = detail_rows if detail_rows else source_rows

    # ---- Count programs ----
    # Filter to only benchmark programs (skip example.cpp)
    bench_rows = [r for r in source_rows
                  if r.get("Program", "").startswith("program_")]
    if not bench_rows:
        bench_rows = source_rows  # use all rows if no benchmarking-specific rows

    stats.total_programs = len(bench_rows)

    if stats.total_programs == 0:
        if logger:
            logger.warn("No benchmark programs found in CSV data")
        return stats

    # ---- Compile stats ----
    stats.compile_pass = sum(1 for r in bench_rows
                             if _safe_bool(r.get("CompilePass", r.get("FinalCompilePass"))))
    stats.compile_fail = stats.total_programs - stats.compile_pass
    stats.compile_success_rate = stats.compile_pass / max(stats.total_programs, 1) * 100

    # ---- Runtime stats ----
    stats.runtime_pass = sum(1 for r in bench_rows
                             if _safe_bool(r.get("RuntimePass")))
    stats.runtime_fail = stats.total_programs - stats.runtime_pass
    stats.runtime_success_rate = stats.runtime_pass / max(stats.total_programs, 1) * 100

    # ---- Functional stats ----
    stats.functional_pass = sum(1 for r in bench_rows
                                if _safe_bool(r.get("FunctionalPass")))
    stats.functional_fail = stats.total_programs - stats.functional_pass
    stats.functional_success_rate = stats.functional_pass / max(stats.total_programs, 1) * 100

    # ---- Final state ----
    stats.final_compile_pass = sum(1 for r in bench_rows
                                   if _safe_bool(r.get("FinalCompilePass")))
    stats.final_runtime_pass = sum(1 for r in bench_rows
                                   if _safe_bool(r.get("RuntimePass")))
    stats.final_functional_pass = sum(1 for r in bench_rows
                                      if _safe_bool(r.get("FunctionalPass")))
    stats.overall_success = sum(
        1 for r in bench_rows
        if _safe_bool(r.get("FinalCompilePass"))
        and _safe_bool(r.get("RuntimePass"))
        and _safe_bool(r.get("FunctionalPass"))
    )
    stats.overall_success_rate = stats.overall_success / max(stats.total_programs, 1) * 100

    # ---- Timing (from all rows: summary + detail fallback) ----
    # Try summary first, then detail, then merge both
    trans_times = [_safe_float(r.get("TranslationTime")) for r in bench_rows
                   if _safe_float(r.get("TranslationTime")) > 0]
    total_times = [_safe_float(r.get("TotalTime")) for r in bench_rows
                   if _safe_float(r.get("TotalTime")) > 0]
    val_times = [_safe_float(r.get("ValidationTime")) for r in bench_rows
                 if _safe_float(r.get("ValidationTime")) > 0]

    # Detail rows for per-phase timing (compile/runtime per round)
    detail_bench = [r for r in use_detail if r.get("Program", "").startswith("program_")]
    if not detail_bench:
        detail_bench = use_detail

    compile_times = [_safe_float(r.get("CompileTime")) for r in detail_bench
                     if _safe_float(r.get("CompileTime")) > 0]
    runtime_ms = [_safe_float(r.get("RuntimeTime")) for r in detail_bench
                  if _safe_float(r.get("RuntimeTime")) > 0]

    if trans_times:
        stats.avg_translation_time = sum(trans_times) / len(trans_times)
    if compile_times:
        stats.avg_compile_time = sum(compile_times) / len(compile_times)
    if runtime_ms:
        stats.avg_runtime_time = sum(runtime_ms) / len(runtime_ms)
    if val_times:
        stats.avg_validation_time = sum(val_times) / len(val_times)
    if total_times:
        stats.avg_total_time = sum(total_times) / len(total_times)
        stats.total_experiment_time = max(total_times)  # wall-clock

    # ---- Repair ----
    repair_rounds = [_safe_int(r.get("RepairRounds")) for r in bench_rows
                     if _safe_int(r.get("RepairRounds")) > 0]
    if repair_rounds:
        stats.programs_repaired = len(repair_rounds)
        stats.avg_repair_rounds = sum(repair_rounds) / len(repair_rounds)
        stats.max_repair_rounds = max(repair_rounds)

    # Repair success gain: programs that passed at repair round > 0
    # (from detail rows, not summary)
    for r in use_detail:
        if not r.get("Program", "").startswith("program_"):
            continue
        round_num = _safe_int(r.get("Round"))
        if round_num > 0 and _safe_bool(r.get("FunctionalPass")):
            stats.repair_success_gain += 1

    # ---- Test counts ----
    gen_tests = [_safe_int(r.get("GeneratedTestCount")) for r in bench_rows
                 if _safe_int(r.get("GeneratedTestCount")) > 0]
    exec_tests = [_safe_int(r.get("ExecutedTestCount")) for r in bench_rows
                  if _safe_int(r.get("ExecutedTestCount")) > 0]
    passed_tests = [_safe_int(r.get("PassedTestCount")) for r in bench_rows
                    if _safe_int(r.get("PassedTestCount")) > 0]
    test_rates = [_safe_float(r.get("SuccessRate")) for r in bench_rows
                  if _safe_float(r.get("SuccessRate")) > 0]

    if gen_tests:
        stats.avg_generated_tests = sum(gen_tests) / len(gen_tests)
    if exec_tests:
        stats.avg_executed_tests = sum(exec_tests) / len(exec_tests)
    if passed_tests:
        stats.avg_passed_tests = sum(passed_tests) / len(passed_tests)
    if test_rates:
        stats.avg_test_success_rate = sum(test_rates) / len(test_rates) * 100

    # ---- Token statistics (from summary rows) ----
    token_prompts: list[int] = []
    token_completions: list[int] = []
    token_totals: list[int] = []
    for r in source_rows:
        pt = _safe_int(r.get("PromptTokens", r.get("prompt_tokens", 0)))
        ct = _safe_int(r.get("CompletionTokens", r.get("completion_tokens", 0)))
        tt = _safe_int(r.get("TotalTokens", r.get("total_tokens", 0)))
        if pt > 0 or ct > 0:
            token_prompts.append(pt)
            token_completions.append(ct)
            token_totals.append(tt)
    stats.total_prompt_tokens = sum(token_prompts)
    stats.total_completion_tokens = sum(token_completions)
    stats.total_tokens = sum(token_totals)
    n_token = max(len(token_prompts), 1)
    stats.avg_prompt_tokens_per_program = stats.total_prompt_tokens / n_token
    stats.avg_completion_tokens_per_program = stats.total_completion_tokens / n_token
    stats.avg_tokens_per_program = stats.total_tokens / n_token

    # ---- Error distribution (from detail rows) ----
    error_counter: Counter = Counter()
    category_counter: Counter = Counter()
    failure_reasons: list[str] = []

    for r in use_detail:
        if not r.get("Program", "").startswith("program_"):
            continue
        error_type = r.get("ErrorType", "")
        if error_type and error_type != "None":
            error_counter[error_type] += 1
        error_cat = r.get("ErrorCategory", r.get("FinalErrorType", ""))
        if error_cat and error_cat not in ("None", "none", ""):
            category_counter[error_cat] += 1
        failure = r.get("FailureReason", "")
        if failure:
            failure_reasons.append(failure)

    stats.error_counts = dict(error_counter.most_common())
    stats.error_categories = dict(category_counter.most_common())
    stats.failure_reasons = failure_reasons

    # ---- Per-program stats (from summary) ----
    stats.program_stats = bench_rows

    # ---- Metadata ----
    config_path = os.path.join(exp_dir, "config", "experiment_configuration.json")
    try:
        import json
        if os.path.isfile(config_path):
            with open(config_path) as f:
                stats.metadata = json.load(f)
    except Exception:
        pass

    if logger:
        logger.info(
            f"Stats computed: {stats.total_programs} programs, "
            f"compile={stats.compile_success_rate:.1f}%, "
            f"runtime={stats.runtime_success_rate:.1f}%, "
            f"functional={stats.functional_success_rate:.1f}%"
        )

    return stats


# ============================================================================
# Public API — load stats from experiment directory
# ============================================================================

def load_stats(exp_dir: str | None = None, logger: Logger | None = None) -> ExperimentStats:
    """Load statistics from the specified or latest experiment.

    Args:
        exp_dir: Path to an experiment output directory.  When ``None``,
            the latest directory under ``experiment_results/`` is used.
        logger: Optional :class:`Logger` instance.

    Returns:
        :class:`ExperimentStats` with all computed metrics.  Returns an
        empty stats object when no experiment data is found.

    Raises:
        FileNotFoundError: When *exp_dir* is explicitly given but does
            not exist.
    """
    if exp_dir is None:
        exp_dir = _find_latest_experiment()

    if exp_dir is None:
        if logger:
            logger.warn("No experiment results found. "
                        "Run `python experiment_runner.py --limit N` first.")
        return ExperimentStats()

    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    if logger:
        logger.info(f"Loading experiment: {os.path.basename(exp_dir)}")

    csv_dir = _find_csv_dir(exp_dir)
    detail_path = os.path.join(csv_dir, "experiment_results.csv")
    summary_path = os.path.join(csv_dir, "summary_results.csv")

    return compute_stats(detail_path, summary_path, exp_dir, logger)


# ============================================================================
# Summary dict — for use by report generators
# ============================================================================

def stats_summary(stats: ExperimentStats) -> Dict[str, Any]:
    """Return a flat dict of key-value pairs suitable for report tables.

    Args:
        stats: Computed :class:`ExperimentStats`.

    Returns:
        Dict with human-readable metric names and formatted values.
    """
    p = stats.total_programs
    return {
        "Programs": p,
        "Compile Success": f"{stats.compile_pass}/{p} ({stats.compile_success_rate:.1f}%)",
        "Runtime Success": f"{stats.runtime_pass}/{p} ({stats.runtime_success_rate:.1f}%)",
        "Functional Success": f"{stats.functional_pass}/{p} ({stats.functional_success_rate:.1f}%)",
        "Overall Success": f"{stats.overall_success}/{p} ({stats.overall_success_rate:.1f}%)",
        "Avg Translation Time": f"{stats.avg_translation_time:.2f}s",
        "Avg Compile Time": f"{stats.avg_compile_time:.4f}s",
        "Avg Runtime": f"{stats.avg_runtime_time:.4f}s",
        "Avg Validation Time": f"{stats.avg_validation_time:.4f}s",
        "Avg Total Time": f"{stats.avg_total_time:.2f}s",
        "Avg Repair Rounds": f"{stats.avg_repair_rounds:.2f}",
        "Programs Repaired": stats.programs_repaired,
        "Repair Gain": stats.repair_success_gain,
        "Avg Generated Tests": f"{stats.avg_generated_tests:.1f}",
        "Avg Executed Tests": f"{stats.avg_executed_tests:.1f}",
        "Avg Passed Tests": f"{stats.avg_passed_tests:.1f}",
        "Avg Test Success Rate": f"{stats.avg_test_success_rate:.1f}%",
    }
