"""
differential_testing.py — Differential Testing Framework
=========================================================

Runs every available test case through both the original C++ program
and the translated Python program, then compares outputs.  The
translation is considered correct **only when all test cases pass**.

Provides detailed mismatch reports and smart failure compression for
inclusion in repair prompts (selects representative failures to avoid
excessive prompt size).

Usage::

    from differential_testing import run_differential_tests, DiffReport

    report = run_differential_tests(
        cpp_file="samples/example.cpp",
        python_code="...",
        test_cases=[("5\\n", None), ("42\\n", None)],
        run_cpp_fn=run_cpp,
        run_python_fn=run_python,
    )
    print(report.summary)
    print(report.passed, report.failed)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# Type alias for a single test case
TestCase = Tuple[str, Optional[str]]
"""``(input_string, expected_output_or_None)``.

When *expected_output* is ``None``, the C++ program's output is used
as the oracle.
"""

# Maximum number of detailed failures to include in a full mismatch report
_MAX_MISMATCH_DETAIL = 10

# Maximum number of failed tests to include in a compact repair summary
_MAX_COMPACT_FAILURES = 8


@dataclass
class SingleTestResult:
    """Result of running one test case through both programs.

    Attributes:
        test_index: Zero-based index of this test case.
        test_input: Raw stdin input provided.
        expected_output: Expected (oracle) output — from ``.out`` file
            or C++ execution.
        actual_output: Output produced by the translated Python program.
        passed: ``True`` when expected and actual outputs match.
        error: Human-readable error description when the test did not pass.
        cpp_ok: ``False`` when the C++ oracle itself failed to execute.
        python_ok: ``False`` when the Python program crashed or timed out.
    """

    test_index: int
    test_input: str
    expected_output: str
    actual_output: str
    passed: bool
    error: str = ""
    cpp_ok: bool = True
    python_ok: bool = True


@dataclass
class DiffReport:
    """Aggregate differential testing report for one translation attempt.

    Attributes:
        test_results: Per-test-case results in order.
    """

    test_results: List[SingleTestResult] = field(default_factory=list)

    # -- derived properties --------------------------------------------------

    @property
    def total(self) -> int:
        """Total number of test cases executed."""
        return len(self.test_results)

    @property
    def passed(self) -> int:
        """Number of test cases that passed."""
        return sum(1 for r in self.test_results if r.passed)

    @property
    def failed(self) -> int:
        """Number of test cases that failed."""
        return self.total - self.passed

    @property
    def all_passed(self) -> bool:
        """``True`` when every test case passed."""
        return self.failed == 0 and self.total > 0

    @property
    def summary(self) -> str:
        """One-line summary string."""
        return (
            f"Differential Testing: {self.passed}/{self.total} passed"
            + ("" if self.all_passed else f", {self.failed} FAILED")
        )

    # -- reporting -----------------------------------------------------------

    def mismatch_report(self, max_cases: int = _MAX_MISMATCH_DETAIL) -> str:
        """Human-readable report of all failing test cases.

        Args:
            max_cases: Maximum number of failures to include in detail.

        Returns:
            Formatted multi-line report string.
        """
        failures = [r for r in self.test_results if not r.passed]
        if not failures:
            return "All test cases passed — no mismatches."

        lines: list[str] = []
        lines.append(f"{'=' * 60}")
        lines.append("DIFFERENTIAL TESTING FAILURE REPORT")
        lines.append(f"{'=' * 60}")
        lines.append(
            f"  Total: {self.total}  |  Passed: {self.passed}"
            f"  |  Failed: {self.failed}"
        )
        lines.append("")

        for r in failures[:max_cases]:
            lines.append(
                f"--- Failed Test #{r.test_index + 1} "
                f"(index {r.test_index}) ---"
            )
            lines.append("  Input:")
            for line in r.test_input.rstrip("\n").split("\n"):
                lines.append(f"    {line}")
            lines.append("  Expected Output:")
            for line in r.expected_output.split("\n"):
                lines.append(f"    {line}")
            lines.append("  Actual Output:")
            for line in r.actual_output.split("\n"):
                lines.append(f"    {line}")
            if r.error:
                lines.append("  Error:")
                for line in r.error.split("\n"):
                    lines.append(f"    {line}")
            lines.append("")

        if len(failures) > max_cases:
            lines.append(
                f"  ... and {len(failures) - max_cases} more failures "
                f"(showing first {max_cases})."
            )
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def compact_failure_summary(self) -> str:
        """Compact failure summary for inclusion in a repair prompt.

        Uses **smart selection**: when many tests fail, only the most
        representative failures are included — the first failure, the
        test with the smallest input, the test with the largest input,
        and a random sample from the remainder.  This keeps repair
        prompts manageable while preserving diagnostic value.

        Returns:
            A string suitable for pasting into an LLM context window.
        """
        failures = [r for r in self.test_results if not r.passed]
        if not failures:
            return "All tests passed."

        selected = _select_representative_failures(failures)
        total_failures = len(failures)

        lines = [f"FAILED TEST CASES ({len(selected)} shown of {total_failures} total):"]
        for r in selected:
            lines.append(
                f"  Test {r.test_index}: "
                f"input={r.test_input.strip()!r} | "
                f"expected={r.expected_output!r} | "
                f"got={r.actual_output!r}"
                + (f" | error={r.error!r}" if r.error else "")
            )

        if total_failures > len(selected):
            lines.append(
                f"  ... ({total_failures - len(selected)} more failures omitted "
                f"— the {len(selected)} most representative are shown above)"
            )
        return "\n".join(lines)


# ======================================================================
# Smart failure selection
# ======================================================================

def _select_representative_failures(
    failures: list[SingleTestResult],
) -> list[SingleTestResult]:
    """Select a representative subset of failing test cases.

    Selection strategy:
        1. The **first** failure (earliest test index).
        2. The failure with the **smallest** input (shortest string).
        3. The failure with the **largest** input (longest string).
        4. Up to 2 additional failures chosen at **random** from the
           remainder.

    This produces a diverse set of 4–5 failures that cover different
    input sizes while keeping repair prompts concise.

    Args:
        failures: All failing :class:`SingleTestResult` objects.

    Returns:
        A deduplicated subset of representative failures.
    """
    n = len(failures)
    if n <= _MAX_COMPACT_FAILURES:
        return failures

    selected: list[SingleTestResult] = []
    used_indices: set[int] = set()

    # 1. First failure
    selected.append(failures[0])
    used_indices.add(0)

    # 2. Smallest input (by string length)
    if n > 1:
        sorted_by_len = sorted(
            enumerate(failures), key=lambda x: len(x[1].test_input)
        )
        for idx, _ in sorted_by_len:
            if idx not in used_indices:
                selected.append(failures[idx])
                used_indices.add(idx)
                break

    # 3. Largest input
    if n > 2:
        for idx, _ in reversed(sorted_by_len):
            if idx not in used_indices:
                selected.append(failures[idx])
                used_indices.add(idx)
                break

    # 4. Up to 2 random from the remainder
    remaining = [i for i in range(n) if i not in used_indices]
    if remaining:
        rng = random.Random(42)  # deterministic seed for reproducibility
        for _ in range(min(2, len(remaining))):
            pick = rng.choice(remaining)
            selected.append(failures[pick])
            remaining.remove(pick)

    # Sort by original test index for consistent output
    selected.sort(key=lambda r: r.test_index)
    return selected


# ======================================================================
# Core differential testing runner
# ======================================================================

def run_differential_tests(
    cpp_file: str,
    python_code: str,
    test_cases: List[TestCase],
    *,
    run_cpp_fn: Callable[[str, str], Tuple[bool, str]],
    run_python_fn: Callable[[str, str], Tuple[bool, str]],
    cache: Optional[object] = None,
) -> DiffReport:
    """Execute all *test_cases* through both C++ and Python, comparing outputs.

    Args:
        cpp_file: Path to the C++ source file.
        python_code: Translated Python source code.
        test_cases: List of ``(input, expected_output_or_None)`` pairs.
        run_cpp_fn: Callable with signature
            ``(cpp_file, input) -> (ok, output)``.
        run_python_fn: Callable with signature
            ``(code, input) -> (ok, output)``.
        cache: Optional :class:`ExecutionCache` instance for avoiding
            duplicate executions.

    Returns:
        A :class:`DiffReport` with per-test results and aggregate
        statistics.
    """
    results: list[SingleTestResult] = []

    for idx, (test_input, expected) in enumerate(test_cases):
        # -- Run C++ oracle -------------------------------------------------
        cpp_ok, cpp_output = _run_cpp_maybe_cached(
            cpp_file, test_input, run_cpp_fn, cache
        )

        if not cpp_ok:
            results.append(
                SingleTestResult(
                    test_index=idx,
                    test_input=test_input,
                    expected_output="",
                    actual_output="",
                    passed=False,
                    error=f"C++ oracle execution failed: {cpp_output}",
                    cpp_ok=False,
                    python_ok=True,
                )
            )
            continue

        oracle_output = (
            expected if expected is not None else cpp_output
        ).strip()

        # -- Run translated Python ------------------------------------------
        py_ok, py_output = _run_python_maybe_cached(
            python_code, test_input, run_python_fn, cache
        )

        if not py_ok:
            results.append(
                SingleTestResult(
                    test_index=idx,
                    test_input=test_input,
                    expected_output=oracle_output,
                    actual_output="",
                    passed=False,
                    error=f"Python execution failed: {py_output}",
                    cpp_ok=True,
                    python_ok=False,
                )
            )
            continue

        actual_output = py_output.strip()

        # -- Compare ---------------------------------------------------------
        if oracle_output == actual_output:
            results.append(
                SingleTestResult(
                    test_index=idx,
                    test_input=test_input,
                    expected_output=oracle_output,
                    actual_output=actual_output,
                    passed=True,
                )
            )
        else:
            results.append(
                SingleTestResult(
                    test_index=idx,
                    test_input=test_input,
                    expected_output=oracle_output,
                    actual_output=actual_output,
                    passed=False,
                    error=(
                        f"Output mismatch: "
                        f"expected {oracle_output!r}, "
                        f"got {actual_output!r}"
                    ),
                )
            )

    return DiffReport(test_results=results)


# ======================================================================
# Internal helpers
# ======================================================================

def _run_cpp_maybe_cached(
    cpp_file: str,
    test_input: str,
    run_cpp_fn: Callable[[str, str], Tuple[bool, str]],
    cache: Optional[object],
) -> Tuple[bool, str]:
    """Run C++ program, using cache if available.

    Args:
        cpp_file: Path to the C++ source.
        test_input: Stdin input.
        run_cpp_fn: Raw executor function.
        cache: Optional :class:`ExecutionCache` instance.

    Returns:
        ``(success, output_or_error)`` tuple.
    """
    if cache is not None:
        return cache.run_cpp(cpp_file, test_input, run_cpp_fn)
    return run_cpp_fn(cpp_file, test_input)


def _run_python_maybe_cached(
    python_code: str,
    test_input: str,
    run_python_fn: Callable[[str, str], Tuple[bool, str]],
    cache: Optional[object],
) -> Tuple[bool, str]:
    """Run Python code, using cache if available.

    Args:
        python_code: Python source code.
        test_input: Stdin input.
        run_python_fn: Raw executor function.
        cache: Optional :class:`ExecutionCache` instance.

    Returns:
        ``(success, output_or_error)`` tuple.
    """
    if cache is not None:
        return cache.run_python(python_code, test_input, run_python_fn)
    return run_python_fn(python_code, test_input)
