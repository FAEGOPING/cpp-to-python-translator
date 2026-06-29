"""
dataset_manager/validate_cpp.py — Compile Validation
=====================================================

Compiles every unique ``.cpp`` file under ``raw_cpp/`` using ``g++``
and records comprehensive per-file compile results including timing,
warnings, errors, and return codes.

Compile failures **never** stop the pipeline — remaining files
continue to be processed.

Outputs:
    ``reports/compile_report.csv`` — per-file compile results.

Usage::

    python dataset_manager/validate_cpp.py
"""

from __future__ import annotations
import os as _os
import sys as _sys
_DEPS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _DEPS not in _sys.path:
    _sys.path.insert(0, _DEPS)

import os
import sys
import tempfile
import time
from typing import List, Tuple

from dataset_manager.utils import (
    RAW_CPP_DIR,
    REPORTS_DIR,
    Logger,
    run_command_full,
    write_csv,
)


def _compile_one_full(
    cpp_file: str,
    timeout: int = 30,
) -> Tuple[bool, str, str, float, int, int, int]:
    """Compile a single C++ file and return full diagnostics.

    Args:
        cpp_file: Absolute path to a ``.cpp`` file.
        timeout: Compilation timeout in seconds.

    Returns:
        ``(passed, stderr, stdout, elapsed, warning_count,
        error_count, return_code)``.
    """
    fd, exe_path = tempfile.mkstemp(suffix=".out")
    os.close(fd)

    rc, stdout, stderr, elapsed = run_command_full(
        ["g++", "-std=c++17", "-w", "-o", exe_path, cpp_file],
        timeout=timeout,
    )

    try:
        os.remove(exe_path)
    except OSError:
        pass

    passed = (rc == 0)
    # Count warnings/errors in stderr (rough heuristic)
    lower_err = stderr.lower()
    warn_count = lower_err.count("warning:")
    error_count = lower_err.count("error:")

    return (passed, stderr.strip(), stdout.strip(), elapsed,
            warn_count, error_count, rc)


def _compile_all(
    files: List[str],
    logger: Logger,
    timeout: int = 30,
) -> Tuple[List[str], List[str], dict[str, float], List[List]]:
    """Compile all *files* and record per-file diagnostics.

    Args:
        files: List of absolute ``.cpp`` paths.
        logger: :class:`Logger` instance.
        timeout: Per-file compilation timeout in seconds.

    Returns:
        ``(passed_paths, failed_paths, compile_times, report_rows)``.
    """
    passed: list[str] = []
    failed: list[str] = []
    compile_times: dict[str, float] = {}
    rows: list[list] = []

    total = len(files)
    for i, cpp_file in enumerate(files):
        ok, stderr, stdout, elapsed, warns, errs, rc = \
            _compile_one_full(cpp_file, timeout=timeout)

        rel = os.path.relpath(cpp_file, RAW_CPP_DIR)
        status = "PASS" if ok else "FAIL"

        rows.append([
            rel, status, f"{elapsed:.3f}",
            stderr[:200] if stderr else "",
            stdout[:200] if stdout else "",
            warns, errs, rc,
        ])

        compile_times[cpp_file] = elapsed

        if ok:
            passed.append(cpp_file)
        else:
            failed.append(cpp_file)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            logger.info(
                f"  Compiled {i + 1}/{total}  "
                f"(pass={len(passed)} fail={len(failed)})"
            )

    logger.count("compile_pass", len(passed))
    logger.count("compile_fail", len(failed))
    return passed, failed, compile_times, rows


# ============================================================================
# Public helpers (used by pipeline and metadata generator)
# ============================================================================

def get_compilable_files(
    file_list: List[str] | None = None,
    timeout: int = 30,
) -> List[str]:
    """Return the subset of files that compile successfully.

    Args:
        file_list: List of absolute ``.cpp`` paths.  When ``None``,
            all unique files under ``raw_cpp/`` are checked.
        timeout: Per-file compilation timeout.

    Returns:
        List of compiled file paths.
    """
    from dataset_manager.deduplicate import get_unique_files

    if file_list is None:
        file_list = sorted(get_unique_files())

    passed: list[str] = []
    for cpp_file in file_list:
        ok, _, _, _, _, _, _ = _compile_one_full(cpp_file, timeout=timeout)
        if ok:
            passed.append(cpp_file)
    return passed


def get_compilable_files_with_times(
    file_list: List[str] | None = None,
    logger: Logger | None = None,
    timeout: int = 30,
) -> Tuple[List[str], dict[str, float]]:
    """Return compilable file paths and per-file compile times.

    Args:
        file_list: List of absolute ``.cpp`` paths.
        logger: Optional logger.
        timeout: Per-file compilation timeout.

    Returns:
        ``(compilable_paths, {path: compile_time_seconds})``.
    """
    from dataset_manager.deduplicate import get_unique_files

    if file_list is None:
        file_list = sorted(get_unique_files())

    passed: list[str] = []
    times: dict[str, float] = {}
    total = len(file_list)

    for i, cpp_file in enumerate(file_list):
        ok, _, _, elapsed, _, _, _ = _compile_one_full(cpp_file, timeout=timeout)
        times[cpp_file] = elapsed
        if ok:
            passed.append(cpp_file)
        if logger and (i + 1) % 1000 == 0:
            logger.info(f"  Compile validation: {i + 1}/{total}")

    return passed, times


# ============================================================================
# Main entry point
# ============================================================================

_COMPILE_HEADER = [
    "File", "Status", "CompileTimeSeconds", "Stderr",
    "Stdout", "WarningCount", "ErrorCount", "ReturnCode",
]


def main() -> None:
    """Compile all unique files and write ``compile_report.csv``."""
    from dataset_manager.deduplicate import get_unique_files

    logger = Logger("validate_cpp")

    if not os.path.isdir(RAW_CPP_DIR):
        logger.error(f"raw_cpp/ not found: {RAW_CPP_DIR}")
        sys.exit(1)

    unique = sorted(get_unique_files())
    if not unique:
        logger.error("No unique C++ files found.")
        sys.exit(1)

    logger.info(f"Unique files to validate: {len(unique)}")
    logger.info("Compiling with g++ -std=c++17 -w …")

    passed, failed, _, rows = _compile_all(unique, logger)

    compile_rate = len(passed) / max(len(unique), 1) * 100
    logger.info(
        f"Compile validation complete: "
        f"{len(passed)} passed, {len(failed)} failed "
        f"({compile_rate:.1f}% pass rate)"
    )

    report_path = os.path.join(REPORTS_DIR, "compile_report.csv")
    write_csv(report_path, list(_COMPILE_HEADER), rows)
    logger.info(f"Report written: {report_path}")

    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
