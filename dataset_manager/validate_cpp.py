"""
dataset_manager/validate_cpp.py — Compile Validation
=====================================================

Compiles every unique ``.cpp`` file under ``raw_cpp/`` using ``g++``
and records per-file compile results.

Only files that **compile successfully** are retained for the final
benchmark dataset.  Failed files are logged but never deleted.

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
from typing import List, Set, Tuple

from dataset_manager.utils import (
    RAW_CPP_DIR,
    REPORTS_DIR,
    Logger,
    run_command,
    write_csv,
)


# ============================================================================
# Single-file compilation
# ============================================================================

def _compile_one(cpp_file: str, timeout: int = 30) -> Tuple[bool, str, float]:
    """Attempt to compile a single C++ file with ``g++ -std=c++17``.

    Args:
        cpp_file: Absolute path to a ``.cpp`` file.
        timeout: Compilation timeout in seconds.

    Returns:
        ``(passed, error_message, elapsed_seconds)``.
    """
    fd, exe_path = tempfile.mkstemp(suffix=".out")
    os.close(fd)

    t0 = time.time()
    ok, stdout, stderr = run_command(
        ["g++", "-std=c++17", "-w", "-o", exe_path, cpp_file],
        timeout=timeout,
    )
    elapsed = time.time() - t0

    # Clean up the temporary executable
    try:
        os.remove(exe_path)
    except OSError:
        pass

    if ok:
        return (True, "", elapsed)
    else:
        # Truncate long error messages
        err = (stderr or stdout or "Unknown compilation error")[:300]
        return (False, err, elapsed)


# ============================================================================
# Batch compilation
# ============================================================================

def _compile_all(
    files: list[str],
    logger: Logger,
    timeout: int = 30,
) -> Tuple[List[str], List[str], List[List[str]]]:
    """Compile a batch of C++ files and record results.

    Args:
        files: List of absolute file paths.
        logger: :class:`Logger` instance.
        timeout: Per-file compilation timeout.

    Returns:
        ``(passed_paths, failed_paths, report_rows)``.
    """
    passed: list[str] = []
    failed: list[str] = []
    rows: list[list[str]] = []

    total = len(files)
    for i, cpp_file in enumerate(files):
        ok, err, elapsed = _compile_one(cpp_file, timeout=timeout)

        rel = os.path.relpath(cpp_file, RAW_CPP_DIR)
        rows.append([rel, "PASS" if ok else "FAIL", f"{elapsed:.3f}", err])

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
    return passed, failed, rows


# ============================================================================
# Public helper (used by pipeline)
# ============================================================================

def get_compilable_files(file_list: List[str] | None = None) -> list[str]:
    """Return the subset of *file_list* that compile successfully.

    Args:
        file_list: List of absolute ``.cpp`` paths.  When ``None``,
            all files under ``raw_cpp/`` are checked.

    Returns:
        List of compiled file paths.
    """
    from dataset_manager.deduplicate import get_unique_files

    if file_list is None:
        file_list = sorted(get_unique_files())

    passed: list[str] = []
    for cpp_file in file_list:
        ok, _, _ = _compile_one(cpp_file)
        if ok:
            passed.append(cpp_file)
    return passed


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    """Compile all unique C++ files and write ``compile_report.csv``."""
    from dataset_manager.deduplicate import get_unique_files

    logger = Logger("validate_cpp")

    if not os.path.isdir(RAW_CPP_DIR):
        logger.error(f"raw_cpp/ directory not found: {RAW_CPP_DIR}")
        sys.exit(1)

    # Only validate unique files (skip duplicates)
    unique = sorted(get_unique_files())
    if not unique:
        logger.error("No unique C++ files found in raw_cpp/")
        sys.exit(1)

    logger.info(f"Unique files to validate: {len(unique)}")
    logger.info("Compiling with g++ -std=c++17 -w …")

    passed, failed, rows = _compile_all(unique, logger)

    logger.info(
        f"Compile validation complete: "
        f"{len(passed)} passed, {len(failed)} failed "
        f"({len(passed) / max(len(unique), 1) * 100:.1f}% pass rate)"
    )

    # Write compile report
    report_path = os.path.join(REPORTS_DIR, "compile_report.csv")
    write_csv(
        report_path,
        ["File", "Status", "CompileTimeSeconds", "Error"],
        rows,
    )
    logger.info(f"Report written: {report_path}")

    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
