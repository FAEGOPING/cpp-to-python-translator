"""
dataset_manager/validate_cpp.py — Compile Validation
=====================================================

Compiles every unique executable ``.cpp`` file using its **original
repository path** (``~/datasets/...``) so that local headers
(``#include "xxx.h"``, ``#include "../common.hpp"``) resolve
correctly within the repository's directory structure.

Compilation uses the original repository directory as the working
directory, preserving the exact layout that the author intended.

Compile failures **never** stop the pipeline.

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
from typing import Dict, List, Tuple

from dataset_manager.utils import (
    RAW_CPP_DIR,
    DATASETS_DIR,
    REPORTS_DIR,
    Logger,
    run_command_full,
    write_csv,
    get_compiler,
)


# ============================================================================
# Path mapping — raw_cpp → original repository
# ============================================================================

def _raw_to_original(raw_cpp_path: str) -> str:
    """Map a ``raw_cpp/`` path back to the original repository path.

    ``raw_cpp/algorithms/C-Plus-Plus/search/binary.cpp``
    →
    ``~/datasets/algorithms/C-Plus-Plus/search/binary.cpp``

    Args:
        raw_cpp_path: Absolute path under ``raw_cpp/``.

    Returns:
        The corresponding absolute path under ``~/datasets/``.
        Falls back to *raw_cpp_path* if the mapping is not possible.
    """
    # Normalise both for reliable prefix replacement
    raw_norm = os.path.normpath(RAW_CPP_DIR)
    file_norm = os.path.normpath(raw_cpp_path)

    if file_norm.startswith(raw_norm + os.sep):
        rel = os.path.relpath(file_norm, raw_norm)
        original = os.path.join(DATASETS_DIR, rel)
        if os.path.isfile(original):
            return original

    # Fallback: try to find by relative path
    rel = os.path.relpath(raw_cpp_path, RAW_CPP_DIR)
    original = os.path.join(DATASETS_DIR, rel)
    if os.path.isfile(original):
        return original

    # Last resort: return the raw_cpp path itself
    return raw_cpp_path


def _get_compile_dir(original_path: str) -> str:
    """Return the directory from which to compile *original_path*.

    This is the directory containing the ``.cpp`` file, so that
    ``#include "local.h"`` resolves correctly.

    Args:
        original_path: Absolute path in ``~/datasets/``.

    Returns:
        The parent directory of *original_path*.
    """
    return os.path.dirname(original_path)


# ============================================================================
# Single-file compilation (compiles from ORIGINAL repo path)
# ============================================================================

def _compile_one_full(
    raw_cpp_file: str,
    timeout: int = 30,
) -> Tuple[bool, str, str, float, int, int, int, str, str, str, str, str, str, str]:
    """Compile a single C++ file using the **detected** C++ compiler.

    Maps ``raw_cpp_file`` to its original location under
    ``~/datasets/``, then compiles from that file's parent directory
    so local headers resolve correctly.

    Args:
        raw_cpp_file: Absolute path under ``raw_cpp/``.
        timeout: Compilation timeout in seconds.

    Returns:
        ``(passed, stderr, stdout, elapsed, warning_count,
        error_count, return_code, original_path, compile_dir,
        compiler_command, compiler_name, compiler_version,
        compiler_executable, compiler_standard)``.
    """
    original_path = _raw_to_original(raw_cpp_file)
    compile_dir = _get_compile_dir(original_path)
    compiler = get_compiler()

    fd, exe_path = tempfile.mkstemp(suffix=".out")
    os.close(fd)

    # Compile using the detected compiler from the original file's directory
    compile_cmd = [compiler["executable"], "-std=c++17", "-w", "-o", exe_path, original_path]

    rc, stdout, stderr, elapsed = run_command_full(
        compile_cmd,
        timeout=timeout,
    )

    try:
        os.remove(exe_path)
    except OSError:
        pass

    passed = (rc == 0)
    lower_err = stderr.lower()
    warn_count = lower_err.count("warning:")
    error_count = lower_err.count("error:")

    cmd_str = " ".join(compile_cmd)

    return (passed, stderr.strip(), stdout.strip(), elapsed,
            warn_count, error_count, rc,
            original_path, compile_dir, cmd_str,
            compiler["name"], compiler["version"],
            compiler["executable"], compiler["standard"])


# ============================================================================
# Batch compilation
# ============================================================================

def _compile_all(
    files: List[str],
    logger: Logger,
    timeout: int = 30,
) -> Tuple[List[str], List[str], dict[str, float], List[List]]:
    """Compile all *files* using their original repository paths.

    Args:
        files: List of absolute ``raw_cpp/`` paths.
        logger: :class:`Logger` instance.
        timeout: Per-file compilation timeout.

    Returns:
        ``(passed_raw_cpp_paths, failed_raw_cpp_paths, compile_times,
        report_rows)``.

        The returned paths are still ``raw_cpp/`` paths for downstream
        compatibility.
    """
    passed: list[str] = []
    failed: list[str] = []
    compile_times: dict[str, float] = {}
    rows: list[list] = []

    total = len(files)
    for i, raw_cpp_file in enumerate(files):
        (ok, stderr, stdout, elapsed, warns, errs, rc,
         original_path, compile_dir, cmd_str,
         compiler_name, compiler_ver, compiler_exe, compiler_std) = \
            _compile_one_full(raw_cpp_file, timeout=timeout)

        # Use raw_cpp path for reporting (consistent with other stages)
        rel = os.path.relpath(raw_cpp_file, RAW_CPP_DIR)
        status = "PASS" if ok else "FAIL"

        rows.append([
            rel,                    # File (relative in raw_cpp)
            status,                 # Status
            f"{elapsed:.3f}",       # CompileTimeSeconds
            stderr[:200] if stderr else "",              # Stderr
            stdout[:200] if stdout else "",              # Stdout
            warns,                  # WarningCount
            errs,                   # ErrorCount
            rc,                     # ReturnCode
            original_path,          # OriginalPath
            compile_dir,            # CompileDirectory
            cmd_str,                # CompilerCommand
            compiler_name,          # Compiler
            compiler_ver,           # CompilerVersion
            compiler_exe,           # CompilerExecutable
            compiler_std,           # CompilerStandard
        ])

        # Track compile time keyed by raw_cpp path (downstream consumers
        # use raw_cpp paths to identify files)
        compile_times[raw_cpp_file] = elapsed

        if ok:
            passed.append(raw_cpp_file)
        else:
            failed.append(raw_cpp_file)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            logger.info(
                f"  Compiled {i + 1}/{total}  "
                f"(pass={len(passed)} fail={len(failed)})"
            )

    logger.count("compile_pass", len(passed))
    logger.count("compile_fail", len(failed))
    return passed, failed, compile_times, rows


# ============================================================================
# Public helpers (used by build_dataset and metadata_generator)
# ============================================================================

def get_compilable_files(
    file_list: List[str] | None = None,
    timeout: int = 30,
) -> List[str]:
    """Return the subset of files that compile successfully.

    Compiles from original repository paths but returns ``raw_cpp/``
    paths for downstream compatibility.

    Args:
        file_list: List of absolute ``raw_cpp/`` paths.  When ``None``,
            all unique executable files are checked.
        timeout: Per-file compilation timeout.

    Returns:
        List of ``raw_cpp/`` file paths that compile successfully.
    """
    from dataset_manager.deduplicate import get_unique_files
    from dataset_manager.filter_programs import get_executable_files

    if file_list is None:
        executable = get_executable_files(RAW_CPP_DIR, None)
        unique = get_unique_files() if os.path.isdir(RAW_CPP_DIR) else set()
        file_list = sorted(executable & unique) if unique else sorted(executable)

    passed: list[str] = []
    for raw_cpp_file in file_list:
        (ok, _, _, _, _, _, _, _, _, _, _, _, _, _) = \
            _compile_one_full(raw_cpp_file, timeout=timeout)
        if ok:
            passed.append(raw_cpp_file)
    return passed


def get_compilable_files_with_times(
    file_list: List[str] | None = None,
    logger: Logger | None = None,
    timeout: int = 30,
) -> Tuple[List[str], dict[str, float]]:
    """Return compilable files and per-file compile times.

    Compiles from original repository paths.  Returned paths are
    ``raw_cpp/`` paths; times are keyed by ``raw_cpp/`` paths.

    Args:
        file_list: List of absolute ``raw_cpp/`` paths.
        logger: Optional logger.
        timeout: Per-file compilation timeout.

    Returns:
        ``(compilable_raw_cpp_paths, {raw_cpp_path: compile_time})``.
    """
    from dataset_manager.deduplicate import get_unique_files
    from dataset_manager.filter_programs import get_executable_files

    if file_list is None:
        executable = get_executable_files(RAW_CPP_DIR, logger)
        unique = get_unique_files() if os.path.isdir(RAW_CPP_DIR) else set()
        file_list = sorted(executable & unique) if unique else sorted(executable)

    passed: list[str] = []
    times: dict[str, float] = {}
    total = len(file_list)

    for i, raw_cpp_file in enumerate(file_list):
        (ok, _, _, elapsed, _, _, _, _, _, _, _, _, _, _) = \
            _compile_one_full(raw_cpp_file, timeout=timeout)
        times[raw_cpp_file] = elapsed
        if ok:
            passed.append(raw_cpp_file)
        if logger and (i + 1) % 1000 == 0:
            logger.info(f"  Compile validation: {i + 1}/{total}")

    return passed, times


# ============================================================================
# Main entry point
# ============================================================================

_COMPILE_HEADER = [
    "File", "Status", "CompileTimeSeconds", "Stderr",
    "Stdout", "WarningCount", "ErrorCount", "ReturnCode",
    "OriginalPath", "CompileDirectory", "CompilerCommand",
    "Compiler", "CompilerVersion", "CompilerExecutable", "CompilerStandard",
]


def main() -> None:
    """Compile all executable unique files from original repo paths."""
    from dataset_manager.deduplicate import get_unique_files
    from dataset_manager.filter_programs import get_executable_files

    logger = Logger("validate_cpp")

    if not os.path.isdir(RAW_CPP_DIR):
        logger.error(f"raw_cpp/ not found: {RAW_CPP_DIR}")
        sys.exit(1)

    # Only validate executable programs
    executable = get_executable_files(RAW_CPP_DIR, logger)
    if not executable:
        logger.error("No executable C++ files found. Run filtering first.")
        sys.exit(1)

    logger.info(f"Executable files found: {len(executable)}")
    logger.info(f"Compiling from original repository paths under: {DATASETS_DIR}")

    # Intersect with unique files (post-dedup)
    unique = get_unique_files() if os.path.isdir(RAW_CPP_DIR) else set()
    candidates = sorted(executable & unique) if unique else sorted(executable)

    if not candidates:
        logger.error("No executable unique files found.")
        sys.exit(1)

    # Log compiler information
    compiler = get_compiler()
    logger.info(f"Compiler:")
    logger.info(f"  Name:       {compiler['name']}")
    logger.info(f"  Executable: {compiler['executable']}")
    logger.info(f"  Standard:   C++{compiler['standard'].replace('c++', '')}")

    logger.info(f"Unique executable files to validate: {len(candidates)}")
    logger.info(f"Compiling with {compiler['executable']} -std=c++17 "
                f"from original repo directories …")

    passed, failed, _, rows = _compile_all(candidates, logger)

    compile_rate = len(passed) / max(len(candidates), 1) * 100
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
