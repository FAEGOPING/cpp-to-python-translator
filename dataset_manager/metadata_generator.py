"""
dataset_manager/metadata_generator.py — Metadata Generator
============================================================

Analyses compiled C++ source files and generates ``metadata.csv``
with comprehensive per-file software-engineering metrics including
code structure, control flow, complexity, STL usage, and provenance.

Outputs:
    ``reports/metadata.csv``

Usage::

    python dataset_manager/metadata_generator.py
"""

from __future__ import annotations
import os as _os
import sys as _sys
_DEPS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _DEPS not in _sys.path:
    _sys.path.insert(0, _DEPS)

import os
import sys
from typing import List, Optional, Set

from dataset_manager.utils import (
    RAW_CPP_DIR,
    REPORTS_DIR,
    DATASETS_DIR,
    Logger,
    count_lines,
    count_patterns,
    cyclomatic_complexity,
    nesting_depth,
    detect_stl_usage,
    extract_source_info,
    write_csv,
)

# Compiler version cache
_COMPILER_VERSION: str = ""


def _get_compiler_version() -> str:
    """Get g++ version string (cached).

    Returns:
        Compiler version string, or ``"unknown"``.
    """
    global _COMPILER_VERSION
    if _COMPILER_VERSION:
        return _COMPILER_VERSION
    from dataset_manager.utils import run_command
    ok, stdout, _ = run_command(["g++", "--version"], timeout=5)
    _COMPILER_VERSION = stdout.split("\n")[0].strip() if ok else "unknown"
    return _COMPILER_VERSION


# ============================================================================
# Per-file analysis
# ============================================================================

def _analyse_one(
    file_path: str,
    compile_passed: Set[str],
    compile_times: dict[str, float],
) -> Optional[list]:
    """Analyse a single C++ file and return a metadata row.

    Args:
        file_path: Absolute path to ``.cpp`` file.
        compile_passed: Set of paths known to compile.
        compile_times: Mapping file_path → compile time in seconds.

    Returns:
        List of values for the metadata CSV, or ``None`` on error.
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    except OSError:
        return None

    size_bytes = len(source.encode("utf-8"))

    # Line counts
    lines = count_lines(source)

    # Patterns
    patterns = count_patterns(source)

    # Complexity
    cyclo = cyclomatic_complexity(source)
    nest = nesting_depth(source)

    # STL
    stl = detect_stl_usage(source)

    # Compile info
    compiles = "PASS" if file_path in compile_passed else "FAIL"
    compile_time = compile_times.get(file_path, 0.0)

    # Source info
    category, repo, rel_path, gh_url = extract_source_info(file_path, RAW_CPP_DIR)

    # Program type classification (from filter stage)
    from dataset_manager.filter_programs import get_filter_classification
    classification = get_filter_classification()
    prog_type = "unknown"
    compile_eligible = "FALSE"
    if file_path in classification.get("executable", set()):
        prog_type = "executable"
        compile_eligible = "TRUE"
    elif file_path in classification.get("library", set()):
        prog_type = "library"
    elif file_path in classification.get("test", set()):
        prog_type = "test"
    elif file_path in classification.get("dependency", set()):
        prog_type = "dependency"

    return [
        rel_path,
        lines["total"], lines["code"], lines["blank"], lines["comments"],
        patterns["functions"], patterns["classes"], patterns["namespaces"],
        patterns["templates"], patterns["loops"], patterns["conditionals"],
        patterns["lambdas"], patterns["includes"],
        cyclo, nest,
        stl,
        compiles, f"{compile_time:.3f}",
        category, repo, gh_url,
        size_bytes,
        prog_type, compile_eligible,
    ]


# ============================================================================
# Batch generation
# ============================================================================

def generate_metadata(
    file_list: List[str],
    compile_passed: Optional[Set[str]] = None,
    compile_times: Optional[dict[str, float]] = None,
    logger: Optional[Logger] = None,
) -> List[List]:
    """Analyse all files in *file_list* and produce metadata rows.

    Args:
        file_list: List of absolute ``.cpp`` paths.
        compile_passed: Set of files known to compile.
        compile_times: Mapping of file → compile time.
        logger: Optional :class:`Logger` instance.

    Returns:
        List of row lists matching the metadata CSV header.
    """
    if compile_passed is None:
        compile_passed = set()
    if compile_times is None:
        compile_times = {}

    rows: list[list] = []
    total = len(file_list)
    compiled_count = 0

    for i, fpath in enumerate(file_list):
        row = _analyse_one(fpath, compile_passed, compile_times)
        if row is not None:
            rows.append(row)
            if fpath in compile_passed:
                compiled_count += 1

        if logger and (i + 1) % 1000 == 0:
            logger.info(
                f"  Analysed {i + 1}/{total}  "
                f"(compiled: {compiled_count})"
            )

    if logger:
        logger.count("files_analysed", len(rows))
        logger.count("files_compilable", compiled_count)

    return rows


# ============================================================================
# Main entry point
# ============================================================================

_HEADER = [
    "File", "TotalLines", "CodeLines", "BlankLines", "CommentLines",
    "Functions", "Classes", "Namespaces", "Templates",
    "Loops", "Conditionals", "Lambdas", "Includes",
    "CyclomaticComplexity", "NestingDepth", "STLUsage",
    "CompileStatus", "CompileTimeSeconds",
    "Category", "Repository", "GitHubURL",
    "FileSizeBytes",
    "ProgramType", "CompileEligible",
]


def main() -> None:
    """Analyse all unique C++ files and write ``metadata.csv``."""
    from dataset_manager.deduplicate import get_unique_files
    from dataset_manager.validate_cpp import get_compilable_files_with_times

    logger = Logger("metadata_generator")

    if not os.path.isdir(RAW_CPP_DIR):
        logger.error(f"raw_cpp/ not found: {RAW_CPP_DIR}")
        sys.exit(1)

    unique = sorted(get_unique_files())
    if not unique:
        logger.error("No unique C++ files found.")
        sys.exit(1)

    logger.info(f"Analysing {len(unique)} unique files …")

    # Get compile status + timing
    logger.info("Running compile validation …")
    compilable, compile_times = get_compilable_files_with_times(unique, logger)
    logger.info(f"  {len(compilable)}/{len(unique)} compile successfully")

    # Generate metadata
    rows = generate_metadata(unique, set(compilable), compile_times, logger)
    logger.info(f"Generated metadata for {len(rows)} files")

    report_path = os.path.join(REPORTS_DIR, "metadata.csv")
    write_csv(report_path, list(_HEADER), rows)
    logger.info(f"Report written: {report_path}")

    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
