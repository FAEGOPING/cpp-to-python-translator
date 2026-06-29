"""
dataset_manager/metadata_generator.py — Metadata Generator
============================================================

Analyses compiled C++ source files and generates ``metadata.csv``
with per-file statistics:

* Lines of Code (LOC)
* Function count
* Loop count (``for``, ``while``)
* Conditional count (``if``, ``else if``, ``switch``, ``case``)
* Header / ``#include`` count
* STL usage (vector, map, set, queue, stack, algorithm, etc.)
* File size (bytes)
* Source repository
* Dataset category
* Compile status

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
import re
import sys
from typing import Dict, List

from dataset_manager.utils import (
    RAW_CPP_DIR,
    REPORTS_DIR,
    Logger,
    write_csv,
)


# ============================================================================
# Analysis helpers
# ============================================================================

# Regex patterns (compiled once for performance)
_RE_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]', re.MULTILINE)
_RE_FUNCTION = re.compile(
    r'\b(?:void|int|long|float|double|char|bool|string|auto|'
    r'unsigned|size_t|ll|pair|vector|map|set)\s+'
    r'[a-zA-Z_]\w*\s*\([^)]*\)\s*\{'
)
_RE_FOR = re.compile(r'\bfor\s*\(', re.MULTILINE)
_RE_WHILE = re.compile(r'\bwhile\s*\(', re.MULTILINE)
_RE_IF = re.compile(r'\bif\s*\(', re.MULTILINE)
_RE_ELSE_IF = re.compile(r'\belse\s+if\s*\(', re.MULTILINE)
_RE_SWITCH = re.compile(r'\bswitch\s*\(', re.MULTILINE)
_RE_COMMENT_LINE = re.compile(r'^\s*//')
_RE_BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)

# STL identifiers to detect
_STL_TOKENS: tuple[str, ...] = (
    "vector", "map", "set", "queue", "stack", "deque",
    "priority_queue", "unordered_map", "unordered_set",
    "algorithm", "sort", "lower_bound", "upper_bound",
    "pair", "tuple", "string", "iostream", "fstream",
    "sstream", "cmath", "cstdio", "cstdlib", "cstring",
    "bits/stdc++.h",
)


def _count_lines_of_code(source: str) -> int:
    """Count non-blank, non-comment-only lines in *source*.

    Args:
        source: C++ source code string.

    Returns:
        Number of significant lines.
    """
    lines = source.split("\n")
    count = 0
    in_block = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            continue
        if "/*" in stripped:
            in_block = True
        if "*/" in stripped:
            in_block = False
            continue
        if in_block:
            continue
        count += 1
    return count


def _count_patterns(source: str) -> Dict[str, int]:
    """Count structural patterns in C++ source.

    Args:
        source: C++ source code string.

    Returns:
        Dict with keys ``functions``, ``loops``, ``conditionals``,
        ``includes``.
    """
    # Remove block comments to avoid matching patterns in them
    clean = _RE_BLOCK_COMMENT.sub(" ", source)

    functions = len(_RE_FUNCTION.findall(clean))
    loops = len(_RE_FOR.findall(clean)) + len(_RE_WHILE.findall(clean))
    conditionals = (
        len(_RE_IF.findall(clean))
        + len(_RE_ELSE_IF.findall(clean))
        + len(_RE_SWITCH.findall(clean))
    )
    includes = len(_RE_INCLUDE.findall(clean))

    return {
        "functions": functions,
        "loops": loops,
        "conditionals": conditionals,
        "includes": includes,
    }


def _detect_stl_usage(source: str) -> str:
    """Detect which STL headers/features are used.

    Args:
        source: C++ source code string.

    Returns:
        Comma-separated list of detected STL tokens, or ``"none"``.
    """
    found: list[str] = []
    lower = source.lower()
    for token in _STL_TOKENS:
        if token in lower:
            found.append(token)
    return ", ".join(sorted(found)) if found else "none"


def _extract_source_info(file_path: str) -> tuple[str, str, str]:
    """Extract category, repo, and author from a file path.

    Uses the convention ``raw_cpp/<category>/<repo>/...``.

    Args:
        file_path: Absolute path to a C++ file under ``raw_cpp/``.

    Returns:
        ``(category, repository, relative_source_path)``.
    """
    rel = os.path.relpath(file_path, RAW_CPP_DIR)
    parts = rel.split(os.sep)
    category = parts[0] if len(parts) > 0 else "unknown"
    repo = parts[1] if len(parts) > 1 else "unknown"
    return category, repo, rel


# ============================================================================
# Batch analysis
# ============================================================================

def generate_metadata(
    file_list: List[str],
    compile_passed: set[str] | None = None,
    logger: Logger | None = None,
) -> List[List[str]]:
    """Analyse each file in *file_list* and produce metadata rows.

    Args:
        file_list: List of absolute paths to ``.cpp`` files.
        compile_passed: Set of file paths known to compile.  When
            ``None``, compile status is marked as ``"unknown"``.
        logger: Optional :class:`Logger` instance.

    Returns:
        List of rows, each with columns matching the metadata CSV header:
        ``[File, LOC, Functions, Loops, Conditionals, Includes, STLUsage,
        FileSizeBytes, Category, Repository, CompileStatus, Author]``.
    """
    if compile_passed is None:
        compile_passed = set()

    rows: list[list[str]] = []
    total = len(file_list)

    for i, fpath in enumerate(file_list):
        try:
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError:
            continue

        size = len(source.encode("utf-8"))
        loc = _count_lines_of_code(source)
        patterns = _count_patterns(source)
        stl = _detect_stl_usage(source)
        category, repo, rel_path = _extract_source_info(fpath)
        compiles = "PASS" if fpath in compile_passed else "unknown"

        rows.append([
            rel_path,
            str(loc),
            str(patterns["functions"]),
            str(patterns["loops"]),
            str(patterns["conditionals"]),
            str(patterns["includes"]),
            stl,
            str(size),
            category,
            repo,
            compiles,
            "",  # Author (reserved for future use)
        ])

        if logger and (i + 1) % 1000 == 0:
            logger.info(f"  Analysed {i + 1}/{total} files …")

    if logger:
        logger.count("files_analysed", len(rows))

    return rows


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    """Analyse all unique C++ files and write ``metadata.csv``."""
    from dataset_manager.deduplicate import get_unique_files
    from dataset_manager.validate_cpp import get_compilable_files

    logger = Logger("metadata_generator")

    if not os.path.isdir(RAW_CPP_DIR):
        logger.error(f"raw_cpp/ directory not found: {RAW_CPP_DIR}")
        sys.exit(1)

    unique = sorted(get_unique_files())
    if not unique:
        logger.error("No unique C++ files found.")
        sys.exit(1)

    logger.info(f"Analysing {len(unique)} unique files …")

    # Determine compile-passed set
    logger.info("Identifying compilable files …")
    compilable = set(get_compilable_files(unique))
    logger.info(f"  {len(compilable)}/{len(unique)} files compile successfully")

    # Generate metadata
    rows = generate_metadata(unique, compilable, logger)
    logger.info(f"Generated metadata for {len(rows)} files")

    # Write report
    report_path = os.path.join(REPORTS_DIR, "metadata.csv")
    write_csv(
        report_path,
        [
            "File", "LOC", "Functions", "Loops", "Conditionals",
            "Includes", "STLUsage", "FileSizeBytes", "Category",
            "Repository", "CompileStatus", "Author",
        ],
        rows,
    )
    logger.info(f"Report written: {report_path}")

    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
