"""
dataset_manager/filter_programs.py — Executable Program Filter
================================================================

Analyses every extracted C++ file and classifies it as one of:

* **executable** — contains a ``main()`` entry point
* **library** — no entry point (helper/library code)
* **test** — uses a testing framework (Google Test, Catch2, etc.)
* **dependency** — requires unresolvable local ``#include`` headers

Only executable programs advance to the benchmark dataset.

Outputs:
    ``reports/filter_report.csv``      — aggregate classification counts
    ``reports/library_files.csv``      — excluded library files
    ``reports/test_files.csv``         — excluded test files
    ``reports/dependency_report.csv``  — files with missing local headers

Public API (used by downstream stages):
    :func:`get_executable_files` → ``set[str]`` of executable ``.cpp`` paths

Usage::

    python dataset_manager/filter_programs.py
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
from typing import Dict, List, Set, Tuple

from dataset_manager.utils import (
    RAW_CPP_DIR,
    REPORTS_DIR,
    Logger,
    write_csv,
    timestamp,
)


# ============================================================================
# Detection patterns (compiled once)
# ============================================================================

# Entry-point patterns — must match the main() declaration, not a call to it
_RE_MAIN = re.compile(
    r'\b(?:int|signed)\s+main\s*\(\s*(?:void)?\s*\)',
    re.MULTILINE,
)
"""Matches ``int main()``, ``int main(void)``, ``signed main()``."""

_RE_MAIN_ARGS = re.compile(
    r'\b(?:int|signed)\s+main\s*\([^)]*\bchar\b',
    re.MULTILINE,
)
"""Matches ``int main(int argc, char* argv[])`` and variants."""

_RE_VOID_MAIN = re.compile(
    r'\bvoid\s+main\s*\(\s*(?:void)?\s*\)',
    re.MULTILINE,
)
"""Matches ``void main()`` (common in competitive programming)."""

# Test framework patterns — broad enough to catch macro names
_TEST_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("GoogleTest", re.compile(
        r'\bTEST\s*\(\s*\w+\s*,\s*\w+\s*\)'
        r'|\bTEST_F\s*\(\s*\w+\s*,\s*\w+\s*\)'
        r'|\bTEST_P\s*\(\s*\w+\s*,\s*\w+\s*\)'
        r'|\bINSTANTIATE_TEST_SUITE_P\s*\(',
    )),
    ("Catch2", re.compile(
        r'\bTEST_CASE\s*\(\s*"[^"]*"\s*\)'
        r'|\bSECTION\s*\(\s*"[^"]*"\s*\)'
        r'|\bREQUIRE\s*\(|\bCHECK\s*\(',
    )),
    ("doctest", re.compile(
        r'\bTEST_CASE\s*\(\s*"[^"]*"\s*\)'
        r'|\bSUBCASE\s*\(\s*"[^"]*"\s*\)'
        r'|\bDOCTEST_',
    )),
    ("BoostTest", re.compile(
        r'\bBOOST_AUTO_TEST_CASE\s*\('
        r'|\bBOOST_FIXTURE_TEST_CASE\s*\('
        r'|\bBOOST_TEST_MODULE\b'
        r'|\bBOOST_AUTO_TEST_SUITE\s*\(',
    )),
    ("Benchmark", re.compile(
        r'\bBENCHMARK\s*\(\s*\w+\s*\)'
        r'|\bBENCHMARK_F\s*\(',
    )),
)

# Local include patterns (relative paths — may be unresolvable)
_RE_LOCAL_INCLUDE = re.compile(
    r'^\s*#\s*include\s*"([^"]+)"',
    re.MULTILINE,
)
"""Matches ``#include "local_header.h"`` and ``#include "../path/header.h"``."""


# ============================================================================
# Classification logic
# ============================================================================

def _read_source(file_path: str) -> str:
    """Read a C++ source file safely.

    Args:
        file_path: Absolute path to a ``.cpp`` file.

    Returns:
        Source text, or ``""`` on error.
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _strip_cpp_comments(source: str) -> str:
    """Remove C++ comments and string literals from *source*.

    Replaces block comments, line comments, and string contents with
    spaces to avoid false matches inside comments/strings.

    Args:
        source: C++ source code.

    Returns:
        Cleaned source with comments and strings blanked out.
    """
    # Remove block comments `/* ... */`
    source = re.sub(r'/\*.*?\*/', ' ', source, flags=re.DOTALL)
    # Remove line comments `// ...`
    source = re.sub(r'//[^\n]*', ' ', source)
    # Remove string literals `"..."` (naive — handles escaped quotes)
    source = re.sub(r'"(?:[^"\\]|\\.)*"', '""', source)
    return source


def _has_main(source: str) -> bool:
    """Check whether *source* contains a ``main()`` entry point.

    Assumes comments and string literals have already been stripped
    from *source* (use :func:`_strip_cpp_comments` first).

    Matches ``int main()``, ``int main(void)``, ``signed main()``,
    ``int main(int argc, char* argv[])``, and ``void main()``.

    Args:
        source: C++ source code string (comments already removed).

    Returns:
        ``True`` if a main function declaration is found.
    """
    if _RE_MAIN.search(source):
        return True
    if _RE_MAIN_ARGS.search(source):
        return True
    if _RE_VOID_MAIN.search(source):
        return True
    return False


def _detect_test_framework(source: str) -> Tuple[bool, str]:
    """Check whether *source* uses a C++ testing framework.

    Args:
        source: C++ source code string.

    Returns:
        ``(is_test, framework_name)`` — *framework_name* is the first
        matching framework, or ``""``.
    """
    for name, pattern in _TEST_PATTERNS:
        if pattern.search(source):
            return True, name
    return False, ""


def _get_local_includes(source: str) -> List[str]:
    """Extract all local ``#include`` paths from *source*.

    Only matches quoted includes, not angle-bracket includes.

    Args:
        source: C++ source code string.

    Returns:
        List of ``"header.h"`` or ``"../path/header.h"`` strings.
    """
    return [m.group(1) for m in _RE_LOCAL_INCLUDE.finditer(source)]


def _has_unresolved_dependencies(
    file_path: str,
    source: str,
    raw_cpp_root: str,
) -> bool:
    """Check whether *file_path* includes local headers that don't exist.

    For each local ``#include "..."`` directive, this attempts to resolve
    the header relative to:
        1. The directory containing *file_path*.
        2. The ``raw_cpp/`` root (for cross-directory includes).

    Args:
        file_path: Absolute path to the ``.cpp`` file.
        source: C++ source code string.
        raw_cpp_root: Root of the extracted ``.cpp`` staging area.

    Returns:
        ``True`` if at least one local include is unresolvable.
    """
    local_includes = _get_local_includes(source)
    if not local_includes:
        return False

    file_dir = os.path.dirname(file_path)

    for inc in local_includes:
        # Skip system-level local includes that are universally available
        if inc.startswith("<") or inc in ("bits/stdc++.h",):
            continue

        # Resolve: first relative to the file's directory
        candidate1 = os.path.normpath(os.path.join(file_dir, inc))
        if os.path.isfile(candidate1):
            continue

        # Resolve: relative to raw_cpp root (cross-directory)
        candidate2 = os.path.normpath(os.path.join(raw_cpp_root, inc))
        if os.path.isfile(candidate2):
            continue

        # Try searching for the basename anywhere under raw_cpp/
        basename = os.path.basename(inc)
        found = False
        for dirpath, _, filenames in os.walk(raw_cpp_root):
            if basename in filenames:
                found = True
                break
        if found:
            continue

        # Unresolvable
        return True

    return False


# ============================================================================
# Batch classification
# ============================================================================

def _is_cpp_file(filename: str) -> bool:
    """Check whether *filename* is a C++ source file.

    Args:
        filename: File name to check.

    Returns:
        ``True`` for ``.cpp``, ``.cc``, ``.cxx``.
    """
    return filename.lower().endswith((".cpp", ".cc", ".cxx"))


def classify_all_files(
    raw_cpp_root: str,
    logger: Logger | None = None,
) -> Dict[str, Set[str]]:
    """Classify every C++ file under *raw_cpp_root*.

    Args:
        raw_cpp_root: Path to ``raw_cpp/``.
        logger: Optional :class:`Logger` instance.

    Returns:
        Dict with keys ``executable``, ``library``, ``test``,
        ``dependency`` — each a ``set[str]`` of absolute file paths.
    """
    executable: set[str] = set()
    library: set[str] = set()
    tests: set[str] = set()
    dependency: set[str] = set()

    # Collect all .cpp files
    all_files: list[str] = []
    for dirpath, _, filenames in os.walk(raw_cpp_root):
        for fname in filenames:
            if _is_cpp_file(fname):
                all_files.append(os.path.join(dirpath, fname))

    total = len(all_files)
    if logger:
        logger.info(f"Classifying {total} C++ files …")

    for i, fpath in enumerate(sorted(all_files)):
        source = _read_source(fpath)
        if not source:
            continue

        # Strip comments once for all checks (avoids false positives)
        clean = _strip_cpp_comments(source)
        has_main = _has_main(clean)
        is_test, framework = _detect_test_framework(clean)
        has_deps = _has_unresolved_dependencies(fpath, clean, raw_cpp_root)

        # Classification order: test > dependency > executable > library
        if is_test:
            tests.add(fpath)
        elif has_deps and not has_main:
            # Missing local headers AND no main → truly dependency-only
            dependency.add(fpath)
        elif has_main:
            executable.add(fpath)
        else:
            library.add(fpath)

        if logger and (i + 1) % 2000 == 0:
            logger.info(
                f"  Classified {i + 1}/{total}  "
                f"(exec={len(executable)} lib={len(library)} "
                f"test={len(tests)} dep={len(dependency)})"
            )

    if logger:
        logger.count("total_files", total)
        logger.count("executable", len(executable))
        logger.count("library", len(library))
        logger.count("test", len(tests))
        logger.count("dependency", len(dependency))

    return {
        "executable": executable,
        "library": library,
        "test": tests,
        "dependency": dependency,
    }


# ============================================================================
# Report generation
# ============================================================================

def _write_list_report(path: str, header: List[str], files: Set[str],
                       raw_cpp_root: str, extra_col: str = "") -> None:
    """Write a CSV listing all files in a category.

    Args:
        path: Output CSV path.
        header: Column names.
        files: Set of file paths.
        raw_cpp_root: Root for computing relative paths.
        extra_col: Optional extra column value.
    """
    rows: list[list] = []
    for f in sorted(files):
        rel = os.path.relpath(f, raw_cpp_root)
        row = [f, rel]
        if extra_col:
            row.append(extra_col)
        rows.append(row)
    write_csv(path, list(header), rows)


def _write_filter_report(
    counts: Dict[str, int],
    path: str,
) -> None:
    """Write the aggregate filter summary CSV.

    Args:
        counts: Dict of category → count.
        path: Output CSV path.
    """
    total = counts.get("total_files", 0)
    executable = counts.get("executable", 0)
    library = counts.get("library", 0)
    test = counts.get("test", 0)
    dependency = counts.get("dependency", 0)
    remaining = executable  # only executables advance

    rows = [
        ["OriginalCPPFiles", str(total)],
        ["ExecutablePrograms", str(executable)],
        ["Remove_LibraryFiles", str(library)],
        ["Remove_TestFiles", str(test)],
        ["Remove_DependencyFiles", str(dependency)],
        ["Remaining_BenchmarkPrograms", str(remaining)],
        ["FilterRate", f"{remaining / max(total, 1) * 100:.1f}%"],
    ]
    write_csv(path, ["Metric", "Value"], rows)


# ============================================================================
# Public API (used by downstream stages)
# ============================================================================

# Module-level cache — computed on first call
_classified: Dict[str, Set[str]] | None = None


def get_executable_files(
    raw_cpp_root: str | None = None,
    logger: Logger | None = None,
) -> Set[str]:
    """Return the set of executable (main-containing) C++ file paths.

    Results are cached after the first call.

    Args:
        raw_cpp_root: Path to ``raw_cpp/``.  Defaults to :data:`RAW_CPP_DIR`.
        logger: Optional :class:`Logger` instance.

    Returns:
        ``set`` of absolute paths to executable ``.cpp`` files.
    """
    global _classified
    if raw_cpp_root is None:
        raw_cpp_root = RAW_CPP_DIR

    if _classified is None:
        _classified = classify_all_files(raw_cpp_root, logger)

    return _classified["executable"]


def get_filter_classification() -> Dict[str, Set[str]]:
    """Return the full classification dict (cached).

    Returns:
        Dict with keys ``executable``, ``library``, ``test``,
        ``dependency``.
    """
    global _classified
    if _classified is None:
        _classified = classify_all_files(RAW_CPP_DIR, None)
    return _classified


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    """Classify all files and write all filter reports."""
    logger = Logger("filter_programs")

    if not os.path.isdir(RAW_CPP_DIR):
        logger.error(f"raw_cpp/ not found: {RAW_CPP_DIR}")
        sys.exit(1)

    logger.info(f"Scanning: {RAW_CPP_DIR}")

    classified = classify_all_files(RAW_CPP_DIR, logger)

    executable = classified["executable"]
    library = classified["library"]
    tests = classified["test"]
    dependency = classified["dependency"]
    total = len(executable) + len(library) + len(tests) + len(dependency)

    logger.info(
        f"Classification complete: "
        f"exec={len(executable)} lib={len(library)} "
        f"test={len(tests)} dep={len(dependency)} "
        f"(out of {total})"
    )

    # Write per-category reports
    _write_list_report(
        os.path.join(REPORTS_DIR, "library_files.csv"),
        ["FilePath", "RelativePath"],
        library, RAW_CPP_DIR,
    )
    logger.info(f"  library_files.csv: {len(library)} files")

    _write_list_report(
        os.path.join(REPORTS_DIR, "test_files.csv"),
        ["FilePath", "RelativePath"],
        tests, RAW_CPP_DIR,
    )
    logger.info(f"  test_files.csv: {len(tests)} files")

    _write_list_report(
        os.path.join(REPORTS_DIR, "dependency_report.csv"),
        ["FilePath", "RelativePath"],
        dependency, RAW_CPP_DIR,
    )
    logger.info(f"  dependency_report.csv: {len(dependency)} files")

    # Write aggregate filter report
    counts = {
        "total_files": total,
        "executable": len(executable),
        "library": len(library),
        "test": len(tests),
        "dependency": len(dependency),
    }
    _write_filter_report(counts, os.path.join(REPORTS_DIR, "filter_report.csv"))
    logger.info(f"  filter_report.csv: {len(executable)}/{total} executable "
                f"({len(executable) / max(total, 1) * 100:.1f}%)")

    # Write program type classification CSV (used by metadata generator)
    _write_program_types(classified, RAW_CPP_DIR)

    print(f"\n{logger.summary()}")


def _write_program_types(classified: Dict[str, Set[str]],
                         raw_cpp_root: str) -> None:
    """Write ``program_type.csv`` mapping each file to its classification.

    Args:
        classified: Dict from :func:`classify_all_files`.
        raw_cpp_root: Root for relative paths.
    """
    rows: list[list] = []
    for category, files in classified.items():
        for fpath in files:
            rel = os.path.relpath(fpath, raw_cpp_root)
            rows.append([fpath, rel, category, "TRUE" if category == "executable" else "FALSE"])
    write_csv(
        os.path.join(REPORTS_DIR, "program_type.csv"),
        ["FilePath", "RelativePath", "ProgramType", "CompileEligible"],
        rows,
    )


if __name__ == "__main__":
    main()
