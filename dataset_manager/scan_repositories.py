"""
dataset_manager/scan_repositories.py — Repository Scanner
==========================================================

Recursively scans ``~/datasets/`` and generates a statistical summary:
repository counts, file-type counts, size distribution, directory tree.

Outputs:
    ``reports/repository_statistics.csv``

Usage::

    python dataset_manager/scan_repositories.py
"""

from __future__ import annotations
import os as _os
import sys as _sys
_DEPS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _DEPS not in _sys.path:
    _sys.path.insert(0, _DEPS)


import os
import sys
from typing import List

from dataset_manager.utils import (
    DATASETS_DIR,
    REPORTS_DIR,
    Logger,
    write_csv,
)


# ============================================================================
# Scanner helpers
# ============================================================================

def _scan_directory(root: str, logger: Logger) -> dict:
    """Recursively scan *root* and return aggregate statistics.

    Args:
        root: The directory to scan.
        logger: :class:`Logger` instance.

    Returns:
        Dict with keys ``repo_count``, ``cpp_count``, ``h_count``,
        ``readme_count``, ``other_count``, ``total_size_mb``,
        ``max_depth``.
    """
    stats: dict[str, int | float] = {
        "repo_count": 0,
        "cpp_count": 0,
        "h_count": 0,
        "readme_count": 0,
        "other_count": 0,
        "total_size_mb": 0.0,
        "max_depth": 0,
    }
    total_bytes: int = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip .git directories
        if ".git" in dirpath.split(os.sep):
            continue

        # Track repository roots (directories containing .git/)
        git_dir = os.path.join(dirpath, ".git")
        if os.path.isdir(git_dir):
            stats["repo_count"] += 1

        # Depth
        rel_depth = dirpath[len(root):].count(os.sep)
        if rel_depth > stats["max_depth"]:
            stats["max_depth"] = rel_depth

        for fname in filenames:
            fpath = os.path.join(dirpath, fname)

            # File size
            try:
                total_bytes += os.path.getsize(fpath)
            except OSError:
                pass

            # File type classification
            if fname.endswith(".cpp") or fname.endswith(".cc") or fname.endswith(".cxx"):
                stats["cpp_count"] += 1
            elif fname.endswith(".h") or fname.endswith(".hpp") or fname.endswith(".hxx"):
                stats["h_count"] += 1
            elif fname.upper().startswith("README"):
                stats["readme_count"] += 1
            else:
                stats["other_count"] += 1

    stats["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)
    return stats


def _per_category_scan(root: str, logger: Logger) -> List[dict]:
    """Scan each category subdirectory individually.

    Args:
        root: The ``datasets/`` directory containing category subdirs.
        logger: :class:`Logger` instance.

    Returns:
        List of per-category statistics dicts.
    """
    results: list[dict] = []
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        logger.error(f"Cannot list directory: {root}")
        return results

    for entry in entries:
        cat_path = os.path.join(root, entry)
        if not os.path.isdir(cat_path):
            continue

        logger.info(f"Scanning category: {entry}")
        stats = _scan_directory(cat_path, logger)
        stats["category"] = entry
        results.append(stats)

    return results


# ============================================================================
# Report generation
# ============================================================================

def _build_report(per_category: List[dict], overall: dict) -> list[list[str]]:
    """Build CSV rows from scan statistics.

    Args:
        per_category: Per-category stats.
        overall: Aggregate stats.

    Returns:
        List of rows for CSV output.
    """
    rows: list[list[str]] = []
    for cat in per_category:
        rows.append([
            cat.get("category", "unknown"),
            str(cat.get("repo_count", 0)),
            str(cat.get("cpp_count", 0)),
            str(cat.get("h_count", 0)),
            str(cat.get("readme_count", 0)),
            str(cat.get("other_count", 0)),
            str(cat.get("total_size_mb", 0)),
            str(cat.get("max_depth", 0)),
        ])
    # Add total row
    rows.append([
        "TOTAL",
        str(overall.get("repo_count", 0)),
        str(overall.get("cpp_count", 0)),
        str(overall.get("h_count", 0)),
        str(overall.get("readme_count", 0)),
        str(overall.get("other_count", 0)),
        str(overall.get("total_size_mb", 0)),
        str(overall.get("max_depth", 0)),
    ])
    return rows


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    """Scan ``~/datasets/`` and write ``repository_statistics.csv``."""
    logger = Logger("scan_repositories")

    if not os.path.isdir(DATASETS_DIR):
        logger.error(f"Datasets directory not found: {DATASETS_DIR}")
        logger.info("Clone repositories first: python dataset_manager/clone_repositories.py")
        sys.exit(1)

    logger.info(f"Scanning: {DATASETS_DIR}")

    # Overall scan
    logger.info("Running overall scan …")
    overall = _scan_directory(DATASETS_DIR, logger)

    # Per-category scan
    per_category = _per_category_scan(DATASETS_DIR, logger)

    logger.info(
        f"Scan complete: {overall['repo_count']} repos, "
        f"{overall['cpp_count']} .cpp files, "
        f"{overall['total_size_mb']} MB"
    )

    # Write report
    rows = _build_report(per_category, overall)
    report_path = os.path.join(REPORTS_DIR, "repository_statistics.csv")
    write_csv(
        report_path,
        [
            "Category", "Repositories", "CPPFiles", "HeaderFiles",
            "ReadmeFiles", "OtherFiles", "TotalSizeMB", "MaxDepth",
        ],
        rows,
    )
    logger.info(f"Report written: {report_path}")

    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
