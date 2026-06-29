"""
dataset_manager/deduplicate.py — SHA-256 Deduplication
=======================================================

Scans ``dataset_manager/raw_cpp/`` for duplicate C++ files using
SHA-256 content hashing.  Only the first occurrence of each unique
file is kept; duplicates are recorded in ``duplicate_report.csv``
and **not** deleted (they are skipped during dataset building).

This is a **non-destructive** process — the ``raw_cpp/`` directory
is never modified.

Outputs:
    ``reports/duplicate_report.csv`` — maps each duplicate to its original.

Usage::

    python dataset_manager/deduplicate.py
"""

from __future__ import annotations
import os as _os
import sys as _sys
_DEPS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _DEPS not in _sys.path:
    _sys.path.insert(0, _DEPS)


import os
import sys
from typing import Dict, List, Tuple

from dataset_manager.utils import (
    RAW_CPP_DIR,
    REPORTS_DIR,
    Logger,
    sha256_file,
    write_csv,
)


# ============================================================================
# Deduplication logic
# ============================================================================

def _hash_all_files(root: str, logger: Logger) -> Tuple[Dict[str, str], List[Tuple[str, str, str]]]:
    """Compute SHA-256 hashes for all files under *root*.

    Args:
        root: Directory to scan recursively.
        logger: :class:`Logger` instance.

    Returns:
        ``(hash_to_first_path, duplicates)`` where *duplicates* is
        ``[(original_path, duplicate_path, hash), ...]``.
    """
    hash_to_path: dict[str, str] = {}
    """hash -> first file path that produced this hash."""

    duplicates: list[tuple[str, str, str]] = []
    """(original_path, duplicate_path, sha256) triples."""

    total = 0

    for dirpath, _, filenames in os.walk(root):
        # Skip .git even inside raw_cpp (shouldn't be there, but be safe)
        dirnames_to_skip = [d for d in os.listdir(dirpath) if d == ".git"]
        for fname in sorted(filenames):
            if not _is_cpp_file(fname):
                continue
            fpath = os.path.join(dirpath, fname)
            total += 1

            digest = sha256_file(fpath)
            if not digest:
                logger.warn(f"Could not hash: {fpath}")
                continue

            if digest in hash_to_path:
                duplicates.append((hash_to_path[digest], fpath, digest))
            else:
                hash_to_path[digest] = fpath

            if total % 1000 == 0:
                logger.info(f"  Hashed {total} files …")

    logger.count("files_hashed", total)
    logger.count("unique_files", len(hash_to_path))
    logger.count("duplicate_files", len(duplicates))

    return hash_to_path, duplicates


def _is_cpp_file(filename: str) -> bool:
    """Check whether *filename* is a C++ source file.

    Args:
        filename: File name to check.

    Returns:
        ``True`` if the extension is ``.cpp``, ``.cc``, or ``.cxx``.
    """
    return filename.lower().endswith((".cpp", ".cc", ".cxx"))


# ============================================================================
# Report generation
# ============================================================================

def _build_duplicate_rows(duplicates: List[Tuple[str, str, str]]) -> List[List[str]]:
    """Convert duplicate tuples into CSV rows.

    Args:
        duplicates: ``[(original, duplicate, sha256), ...]``.

    Returns:
        List of 3-column rows for CSV output.
    """
    rows: list[list[str]] = []
    for orig, dup, digest in duplicates:
        rows.append([orig, dup, digest])
    return rows


# ============================================================================
# Public helper (used by pipeline)
# ============================================================================

def get_unique_files() -> set[str]:
    """Return the set of unique (non-duplicate) file paths in ``raw_cpp/``.

    Returns:
        ``set`` of absolute file paths — the first occurrence of each
        unique hash.
    """
    root = RAW_CPP_DIR
    if not os.path.isdir(root):
        return set()

    seen_hashes: set[str] = set()
    unique: set[str] = set()

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not _is_cpp_file(fname):
                continue
            fpath = os.path.join(dirpath, fname)
            digest = sha256_file(fpath)
            if digest and digest not in seen_hashes:
                seen_hashes.add(digest)
                unique.add(fpath)

    return unique


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    """Hash all files, identify duplicates, write report."""
    logger = Logger("deduplicate")

    if not os.path.isdir(RAW_CPP_DIR):
        logger.error(f"raw_cpp/ directory not found: {RAW_CPP_DIR}")
        logger.info("Run extraction first: python dataset_manager/extract_cpp.py")
        sys.exit(1)

    logger.info(f"Scanning: {RAW_CPP_DIR}")
    logger.info("Computing SHA-256 hashes …")

    hash_to_path, duplicates = _hash_all_files(RAW_CPP_DIR, logger)

    logger.info(
        f"Deduplication complete: {len(hash_to_path)} unique, "
        f"{len(duplicates)} duplicates "
        f"({len(duplicates) / max(len(hash_to_path) + len(duplicates), 1) * 100:.1f}% dup rate)"
    )

    # Write report
    rows = _build_duplicate_rows(duplicates)
    report_path = os.path.join(REPORTS_DIR, "duplicate_report.csv")
    write_csv(
        report_path,
        ["OriginalFile", "DuplicateFile", "SHA256Hash"],
        rows,
    )
    logger.info(f"Report written: {report_path}")

    # Also write summary
    summary_path = os.path.join(REPORTS_DIR, "dedup_summary.csv")
    write_csv(
        summary_path,
        ["Metric", "Value"],
        [
            ["TotalFilesHashed", str(len(hash_to_path) + len(duplicates))],
            ["UniqueFiles", str(len(hash_to_path))],
            ["DuplicateFiles", str(len(duplicates))],
            ["DedupRatio", f"{len(duplicates) / max(len(hash_to_path) + len(duplicates), 1) * 100:.2f}%"],
        ],
    )
    logger.info(f"Summary written: {summary_path}")

    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
