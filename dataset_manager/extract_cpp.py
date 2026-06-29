"""
dataset_manager/extract_cpp.py — C++ File Extractor
=====================================================

Recursively scans ``~/datasets/`` for all ``*.cpp``, ``*.cc``, and
``*.cxx`` files and copies them into ``dataset_manager/raw_cpp/``,
preserving the directory mapping:

    ~/datasets/<category>/<repo>/.../file.cpp
    ->
    raw_cpp/<category>/<repo>/.../file.cpp

Usage::

    python dataset_manager/extract_cpp.py
"""

from __future__ import annotations
import os as _os
import sys as _sys
_DEPS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _DEPS not in _sys.path:
    _sys.path.insert(0, _DEPS)


import os
import shutil
import sys
from typing import List, Tuple

from dataset_manager.utils import (
    DATASETS_DIR,
    RAW_CPP_DIR,
    REPORTS_DIR,
    Logger,
    write_csv,
)


# File extensions considered as C++ source
_CPP_EXTENSIONS: tuple[str, ...] = (".cpp", ".cc", ".cxx")


# ============================================================================
# Extraction logic
# ============================================================================

def _find_cpp_files(root: str) -> list[tuple[str, str]]:
    """Find all C++ source files under *root*.

    Args:
        root: Directory to search recursively.

    Returns:
        List of ``(absolute_source_path, relative_path)`` pairs.
        ``.git/`` directories are skipped.
    """
    results: list[tuple[str, str]] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip .git directories
            dirnames[:] = [d for d in dirnames if d != ".git"]

            for fname in sorted(filenames):
                if fname.lower().endswith(_CPP_EXTENSIONS):
                    abs_path = os.path.join(dirpath, fname)
                    rel_path = os.path.relpath(abs_path, root)
                    results.append((abs_path, rel_path))
    except OSError as exc:
        pass  # handled by caller

    return results


def _extract_files(
    files: list[tuple[str, str]],
    dest_root: str,
    logger: Logger,
) -> Tuple[int, int, int]:
    """Copy source files to *dest_root*, maintaining relative paths.

    Args:
        files: ``(source_abs, rel_path)`` pairs.
        dest_root: Destination root directory.
        logger: :class:`Logger` instance.

    Returns:
        ``(copied, skipped, failed)`` counts.
    """
    copied = skipped = failed = 0

    for abs_path, rel_path in files:
        dest_path = os.path.join(dest_root, rel_path)
        dest_dir = os.path.dirname(dest_path)

        # Skip if already copied (same size — simple dedup check)
        if os.path.isfile(dest_path):
            try:
                if os.path.getsize(abs_path) == os.path.getsize(dest_path):
                    skipped += 1
                    continue
            except OSError:
                pass

        try:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(abs_path, dest_path)
            copied += 1
        except OSError as exc:
            logger.warn(f"Failed to copy {abs_path}: {exc}")
            failed += 1

    return copied, skipped, failed


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    """Discover and copy all C++ source files to ``raw_cpp/``."""
    logger = Logger("extract_cpp")

    if not os.path.isdir(DATASETS_DIR):
        logger.error(f"Datasets directory not found: {DATASETS_DIR}")
        sys.exit(1)

    logger.info(f"Source:      {DATASETS_DIR}")
    logger.info(f"Destination: {RAW_CPP_DIR}")
    logger.info("Searching for C++ source files …")

    files = _find_cpp_files(DATASETS_DIR)
    logger.info(f"Found {len(files)} C++ source files")
    logger.count("files_found", len(files))

    if not files:
        logger.warn("No C++ files found — nothing to extract.")
        return

    logger.info("Copying files …")
    copied, skipped, failed = _extract_files(files, RAW_CPP_DIR, logger)

    logger.count("files_copied", copied)
    logger.count("files_skipped", skipped)
    logger.count("files_failed", failed)

    logger.info(
        f"Extraction complete: {copied} copied, "
        f"{skipped} skipped, {failed} failed"
    )

    # Write summary report
    report_path = os.path.join(REPORTS_DIR, "extraction_report.csv")
    write_csv(
        report_path,
        ["Metric", "Value"],
        [
            ["TotalFound", str(len(files))],
            ["Copied", str(copied)],
            ["Skipped", str(skipped)],
            ["Failed", str(failed)],
            ["Destination", RAW_CPP_DIR],
        ],
    )
    logger.info(f"Report written: {report_path}")

    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
