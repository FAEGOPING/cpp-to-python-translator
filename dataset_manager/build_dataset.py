"""
dataset_manager/build_dataset.py — Benchmark Dataset Builder
==============================================================

Assembles the final benchmark dataset from the set of compiled,
deduplicated C++ source files.

Each file is copied to ``dataset_manager/benchmark_dataset/`` with a
zero-padded sequential filename:

    program_000001.cpp
    program_000002.cpp
    ...

A ``metadata.csv`` is also written to the benchmark directory mapping
each program ID back to its original source file, category, and
repository.

Outputs:
    ``dataset_manager/benchmark_dataset/program_NNNNNN.cpp``
    ``dataset_manager/benchmark_dataset/metadata.csv``

Usage::

    python dataset_manager/build_dataset.py
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
    BENCHMARK_DIR,
    RAW_CPP_DIR,
    Logger,
    write_csv,
)

# How many digits for program filenames (supports up to 999,999)
_PROGRAM_ID_WIDTH = 6


# ============================================================================
# Dataset assembly
# ============================================================================

def _build_programs(
    file_list: List[str],
    dest_dir: str,
    logger: Logger,
) -> Tuple[List[List[str]], int]:
    """Copy each file to *dest_dir* as ``program_NNNNNN.cpp``.

    Args:
        file_list: Sorted list of absolute source paths.
        dest_dir: Destination directory (``benchmark_dataset/``).
        logger: :class:`Logger` instance.

    Returns:
        ``(metadata_rows, count)``.
    """
    metadata_rows: list[list[str]] = []
    count = 0

    for i, src_path in enumerate(file_list):
        program_id = f"program_{i + 1:0{_PROGRAM_ID_WIDTH}d}"
        dest_name = f"{program_id}.cpp"
        dest_path = os.path.join(dest_dir, dest_name)

        try:
            shutil.copy2(src_path, dest_path)
        except OSError as exc:
            logger.warn(f"Failed to copy {src_path}: {exc}")
            continue

        # Extract source info
        rel = os.path.relpath(src_path, RAW_CPP_DIR) if RAW_CPP_DIR in src_path else src_path
        parts = rel.split(os.sep)
        category = parts[0] if len(parts) > 0 else "unknown"
        repo = parts[1] if len(parts) > 1 else "unknown"

        try:
            size = os.path.getsize(src_path)
        except OSError:
            size = 0

        metadata_rows.append([
            program_id,
            dest_name,
            rel,
            category,
            repo,
            str(size),
        ])
        count += 1

        if (count) % 1000 == 0:
            logger.info(f"  Built {count} programs …")

    return metadata_rows, count


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    """Build the final benchmark dataset."""
    from dataset_manager.deduplicate import get_unique_files
    from dataset_manager.validate_cpp import get_compilable_files

    logger = Logger("build_dataset")

    if not os.path.isdir(RAW_CPP_DIR):
        logger.error(f"raw_cpp/ directory not found: {RAW_CPP_DIR}")
        sys.exit(1)

    # Step 1: get unique files
    unique = sorted(get_unique_files())
    logger.info(f"Unique files: {len(unique)}")

    # Step 2: filter to compilable only
    logger.info("Filtering to compilable files …")
    compilable = get_compilable_files(unique)
    logger.info(f"Compilable files: {len(compilable)}")

    if not compilable:
        logger.warn("No compilable files — cannot build dataset.")
        return

    # Step 3: build the dataset
    logger.info(f"Building benchmark dataset in {BENCHMARK_DIR} …")

    # Clean previous build (if any)
    for old_file in os.listdir(BENCHMARK_DIR):
        if old_file.startswith("program_") or old_file == "metadata.csv":
            try:
                os.remove(os.path.join(BENCHMARK_DIR, old_file))
            except OSError:
                pass

    metadata_rows, count = _build_programs(compilable, BENCHMARK_DIR, logger)

    # Write metadata
    metadata_path = os.path.join(BENCHMARK_DIR, "metadata.csv")
    write_csv(
        metadata_path,
        ["ProgramID", "Filename", "OriginalSource", "Category", "Repository", "FileSizeBytes"],
        metadata_rows,
    )

    logger.info(f"Dataset built: {count} programs")
    logger.info(f"Metadata:     {metadata_path}")
    logger.count("programs_in_dataset", count)

    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
