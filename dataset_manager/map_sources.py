"""
dataset_manager/map_sources.py — Source Mapping
=================================================

Generates ``source_mapping.csv`` that maps every benchmark program ID
back to its original repository, path, and GitHub URL, guaranteeing
full reproducibility of the dataset.

Outputs:
    ``reports/source_mapping.csv``

Usage::

    python dataset_manager/map_sources.py
"""

from __future__ import annotations
import os as _os
import sys as _sys
_DEPS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _DEPS not in _sys.path:
    _sys.path.insert(0, _DEPS)

import os
import sys
from typing import List, Tuple

from dataset_manager.utils import (
    BENCHMARK_DIR,
    RAW_CPP_DIR,
    REPORTS_DIR,
    DATASETS_DIR,
    Logger,
    git_remote_url,
    write_csv,
    read_csv,
    sha256_file,
)


_PROGRAM_ID_WIDTH = 6


def _find_original_source(
    bench_file: str,
    raw_cpp_root: str,
) -> Tuple[str, str, str, str]:
    """Find the original source path for a benchmark file.

    Uses SHA-256 matching to reliably identify the original source
    (handles renamed files).

    Args:
        bench_file: Path to a ``program_NNNNNN.cpp`` file.
        raw_cpp_root: Root of the ``raw_cpp/`` staging directory.

    Returns:
        ``(original_rel_path, category, repo_name, github_url)``.
    """
    bench_hash = sha256_file(bench_file)
    bench_name = os.path.basename(bench_file)

    # First: try to find by name in raw_cpp/
    for dirpath, _, filenames in os.walk(raw_cpp_root):
        for fname in filenames:
            if fname == bench_name:
                # Verify by hash
                full = os.path.join(dirpath, fname)
                if sha256_file(full) == bench_hash:
                    rel = os.path.relpath(full, raw_cpp_root)
                    parts = rel.split(os.sep)
                    cat = parts[0] if len(parts) > 0 else "unknown"
                    repo = parts[1] if len(parts) > 1 else "unknown"
                    repo_dir = os.path.join(DATASETS_DIR, cat, repo)
                    url = git_remote_url(repo_dir) if os.path.isdir(repo_dir) else "unknown"
                    return (rel, cat, repo, url)

    # Fallback: hash-only search (slower)
    for dirpath, _, filenames in os.walk(raw_cpp_root):
        for fname in filenames:
            if fname.endswith((".cpp", ".cc", ".cxx")):
                full = os.path.join(dirpath, fname)
                if sha256_file(full) == bench_hash:
                    rel = os.path.relpath(full, raw_cpp_root)
                    parts = rel.split(os.sep)
                    cat = parts[0] if len(parts) > 0 else "unknown"
                    repo = parts[1] if len(parts) > 1 else "unknown"
                    repo_dir = os.path.join(DATASETS_DIR, cat, repo)
                    url = git_remote_url(repo_dir) if os.path.isdir(repo_dir) else "unknown"
                    return (rel, cat, repo, url)

    return ("unknown", "unknown", "unknown", "unknown")


def generate_mapping(bench_dir: str, raw_dir: str, logger: Logger | None = None) -> List[List]:
    """Generate source mapping rows for all benchmark programs.

    Args:
        bench_dir: Path to ``benchmark_dataset/``.
        raw_dir: Path to ``raw_cpp/``.
        logger: Optional logger.

    Returns:
        List of CSV row lists.
    """
    rows: list[list] = []
    bench_files = sorted(
        f for f in os.listdir(bench_dir)
        if f.startswith("program_") and f.endswith(".cpp")
    )

    total = len(bench_files)
    for i, fname in enumerate(bench_files):
        bench_path = os.path.join(bench_dir, fname)
        program_id = fname.replace(".cpp", "")
        orig_rel, category, repo, url = _find_original_source(bench_path, raw_dir)
        rows.append([program_id, orig_rel, category, repo, url])

        if logger and (i + 1) % 1000 == 0:
            logger.info(f"  Mapped {i + 1}/{total} programs …")

    if logger:
        logger.count("programs_mapped", len(rows))
    return rows


def main() -> None:
    """Generate ``source_mapping.csv`` from the benchmark dataset."""
    logger = Logger("map_sources")

    if not os.path.isdir(BENCHMARK_DIR):
        logger.error(f"Benchmark dir not found: {BENCHMARK_DIR}")
        sys.exit(1)

    bench_files = [f for f in os.listdir(BENCHMARK_DIR)
                   if f.startswith("program_") and f.endswith(".cpp")]

    if not bench_files:
        logger.error("No benchmark programs found. Run build_dataset first.")
        sys.exit(1)

    logger.info(f"Benchmark programs: {len(bench_files)}")
    logger.info(f"Raw C++ root:      {RAW_CPP_DIR}")

    rows = generate_mapping(BENCHMARK_DIR, RAW_CPP_DIR, logger)

    report_path = os.path.join(REPORTS_DIR, "source_mapping.csv")
    write_csv(
        report_path,
        ["ProgramID", "OriginalPath", "Category", "Repository", "GitHubURL"],
        rows,
    )
    logger.info(f"Mapping written: {report_path}")
    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
