"""
dataset_manager/scan_repositories.py — Repository Scanner
==========================================================

Recursively scans ``~/datasets/`` and generates comprehensive
per-repository statistics.

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
from typing import Any, Dict, List

from dataset_manager.utils import (
    DATASETS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    Logger,
    count_lines,
    git_last_commit,
    git_remote_url,
    write_csv,
)


# File extensions considered C++ source or header
_CPP_EXTS = (".cpp", ".cc", ".cxx")
_HDR_EXTS = (".h", ".hpp", ".hxx")


def _scan_one_repo(repo_dir: str, dataset_source: str) -> Dict[str, Any]:
    """Scan a single repository directory and return statistics.

    Args:
        repo_dir: Path to the repository root.
        dataset_source: Dataset category name (e.g. ``"algorithms"``).

    Returns:
        Dict of stat name → value.
    """
    repo_name = os.path.basename(repo_dir)
    cpp_files: list[str] = []
    hdr_files: list[str] = []
    readme_found = False
    total_bytes: int = 0
    total_loc: int = 0
    max_loc: int = 0

    for dirpath, dirnames, filenames in os.walk(repo_dir):
        # Skip .git
        dirnames[:] = [d for d in dirnames if d != ".git"]

        for fname in filenames:
            fpath = os.path.join(dirpath, fname)

            try:
                total_bytes += os.path.getsize(fpath)
            except OSError:
                pass

            if fname.upper().startswith("README"):
                readme_found = True

            if fname.lower().endswith(_CPP_EXTS):
                cpp_files.append(fpath)
                try:
                    with open(fpath, encoding="utf-8", errors="replace") as fh:
                        source = fh.read()
                    loc = count_lines(source)["code"]
                    total_loc += loc
                    if loc > max_loc:
                        max_loc = loc
                except OSError:
                    pass

            elif fname.lower().endswith(_HDR_EXTS):
                hdr_files.append(fpath)

    n_cpp = len(cpp_files)
    return {
        "repository_name": repo_name,
        "repository_owner": _guess_owner(repo_dir),
        "repository_url": git_remote_url(repo_dir),
        "dataset_source": dataset_source,
        "repository_size_mb": round(total_bytes / (1024 * 1024), 2),
        "cpp_files": n_cpp,
        "header_files": len(hdr_files),
        "total_loc": total_loc,
        "average_loc": round(total_loc / max(n_cpp, 1), 1),
        "maximum_loc": max_loc,
        "readme_exists": readme_found,
        "last_commit": git_last_commit(repo_dir),
    }


def _guess_owner(repo_dir: str) -> str:
    """Guess the repository owner from the git remote URL.

    Args:
        repo_dir: Path to a git repository.

    Returns:
        Owner name string, or ``"unknown"``.
    """
    url = git_remote_url(repo_dir)
    if url == "unknown":
        return "unknown"
    # Extract owner from GitHub URL: https://github.com/OWNER/REPO
    for part in url.split("/"):
        if part and not part.startswith("http") and "github" not in part and "." not in part:
            return part
    return "unknown"


def _scan_all(logger: Logger) -> List[Dict[str, Any]]:
    """Scan every repository under ``~/datasets/``.

    Args:
        logger: :class:`Logger` instance.

    Returns:
        List of per-repository statistics dicts.
    """
    results: list[dict] = []
    try:
        categories = sorted(os.listdir(DATASETS_DIR))
    except OSError:
        logger.error(f"Cannot list {DATASETS_DIR}")
        return results

    for category in categories:
        cat_path = os.path.join(DATASETS_DIR, category)
        if not os.path.isdir(cat_path):
            continue

        try:
            repos = sorted(os.listdir(cat_path))
        except OSError:
            continue

        for repo in repos:
            repo_dir = os.path.join(cat_path, repo)
            if not os.path.isdir(repo_dir):
                continue

            logger.info(f"Scanning: {category}/{repo}")
            try:
                stats = _scan_one_repo(repo_dir, category)
                results.append(stats)
            except Exception as exc:
                logger.error(f"Failed to scan {category}/{repo}: {exc}")
                continue

    logger.count("repositories_scanned", len(results))
    return results


def _build_csv_rows(stats_list: List[Dict[str, Any]]) -> List[List[Any]]:
    """Convert stats dicts to CSV rows.

    Uses the same keys as those in :func:`_scan_one_repo`'s return dict.

    Args:
        stats_list: List of per-repo stats.

    Returns:
        Row list suitable for :func:`write_csv`.
    """
    # These must match the keys in _scan_one_repo's return dict
    keys = [
        "repository_name", "repository_owner", "repository_url",
        "dataset_source", "repository_size_mb", "cpp_files", "header_files",
        "total_loc", "average_loc", "maximum_loc", "readme_exists",
        "last_commit",
    ]
    rows: list[list] = []
    for s in stats_list:
        rows.append([s.get(k, "") for k in keys])
    return rows


def main() -> None:
    """Scan all repositories and write ``repository_statistics.csv``."""
    logger = Logger("scan_repositories")

    if not os.path.isdir(DATASETS_DIR):
        logger.error(f"Datasets directory not found: {DATASETS_DIR}")
        sys.exit(1)

    logger.info(f"Scanning: {DATASETS_DIR}")

    stats = _scan_all(logger)
    logger.info(f"Scan complete: {len(stats)} repositories")

    # Print aggregate summary
    total_cpp = sum(s["cpp_files"] for s in stats)
    total_loc = sum(s["total_loc"] for s in stats)
    total_mb = sum(s["repository_size_mb"] for s in stats)
    logger.info(f"  Total .cpp files: {total_cpp}")
    logger.info(f"  Total LOC:        {total_loc}")
    logger.info(f"  Total size:       {total_mb:.1f} MB")

    rows = _build_csv_rows(stats)
    report_path = os.path.join(REPORTS_DIR, "repository_statistics.csv")
    write_csv(
        report_path,
        [
            "RepositoryName", "RepositoryOwner", "RepositoryURL",
            "DatasetSource", "RepositorySizeMB", "CppFiles", "HeaderFiles",
            "TotalLOC", "AverageLOC", "MaximumLOC", "ReadmeExists",
            "LastCommit",
        ],
        rows,
    )
    logger.info(f"Report written: {report_path}")
    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
