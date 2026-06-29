"""
dataset_manager/clone_repositories.py — Repository Cloner
==========================================================

Reads ``repositories.txt`` and clones every listed GitHub repository
into ``~/datasets/<category>/``.

Features:
    - Skips repositories that already exist on disk.
    - Records clone results (success / already-exists / 404 / error).
    - Generates ``download_report.csv``.

Usage::

    python dataset_manager/clone_repositories.py
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
    DATASETS_DIR,
    REPORTS_DIR,
    Logger,
    run_command,
    timestamp,
    write_csv,
)


# ============================================================================
# Repository list parser
# ============================================================================

def _parse_repositories(path: str) -> List[Tuple[str, str]]:
    """Parse ``repositories.txt`` into ``[(category, url), ...]``.

    Args:
        path: Path to ``repositories.txt``.

    Returns:
        List of ``(category, url)`` pairs.
    """
    repos: list[tuple[str, str]] = []
    if not os.path.isfile(path):
        return repos

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                category = parts[0]
                url = parts[1]
                repos.append((category, url))
    return repos


# ============================================================================
# Single-repo clone
# ============================================================================

def _clone_one(
    category: str,
    url: str,
    logger: Logger,
) -> Tuple[str, str, str]:
    """Clone a single repository and return ``(status, repo_name, message)``.

    Args:
        category: Dataset category (e.g. ``"algorithms"``).
        url: Git clone URL.
        logger: :class:`Logger` instance.

    Returns:
        ``(status, repo_name, message)`` where *status* is one of
        ``"success"``, ``"exists"``, ``"404"``, ``"error"``.
    """
    repo_name = url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    target_dir = os.path.join(DATASETS_DIR, category, repo_name)

    # Already exists — skip
    if os.path.isdir(target_dir) and os.path.isdir(os.path.join(target_dir, ".git")):
        logger.info(f"[EXISTS] {category}/{repo_name}")
        return ("exists", repo_name, "Already cloned")

    # Clone
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    logger.info(f"[CLONE] {category}/{repo_name} <- {url}")

    ok, stdout, stderr = run_command(
        ["git", "clone", "--depth=1", url, target_dir],
        timeout=300,  # 5 minutes for large repos
    )

    if ok:
        logger.count("success")
        logger.info(f"  -> success: {category}/{repo_name}")
        return ("success", repo_name, "Clone successful")
    else:
        combined = (stderr + stdout).lower()
        if "not found" in combined or "repository does not exist" in combined:
            logger.warn(f"  -> 404: {url}")
            logger.count("not_found")
            return ("404", repo_name, "Repository not found")
        logger.error(f"  -> error: {stderr[:200]}")
        logger.count("error")
        return ("error", repo_name, stderr[:500])


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    """Clone all repositories listed in ``repositories.txt``."""
    logger = Logger("clone_repositories")

    repo_file = os.path.join(os.path.dirname(__file__), "repositories.txt")
    repos = _parse_repositories(repo_file)

    if not repos:
        logger.warn(f"No repositories found in {repo_file}")
        return

    logger.info(f"Found {len(repos)} repositories in repositories.txt")
    logger.info(f"Target directory: {DATASETS_DIR}")

    rows: list[list[str]] = []
    for category, url in repos:
        try:
            status, name, msg = _clone_one(category, url, logger)
            rows.append([category, name, url, status, msg])
        except Exception as exc:
            logger.error(f"Unexpected error for {url}: {exc}")
            rows.append([category, "unknown", url, "error", str(exc)])

    # Write download report
    report_path = os.path.join(REPORTS_DIR, "download_report.csv")
    write_csv(
        report_path,
        ["Category", "Repository", "URL", "Status", "Message"],
        rows,
    )
    logger.info(f"Download report written: {report_path}")

    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
