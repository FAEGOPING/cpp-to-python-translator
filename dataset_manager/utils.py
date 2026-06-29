"""
dataset_manager/utils.py — Shared Utilities
============================================

Common helpers used by all Dataset Manager modules: logging, CSV I/O,
SHA256 hashing, subprocess execution, and path resolution.

All modules in ``dataset_manager/`` import from this module rather than
duplicating logic.
"""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

# ============================================================================
# Path resolution
# ============================================================================

PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""Absolute path to the project root (parent of ``dataset_manager/``)."""

DM_DIR: str = os.path.join(PROJECT_ROOT, "dataset_manager")
"""Path to the ``dataset_manager/`` directory."""

RAW_CPP_DIR: str = os.path.join(DM_DIR, "raw_cpp")
"""Path where extracted ``.cpp`` files are staged."""

BENCHMARK_DIR: str = os.path.join(DM_DIR, "benchmark_dataset")
"""Path where the final benchmark dataset is assembled."""

REPORTS_DIR: str = os.path.join(DM_DIR, "reports")
"""Path for generated CSV reports."""

LOGS_DIR: str = os.path.join(DM_DIR, "logs")
"""Path for pipeline run logs."""

DATASETS_DIR: str = os.path.expanduser("~/datasets")
"""Path to the user's local datasets directory (source of C++ repos)."""

# Ensure output directories exist
for _d in (RAW_CPP_DIR, BENCHMARK_DIR, REPORTS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)


# ============================================================================
# Logging
# ============================================================================

class Logger:
    """Simple file + stdout logger for pipeline modules.

    Each pipeline run gets a timestamped log file under ``logs/``.

    Args:
        module_name: Human-readable module identifier (e.g.
            ``"clone_repositories"``).
    """

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(LOGS_DIR, f"{module_name}_{ts}.log")
        self._start_time = time.time()
        self._counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def info(self, msg: str) -> None:
        """Log an informational message."""
        self._write("INFO", msg)

    def warn(self, msg: str) -> None:
        """Log a warning message."""
        self._write("WARN", msg)

    def error(self, msg: str) -> None:
        """Log an error message."""
        self._write("ERROR", msg)

    def count(self, label: str, amount: int = 1) -> None:
        """Increment a named counter (for summary reporting)."""
        self._counts[label] = self._counts.get(label, 0) + amount

    def summary(self) -> str:
        """Return a human-readable summary of the run."""
        elapsed = time.time() - self._start_time
        lines = [
            f"Module:    {self.module_name}",
            f"Duration:  {elapsed:.2f}s",
        ]
        for label, cnt in sorted(self._counts.items()):
            lines.append(f"  {label}: {cnt}")
        lines.append(f"Log file:  {self._path}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, level: str, msg: str) -> None:
        """Write a timestamped log entry to file and print it."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        print(line)
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass  # best-effort


# ============================================================================
# CSV helpers
# ============================================================================

def write_csv(path: str, header: List[str], rows: List[List[str]]) -> None:
    """Write a CSV file with *header* and data *rows*.

    Args:
        path: Absolute or relative file path.
        header: List of column name strings.
        rows: List of row lists (each inner list must have the same
            length as *header*).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def read_csv(path: str) -> List[dict]:
    """Read a CSV file into a list of dicts.

    Args:
        path: Path to a CSV file.

    Returns:
        List of ``colname → value`` dicts.  Returns an empty list when
        the file does not exist or cannot be read.
    """
    if not os.path.isfile(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return []


# ============================================================================
# Hashing
# ============================================================================

def sha256_file(file_path: str) -> str:
    """Return the SHA-256 hex digest of a file's contents.

    Args:
        file_path: Path to the file to hash.

    Returns:
        64-character lowercase hex digest, or ``""`` on error.
    """
    try:
        h = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of a string.

    Args:
        text: Arbitrary string.

    Returns:
        64-character lowercase hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================================
# Subprocess
# ============================================================================

def run_command(
    cmd: List[str],
    timeout: int = 60,
    capture: bool = True,
) -> Tuple[bool, str, str]:
    """Run a shell command and return its result.

    Args:
        cmd: Command and arguments as a list (e.g.
            ``["g++", "-std=c++17", "file.cpp"]``).
        timeout: Maximum execution time in seconds.
        capture: When ``True``, capture stdout/stderr; when ``False``,
            inherit the parent process's streams.

    Returns:
        ``(success, stdout, stderr)`` tuple.  *success* is ``True``
        when the return code is 0.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
        return (result.returncode == 0, result.stdout or "", result.stderr or "")
    except subprocess.TimeoutExpired:
        return (False, "", f"Command timed out after {timeout}s: {' '.join(cmd)}")
    except FileNotFoundError:
        return (False, "", f"Command not found: {cmd[0]}")
    except Exception as exc:
        return (False, "", str(exc))


# ============================================================================
# Misc
# ============================================================================

def timestamp() -> str:
    """Return a human-readable UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
