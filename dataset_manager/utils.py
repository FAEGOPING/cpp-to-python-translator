"""
dataset_manager/utils.py — Shared Utilities
============================================

Common helpers used by all Dataset Manager modules: logging, CSV I/O,
SHA256 hashing, subprocess execution, path resolution, code metrics,
and memory profiling.

All modules in ``dataset_manager/`` import from this module rather than
duplicating logic.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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

FIGURES_DIR: str = os.path.join(REPORTS_DIR, "figures")
"""Path for generated figures (PNG and PDF)."""

LOGS_DIR: str = os.path.join(DM_DIR, "logs")
"""Path for pipeline run logs."""

DATASETS_DIR: str = os.path.expanduser("~/datasets")
"""Path to the user's local datasets directory (source of C++ repos)."""

# Ensure output directories exist
for _d in (RAW_CPP_DIR, BENCHMARK_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR):
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
        self._warnings: list[str] = []
        self._errors: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def info(self, msg: str) -> None:
        """Log an informational message."""
        self._write("INFO", msg)

    def warn(self, msg: str) -> None:
        """Log a warning message."""
        self._warnings.append(msg)
        self._write("WARN", msg)

    def error(self, msg: str) -> None:
        """Log an error message."""
        self._errors.append(msg)
        self._write("ERROR", msg)

    def count(self, label: str, amount: int = 1) -> None:
        """Increment a named counter (for summary reporting).

        Args:
            label: Counter name.
            amount: Value to add (default 1).
        """
        self._counts[label] = self._counts.get(label, 0) + amount

    @property
    def warnings(self) -> List[str]:
        """List of warning messages collected so far."""
        return list(self._warnings)

    @property
    def errors(self) -> List[str]:
        """List of error messages collected so far."""
        return list(self._errors)

    @property
    def elapsed(self) -> float:
        """Seconds since this logger was created."""
        return time.time() - self._start_time

    @property
    def log_path(self) -> str:
        """Path to the underlying log file."""
        return self._path

    def summary(self) -> str:
        """Return a human-readable summary of the run."""
        elapsed = self.elapsed
        lines = [
            f"Module:       {self.module_name}",
            f"Duration:     {elapsed:.2f}s",
            "",
            "Counters:",
        ]
        for label, cnt in sorted(self._counts.items()):
            lines.append(f"  {label}: {cnt}")
        if self._warnings:
            lines.append(f"\nWarnings: ({len(self._warnings)} total)")
        if self._errors:
            lines.append(f"\nErrors: ({len(self._errors)} total)")
            for e in self._errors[-5:]:
                lines.append(f"  {e[:120]}")
        lines.append(f"\nLog file:     {self._path}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, level: str, msg: str) -> None:
        """Write a timestamped log entry to file and print it.

        Args:
            level: Log level (``"INFO"``, ``"WARN"``, ``"ERROR"``).
            msg: The message text.
        """
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

def write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    """Write a CSV file with *header* and data *rows*.

    Args:
        path: Absolute or relative file path.
        header: List of column name strings.
        rows: List of row lists (each inner list must have the same
            length as *header*).  Values are converted to strings
            automatically.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow([str(v) for v in row])


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


def run_command_full(
    cmd: List[str],
    timeout: int = 60,
) -> Tuple[int, str, str, float]:
    """Run a shell command and return full result with return code and timing.

    Args:
        cmd: Command and arguments.
        timeout: Maximum execution time in seconds.

    Returns:
        ``(return_code, stdout, stderr, elapsed_seconds)``.
    """
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        return (result.returncode, result.stdout or "", result.stderr or "", elapsed)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return (-1, "", f"Timed out after {timeout}s", elapsed)
    except FileNotFoundError:
        elapsed = time.time() - t0
        return (-2, "", f"Command not found: {cmd[0]}", elapsed)
    except Exception as exc:
        elapsed = time.time() - t0
        return (-3, "", str(exc), elapsed)


# ============================================================================
# C++ source code metrics
# ============================================================================

# Pre-compiled regex patterns for efficiency
_RE_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]', re.MULTILINE)
_RE_FUNCTION = re.compile(
    r'\b(?:void|int|long|float|double|char|bool|string|auto|'
    r'unsigned|size_t|ll|pair|vector|map|set|struct|class)\s+'
    r'[a-zA-Z_]\w*\s*\([^)]*\)\s*\{'
)
_RE_FOR = re.compile(r'\bfor\s*\(', re.MULTILINE)
_RE_WHILE = re.compile(r'\bwhile\s*\(', re.MULTILINE)
_RE_IF = re.compile(r'\bif\s*\(', re.MULTILINE)
_RE_ELSE_IF = re.compile(r'\belse\s+if\s*\(', re.MULTILINE)
_RE_SWITCH = re.compile(r'\bswitch\s*\(', re.MULTILINE)
_RE_CLASS = re.compile(r'\bclass\s+[a-zA-Z_]\w*', re.MULTILINE)
_RE_NAMESPACE = re.compile(r'\bnamespace\s+[a-zA-Z_]\w*', re.MULTILINE)
_RE_TEMPLATE = re.compile(r'\btemplate\s*<', re.MULTILINE)
_RE_LAMBDA = re.compile(r'\[\s*(?:&|=)?\s*\]\s*\(', re.MULTILINE)
_RE_RETURN = re.compile(r'\breturn\b', re.MULTILINE)
_RE_RECURSION = re.compile(r'(\w+)\s*\([^)]*\)\s*\{(?:[^}]|\{[^}]*\})*\1\s*\(', re.DOTALL)
_RE_BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)
_RE_LINE_COMMENT = re.compile(r'//[^\n]*')

# STL tokens
_STL_TOKENS: tuple[str, ...] = (
    "vector", "map", "unordered_map", "queue", "priority_queue",
    "stack", "deque", "set", "unordered_set", "string", "iostream",
    "fstream", "sstream", "algorithm", "sort", "lower_bound",
    "upper_bound", "pair", "tuple", "cmath", "cstdio", "cstdlib",
    "cstring", "bits/stdc++.h", "iterator", "numeric", "functional",
)

# Decision points (for cyclomatic complexity)
_RE_DECISION = re.compile(
    r'\b(?:if|else\s+if|for|while|switch|case\s+(?!.*:.*=)|'
    r'&&|\|\||\?\s*.*\s*:)\b'
)


def count_lines(source: str) -> Dict[str, int]:
    """Count LOC, blank lines, and comment lines in C++ source.

    Args:
        source: C++ source code string.

    Returns:
        Dict with keys ``total``, ``code``, ``blank``, ``comments``.
    """
    lines = source.split("\n")
    total = len(lines)
    blank = 0
    comments = 0
    code = 0
    in_block = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank += 1
            continue

        # Track block comments
        if in_block:
            comments += 1
            if "*/" in stripped:
                in_block = False
            continue

        if stripped.startswith("//"):
            comments += 1
            continue

        if "/*" in stripped:
            comments += 1
            if "*/" not in stripped:
                in_block = True
            continue

        if stripped.startswith("#"):
            # Preprocessor — count as code
            code += 1
            continue

        code += 1

    return {"total": total, "code": code, "blank": blank, "comments": comments}


def count_patterns(source: str) -> Dict[str, int]:
    """Count structural patterns in C++ source.

    Args:
        source: C++ source code string.

    Returns:
        Dict with counts for functions, loops, conditionals, classes,
        namespaces, templates, lambdas, includes, and returns.
    """
    clean = _RE_BLOCK_COMMENT.sub(" ", source)
    clean = _RE_LINE_COMMENT.sub(" ", clean)

    return {
        "functions": len(_RE_FUNCTION.findall(clean)),
        "loops": len(_RE_FOR.findall(clean)) + len(_RE_WHILE.findall(clean)),
        "conditionals": (
            len(_RE_IF.findall(clean))
            + len(_RE_ELSE_IF.findall(clean))
            + len(_RE_SWITCH.findall(clean))
        ),
        "classes": len(_RE_CLASS.findall(clean)),
        "namespaces": len(_RE_NAMESPACE.findall(clean)),
        "templates": len(_RE_TEMPLATE.findall(clean)),
        "lambdas": len(_RE_LAMBDA.findall(clean)),
        "includes": len(_RE_INCLUDE.findall(clean)),
        "returns": len(_RE_RETURN.findall(clean)),
    }


def cyclomatic_complexity(source: str) -> int:
    """Calculate McCabe's cyclomatic complexity.

    Starts at 1 (the function entry point), then adds 1 for each
    decision point (if, for, while, switch, &&, ||, ?:).

    Args:
        source: C++ source code string.

    Returns:
        Cyclomatic complexity integer.
    """
    clean = _RE_BLOCK_COMMENT.sub(" ", source)
    clean = _RE_LINE_COMMENT.sub(" ", clean)
    decisions = len(_RE_DECISION.findall(clean))
    return 1 + decisions


def nesting_depth(source: str) -> int:
    """Estimate maximum brace nesting depth.

    Args:
        source: C++ source code string.

    Returns:
        Maximum nesting depth found.
    """
    clean = _RE_BLOCK_COMMENT.sub(" ", source)
    clean = _RE_LINE_COMMENT.sub(" ", clean)
    depth = 0
    max_depth = 0
    for ch in clean:
        if ch == "{":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch == "}":
            depth -= 1
    return max_depth


def detect_stl_usage(source: str) -> str:
    """Detect STL headers/features used in the source.

    Args:
        source: C++ source code string.

    Returns:
        Comma-separated list of detected STL tokens, or ``"none"``.
    """
    lower = source.lower()
    found: list[str] = []
    for token in _STL_TOKENS:
        if token in lower:
            found.append(token)
    return ", ".join(sorted(found)) if found else "none"


# ============================================================================
# Git helpers
# ============================================================================

def git_last_commit(repo_path: str) -> str:
    """Get the last commit date for a repository.

    Args:
        repo_path: Path to a git repository (contains ``.git/``).

    Returns:
        ISO-format date string, or ``"unknown"``.
    """
    ok, stdout, _ = run_command(
        ["git", "-C", repo_path, "log", "-1", "--format=%aI"],
        timeout=10,
    )
    return stdout.strip() if ok and stdout.strip() else "unknown"


def git_remote_url(repo_path: str) -> str:
    """Get the remote origin URL for a repository.

    Args:
        repo_path: Path to a git repository.

    Returns:
        Remote URL string, or ``"unknown"``.
    """
    ok, stdout, _ = run_command(
        ["git", "-C", repo_path, "remote", "get-url", "origin"],
        timeout=10,
    )
    return stdout.strip() if ok else "unknown"


# ============================================================================
# Memory profiling
# ============================================================================

def memory_usage_mb() -> float:
    """Return current process resident memory in MB.

    Returns:
        Memory usage in megabytes, or ``0.0`` on error.
    """
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, AttributeError):
        return 0.0


# ============================================================================
# Misc
# ============================================================================

def timestamp() -> str:
    """Return a human-readable UTC timestamp string.

    Returns:
        ISO-format UTC timestamp.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def extract_source_info(file_path: str, base_dir: str = "") -> Tuple[str, str, str, str]:
    """Extract category, repo, and relative path from a file path.

    Uses the convention ``<base>/<category>/<repo>/.../file.cpp``.

    Args:
        file_path: Absolute path to a C++ file.
        base_dir: The base directory (e.g. ``RAW_CPP_DIR``).

    Returns:
        ``(category, repository, relative_path, github_url)``.
    """
    if not base_dir:
        base_dir = RAW_CPP_DIR

    rel = os.path.relpath(file_path, base_dir) if base_dir in file_path else file_path
    parts = rel.split(os.sep)
    category = parts[0] if len(parts) > 0 else "unknown"
    repo = parts[1] if len(parts) > 1 else "unknown"

    # Try to resolve GitHub URL
    repo_path = os.path.join(DATASETS_DIR, category, repo)
    url = git_remote_url(repo_path)

    return category, repo, rel, url
