"""
skip_list.py — Persistent Skip List for Failed Programs
=========================================================

Maintains a JSON file (``skip_programs.json``) mapping program
filenames to their failure reason.  Programs with persistent
infrastructure failures (API timeout / API error) are automatically
recorded and can be skipped in future runs.

Usage::

    from skip_list import should_skip, record_failure, load_skip_list, clear_skip_list

    if should_skip(program_name):
        continue  # immediately skip this sample

    record_failure(program_name, "RepairTimeout")
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SKIP_FILE = PROJECT_ROOT / "skip_programs.json"

_lock = threading.Lock()
_skip_map: dict[str, str] | None = None  # None = not loaded yet


def load_skip_list() -> dict[str, str]:
    """Load the skip list from disk (lazy, cached).

    Returns:
        Dict mapping ``program_000123.cpp`` → ``"RepairTimeout"`` etc.
    """
    global _skip_map
    if _skip_map is not None:
        return _skip_map
    with _lock:
        if _skip_map is not None:
            return _skip_map
        if SKIP_FILE.is_file():
            try:
                with open(SKIP_FILE, encoding="utf-8") as fh:
                    _skip_map = json.load(fh)
            except (json.JSONDecodeError, OSError):
                _skip_map = {}
        else:
            _skip_map = {}
    return _skip_map


def _save_skip_list(data: dict[str, str]) -> None:
    """Persist the skip list to disk (atomic via temp + rename)."""
    try:
        tmp = str(SKIP_FILE) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, str(SKIP_FILE))
    except OSError:
        pass  # best-effort


def should_skip(program_name: str) -> str | None:
    """Check whether *program_name* should be skipped.

    Args:
        program_name: e.g. ``"program_000029.cpp"``.

    Returns:
        The failure reason string (e.g. ``"RepairTimeout"``), or
        ``None`` if the program is not in the skip list.
    """
    return load_skip_list().get(program_name)


def record_failure(program_name: str, reason: str) -> None:
    """Persistently record a program failure.

    Only records infrastructure failures (API timeout / API error).
    Idempotent — re-recording the same program is a no-op.

    Args:
        program_name: e.g. ``"program_000029.cpp"``.
        reason: One of ``RepairTimeout``, ``TranslationTimeout``,
            ``RepairAPIError``, ``TranslationAPIError``.
    """
    data = load_skip_list()
    if data.get(program_name) == reason:
        return  # already recorded
    with _lock:
        data[program_name] = reason
        _save_skip_list(data)


def clear_skip_list() -> None:
    """Delete the skip list file entirely."""
    global _skip_map
    with _lock:
        _skip_map = {}
        try:
            os.remove(str(SKIP_FILE))
        except OSError:
            pass


def skip_count() -> int:
    """Return the number of entries in the skip list."""
    return len(load_skip_list())
