"""
skip_list.py — Persistent Skip List for Failed Programs (v2.0)
================================================================

Maintains a JSON file (``skip_programs.json``) mapping program
filenames to their failure metadata.  Programs with persistent
infrastructure failures are automatically recorded and can be
skipped in future runs.

V2.0: Rich records with timeout_count.  A program only enters the
skip list after TWO consecutive infrastructure failures on
different runs.  Old flat-format files are migrated automatically.

Usage::

    from skip_list import should_skip, record_failure, load_skip_list, clear_skip_list

    reason = should_skip(program_name)
    if reason is not None:
        continue  # immediately skip this sample

    record_failure(program_name, "RepairTimeout", "repair")
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SKIP_FILE = PROJECT_ROOT / "skip_programs.json"

_lock = threading.Lock()
_skip_map: dict[str, dict[str, Any]] | None = None


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_skip_list() -> dict[str, dict[str, Any]]:
    """Load + migrate the skip list from disk (lazy, cached).

    Returns:
        Dict mapping ``program_000123.cpp`` → ``{reason, count, stage, ...}``.
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
                    raw = json.load(fh)
            except (json.JSONDecodeError, OSError):
                raw = {}
        else:
            raw = {}

        # Migrate flat records → rich records
        migrated = False
        for k, v in raw.items():
            if isinstance(v, str):
                raw[k] = {
                    "reason": v,
                    "count": 1,
                    "stage": "unknown",
                    "first_seen": _now_utc(),
                    "last_seen": _now_utc(),
                }
                migrated = True
        if migrated:
            _save_skip_list(raw)
        _skip_map = raw
    return _skip_map


def _save_skip_list(data: dict[str, Any]) -> None:
    """Persist the skip list to disk (atomic via temp + rename)."""
    try:
        tmp = str(SKIP_FILE) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, str(SKIP_FILE))
    except OSError:
        pass


def should_skip(program_name: str) -> str | None:
    """Check whether *program_name* should be skipped.

    A program is only skipped when its failure count >= 2 (two
    confirmations on separate runs).

    Args:
        program_name: e.g. ``"program_000029.cpp"``.

    Returns:
        The failure reason string, or ``None`` if not in skip list
        or count < 2.
    """
    entry = load_skip_list().get(program_name)
    if entry is None:
        return None
    if entry.get("count", 1) < 2:
        return None
    return str(entry.get("reason", "Unknown"))


def record_failure(
    program_name: str,
    reason: str,
    stage: str = "unknown",
) -> None:
    """Record a program failure, incrementing its timeout_count.

    - First failure: count=1, NOT added to skip list.
    - Second failure: count=2, NOW added to skip list.
    - Subsequent: count incremented.

    Args:
        program_name: e.g. ``"program_000029.cpp"``.
        reason: One of ``RepairTimeout``, ``TranslationTimeout``,
            ``RepairAPIError``, ``TranslationAPIError``.
        stage: Pipeline stage where the failure occurred.
    """
    data = load_skip_list()
    now = _now_utc()
    with _lock:
        entry = data.get(program_name)
        if entry is None:
            data[program_name] = {
                "reason": reason,
                "count": 1,
                "stage": stage,
                "first_seen": now,
                "last_seen": now,
            }
        else:
            entry["count"] = entry.get("count", 0) + 1
            entry["last_seen"] = now
            entry["reason"] = reason
            entry["stage"] = stage
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
