"""
translation_cache.py — Translation Cache System
==================================================

Deterministic, thread-safe cache for LLM-generated Python translations.

Each translated program is stored on disk under
``translation_cache/<model>/<program_name>.py``.  Cache hits skip the
LLM API call entirely; cache misses trigger a normal translation and
the result is persisted atomically.

The cache has three independently togglable capabilities:

    read    — load cached translations (skip API calls)
    write   — persist newly generated translations
    both    — full read + write (default)

Usage::

    from translation_cache import configure, lookup, save, get_stats

    configure(read=True, write=True)

    cached = lookup("program_000123.cpp")
    if cached is not None:
        python_code = cached          # [CACHE HIT]
    else:
        python_code = translate_cpp(...)   # [CACHE MISS]
        save("program_000123.cpp", python_code)  # [CACHE SAVE]

    hits, misses, writes = get_stats()
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "translation_cache" / "deepseek-v4-pro"

# ---------------------------------------------------------------------------
# Configuration (thread-safe)
# ---------------------------------------------------------------------------

_read_enabled: bool = True
_write_enabled: bool = True
_lock = threading.Lock()

# ---- statistics ----
_hits: int = 0
_misses: int = 0
_writes: int = 0
_stats_lock = threading.Lock()


def configure(*, read: bool = True, write: bool = True) -> None:
    """Enable or disable cache read and write independently.

    Args:
        read: When ``True``, cached translations are used (skip API call).
        write: When ``True``, newly generated translations are persisted.
    """
    global _read_enabled, _write_enabled
    with _lock:
        _read_enabled = read
        _write_enabled = write


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------

def lookup(program_name: str) -> str | None:
    """Attempt to load a cached translation.

    The cache key is the benchmark filename (e.g. ``program_000123.cpp``).
    The cached file is ``translation_cache/<model>/program_000123.py``.

    Args:
        program_name: Benchmark filename (e.g. ``"program_000123.cpp"``).

    Returns:
        The cached Python source as a string, or ``None`` when no cache
        entry exists or read is disabled.
    """
    global _hits, _misses
    if not _read_enabled:
        return None

    cache_path = _cache_path(program_name)
    if cache_path.is_file():
        try:
            code = cache_path.read_text(encoding="utf-8")
            with _stats_lock:
                _hits += 1
            print(f"  [CACHE HIT] {program_name}")
            return code
        except (OSError, UnicodeDecodeError):
            # Corrupt or unreadable — treat as miss
            return None

    with _stats_lock:
        _misses += 1
    print(f"  [CACHE MISS] {program_name}")
    return None


def save(program_name: str, python_code: str) -> None:
    """Persist a newly translated program to the cache.

    Uses atomic write (temp file + rename) to prevent corruption when
    multiple workers attempt to write the same file concurrently.

    Args:
        program_name: Benchmark filename (e.g. ``"program_000123.cpp"``).
        python_code: Translated Python source to cache.
    """
    global _writes
    if not _write_enabled:
        return

    cache_path = _cache_path(program_name)
    try:
        # Atomic write: write to a temp file in the same directory,
        # then rename into place.  On most filesystems rename is atomic.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            suffix=".py", prefix=".tmp_", dir=str(cache_path.parent)
        )
        try:
            os.write(fd, python_code.encode("utf-8"))
        finally:
            os.close(fd)
        os.replace(tmp, str(cache_path))  # atomic on POSIX
        with _stats_lock:
            _writes += 1
        print(f"  [CACHE SAVE] {program_name}")
    except OSError:
        # Best-effort — don't crash the pipeline over cache persistence
        pass


def get_stats() -> tuple[int, int, int]:
    """Return ``(hits, misses, writes)`` since the last :func:`reset_stats`."""
    with _stats_lock:
        return (_hits, _misses, _writes)


def reset_stats() -> None:
    """Reset cache hit/miss/write counters to zero."""
    global _hits, _misses, _writes
    with _stats_lock:
        _hits = 0
        _misses = 0
        _writes = 0


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _cache_path(program_name: str) -> Path:
    """Map a benchmark filename to its cache path.

    ``program_000123.cpp`` → ``translation_cache/<model>/program_000123.py``.
    """
    py_name = program_name.replace(".cpp", ".py")
    return CACHE_DIR / py_name
