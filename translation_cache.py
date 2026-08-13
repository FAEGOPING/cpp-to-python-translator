"""
translation_cache.py — Translation Cache System
==================================================

Deterministic, thread-safe cache for LLM-generated Python translations.

Each translated program is stored on disk under
``translation_cache/<model>/<program_name>.py.<hash16>``, where
``<hash16>`` is the first 16 hex characters of the SHA-256 of the
**C++ source**.  Content-addressing the cache key means a cached
translation is only reused when the *same* program name maps to the
*same* source — if the benchmark dataset is regenerated and a program's
content changes, the old translation is automatically treated as a miss
rather than silently served against the wrong source.

Cache hits skip the LLM API call entirely; cache misses trigger a normal
translation and the result is persisted atomically.

The cache has three independently togglable capabilities:

    read    — load cached translations (skip API calls)
    write   — persist newly generated translations
    both    — full read + write (default)

Usage::

    from translation_cache import configure, lookup, save, get_stats

    configure(read=True, write=True)

    cached = lookup("program_000123.cpp", cpp_source)
    if cached is not None:
        python_code = cached          # [CACHE HIT]
    else:
        python_code = translate_cpp(cpp_source)   # [CACHE MISS]
        save("program_000123.cpp", cpp_source, python_code)  # [CACHE SAVE]

    hits, misses, writes = get_stats()
"""

from __future__ import annotations

import hashlib
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

def lookup(program_name: str, cpp_source: str) -> str | None:
    """Attempt to load a cached translation for *cpp_source*.

    The cache key is ``(program_name, sha256(cpp_source))``: a cached
    translation is only returned when the program name *and* the exact C++
    source both match.  If the source has changed (e.g. the benchmark
    dataset was regenerated), the stale entry is treated as a miss.

    Args:
        program_name: Benchmark filename (e.g. ``"program_000123.cpp"``).
        cpp_source: Full C++ source code the program was translated from.

    Returns:
        The cached Python source as a string, or ``None`` when no cache
        entry exists or read is disabled.
    """
    global _hits, _misses
    if not _read_enabled:
        return None

    cache_path = _cache_path(program_name, cpp_source)
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


def save(program_name: str, cpp_source: str, python_code: str) -> None:
    """Persist a newly translated program to the cache.

    Uses atomic write (temp file + rename) to prevent corruption when
    multiple workers attempt to write the same file concurrently.

    Args:
        program_name: Benchmark filename (e.g. ``"program_000123.cpp"``).
        cpp_source: Full C++ source code the translation was produced from.
        python_code: Translated Python source to cache.
    """
    global _writes
    if not _write_enabled:
        return

    cache_path = _cache_path(program_name, cpp_source)
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

def _cache_path(program_name: str, cpp_source: str) -> Path:
    """Map ``(program_name, cpp_source)`` to a content-addressed cache path.

    ``program_000123.cpp`` + source → ``translation_cache/<model>/program_000123.py.<hash16>``.
    """
    py_name = program_name.replace(".cpp", ".py")
    digest = _content_hash(cpp_source)[:16]
    return CACHE_DIR / f"{py_name}.{digest}"


def _content_hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
