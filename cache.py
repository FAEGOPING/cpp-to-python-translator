"""
cache.py — Execution Result Cache & C++ Binary Cache
=====================================================

Two complementary caches for the translation & evaluation framework:

``ExecutionCache``
    Caches the outputs of C++ and Python program executions to avoid
    duplicate subprocess invocations.  Cache keys are deterministic
    (source-hash, input) pairs.

``CppBinaryCache``
    Caches compiled C++ binaries so that the source file is compiled
    **once** and the resulting executable is reused for every test
    case.  The compilation is invalidated only when the source file's
    modification time changes.

Both caches are process-local and ephemeral — they do not persist to disk.

Usage::

    from cache import ExecutionCache, CppBinaryCache

    bin_cache = CppBinaryCache()
    exe_path = bin_cache.get_or_compile("samples/example.cpp", timeout=10)
    # ... run many test inputs against exe_path ...
    bin_cache.cleanup()
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from typing import Callable


# ======================================================================
# Execution result cache
# ======================================================================

class ExecutionCache:
    """In-memory cache for subprocess execution (stdout) results.

    Avoids re-running the same program with the same input, which is
    common during iterative repair where the C++ baseline is executed
    repeatedly for the same test case.
    """

    def __init__(self) -> None:
        self._cpp: dict[tuple[str, str], tuple[bool, str]] = {}
        self._py: dict[tuple[str, str], tuple[bool, str]] = {}
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cpp(
        self,
        cpp_file: str,
        test_input: str,
        executor: Callable[[str, str], tuple[bool, str]],
    ) -> tuple[bool, str]:
        """Execute a C++ program, returning a cached result when available.

        Args:
            cpp_file: Path to the ``.cpp`` source file.
            test_input: Stdin input string.
            executor: Callable that compiles & runs the program
                (e.g. :func:`run_cpp` from ``run.py``).

        Returns:
            ``(success, output_or_error)`` — identical contract to
            the underlying executor.
        """
        key = self._make_key(cpp_file, test_input)
        if key in self._cpp:
            self._hits += 1
            return self._cpp[key]
        self._misses += 1
        result = executor(cpp_file, test_input)
        self._cpp[key] = result
        return result

    def run_python(
        self,
        python_code: str,
        test_input: str,
        executor: Callable[[str, str], tuple[bool, str]],
    ) -> tuple[bool, str]:
        """Execute Python code, returning a cached result when available.

        Args:
            python_code: Python source code string.
            test_input: Stdin input string.
            executor: Callable that runs the code
                (e.g. :func:`run_python` from ``run.py``).

        Returns:
            ``(success, output_or_error)`` tuple.
        """
        key = self._make_key(python_code, test_input)
        if key in self._py:
            self._hits += 1
            return self._py[key]
        self._misses += 1
        result = executor(python_code, test_input)
        self._py[key] = result
        return result

    def invalidate_python(self, python_code: str) -> None:
        """Remove all cached results for a specific Python code version.

        Call this after the code has been repaired, so stale cached
        outputs are not reused.

        Args:
            python_code: The Python source code whose cached entries
                should be purged.
        """
        to_remove: list[tuple[str, str]] = []
        source_hash = _hash_text(python_code)
        for key in self._py:
            if key[0] == source_hash:
                to_remove.append(key)
        for k in to_remove:
            del self._py[k]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        """Number of cache hits so far."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of cache misses so far."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 – 1.0).  Returns 0.0 when the cache is cold."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def clear(self) -> None:
        """Purge all cached entries and reset statistics."""
        self._cpp.clear()
        self._py.clear()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(source: str, test_input: str) -> tuple[str, str]:
        """Build a deterministic cache key from source code and input.

        Returns:
            ``(source_hash, input_hash)`` tuple of hex digests.
        """
        return (_hash_text(source), _hash_text(test_input))


# ======================================================================
# C++ binary compilation cache (v2.1 — compile once, execute many)
# ======================================================================

class CppBinaryCache:
    """Caches compiled C++ executables so compilation happens once per file.

    The compiled binary is reused for every test case.  Compilation is
    invalidated when the source file's modification time changes.
    Temp files are cleaned up automatically or on-demand.
    """

    def __init__(self) -> None:
        self._binaries: dict[str, str] = {}
        """Mapping: cpp_file → path to compiled executable."""

        self._mtime: dict[str, float] = {}
        """Last-known mtime for each cached file."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_compile(
        self,
        cpp_file: str,
        timeout: int = 10,
    ) -> tuple[bool, str]:
        """Return a compiled executable for *cpp_file*, compiling if needed.

        Args:
            cpp_file: Path to the ``.cpp`` source file.
            timeout: Compilation timeout in seconds.

        Returns:
            ``(ok, exe_path_or_error)`` — on success the second element
            is the path to a compiled executable; on failure it is a
            human-readable error message.
        """
        # Check if source has changed since last compilation
        try:
            current_mtime = os.path.getmtime(cpp_file)
        except OSError as exc:
            return False, f"Cannot access source file: {exc}"

        if cpp_file in self._binaries and self._mtime.get(cpp_file) == current_mtime:
            # Binary is fresh — reuse
            exe_path = self._binaries[cpp_file]
            if os.path.isfile(exe_path):
                return True, exe_path
            # Binary was deleted externally — recompile

        # Compile to a persistent temp file
        fd, exe_path = tempfile.mkstemp(suffix=".cpp_bin")
        os.close(fd)

        try:
            compile_result = subprocess.run(
                ["g++", "-std=c++17", cpp_file, "-o", exe_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            _try_remove(exe_path)
            return False, f"C++ compilation timed out after {timeout}s"
        except FileNotFoundError:
            _try_remove(exe_path)
            return False, (
                "g++ not found — please install g++ to enable "
                "C++ compilation and functional validation."
            )

        if compile_result.returncode != 0:
            _try_remove(exe_path)
            return False, (
                f"C++ compilation failed:\n{compile_result.stderr.strip()}"
            )

        # Store in cache
        self._binaries[cpp_file] = exe_path
        self._mtime[cpp_file] = current_mtime
        return True, exe_path

    def invalidate(self, cpp_file: str) -> None:
        """Remove cached binary for *cpp_file*, deleting the temp file.

        Args:
            cpp_file: Path to the C++ source whose binary should be purged.
        """
        exe_path = self._binaries.pop(cpp_file, None)
        self._mtime.pop(cpp_file, None)
        if exe_path is not None:
            _try_remove(exe_path)

    def cleanup(self) -> None:
        """Remove all cached binaries from disk and clear the cache."""
        for exe_path in self._binaries.values():
            _try_remove(exe_path)
        self._binaries.clear()
        self._mtime.clear()

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass


# ======================================================================
# Internal helpers
# ======================================================================

def _hash_text(text: str) -> str:
    """Return a short hex digest for *text*.

    Args:
        text: Arbitrary string to hash.

    Returns:
        64-character SHA-256 hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _try_remove(path: str) -> None:
    """Remove a file, silently ignoring errors."""
    try:
        os.remove(path)
    except OSError:
        pass
