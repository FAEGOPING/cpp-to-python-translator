"""
run.py — Compiler-Assisted C++ to Python Translation System
================================================================
Research-Grade Code Translation Evaluation Framework

Supports:
  - Compilation Validation  (py_compile)
  - Runtime Validation      (subprocess + timeout)
  - Functional Equivalence  (output comparison with original C++)
  - Iterative Self-Repair   (LLM feedback loop)
  - Experiment Logging      (per-round + summary CSV)
  - Multi-Test Validation   (multiple .in/.out pairs)
  - Differential Testing    (C++ vs Python per test case)
  - Automatic Test Generation (random, boundary, edge, heuristic, LLM)
  - Execution Result Caching  (avoids duplicate runs)
  - C++ Binary Compilation Cache (compile once, execute many)
  - Enhanced Repair Prompts   (error categories, history, smart failure selection)
  - Extended Experiment Logging (timing per phase, test counts, error details)

Workflow:
  C++ Source → LLM Translation → Compile Check → Runtime Check
  → Functional Validation (all test cases) → Self-Repair → Repeat (max N rounds)

Author: Research-Grade Translation Framework
Python: 3.10+
Version: 2.1
"""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
import threading
import time
from typing import List, Optional, Tuple

# Thread-safe CSV writes for parallel experiment execution
_csv_lock = threading.Lock()

# Buffered CSV mode — rows are collected in memory instead of written
# to disk immediately. The main thread sorts and flushes them after
# all workers complete, guaranteeing deterministic output ordering.
_buffer_mode: bool = False
_result_buffer: list[list] = []
_summary_buffer: list[list] = []
_buffer_header: list[str] | None = None
_buffer_summary_header: list[str] | None = None


def enable_buffer_mode() -> None:
    """Switch CSV logging to in-memory buffering.

    When enabled, :func:`log_result` and :func:`log_summary` append rows
    to internal lists instead of writing to files.  Call
    :func:`flush_sorted_csvs` after all workers finish to write the
    aggregated CSV files sorted by program name.

    This has no effect on the sequential path — direct file writes are
    the default.
    """
    global _buffer_mode, _result_buffer, _summary_buffer
    global _buffer_header, _buffer_summary_header
    _buffer_mode = True
    _result_buffer = []
    _summary_buffer = []
    _buffer_header = None
    _buffer_summary_header = None


def flush_sorted_csvs() -> None:
    """Write buffered CSV rows to disk, sorted by program filename.

    Sorts both the per-round experiment log and the per-program summary
    log alphabetically by ``Program`` (the first column), then writes
    complete CSV files with headers.

    Idempotent — calling when buffer mode is not active or buffers are
    empty is a no-op.
    """
    global _buffer_mode, _result_buffer, _summary_buffer
    global _buffer_header, _buffer_summary_header

    if not _buffer_mode:
        return

    cfg = _cfg()

    # --- detailed experiment CSV ---
    if _result_buffer:
        _result_buffer.sort(key=lambda row: _sort_key(row[0]))
        try:
            with open(cfg.csv_file, mode="w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if _buffer_header:
                    writer.writerow(_buffer_header)
                for row in _result_buffer:
                    writer.writerow(row)
        except OSError:
            pass

    # --- summary CSV ---
    if _summary_buffer:
        _summary_buffer.sort(key=lambda row: _sort_key(row[0]))
        try:
            with open(cfg.summary_csv, mode="w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if _buffer_summary_header:
                    writer.writerow(_buffer_summary_header)
                for row in _summary_buffer:
                    writer.writerow(row)
        except OSError:
            pass

    # Reset state
    _buffer_mode = False
    _result_buffer = []
    _summary_buffer = []
    _buffer_header = None
    _buffer_summary_header = None


import re as _re

_SORT_SPLIT = _re.compile(r"(\d+)")

def _sort_key(program_name: str) -> tuple:
    """Natural sort key for program filenames.

    Ensures ``program_000002.cpp`` sorts after ``program_000001.cpp``
    rather than lexicographically after ``program_000010.cpp``.

    Args:
        program_name: A filename like ``"program_000123.cpp"``.

    Returns:
        A tuple suitable for use as a sort key.
    """
    base = program_name.replace(".cpp", "").replace(".py", "")
    parts = _SORT_SPLIT.split(base)
    key: list = []
    for p in parts:
        if p.isdigit():
            key.append((0, int(p)))
        else:
            key.append((1, p.lower()))
    return tuple(key)

from gpt_api import call_gpt, call_gpt_structured, CallResult
from perf_monitor import get_monitor
from skip_list import record_failure as _record_skip
from token_tracker import (
    reset_program_tokens, get_program_tokens, get_last_call_tokens,
)
from translation_cache import lookup as _cache_lookup, save as _cache_save

# Framework modules
from config import Config, DEFAULT_CONFIG, _resolve_legacy
from cache import ExecutionCache, CppBinaryCache
from test_generator import TestGenerator, TestCase
from differential_testing import (
    run_differential_tests,
    DiffReport,
)

# ============================================================================
# Configuration — defaults reproduce original behaviour exactly
# ============================================================================

_active_config: Config | None = None


def get_config() -> Config:
    """Return the active (possibly user-overridden) configuration.

    By default this is :data:`DEFAULT_CONFIG`.  Users may call
    :func:`set_config` before :func:`main` to customise the run.

    Returns:
        The active :class:`Config` instance.
    """
    global _active_config
    if _active_config is None:
        _active_config = _resolve_legacy()
    return _active_config


def set_config(cfg: Config) -> None:
    """Replace the active configuration (call before :func:`main`).

    Args:
        cfg: The :class:`Config` instance to use for all subsequent
            pipeline operations.
    """
    global _active_config
    _active_config = cfg


def _cfg() -> Config:
    """Convenience: return the active config."""
    return get_config()


# ============================================================================
# C++ binary compilation cache (compile once, execute many)
# ============================================================================

_cpp_bin_cache: CppBinaryCache = CppBinaryCache()
"""Module-level cache so each C++ file is compiled once per run."""


def _cleanup_cpp_cache() -> None:
    """Remove all cached C++ binaries.  Called at end of main()."""
    _cpp_bin_cache.cleanup()


# ============================================================================
# Utility: test input loading (original — preserved)
# ============================================================================

def load_test_input(cpp_file: str) -> str:
    """Load the corresponding ``.in`` file for a C++ source file.

    Looks for a file with the same base name but ``.in`` extension
    alongside the C++ source.  Returns its content as a string, or
    an empty string when no ``.in`` file exists.

    Args:
        cpp_file: Absolute path to a ``.cpp`` source file.

    Returns:
        Test input string (may be empty).
    """
    in_path = os.path.splitext(cpp_file)[0] + ".in"
    if os.path.isfile(in_path):
        try:
            with open(in_path, encoding="utf-8") as fh:
                return fh.read()
        except (OSError, UnicodeDecodeError):
            return ""
    return ""


# ============================================================================
# Utility: multi-test loading (extends single-test behaviour)
# ============================================================================

def load_test_cases(cpp_file: str) -> List[TestCase]:
    """Discover all test cases for *cpp_file*.

    Supports two modes:

    **Mode A — Numbered files (preferred):**
        ``example_1.in``, ``example_1.out``,
        ``example_2.in``, ``example_2.out``, ...

    **Mode B — Single ``.in`` (legacy):**
        ``example.in`` — used as the sole test case.  When an
        ``example.out`` file also exists its content is used as the
        expected output; otherwise the C++ program's output at runtime
        serves as the oracle.

    Args:
        cpp_file: Absolute path to a ``.cpp`` source file.

    Returns:
        List of ``(input_text, expected_output_or_None)`` pairs.
        ``None`` for expected output means "use C++ execution as oracle".
    """
    base = os.path.splitext(cpp_file)[0]
    cases: list[TestCase] = []

    # Mode A: numbered files  example_1.in, example_2.in, ...
    idx = 1
    while True:
        in_path = f"{base}_{idx}.in"
        out_path = f"{base}_{idx}.out"
        if os.path.isfile(in_path):
            try:
                with open(in_path, encoding="utf-8") as fh:
                    inp = fh.read()
            except (OSError, UnicodeDecodeError):
                idx += 1
                continue
            expected: str | None = None
            if os.path.isfile(out_path):
                try:
                    with open(out_path, encoding="utf-8") as fh:
                        expected = fh.read()
                except (OSError, UnicodeDecodeError):
                    expected = None
            cases.append((inp, expected))
            idx += 1
        else:
            break

    if cases:
        return cases

    # Mode B: legacy single .in file
    inp = load_test_input(cpp_file)
    out_path = base + ".out"
    expected = None
    if os.path.isfile(out_path):
        try:
            with open(out_path, encoding="utf-8") as fh:
                expected = fh.read()
        except (OSError, UnicodeDecodeError):
            expected = None

    # Always return at least one case (even with empty input)
    cases.append((inp, expected))
    return cases


# ============================================================================
# Utility: error classification (original — preserved)
# ============================================================================

_KNOWN_PYTHON_ERRORS: tuple[str, ...] = (
    "SyntaxError",
    "IndentationError",
    "TabError",
    "NameError",
    "TypeError",
    "ValueError",
    "IndexError",
    "KeyError",
    "AttributeError",
    "ImportError",
    "ModuleNotFoundError",
    "ZeroDivisionError",
    "RecursionError",
    "RuntimeError",
    "FileNotFoundError",
    "OSError",
    "MemoryError",
    "OverflowError",
    "UnboundLocalError",
    "StopIteration",
    "AssertionError",
    "EOFError",
)


def _classify_error(error_text: str | None) -> str:
    """Heuristically classify an error message into a known category.

    Args:
        error_text: Raw stderr / error string.  May be ``None``.

    Returns:
        One of the known Python exception names, ``"Timeout"``,
        ``"FunctionalMismatch"``, or ``"Unknown"``.
    """
    if not error_text:
        return "Unknown"
    for err in _KNOWN_PYTHON_ERRORS:
        if err in error_text:
            return err
    if "Timeout" in error_text or "timed out" in error_text.lower():
        return "Timeout"
    return "Unknown"


# ============================================================================
# Utility: error category (maps errors to broad categories)
# ============================================================================

def _classify_error_category(error_type: str) -> str:
    """Map a concrete error type to a broad category for the repair prompt.

    Args:
        error_type: Classified error name (e.g. ``"SyntaxError"``).

    Returns:
        One of ``"syntax"``, ``"runtime"``, ``"semantic"``,
        ``"timeout"``, or ``"unknown"``.
    """
    syntax_errors = {
        "SyntaxError", "IndentationError", "TabError",
    }
    runtime_errors = {
        "NameError", "TypeError", "ValueError", "IndexError",
        "KeyError", "AttributeError", "ImportError",
        "ModuleNotFoundError", "ZeroDivisionError", "RecursionError",
        "RuntimeError", "FileNotFoundError", "OSError", "MemoryError",
        "OverflowError", "UnboundLocalError", "StopIteration",
        "AssertionError", "EOFError",
    }

    if error_type in syntax_errors:
        return "syntax"
    if error_type in runtime_errors:
        return "runtime"
    if error_type == "FunctionalMismatch":
        return "semantic"
    if error_type == "Timeout":
        return "timeout"
    return "unknown"


# ============================================================================
# C++ compilation & execution
# ============================================================================

def run_cpp(cpp_file: str, test_input: str) -> tuple[bool, str]:
    """Compile and execute a C++ program, capturing its stdout.

    Uses an internal compilation cache so the C++ source is compiled
    **once** and the resulting executable is reused for every test
    case.  The cache is invalidated automatically when the source file
    changes.

    Args:
        cpp_file: Path to the ``.cpp`` source file.
        test_input: String to pipe to the program's stdin.

    Returns:
        ``(success, output_or_error)`` — on success *output_or_error*
        is the program's stdout; on failure it is a human-readable
        error message.
    """
    cfg = _cfg()

    # -- get or compile the binary (cached) ----------------------------------
    ok, exe_path_or_err = _cpp_bin_cache.get_or_compile(
        cpp_file, timeout=cfg.execution_timeout
    )

    if not ok:
        return False, exe_path_or_err

    exe_path = exe_path_or_err

    # -- run the binary ------------------------------------------------------
    try:
        run_result = subprocess.run(
            [exe_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=cfg.execution_timeout,
        )

        if run_result.returncode != 0:
            err = run_result.stderr.strip() or "(no stderr)"
            return False, (
                f"C++ runtime error (exit {run_result.returncode}):\n{err}"
            )

        return True, run_result.stdout

    except subprocess.TimeoutExpired:
        return False, (
            f"Timeout: C++ execution exceeded {cfg.execution_timeout}s"
        )


def clean_code(text: str) -> str:
    """Strip markdown fences, leading/trailing blank lines, and
    normalise whitespace from an LLM-generated code response.

    The LLM sometimes wraps code in `` ```python … ``` `` or
    `` ```cpp … ``` ``` `` despite explicit instructions not to.
    This function removes those artifacts so the code compiles
    cleanly.

    Args:
        text: Raw LLM response text (may contain markdown fences).

    Returns:
        Clean Python source code.
    """
    import re

    code = text.strip()
    if not code:
        return code

    # Remove opening markdown fence: ```python, ```cpp, ```py, or bare ```
    code = re.sub(r"^```(?:python|cpp|c\+\+|py|python3)?\s*", "", code, flags=re.IGNORECASE)

    # Remove closing markdown fence
    code = re.sub(r"\s*```\s*$", "", code)

    # Normalise: strip leading/trailing blank lines again
    code = code.strip()

    return code


# ============================================================================
# Translation prompt (original — preserved)
# ============================================================================

def translate_cpp(cpp_code: str) -> "tuple[str, CallResult]":
    """Translate C++ source code to Python via the LLM.

    Args:
        cpp_code: Full C++ source code as a string.

    Returns:
        ``(python_code, api_result)`` tuple with token + timing data.
    """
    prompt = f"""\
You are an expert software engineer.

Translate the following C++ code into correct, idiomatic Python 3.

Requirements:
1. Preserve the original functionality — the Python program must produce
   **identical output** to the C++ program for any given input.
2. Return **only** the Python code.
3. Do **not** include explanations, markdown fences, or comments.

C++ Code:

{cpp_code}"""

    result = call_gpt_structured(prompt)
    cleaned = clean_code(result.text)
    return cleaned, result


# ============================================================================
# Repair prompt (extended — original params preserved, new optional params)
# ============================================================================

def fix_code(
    cpp_code: str,
    python_code: str,
    *,
    compile_error: str | None = None,
    runtime_error: str | None = None,
    functional_mismatch: str | None = None,
    error_category: str | None = None,
    failed_test_inputs: str | None = None,
    expected_outputs: str | None = None,
    actual_outputs: str | None = None,
    repair_history: str | None = None,
    previous_repair_count: int = 0,
) -> "tuple[str, CallResult]":
    """Repair translated Python code with a strictly bounded LLM response.

    V5.3 — max_tokens=4000 prevents 20k+ token completions.
    The prompt explicitly demands ONLY Python code output, stated at
    both the beginning and end of the prompt.  Completion sanity
    check warns when the response is abnormally large.
    """
    _MAX_ERR  = 1000   # chars — truncate long error messages
    _MAX_CPP  = 300    # lines — compact C++ source above this
    _MAX_TOK  = 4000   # max completion tokens (prevents 20k+ responses)

    # ---- helpers ----
    def _trunc(s: str) -> str:
        if len(s) > _MAX_ERR:
            return s[:_MAX_ERR] + "\n... (truncated)"
        return s

    # ---- error summary ----
    error_parts: list[str] = []
    if compile_error:
        error_parts.append(f"Compilation Error:\n{_trunc(compile_error)}")
    if runtime_error:
        error_parts.append(f"Runtime Error:\n{_trunc(runtime_error)}")
    if functional_mismatch:
        error_parts.append(f"Output Mismatch:\n{_trunc(functional_mismatch)}")
    error_summary = "\n\n".join(error_parts) if error_parts else "No error details."

    # ---- 1 representative failed test case only ----
    test_case_block = ""
    if failed_test_inputs:
        fl = [l for l in failed_test_inputs.strip().split("\n") if l.strip()]
        first_input = fl[0] if fl else ""
        exp_line = ""
        act_line = ""
        if expected_outputs:
            el = [l for l in expected_outputs.strip().split("\n") if l.strip()]
            exp_line = el[0] if el else ""
        if actual_outputs:
            al = [l for l in actual_outputs.strip().split("\n") if l.strip()]
            act_line = al[0] if al else ""
        if first_input:
            test_case_block = (
                f"Failed Test:\n"
                f"  Input:    {first_input}\n"
                f"  Expected: {exp_line}\n"
                f"  Got:      {act_line}"
            )

    # ---- last 2 repair entries only ----
    compact_history = ""
    if repair_history:
        hl = [l for l in repair_history.strip().split("\n") if l.strip()]
        compact_history = "\n".join(hl[-2:]) if len(hl) > 2 else "\n".join(hl)

    # ---- compact C++ (keep first 50 + signatures + last 30 when >300 lines) ----
    cpp_display = cpp_code
    cpp_lines = cpp_code.count("\n") + 1
    if cpp_lines > _MAX_CPP:
        all_lines = cpp_code.split("\n")
        kept = (
            all_lines[:50] +
            ["// ... (omitted middle section) ..."] +
            [l for l in all_lines if (
                any(l.strip().startswith(p) for p in (
                    "int ", "void ", "bool ", "double ", "float ",
                    "long ", "char ", "string ", "auto ", "vector",
                    "class ", "struct ", "template"))
                and "(" in l and ")" in l
            )] +
            all_lines[-30:]
        )
        seen: set[str] = set()
        deduped: list[str] = []
        for l in kept:
            s = l.strip()
            if s not in seen or s.startswith("//"):
                seen.add(s)
                deduped.append(l)
        cpp_display = "\n".join(deduped)

    # ---- category hint ----
    category_hint = ""
    _HINTS = {
        "syntax":   "Syntax/compilation error — check Python grammar.",
        "runtime":  "Runtime error — check variables, types, imports.",
        "semantic": "Logic error — code runs but produces wrong output.",
        "timeout":  "Timeout — check for infinite loops.",
    }
    if error_category:
        category_hint = _HINTS.get(error_category, "")

    # ==== BUILD PROMPT — output constraint at BOTH start and end ====
    OUTPUT_RULE = (
        "IMPORTANT: Return ONLY the repaired Python source code. "
        "No explanations. No markdown. No code fences. "
        "No change descriptions. No commentary of any kind. "
        "Only the final Python program."
    )

    parts: list[str] = [OUTPUT_RULE]
    if category_hint:
        parts.append(category_hint)
    if compact_history:
        parts.append(f"Last repairs:\n{compact_history}")
    if previous_repair_count > 0:
        parts.append(
            f"Repair attempt #{previous_repair_count + 1} — "
            "previous attempts failed, use a different approach."
        )

    parts.append(
        f"Original C++:\n```cpp\n{cpp_display}\n```\n\n"
        f"Current Python (to fix):\n```python\n{python_code}\n```\n\n"
        f"Errors:\n{error_summary}"
    )
    if test_case_block:
        parts.append(test_case_block)
    parts.append(OUTPUT_RULE)

    prompt = "\n\n".join(parts)

    # ---- log prompt size ----
    prompt_kb = len(prompt) / 1024
    est_tok = len(prompt) // 4
    print(f"     Repair prompt: {prompt_kb:.1f} KB (~{est_tok} tokens)", flush=True)

    # ==== CALL API with max_tokens limit ====
    result = call_gpt_structured(prompt, max_tokens=_MAX_TOK)
    cleaned = clean_code(result.text)

    # ---- completion sanity check ----
    if result.completion_tokens > 5000:
        print(f"     WARNING: Abnormally large LLM response. "
              f"Completion: {result.completion_tokens} tokens", flush=True)

    # ---- repair efficiency ----
    ratio = (result.completion_tokens / max(result.prompt_tokens, 1))
    print(f"     Repair done: {result.elapsed_seconds:.1f}s | "
          f"P:{result.prompt_tokens} C:{result.completion_tokens} "
          f"T:{result.total_tokens} | C/P ratio: {ratio:.2f}", flush=True)
    if ratio > 3:
        print(f"     WARNING: Repair response unusually long "
              f"(C/P ratio {ratio:.2f})", flush=True)

    return cleaned, result

def check_compile(code: str) -> tuple[bool, str | None]:
    """Check whether Python *code* passes ``py_compile``.

    The check is bounded by ``execution_timeout`` seconds to prevent a
    hung subprocess from blocking the worker thread indefinitely.  On
    timeout the compilation is treated as a failure and the worker
    continues to the next stage.

    Args:
        code: Python source code string.

    Returns:
        ``(passed, error_or_none)`` — *error_or_none* is the compiler
        stderr on failure, ``None`` on success.
    """
    timeout = _cfg().execution_timeout

    fd, file_path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(code)
    except OSError:
        _try_remove(file_path)
        return False, "Failed to write temporary Python file for compilation check"

    try:
        result = subprocess.run(
            ["python3", "-m", "py_compile", file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            return False, result.stderr.strip()

        return True, None

    except subprocess.TimeoutExpired:
        return False, (
            f"[Timeout] py_compile exceeded {timeout}s"
        )
    finally:
        _try_remove(file_path)


# ============================================================================
# Runtime execution (original — preserved)
# ============================================================================

def run_python(code: str, test_input: str) -> tuple[bool, str]:
    """Execute Python *code* with *test_input* on stdin and a timeout.

    Args:
        code: Python source code string.
        test_input: String to pipe to stdin.

    Returns:
        ``(success, output_or_error)`` — on success *output_or_error*
        is stdout; on failure it is a classified error message.
    """
    fd, file_path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(code)
    except OSError:
        _try_remove(file_path)
        return False, "[OSError] Failed to write temporary Python file"

    try:
        result = subprocess.run(
            ["python3", file_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=_cfg().execution_timeout,
        )

        if result.returncode != 0:
            raw = result.stderr.strip()
            error_type = _classify_error(raw)
            return False, f"[{error_type}] {raw}"

        return True, result.stdout

    except subprocess.TimeoutExpired:
        return False, (
            f"[Timeout] Execution exceeded {_cfg().execution_timeout} seconds"
        )
    finally:
        _try_remove(file_path)


# ============================================================================
# Functional equivalence validation (original — preserved)
# ============================================================================

def validate_translation(
    cpp_file: str,
    python_code: str,
    test_input: str,
) -> tuple[bool, str, str, str]:
    """Check whether the translated Python produces the same output as the
    original C++ program for a given input.

    Args:
        cpp_file: Path to the original C++ source.
        python_code: Translated Python source code.
        test_input: Test input string (piped to stdin for both programs).

    Returns:
        ``(passed, cpp_output, python_output, mismatch_details)``.

        - *passed*: ``True`` when outputs match.
        - *cpp_output*: stdout from the C++ program (trimmed).
        - *python_output*: stdout from the Python program (trimmed).
        - *mismatch_details*: Human-readable description when outputs
          differ, empty string otherwise.  Starts with
          ``"C++_EXECUTION_FAILED:"`` when the C++ program itself could
          not be run.
    """
    # -- run original C++ ---------------------------------------------------
    cpp_ok, cpp_result = run_cpp(cpp_file, test_input)
    if not cpp_ok:
        return (
            False,
            "",
            "",
            f"C++_EXECUTION_FAILED: {cpp_result}",
        )

    # -- run translated Python ----------------------------------------------
    py_ok, py_result = run_python(python_code, test_input)
    if not py_ok:
        return (
            False,
            cpp_result.strip(),
            "",
            f"Python execution failed: {py_result}",
        )

    # -- compare outputs ----------------------------------------------------
    cpp_output = cpp_result.strip()
    python_output = py_result.strip()

    if cpp_output == python_output:
        return True, cpp_output, python_output, ""

    mismatch = (
        f"Output mismatch detected:\n"
        f"  Expected (C++):  {cpp_output}\n"
        f"  Got (Python):    {python_output}"
    )
    return False, cpp_output, python_output, mismatch


# ============================================================================
# Multi-test validation (differential testing wrapper)
# ============================================================================

def validate_translation_multi(
    cpp_file: str,
    python_code: str,
    test_cases: List[TestCase],
    cache: ExecutionCache | None = None,
) -> DiffReport:
    """Run differential testing across all *test_cases*.

    Runs every test case through both the C++ oracle and the translated
    Python program, comparing outputs individually.  The translation
    passes only when **all** test cases produce identical output.

    Args:
        cpp_file: Path to the original C++ source.
        python_code: Translated Python source code.
        test_cases: List of ``(input, expected_output_or_None)`` pairs.
        cache: Optional execution cache for avoiding duplicate runs.

    Returns:
        :class:`DiffReport` with per-test results and aggregate
        statistics.
    """
    return run_differential_tests(
        cpp_file=cpp_file,
        python_code=python_code,
        test_cases=test_cases,
        run_cpp_fn=run_cpp,
        run_python_fn=run_python,
        cache=cache,
    )


# ============================================================================
# File output (original — preserved)
# ============================================================================

def save_code(program_name: str, code: str) -> None:
    """Write translated Python *code* to the ``translated/`` directory.

    Args:
        program_name: Original filename (e.g. ``"example.cpp"``).
        code: Python source code to persist.
    """
    out_name = program_name.replace(".cpp", ".py")
    out_path = os.path.join(_cfg().translated_dir, out_name)

    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(code)
    except OSError:
        # Best-effort — don't crash the pipeline over file output
        pass


# ============================================================================
# Per-round experiment log (extended — original params preserved)
# ============================================================================

_EXPERIMENT_HEADER = [
    "Program",
    "Round",
    "CompilePass",
    "RuntimePass",
    "FunctionalPass",
    "ErrorType",
    "TimeSeconds",
    "RepairCount",
]

# Extended columns — appended only when extended_logging is True
_EXPERIMENT_HEADER_EXTENDED = [
    "TranslationTime",
    "CompileTime",
    "RuntimeTime",
    "ValidationTime",
    "RepairTime",
    "LLMResponseTime",
    "GeneratedTestCount",
    "ExecutedTestCount",
    "PassedTestCount",
    "SuccessRate",
    "FailureReason",
    "FinalErrorType",
    "TotalRepairAttempts",
    "PromptTokens",
    "CompletionTokens",
    "TotalTokens",
]


def log_result(
    program: str,
    round_num: int,
    *,
    compile_pass: bool,
    runtime_pass: bool,
    functional_pass: bool,
    error_type: str,
    elapsed_time: float,
    repair_count: int,
    translation_time: float | None = None,
    compile_time: float | None = None,
    runtime_time: float | None = None,
    validation_time: float | None = None,
    repair_time: float | None = None,
    llm_response_time: float | None = None,
    generated_test_count: int | None = None,
    executed_test_count: int | None = None,
    passed_test_count: int | None = None,
    success_rate: float | None = None,
    failure_reason: str | None = None,
    final_error_type: str | None = None,
    total_repair_attempts: int | None = None,
) -> None:
    """Append one row to the detailed experiment CSV.

    Creates the file with the correct header when it does not yet exist.
    When ``extended_logging`` is enabled in the active config, additional
    columns are appended (old columns remain in their original positions).

    Args:
        program: Program name (e.g. ``"example.cpp"``).
        round_num: Current repair round (0-indexed).
        compile_pass: Whether ``py_compile`` succeeded.
        runtime_pass: Whether Python execution succeeded.
        functional_pass: Whether outputs matched the C++ oracle.
        error_type: Classified error name or ``"None"``.
        elapsed_time: Total elapsed seconds since start.
        repair_count: Number of repair attempts so far.
        translation_time: Time spent in LLM translation (seconds).
        compile_time: Time spent in ``py_compile`` check.
        runtime_time: Time spent in Python execution.
        validation_time: Time spent in functional validation.
        repair_time: Time spent in LLM repair call.
        llm_response_time: Wall-clock time for the LLM API call.
        generated_test_count: Number of auto-generated test cases.
        executed_test_count: Total test cases executed.
        passed_test_count: Number of test cases that passed.
        success_rate: ``passed / executed`` ratio.
        failure_reason: Human-readable failure description.
        final_error_type: Error type after all repairs exhausted.
        total_repair_attempts: Cumulative repair attempts.
    """
    csv_path = _cfg().csv_file
    use_extended = _cfg().extended_logging

    header = list(_EXPERIMENT_HEADER)
    if use_extended:
        header.extend(_EXPERIMENT_HEADER_EXTENDED)

    # Build the row identically regardless of write mode
    base_row = [
        program,
        round_num,
        compile_pass,
        runtime_pass,
        functional_pass,
        error_type,
        round(elapsed_time, 2),
        repair_count,
    ]

    if use_extended:
        base_row.extend([
            round(translation_time, 4) if translation_time is not None else "",
            round(compile_time, 4) if compile_time is not None else "",
            round(runtime_time, 4) if runtime_time is not None else "",
            round(validation_time, 4) if validation_time is not None else "",
            round(repair_time, 4) if repair_time is not None else "",
            round(llm_response_time, 4) if llm_response_time is not None else "",
            generated_test_count if generated_test_count is not None else "",
            executed_test_count if executed_test_count is not None else "",
            passed_test_count if passed_test_count is not None else "",
            round(success_rate, 4) if success_rate is not None else "",
            failure_reason or "",
            final_error_type or "",
            total_repair_attempts if total_repair_attempts is not None else "",
            get_last_call_tokens()[0],
            get_last_call_tokens()[1],
            get_last_call_tokens()[2],
        ])

    if _buffer_mode:
        # Append to in-memory buffer — main thread sorts + flushes later
        global _buffer_header
        with _csv_lock:
            if _buffer_header is None:
                _buffer_header = header
            _result_buffer.append(base_row)
    else:
        # Write directly to disk (sequential path)
        file_exists = os.path.isfile(csv_path)
        try:
            with _csv_lock, open(csv_path, mode="a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(base_row)
        except OSError:
            pass  # Logging is best-effort


# ============================================================================
# Summary log (extended — original params preserved)
# ============================================================================

_SUMMARY_HEADER = [
    "Program",
    "InitialCompilePass",
    "FinalCompilePass",
    "RuntimePass",
    "FunctionalPass",
    "RepairRounds",
    "TotalTime",
]

# Extended summary columns
_SUMMARY_HEADER_EXTENDED = [
    "TranslationTime",
    "ValidationTime",
    "AvgRepairTime",
    "GeneratedTestCount",
    "ExecutedTestCount",
    "PassedTestCount",
    "SuccessRate",
    "FinalErrorType",
    "ErrorCategory",
    "TotalRepairAttempts",
    "PromptTokens",
    "CompletionTokens",
    "TotalTokens",
]


def log_summary(
    program: str,
    *,
    initial_compile_pass: bool,
    final_compile_pass: bool,
    runtime_pass: bool,
    functional_pass: bool,
    repair_rounds: int,
    total_time: float,
    translation_time: float | None = None,
    validation_time: float | None = None,
    avg_repair_time: float | None = None,
    generated_test_count: int | None = None,
    executed_test_count: int | None = None,
    passed_test_count: int | None = None,
    success_rate: float | None = None,
    final_error_type: str | None = None,
    error_category: str | None = None,
    total_repair_attempts: int | None = None,
) -> None:
    """Append one summary row per program.

    Creates the file with the correct header when it does not yet exist.
    When ``extended_logging`` is enabled in the active config, additional
    columns are appended (old columns remain in their original positions).

    Args:
        program: Program name.
        initial_compile_pass: Whether the very first translation compiled.
        final_compile_pass: Whether the final version compiled.
        runtime_pass: Whether runtime execution succeeded.
        functional_pass: Whether functional equivalence was achieved.
        repair_rounds: Number of repair rounds used.
        total_time: Total elapsed seconds.
        translation_time: Total time spent in LLM translation.
        validation_time: Total time spent in validation.
        avg_repair_time: Average time per repair round.
        generated_test_count: Number of auto-generated tests.
        executed_test_count: Total tests executed.
        passed_test_count: Tests that passed.
        success_rate: Fraction of tests passed.
        final_error_type: Error type after all repairs.
        error_category: Broad error category.
        total_repair_attempts: Cumulative repair attempts.
    """
    summary_path = _cfg().summary_csv
    use_extended = _cfg().extended_logging

    header = list(_SUMMARY_HEADER)
    if use_extended:
        header.extend(_SUMMARY_HEADER_EXTENDED)

    # Build the row identically regardless of write mode
    base_row = [
        program,
        initial_compile_pass,
        final_compile_pass,
        runtime_pass,
        functional_pass,
        repair_rounds,
        round(total_time, 2),
    ]

    if use_extended:
        base_row.extend([
            round(translation_time, 4) if translation_time is not None else "",
            round(validation_time, 4) if validation_time is not None else "",
            round(avg_repair_time, 4) if avg_repair_time is not None else "",
            generated_test_count if generated_test_count is not None else "",
            executed_test_count if executed_test_count is not None else "",
            passed_test_count if passed_test_count is not None else "",
            round(success_rate, 4) if success_rate is not None else "",
            final_error_type or "",
            error_category or "",
            total_repair_attempts if total_repair_attempts is not None else "",
            get_last_call_tokens()[0],
            get_last_call_tokens()[1],
            get_last_call_tokens()[2],
        ])

    if _buffer_mode:
        # Append to in-memory buffer — main thread sorts + flushes later
        global _buffer_summary_header
        with _csv_lock:
            if _buffer_summary_header is None:
                _buffer_summary_header = header
            _summary_buffer.append(base_row)
    else:
        # Write directly to disk (sequential path)
        file_exists = os.path.isfile(summary_path)
        try:
            with _csv_lock, open(summary_path, mode="a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(base_row)
        except OSError:
            pass  # Logging is best-effort


# ============================================================================
# Core processing loop — one C++ file
# ============================================================================

def _dbg(msg: str, program: str, start: float) -> None:
    """Debug log with timestamp, program name, and elapsed time."""
    elapsed = time.time() - start
    print(f"  DBG [{elapsed:8.2f}s] {program:30s} | {msg}", flush=True)


def process_program(program_path: str) -> None:
    """Run the full translation → validate → repair pipeline for one program.

    Pipeline:
        1. Load C++ source and discover test cases.
        2. Optionally generate additional test cases.
        3. Translate via LLM.
        4. Loop (compile check → runtime check → differential validation
           → smart repair) up to ``max_repair_rounds`` times.
        5. Log per-round and summary results.

    When only a single ``.in`` file is present and the config uses
    ``validation_strategy="single"``, behaviour is **identical** to the
    v1.0 pipeline.

    Args:
        program_path: Absolute path to a ``.cpp`` source file.
    """
    cfg = _cfg()
    program_name = os.path.basename(program_path)
    reset_program_tokens()
    mon = get_monitor()
    start_time = time.time()  # Moved up early for debug logging
    _dbg("START process_program", program_name, start_time)

    print(f"\n{'=' * 60}")
    print(f"Processing: {program_name}")
    print(f"{'=' * 60}")

    # ---- top-level guard: one sample must never stop the experiment ----
    stage: str = "startup"
    try:

            # -- load C++ source -------------------------------------------------
        with open(program_path, encoding="utf-8") as fh:
            cpp_code = fh.read()

        # -- load test cases -----------------------------------------------------
        test_cases = load_test_cases(program_path)

        if len(test_cases) > 1:
            print(f"  Test cases loaded: {len(test_cases)}")
            for i, (inp, exp) in enumerate(test_cases):
                exp_note = " (with .out)" if exp is not None else ""
                print(f"    [{i}] {os.path.splitext(program_name)[0]}_{i + 1}.in{exp_note}")
        else:
            inp = test_cases[0][0] if test_cases else ""
            if inp:
                print(f"  Test input loaded: {os.path.splitext(program_name)[0]}.in")
            else:
                print("  (no .in file — using empty input)")

        # -- optionally generate additional test cases ---------------------------
        generated_cases: list[TestCase] = []
        if cfg.auto_test and cfg.generated_cases > 0:
            print(f"\n  Generating {cfg.generated_cases} test cases "
                  f"(strategies: {', '.join(cfg.test_strategies)}) …")
            gen = TestGenerator()
            llm_cb = call_gpt if "llm" in cfg.test_strategies else None
            generated_cases = gen.generate(
                program_path,
                count=cfg.generated_cases,
                strategies=cfg.test_strategies,
                llm_callback=llm_cb,
            )
            print(f"  Generated {len(generated_cases)} test cases.")

        # Combine manual + generated for differential testing
        all_test_cases = test_cases + generated_cases
        total_test_count = len(all_test_cases)

        # -- execution cache -----------------------------------------------------
        cache = ExecutionCache() if cfg.enable_caching else None

        # -- initial translation -------------------------------------------------

        # Try translation cache first (avoids unnecessary LLM API calls)
        stage = "translation"
        _dbg("before cache lookup", program_name, start_time)
        cached = _cache_lookup(program_name)
        if cached is not None:
            python_code = cached
            translation_time = 0.0
            _trans_result = CallResult(text=cached, prompt_tokens=0,
                                        completion_tokens=0, total_tokens=0,
                                        elapsed_seconds=0.0, retry_count=0)
            _dbg("cache HIT", program_name, start_time)
        else:
            _dbg("cache MISS - before LLM call", program_name, start_time)
            print("  Generating initial translation …")
            t0 = time.time()
            python_code, _trans_result = translate_cpp(cpp_code)
            translation_time = time.time() - t0
            _dbg(f"LLM returned {len(python_code)} chars", program_name, start_time)
            mon.record_translation(translation_time)
            mon.record_api_wait(_trans_result.elapsed_seconds)
            _cache_save(program_name, python_code)
            _dbg("cache SAVED", program_name, start_time)

        initial_compile_pass: bool = False
        repair_history_entries: list[str] = []
        last_error_type: str = "None"
        last_error_category: str = "unknown"
        compile_ok: bool = False  # safe default when max_repair_rounds == 0
        prev_mismatch_sig: str = ""
        prev_compile_err: str = ""
        prev_python_code: str = ""

        # ---- no-repair (single-pass) path ------------------------------------
        # When repair is disabled, validate exactly once and return.
        # Never enter the repair loop, never print "Maximum repair rounds".
        _dbg(f"max_repair_rounds={cfg.max_repair_rounds}", program_name, start_time)

        if cfg.max_repair_rounds == 0:
            elapsed = time.time() - start_time
            _dbg("NO-REPAIR path entered", program_name, start_time)

            # Compile check
            stage = "compile"
            _dbg("before check_compile", program_name, start_time)
            t0_c = time.time()
            compile_ok, compile_error = check_compile(python_code)
            compile_time = time.time() - t0_c
            _dbg(f"check_compile returned ok={compile_ok}", program_name, start_time)
            mon.record_compile(compile_time)

            if not compile_ok:
                err_type = _classify_error(compile_error)
                log_result(
                    program_name, 0,
                    compile_pass=False, runtime_pass=False, functional_pass=False,
                    error_type=err_type, elapsed_time=elapsed, repair_count=0,
                    translation_time=translation_time, compile_time=compile_time,
                    generated_test_count=len(generated_cases),
                    executed_test_count=total_test_count, passed_test_count=0,
                    success_rate=0.0,
                    failure_reason=compile_error[:200] if compile_error else "",
                    final_error_type=err_type, total_repair_attempts=0,
                )
                log_summary(
                    program_name,
                    initial_compile_pass=False, final_compile_pass=False,
                    runtime_pass=False, functional_pass=False,
                    repair_rounds=0, total_time=elapsed,
                    translation_time=translation_time,
                    generated_test_count=len(generated_cases),
                    executed_test_count=total_test_count, passed_test_count=0,
                    success_rate=0.0, final_error_type=err_type,
                    error_category=_classify_error_category(err_type),
                    total_repair_attempts=0,
                )
                save_code(program_name, python_code)
                return

            initial_compile_pass = True

            # Runtime check
            stage = "runtime"
            _dbg("before run_python", program_name, start_time)
            t0_r = time.time()
            first_input = all_test_cases[0][0] if all_test_cases else ""
            runtime_ok, runtime_output = run_python(python_code, first_input)
            runtime_time = time.time() - t0_r
            _dbg(f"run_python returned ok={runtime_ok}", program_name, start_time)
            mon.record_runtime(runtime_time)

            if not runtime_ok:
                err_type = _classify_error(runtime_output)
                log_result(
                    program_name, 0,
                    compile_pass=True, runtime_pass=False, functional_pass=False,
                    error_type=err_type, elapsed_time=elapsed, repair_count=0,
                    translation_time=translation_time, compile_time=compile_time,
                    runtime_time=runtime_time,
                    generated_test_count=len(generated_cases),
                    executed_test_count=total_test_count, passed_test_count=0,
                    success_rate=0.0,
                    failure_reason=runtime_output[:200],
                    final_error_type=err_type, total_repair_attempts=0,
                )
                log_summary(
                    program_name,
                    initial_compile_pass=True, final_compile_pass=True,
                    runtime_pass=False, functional_pass=False,
                    repair_rounds=0, total_time=elapsed,
                    translation_time=translation_time,
                    generated_test_count=len(generated_cases),
                    executed_test_count=total_test_count, passed_test_count=0,
                    success_rate=0.0, final_error_type=err_type,
                    error_category=_classify_error_category(err_type),
                    total_repair_attempts=0,
                )
                save_code(program_name, python_code)
                return

            # Functional validation
            stage = "functional"
            _dbg("before functional validate", program_name, start_time)
            t0_v = time.time()
            mismatch = ""  # safe default for both validation paths
            if cfg.validation_strategy == "differential" and total_test_count > 0:
                _dbg("before validate_translation_multi", program_name, start_time)
                report = validate_translation_multi(
                    program_path, python_code, all_test_cases, cache=cache,
                )
                func_ok = report.all_passed
                validation_time = time.time() - t0_v
                _dbg(f"validate_multi done ok={func_ok}", program_name, start_time)
                passed_count = report.passed
                sr = passed_count / max(total_test_count, 1)
                if not func_ok:
                    mismatch = report.mismatch_report(max_cases=5)
            else:
                single_input = all_test_cases[0][0] if all_test_cases else ""
                func_ok, _, _, mismatch = validate_translation(
                    program_path, python_code, single_input,
                )
                validation_time = time.time() - t0_v
                passed_count = 1 if func_ok else 0
                sr = passed_count / max(total_test_count, 1)
                report = None
                if not func_ok and mismatch.startswith("C++_EXECUTION_FAILED:"):
                    # Cannot validate — C++ baseline failed
                    log_result(
                        program_name, 0,
                        compile_pass=True, runtime_pass=True, functional_pass=False,
                        error_type="CppExecutionFailed", elapsed_time=elapsed,
                        repair_count=0,
                        translation_time=translation_time,
                        compile_time=compile_time, runtime_time=runtime_time,
                        validation_time=validation_time,
                        generated_test_count=len(generated_cases),
                        executed_test_count=total_test_count, passed_test_count=0,
                        success_rate=0.0, failure_reason=mismatch[:200],
                        final_error_type="CppExecutionFailed",
                        total_repair_attempts=0,
                    )
                    log_summary(
                        program_name,
                        initial_compile_pass=True, final_compile_pass=True,
                        runtime_pass=True, functional_pass=False,
                        repair_rounds=0, total_time=elapsed,
                        translation_time=translation_time,
                        validation_time=validation_time,
                        generated_test_count=len(generated_cases),
                        executed_test_count=total_test_count,
                        passed_test_count=0, success_rate=0.0,
                        final_error_type="CppExecutionFailed",
                        error_category="unknown", total_repair_attempts=0,
                    )
                    save_code(program_name, python_code)
                    return
            mon.record_functional(validation_time)
            _dbg("before final CSV logging", program_name, start_time)

            print(f"\n  ✅ Round 0: ALL CHECKS PASSED" if func_ok else
                  f"\n  ❌ Round 0: Functional MISMATCH")
            if func_ok:
                print(f"     Compile ✓ | Runtime ✓ | Functional ✓ "
                      f"({passed_count}/{total_test_count} tests)")
            save_code(program_name, python_code)

            log_result(
                program_name, 0,
                compile_pass=True, runtime_pass=True,
                functional_pass=func_ok,
                error_type="None" if func_ok else "FunctionalMismatch",
                elapsed_time=elapsed, repair_count=0,
                translation_time=translation_time, compile_time=compile_time,
                runtime_time=runtime_time, validation_time=validation_time,
                generated_test_count=len(generated_cases),
                executed_test_count=total_test_count,
                passed_test_count=passed_count, success_rate=sr,
                failure_reason="" if func_ok else mismatch[:200],
                final_error_type="None" if func_ok else "FunctionalMismatch",
                total_repair_attempts=0,
            )

            log_summary(
                program_name,
                initial_compile_pass=True, final_compile_pass=True,
                runtime_pass=True, functional_pass=func_ok,
                repair_rounds=0, total_time=elapsed,
                translation_time=translation_time,
                validation_time=validation_time,
                generated_test_count=len(generated_cases),
                executed_test_count=total_test_count,
                passed_test_count=passed_count, success_rate=sr,
                final_error_type="None" if func_ok else "FunctionalMismatch",
                error_category="none" if func_ok else "semantic",
                total_repair_attempts=0,
            )
            return
        # ---- end no-repair path -----------------------------------------------

        stage = "repair"
        _dbg("ENTERING repair loop", program_name, start_time)
        # -- repair loop ---------------------------------------------------------
        for round_num in range(cfg.max_repair_rounds):
            elapsed = time.time() - start_time
            _dbg(f"repair round {round_num}", program_name, start_time)

            # ---- 1. Compile check ----------------------------------------------
            _dbg(f"repair r{round_num} before check_compile", program_name, start_time)
            t0_c = time.time()
            compile_ok, compile_error = check_compile(python_code)
            compile_time = time.time() - t0_c
            _dbg(f"repair r{round_num} after check_compile ok={compile_ok} ({compile_time:.2f}s)", program_name, start_time)
            mon.record_compile(compile_time)

            if round_num == 0:
                initial_compile_pass = compile_ok

            if not compile_ok:
                # ---- convergence: same compile error twice → stop ----
                if (compile_error and prev_compile_err
                        and compile_error[:200] == prev_compile_err[:200]):
                    print(f"     Repair converged — same compile error twice. "
                          f"Stopping early.", flush=True)
                    break
                prev_compile_err = compile_error or ""

                err_type = _classify_error(compile_error)
                err_cat = _classify_error_category(err_type)
                last_error_type = err_type
                last_error_category = err_cat

                if cfg.verbose_output:
                    print(f"\n  ❌ Round {round_num}: Compilation FAILED")
                    print(f"     Error: {err_type}")
                    print(f"     Category: {err_cat}")

                log_result(
                    program_name,
                    round_num,
                    compile_pass=False,
                    runtime_pass=False,
                    functional_pass=False,
                    error_type=err_type,
                    elapsed_time=elapsed,
                    repair_count=round_num,
                    translation_time=translation_time if round_num == 0 else None,
                    compile_time=compile_time,
                    generated_test_count=len(generated_cases),
                    executed_test_count=total_test_count,
                    passed_test_count=0,
                    success_rate=0.0,
                    failure_reason=compile_error[:200] if compile_error else "",
                    final_error_type=err_type,
                    total_repair_attempts=round_num,
                )

                # Repair
                history = _format_repair_history(repair_history_entries)
                python_code, _repair_result = fix_code(
                    cpp_code,
                    python_code,
                    compile_error=compile_error,
                    error_category=err_cat,
                    repair_history=history,
                    previous_repair_count=round_num,
                )
                mon.record_repair(_repair_result.elapsed_seconds)
                mon.record_api_wait(_repair_result.elapsed_seconds)

                repair_history_entries.append(
                    f"Round {round_num}: Compile error ({err_type}) — {err_cat}"
                )
                if cache:
                    cache.invalidate_python(python_code)
                continue

            _dbg(f"repair r{round_num} before run_python", program_name, start_time)
            t0_r = time.time()
            first_input = all_test_cases[0][0] if all_test_cases else ""
            runtime_ok, runtime_output = run_python(python_code, first_input)
            runtime_time = time.time() - t0_r
            _dbg(f"repair r{round_num} after run_python ok={runtime_ok} ({runtime_time:.2f}s)", program_name, start_time)
            mon.record_runtime(runtime_time)

            if not runtime_ok:
                err_type = _classify_error(runtime_output)
                err_cat = _classify_error_category(err_type)
                last_error_type = err_type
                last_error_category = err_cat

                if cfg.verbose_output:
                    print(f"\n  ❌ Round {round_num}: Runtime FAILED")
                    print(f"     {runtime_output[:120]}")

                log_result(
                    program_name,
                    round_num,
                    compile_pass=True,
                    runtime_pass=False,
                    functional_pass=False,
                    error_type=err_type,
                    elapsed_time=elapsed,
                    repair_count=round_num,
                    translation_time=translation_time if round_num == 0 else None,
                    compile_time=compile_time,
                    runtime_time=runtime_time,
                    generated_test_count=len(generated_cases),
                    executed_test_count=total_test_count,
                    passed_test_count=0,
                    success_rate=0.0,
                    failure_reason=runtime_output[:200],
                    final_error_type=err_type,
                    total_repair_attempts=round_num,
                )

                history = _format_repair_history(repair_history_entries)
                python_code, _repair_result = fix_code(
                    cpp_code,
                    python_code,
                    runtime_error=runtime_output,
                    error_category=err_cat,
                    repair_history=history,
                    previous_repair_count=round_num,
                )
                mon.record_repair(_repair_result.elapsed_seconds)
                mon.record_api_wait(_repair_result.elapsed_seconds)
                repair_history_entries.append(
                    f"Round {round_num}: Runtime error ({err_type}) — {err_cat}"
                )
                if cache:
                    cache.invalidate_python(python_code)
                continue

            # ---- 3. Differential functional validation -------------------------
            _dbg(f"repair r{round_num} before validate_translation_multi", program_name, start_time)
            t0_v = time.time()

            if cfg.validation_strategy == "differential" and total_test_count > 0:
                report = validate_translation_multi(
                    program_path,
                    python_code,
                    all_test_cases,
                    cache=cache,
                )
                func_ok = report.all_passed
                validation_time = time.time() - t0_v
                _dbg(f"repair r{round_num} after validate_multi ok={func_ok} ({validation_time:.2f}s)", program_name, start_time)
                mon.record_functional(validation_time)

                if not func_ok:
                    mismatch = report.mismatch_report(max_cases=5)
                    compact = report.compact_failure_summary()
                    failed_tests = compact
                else:
                    mismatch = ""
                    compact = ""
                    failed_tests = ""
            else:
                # Legacy single-test validation path
                single_input = all_test_cases[0][0] if all_test_cases else ""
                func_ok, cpp_out, py_out, mismatch = validate_translation(
                    program_path, python_code, single_input,
                )
                validation_time = time.time() - t0_v
                report = None
                compact = ""
                failed_tests = (
                    f"input={single_input.strip()!r} | "
                    f"expected={cpp_out!r} | "
                    f"got={py_out!r}"
                    if not func_ok and not mismatch.startswith("C++_EXECUTION_FAILED:")
                    else ""
                )
                # Use local variables for the single-test path (avoids overwriting
                # outer total_test_count and generated_cases)
                _single_total = 1
                _single_passed = 1 if func_ok else 0

            if not func_ok:
                # Distinguish C++ oracle failure from real mismatch
                if mismatch.startswith("C++_EXECUTION_FAILED:"):
                    if cfg.verbose_output:
                        print(f"\n  ⚠️  Round {round_num}: Cannot validate — "
                              f"C++ execution failed")
                        print(f"     {mismatch}")

                    log_result(
                        program_name,
                        round_num,
                        compile_pass=True,
                        runtime_pass=True,
                        functional_pass=False,
                        error_type="CppExecutionFailed",
                        elapsed_time=elapsed,
                        repair_count=round_num,
                        translation_time=translation_time if round_num == 0 else None,
                        compile_time=compile_time,
                        runtime_time=runtime_time,
                        validation_time=validation_time,
                        generated_test_count=len(generated_cases),
                        executed_test_count=total_test_count,
                        passed_test_count=0,
                        success_rate=0.0,
                        failure_reason=mismatch[:200],
                        final_error_type="CppExecutionFailed",
                        total_repair_attempts=round_num,
                    )
                    # Cannot repair — the problem is in the C++ baseline.
                    save_code(program_name, python_code)

                    log_summary(
                        program_name,
                        initial_compile_pass=initial_compile_pass,
                        final_compile_pass=True,
                        runtime_pass=True,
                        functional_pass=False,
                        repair_rounds=round_num,
                        total_time=elapsed,
                        translation_time=translation_time,
                        validation_time=validation_time,
                        generated_test_count=len(generated_cases),
                        executed_test_count=total_test_count,
                        passed_test_count=0,
                        success_rate=0.0,
                        final_error_type="CppExecutionFailed",
                        error_category="unknown",
                        total_repair_attempts=round_num,
                    )
                    return

                # Genuine output mismatch — repair
                _dbg(f"repair r{round_num} functional mismatch - entering repair", program_name, start_time)

                # ---- convergence: same mismatch / identical code → stop ----
                current_sig = mismatch[:300] if mismatch else (compact[:300] if compact else "")
                if current_sig and current_sig == prev_mismatch_sig:
                    print("     Repair converged — same mismatch twice. Stopping early.", flush=True)
                    break
                prev_mismatch_sig = current_sig
                if python_code == prev_python_code and prev_python_code:
                    print("     Repair converged — identical output. Stopping early.", flush=True)
                    break
                prev_python_code = python_code

                err_type = "FunctionalMismatch"
                err_cat = "semantic"
                last_error_type = err_type
                last_error_category = err_cat

                if cfg.verbose_output:
                    print(f"\n  ❌ Round {round_num}: Functional MISMATCH")
                    if report is not None:
                        print(f"     {report.summary}")

                passed_count = report.passed if report is not None else _single_passed
                sr = passed_count / max(total_test_count, 1)

                _dbg(f"repair r{round_num} before log_result (functional mismatch)", program_name, start_time)
                log_result(
                    program_name,
                    round_num,
                    compile_pass=True,
                    runtime_pass=True,
                    functional_pass=False,
                    error_type=err_type,
                    elapsed_time=elapsed,
                    repair_count=round_num,
                    translation_time=translation_time if round_num == 0 else None,
                    compile_time=compile_time,
                    runtime_time=runtime_time,
                    validation_time=validation_time,
                    generated_test_count=len(generated_cases),
                    executed_test_count=total_test_count,
                    passed_test_count=passed_count,
                    success_rate=sr,
                    failure_reason=mismatch[:200] if mismatch else (compact[:200] if compact else ""),
                    final_error_type=err_type,
                    total_repair_attempts=round_num,
                )
                _dbg(f"repair r{round_num} after log_result, before fix_code", program_name, start_time)

                # Enhanced repair with smart failure compression
                history = _format_repair_history(repair_history_entries)
                _dbg(f"repair r{round_num} calling fix_code (LLM repair API)", program_name, start_time)
                python_code, _repair_result = fix_code(
                    cpp_code,
                    python_code,
                    functional_mismatch=mismatch if mismatch else compact,
                    error_category=err_cat,
                    failed_test_inputs=(
                        _format_failed_inputs(report) if report else failed_tests
                    ),
                    expected_outputs=(
                        _format_expected_outputs(report) if report else ""
                    ),
                    actual_outputs=(
                        _format_actual_outputs(report) if report else ""
                    ),
                    repair_history=history,
                    previous_repair_count=round_num,
                )
                _dbg(f"repair r{round_num} fix_code returned {len(python_code)} chars", program_name, start_time)
                mon.record_repair(_repair_result.elapsed_seconds)
                mon.record_api_wait(_repair_result.elapsed_seconds)
                _dbg(f"repair r{round_num} recorded repair timing, appending history", program_name, start_time)
                repair_history_entries.append(
                    f"Round {round_num}: Functional mismatch — {err_cat} "
                    f"({passed_count}/{total_test_count} passed)"
                )
                _dbg(f"repair r{round_num} before cache invalidate + continue", program_name, start_time)
                if cache:
                    cache.invalidate_python(python_code)
                _dbg(f"repair r{round_num} continue to next round", program_name, start_time)
                continue

            # ---- all checks passed ---------------------------------------------
            passed_count = report.passed if report is not None else _single_passed
            sr = passed_count / max(total_test_count, 1)

            print(f"\n  ✅ Round {round_num}: ALL CHECKS PASSED")
            print(f"     Compile ✓ | Runtime ✓ | Functional ✓ "
                  f"({passed_count}/{total_test_count} tests)")
            save_code(program_name, python_code)

            log_result(
                program_name,
                round_num,
                compile_pass=True,
                runtime_pass=True,
                functional_pass=True,
                error_type="None",
                elapsed_time=elapsed,
                repair_count=round_num,
                translation_time=translation_time if round_num == 0 else None,
                compile_time=compile_time,
                runtime_time=runtime_time,
                validation_time=validation_time,
                generated_test_count=len(generated_cases),
                executed_test_count=total_test_count,
                passed_test_count=passed_count,
                success_rate=sr,
                failure_reason="",
                final_error_type="None",
                total_repair_attempts=round_num,
            )

            log_summary(
                program_name,
                initial_compile_pass=initial_compile_pass,
                final_compile_pass=True,
                runtime_pass=True,
                functional_pass=True,
                repair_rounds=round_num,
                total_time=elapsed,
                translation_time=translation_time,
                validation_time=validation_time,
                generated_test_count=len(generated_cases),
                executed_test_count=total_test_count,
                passed_test_count=passed_count,
                success_rate=sr,
                final_error_type="None",
                error_category="none",
                total_repair_attempts=round_num,
            )
            return

        _dbg("EXHAUSTED repair rounds", program_name, start_time)
        # -- exhausted all repair rounds -----------------------------------------
        elapsed = time.time() - start_time

        # Determine final state (the loop always runs at least once, so
        # *compile_ok* from the last iteration is still in scope)
        final_compile_ok = compile_ok
        final_runtime_ok = False
        final_func_ok = False
        final_passed = 0

        if final_compile_ok:
            first_input = all_test_cases[0][0] if all_test_cases else ""
            final_runtime_ok, _ = run_python(python_code, first_input)
            if final_runtime_ok:
                if cfg.validation_strategy == "differential" and total_test_count > 0:
                    final_report = validate_translation_multi(
                        program_path, python_code, all_test_cases, cache=cache,
                    )
                    final_func_ok = final_report.all_passed
                    final_passed = final_report.passed
                else:
                    func_ok, _, _, mismatch = validate_translation(
                        program_path, python_code,
                        all_test_cases[0][0] if all_test_cases else "",
                    )
                    if func_ok:
                        final_func_ok = True
                        final_passed = 1
                    elif mismatch.startswith("C++_EXECUTION_FAILED:"):
                        final_func_ok = False

        final_sr = final_passed / max(total_test_count, 1)

        print(f"\n  ❌ Maximum repair rounds ({cfg.max_repair_rounds}) reached.")
        save_code(program_name, python_code)

        log_result(
            program_name,
            cfg.max_repair_rounds,
            compile_pass=final_compile_ok,
            runtime_pass=final_runtime_ok,
            functional_pass=final_func_ok,
            error_type="MaxRoundsExceeded",
            elapsed_time=elapsed,
            repair_count=cfg.max_repair_rounds,
            translation_time=translation_time,
            executed_test_count=total_test_count,
            passed_test_count=final_passed,
            success_rate=final_sr,
            failure_reason=f"Exhausted {cfg.max_repair_rounds} repair rounds",
            final_error_type=last_error_type,
            total_repair_attempts=cfg.max_repair_rounds,
        )

        log_summary(
            program_name,
            initial_compile_pass=initial_compile_pass,
            final_compile_pass=final_compile_ok,
            runtime_pass=final_runtime_ok,
            functional_pass=final_func_ok,
            repair_rounds=cfg.max_repair_rounds,
            total_time=elapsed,
            translation_time=translation_time,
            generated_test_count=len(generated_cases),
            executed_test_count=total_test_count,
            passed_test_count=final_passed,
            success_rate=final_sr,
            final_error_type=last_error_type,
            error_category=last_error_category,
            total_repair_attempts=cfg.max_repair_rounds,
        )


    except Exception as _exc:
        """Any unhandled failure in this sample — log it and continue."""
        elapsed = time.time() - start_time
        exc_name = type(_exc).__name__
        reason = f"{exc_name}: {str(_exc)[:200]}"

        # ---- record persistent infrastructure failures in the skip list ----
        if stage in ("translation", "repair"):
            is_timeout = "timeout" in exc_name.lower()
            skip_reason = (
                f"{'Translation' if stage == 'translation' else 'Repair'}"
                f"{'Timeout' if is_timeout else 'APIError'}"
            )
            try:
                _record_skip(program_name, skip_reason, stage=stage)
            except Exception:
                pass

        print(f"\n  ⚠ Sample failed.")
        print(f"     Stage:   {stage}")
        print(f"     Reason:  {exc_name}")
        print(f"     Continuing experiment...")
        # Best-effort failure log — may fail if variables are unbound
        try:
            log_result(
                program_name, 0,
                compile_pass=False, runtime_pass=False, functional_pass=False,
                error_type=f"FrameworkError_{exc_name}",
                elapsed_time=elapsed, repair_count=0,
                failure_reason=reason,
                final_error_type=exc_name,
                total_repair_attempts=0,
            )
        except Exception:
            pass  # even the failure log failed — nothing we can do
        try:
            log_summary(
                program_name,
                initial_compile_pass=False, final_compile_pass=False,
                runtime_pass=False, functional_pass=False,
                repair_rounds=0, total_time=elapsed,
                final_error_type=exc_name,
                error_category="framework",
                total_repair_attempts=0,
            )
        except Exception:
            pass
        # Try to save whatever code exists
        try:
            save_code(program_name, "")
        except Exception:
            pass


# ============================================================================
# Internal helpers for repair context building
# ============================================================================

def _format_repair_history(entries: list[str]) -> str:
    """Format repair history entries for inclusion in a prompt.

    Only the last 2 entries are included to keep repair prompts compact.

    Args:
        entries: Chronological list of repair attempt descriptions.

    Returns:
        Newline-separated history string, or ``""`` if empty.
    """
    if not entries:
        return ""
    return "\n".join(f"  - {e}" for e in entries[-2:])


def _format_failed_inputs(report: DiffReport | None) -> str:
    """Extract representative failed test inputs from a diff report.

    Uses :func:`differential_testing._select_representative_failures`
    under the hood (via ``compact_failure_summary``), so when many
    tests fail only the most diagnostic ones are included.

    Args:
        report: A :class:`DiffReport` or ``None``.

    Returns:
        Formatted string of failed inputs, or ``""``.
    """
    if report is None:
        return ""
    failures = [r for r in report.test_results if not r.passed]
    parts: list[str] = []
    for r in failures[:8]:
        parts.append(f"Test {r.test_index}: {r.test_input.strip()!r}")
    return "\n".join(parts)


def _format_expected_outputs(report: DiffReport | None) -> str:
    """Extract expected outputs from a diff report.

    Only the first 8 failures are included to keep prompts concise.

    Args:
        report: A :class:`DiffReport` or ``None``.

    Returns:
        Formatted string of expected outputs, or ``""``.
    """
    if report is None:
        return ""
    failures = [r for r in report.test_results if not r.passed]
    parts: list[str] = []
    for r in failures[:8]:
        parts.append(f"Test {r.test_index}: {r.expected_output!r}")
    return "\n".join(parts)


def _format_actual_outputs(report: DiffReport | None) -> str:
    """Extract actual outputs from a diff report.

    Only the first 8 failures are included to keep prompts concise.

    Args:
        report: A :class:`DiffReport` or ``None``.

    Returns:
        Formatted string of actual outputs, or ``""``.
    """
    if report is None:
        return ""
    failures = [r for r in report.test_results if not r.passed]
    parts: list[str] = []
    for r in failures[:8]:
        parts.append(f"Test {r.test_index}: {r.actual_output!r}")
    return "\n".join(parts)


def _try_remove(path: str) -> None:
    """Remove a file, silently ignoring errors.

    Args:
        path: Filesystem path to remove.
    """
    try:
        os.remove(path)
    except OSError:
        pass


# ============================================================================
# Entry point (original — preserved)
# ============================================================================

def main() -> None:
    """Discover all ``.cpp`` files under ``samples/`` and process each one.

    Runs the full translation → validation → repair pipeline for every
    ``.cpp`` file found.  Handles individual program failures gracefully
    so one broken program does not abort the entire batch.
    """
    cfg = _cfg()

    try:
        cpp_files = sorted([
            os.path.join(cfg.samples_dir, f)
            for f in os.listdir(cfg.samples_dir)
            if f.endswith(".cpp")
        ])
    except FileNotFoundError:
        print(f"\n  ⚠️  Samples directory not found: {cfg.samples_dir}")
        print("     Create it and add C++ source files to begin.\n")
        return
    except OSError as exc:
        print(f"\n  ⚠️  Cannot read samples directory: {exc}\n")
        return

    print(f"\n{'=' * 60}")
    print(f"Research-Grade C++ → Python Translation Framework")
    print(f"{'=' * 60}")
    print(f"  Samples dir    : {cfg.samples_dir}")
    print(f"  Translated dir : {cfg.translated_dir}")
    print(f"  Max rounds     : {cfg.max_repair_rounds}")
    print(f"  Timeout        : {cfg.execution_timeout}s")
    print(f"  Auto-test      : {cfg.auto_test}")
    print(f"  Validation     : {cfg.validation_strategy}")
    print(f"  Caching        : {cfg.enable_caching}")
    print(f"  C++ files found: {len(cpp_files)}")

    if not cpp_files:
        print(f"\n  ⚠️  No .cpp files found in {cfg.samples_dir}")
        print("     Add C++ source files to the samples/ directory.\n")
        return

    try:
        for cpp_file in cpp_files:
            try:
                process_program(cpp_file)
            except Exception as exc:
                print(f"\n  💥 Unexpected error processing "
                      f"{os.path.basename(cpp_file)}: {exc}")
                # Continue with the next program — don't abort the whole batch.
    finally:
        # Best-effort cleanup of cached C++ binaries
        _cleanup_cpp_cache()

    print(f"\n{'=' * 60}")
    print("Experiment completed.")
    print(f"  Detailed log : {cfg.csv_file}")
    print(f"  Summary log  : {cfg.summary_csv}")
    print(f"  Translations : {cfg.translated_dir}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
