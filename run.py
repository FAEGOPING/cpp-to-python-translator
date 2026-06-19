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

Workflow:
  C++ Source → LLM Translation → Compile Check → Runtime Check
  → Functional Validation → Self-Repair → Repeat (max N rounds)

Author: Research-Grade Translation Framework
Python: 3.10+
"""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
import time
from typing import Any

from gpt_api import call_gpt

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT: str = "/Users/tianjabez/Desktop/project"

MAX_REPAIR_ROUNDS: int = 5
"""Maximum number of repair iterations per program."""

EXECUTION_TIMEOUT: int = 10
"""Timeout in seconds for subprocess execution (C++ and Python)."""

SAMPLES_DIR: str = os.path.join(PROJECT_ROOT, "samples")
TRANSLATED_DIR: str = os.path.join(PROJECT_ROOT, "translated")
CSV_FILE: str = os.path.join(PROJECT_ROOT, "experiment_results.csv")
SUMMARY_CSV: str = os.path.join(PROJECT_ROOT, "summary_results.csv")

os.makedirs(TRANSLATED_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ============================================================================
# Utility: test input loading
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
        with open(in_path, encoding="utf-8") as fh:
            return fh.read()
    return ""


# ============================================================================
# Utility: error classification
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
# C++ compilation & execution
# ============================================================================

def run_cpp(cpp_file: str, test_input: str) -> tuple[bool, str]:
    """Compile and execute a C++ program, capturing its stdout.

    The C++ source is compiled with ``g++ -std=c++17`` into a temporary
    executable, then executed with *test_input* piped to stdin.

    Args:
        cpp_file: Path to the ``.cpp`` source file.
        test_input: String to pipe to the program's stdin.

    Returns:
        ``(success, output_or_error)`` — on success *output_or_error*
        is the program's stdout; on failure it is a human-readable
        error message.
    """
    exe_fd, exe_path = tempfile.mkstemp(suffix=".out")
    os.close(exe_fd)

    try:
        # -- compile ---------------------------------------------------------
        compile_result = subprocess.run(
            ["g++", "-std=c++17", cpp_file, "-o", exe_path],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
        )

        if compile_result.returncode != 0:
            return False, (
                f"C++ compilation failed:\n{compile_result.stderr.strip()}"
            )

        # -- run -------------------------------------------------------------
        run_result = subprocess.run(
            [exe_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
        )

        # Merge stderr into output when present (non-zero exit usually
        # means a runtime crash).
        if run_result.returncode != 0:
            err = run_result.stderr.strip() or "(no stderr)"
            return False, f"C++ runtime error (exit {run_result.returncode}):\n{err}"

        return True, run_result.stdout

    except subprocess.TimeoutExpired:
        return False, f"Timeout: C++ execution exceeded {EXECUTION_TIMEOUT}s"
    except FileNotFoundError:
        return False, (
            "g++ not found — please install g++ to enable "
            "C++ compilation and functional validation."
        )
    finally:
        try:
            os.remove(exe_path)
        except OSError:
            pass


# ============================================================================
# Translation prompt
# ============================================================================

def translate_cpp(cpp_code: str) -> str:
    """Translate C++ source code to Python via the LLM.

    Args:
        cpp_code: Full C++ source code as a string.

    Returns:
        LLM-generated Python source code.
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

    return call_gpt(prompt)


# ============================================================================
# Repair prompt
# ============================================================================

def fix_code(
    cpp_code: str,
    python_code: str,
    *,
    compile_error: str | None = None,
    runtime_error: str | None = None,
    functional_mismatch: str | None = None,
) -> str:
    """Repair translated Python code, providing full error context to the LLM.

    Args:
        cpp_code: Original C++ source (ground truth).
        python_code: Current (faulty) Python translation.
        compile_error: Compilation error message, if any.
        runtime_error: Runtime error message, if any.
        functional_mismatch: Output difference description, if any.

    Returns:
        LLM-generated repaired Python source code.
    """
    # Assemble error sections ------------------------------------------------
    error_sections: list[str] = []

    if compile_error:
        error_sections.append(
            f"**Compilation Error:**\n```\n{compile_error}\n```"
        )
    if runtime_error:
        error_sections.append(
            f"**Runtime Error:**\n```\n{runtime_error}\n```"
        )
    if functional_mismatch:
        error_sections.append(
            f"**Functional Mismatch:**\n```\n{functional_mismatch}\n```"
        )

    error_block = "\n\n".join(error_sections) if error_sections else (
        "(No specific error details available — please review the code "
        "for semantic discrepancies.)"
    )

    prompt = f"""\
You are an expert software engineer specialised in C++ → Python translation.

The Python code below was translated from C++, but it contains errors or
produces incorrect output.

============================================================================
Original C++ Code (ground truth):
============================================================================
{cpp_code}

============================================================================
Current Python Code (to repair):
============================================================================
{python_code}

============================================================================
Identified Issues:
============================================================================
{error_block}

============================================================================
Requirements:
============================================================================
1. **Preserve semantic equivalence** — the repaired Python program MUST
   produce IDENTICAL output to the original C++ program for the same input.
2. Fix **all** identified issues.
3. Return **only** the corrected Python code.
4. Do **not** include explanations, markdown fences, or comments."""

    return call_gpt(prompt)


# ============================================================================
# Compilation check
# ============================================================================

def check_compile(code: str) -> tuple[bool, str | None]:
    """Check whether Python *code* passes ``py_compile``.

    Args:
        code: Python source code string.

    Returns:
        ``(passed, error_or_none)`` — *error_or_none* is the compiler
        stderr on failure, ``None`` on success.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as fh:
        fh.write(code)
        file_path = fh.name

    try:
        result = subprocess.run(
            ["python3", "-m", "py_compile", file_path],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return False, result.stderr.strip()

        return True, None
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


# ============================================================================
# Runtime execution
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
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as fh:
        fh.write(code)
        file_path = fh.name

    try:
        result = subprocess.run(
            ["python3", file_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
        )

        if result.returncode != 0:
            raw = result.stderr.strip()
            error_type = _classify_error(raw)
            return False, f"[{error_type}] {raw}"

        return True, result.stdout

    except subprocess.TimeoutExpired:
        return False, (
            f"[Timeout] Execution exceeded {EXECUTION_TIMEOUT} seconds"
        )
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


# ============================================================================
# Functional equivalence validation
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
    # --- run original C++ ---------------------------------------------------
    cpp_ok, cpp_result = run_cpp(cpp_file, test_input)
    if not cpp_ok:
        return (
            False,
            "",
            "",
            f"C++_EXECUTION_FAILED: {cpp_result}",
        )

    # --- run translated Python ----------------------------------------------
    py_ok, py_result = run_python(python_code, test_input)
    if not py_ok:
        return (
            False,
            cpp_result.strip(),
            "",
            f"Python execution failed: {py_result}",
        )

    # --- compare outputs ----------------------------------------------------
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
# File output
# ============================================================================

def save_code(program_name: str, code: str) -> None:
    """Write translated Python *code* to the ``translated/`` directory.

    Args:
        program_name: Original filename (e.g. ``"example.cpp"``).
        code: Python source code to persist.
    """
    out_name = program_name.replace(".cpp", ".py")
    out_path = os.path.join(TRANSLATED_DIR, out_name)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(code)


# ============================================================================
# Per-round experiment log
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
) -> None:
    """Append one row to the detailed experiment CSV.

    Creates the file with the correct header when it does not yet exist.
    """
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)

        if not file_exists:
            writer.writerow(_EXPERIMENT_HEADER)

        writer.writerow([
            program,
            round_num,
            compile_pass,
            runtime_pass,
            functional_pass,
            error_type,
            round(elapsed_time, 2),
            repair_count,
        ])


# ============================================================================
# Summary log (one row per program)
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


def log_summary(
    program: str,
    *,
    initial_compile_pass: bool,
    final_compile_pass: bool,
    runtime_pass: bool,
    functional_pass: bool,
    repair_rounds: int,
    total_time: float,
) -> None:
    """Append one summary row per program.

    Creates the file with the correct header when it does not yet exist.
    """
    file_exists = os.path.isfile(SUMMARY_CSV)

    with open(SUMMARY_CSV, mode="a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)

        if not file_exists:
            writer.writerow(_SUMMARY_HEADER)

        writer.writerow([
            program,
            initial_compile_pass,
            final_compile_pass,
            runtime_pass,
            functional_pass,
            repair_rounds,
            round(total_time, 2),
        ])


# ============================================================================
# Core processing loop — one C++ file
# ============================================================================

def process_program(program_path: str) -> None:
    """Run the full translation → validate → repair pipeline for one program.

    Args:
        program_path: Absolute path to a ``.cpp`` source file.
    """
    program_name = os.path.basename(program_path)

    print(f"\n{'=' * 60}")
    print(f"Processing: {program_name}")
    print(f"{'=' * 60}")

    # -- load C++ source & test input ----------------------------------------
    with open(program_path, encoding="utf-8") as fh:
        cpp_code = fh.read()

    test_input = load_test_input(program_path)
    if test_input:
        print(f"  Test input loaded: {os.path.splitext(program_name)[0]}.in")
    else:
        print("  (no .in file — using empty input)")

    # -- initial translation -------------------------------------------------
    start_time = time.time()
    print("\n  Generating initial translation …")
    python_code = translate_cpp(cpp_code)

    initial_compile_pass: bool = False
    final_result_printed: bool = False

    # -- repair loop ---------------------------------------------------------
    for round_num in range(MAX_REPAIR_ROUNDS):
        elapsed = time.time() - start_time

        # 1. Compile check ---------------------------------------------------
        compile_ok, compile_error = check_compile(python_code)

        if round_num == 0:
            initial_compile_pass = compile_ok

        if not compile_ok:
            print(f"\n  ❌ Round {round_num}: Compilation FAILED")
            print(f"     Error: {_classify_error(compile_error)}")
            log_result(
                program_name,
                round_num,
                compile_pass=False,
                runtime_pass=False,
                functional_pass=False,
                error_type=_classify_error(compile_error),
                elapsed_time=elapsed,
                repair_count=round_num,
            )
            python_code = fix_code(
                cpp_code,
                python_code,
                compile_error=compile_error,
            )
            continue

        # 2. Runtime check ---------------------------------------------------
        runtime_ok, runtime_output = run_python(python_code, test_input)

        if not runtime_ok:
            print(f"\n  ❌ Round {round_num}: Runtime FAILED")
            print(f"     {runtime_output[:120]}")
            log_result(
                program_name,
                round_num,
                compile_pass=True,
                runtime_pass=False,
                functional_pass=False,
                error_type=_classify_error(runtime_output),
                elapsed_time=elapsed,
                repair_count=round_num,
            )
            python_code = fix_code(
                cpp_code,
                python_code,
                runtime_error=runtime_output,
            )
            continue

        # 3. Functional equivalence check ------------------------------------
        func_ok, cpp_out, py_out, mismatch = validate_translation(
            program_path, python_code, test_input,
        )

        if not func_ok:
            # Distinguish "C++ itself failed" from "real mismatch"
            if mismatch.startswith("C++_EXECUTION_FAILED:"):
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
                )
                final_result_printed = True
                return

            # Genuine output mismatch — repair
            print(f"\n  ❌ Round {round_num}: Functional MISMATCH")
            print(f"     Expected: {cpp_out}")
            print(f"     Got:      {py_out}")
            log_result(
                program_name,
                round_num,
                compile_pass=True,
                runtime_pass=True,
                functional_pass=False,
                error_type="FunctionalMismatch",
                elapsed_time=elapsed,
                repair_count=round_num,
            )
            python_code = fix_code(
                cpp_code,
                python_code,
                functional_mismatch=mismatch,
            )
            continue

        # -- all three checks passed -----------------------------------------
        print(f"\n  ✅ Round {round_num}: ALL CHECKS PASSED")
        print(f"     Compile ✓ | Runtime ✓ | Functional ✓")
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
        )
        log_summary(
            program_name,
            initial_compile_pass=initial_compile_pass,
            final_compile_pass=True,
            runtime_pass=True,
            functional_pass=True,
            repair_rounds=round_num,
            total_time=elapsed,
        )
        final_result_printed = True
        return

    # -- exhausted all repair rounds -----------------------------------------
    elapsed = time.time() - start_time

    # Determine final state for summary
    final_compile_ok, _ = check_compile(python_code)
    final_runtime_ok = False
    final_func_ok = False

    if final_compile_ok:
        final_runtime_ok, _ = run_python(python_code, test_input)
        if final_runtime_ok:
            func_ok, _, _, mismatch = validate_translation(
                program_path, python_code, test_input,
            )
            # Only count as functional-pass if it was a genuine match
            # (not a C++ execution failure masking the result)
            if func_ok:
                final_func_ok = True
            elif mismatch.startswith("C++_EXECUTION_FAILED:"):
                final_func_ok = False

    print(f"\n  ❌ Maximum repair rounds ({MAX_REPAIR_ROUNDS}) reached.")
    save_code(program_name, python_code)
    log_result(
        program_name,
        MAX_REPAIR_ROUNDS,
        compile_pass=final_compile_ok,
        runtime_pass=final_runtime_ok,
        functional_pass=final_func_ok,
        error_type="MaxRoundsExceeded",
        elapsed_time=elapsed,
        repair_count=MAX_REPAIR_ROUNDS,
    )
    log_summary(
        program_name,
        initial_compile_pass=initial_compile_pass,
        final_compile_pass=final_compile_ok,
        runtime_pass=final_runtime_ok,
        functional_pass=final_func_ok,
        repair_rounds=MAX_REPAIR_ROUNDS,
        total_time=elapsed,
    )
    final_result_printed = True


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """Discover all ``.cpp`` files under ``samples/`` and process each one."""
    cpp_files = sorted([
        os.path.join(SAMPLES_DIR, f)
        for f in os.listdir(SAMPLES_DIR)
        if f.endswith(".cpp")
    ])

    print(f"\n{'=' * 60}")
    print(f"Research-Grade C++ → Python Translation Framework")
    print(f"{'=' * 60}")
    print(f"  Samples dir    : {SAMPLES_DIR}")
    print(f"  Translated dir : {TRANSLATED_DIR}")
    print(f"  Max rounds     : {MAX_REPAIR_ROUNDS}")
    print(f"  Timeout        : {EXECUTION_TIMEOUT}s")
    print(f"  C++ files found: {len(cpp_files)}")

    if not cpp_files:
        print(f"\n  ⚠️  No .cpp files found in {SAMPLES_DIR}")
        print("     Add C++ source files to the samples/ directory.\n")
        return

    for cpp_file in cpp_files:
        try:
            process_program(cpp_file)
        except Exception as exc:
            print(f"\n  💥 Unexpected error processing "
                  f"{os.path.basename(cpp_file)}: {exc}")
            # Continue with the next program — don't abort the whole batch.

    print(f"\n{'=' * 60}")
    print("Experiment completed.")
    print(f"  Detailed log : {CSV_FILE}")
    print(f"  Summary log  : {SUMMARY_CSV}")
    print(f"  Translations : {TRANSLATED_DIR}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
