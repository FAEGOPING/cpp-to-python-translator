"""
test_generator.py — Automatic Test Case Generator
==================================================

Generates test inputs for C++ programs using multiple strategies:

* **random** — uniformly sampled values within reasonable ranges.
* **boundary** — values at or near typical integer limits.
* **edge** — corner-case inputs (empty, very large, special chars).
* **heuristic** — parse the C++ source for ``cin >>`` patterns and
  produce type-appropriate inputs.  Detects scalar reads, 1D arrays
  (``cin >> a[i]`` inside loops), and 2D arrays (nested loops).
* **llm** — ask the LLM to suggest interesting test inputs based on
  the C++ source.

All generated test cases are **temporary** — they participate in
validation but are never written to disk, so manual ``.in`` / ``.out``
files are never overwritten.

Usage::

    from test_generator import TestGenerator

    gen = TestGenerator(seed=42)
    cases = gen.generate("samples/example.cpp", count=50,
                         strategies=("random", "boundary", "edge", "heuristic"))
    # cases is List[Tuple[str, Optional[str]]] — (input, expected_output_or_None)
"""

from __future__ import annotations

import random
import re
from typing import List, Optional, Tuple

# Type alias for a single test case: (input_string, expected_output_or_None)
TestCase = Tuple[str, Optional[str]]

# Maximum number of array elements to generate in heuristic mode
_MAX_ARRAY_SIZE = 10
_MAX_INPUT_SIZE = 4096  # safety cap on generated input size


class TestGenerator:
    """Generates temporary test cases for a given C++ program.

    Args:
        seed: Random seed for reproducibility.  ``None`` means
            non-deterministic.
    """

    def __init__(self, seed: int | None = 42) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        cpp_file: str,
        count: int = 50,
        strategies: Tuple[str, ...] = (
            "random",
            "boundary",
            "edge",
            "heuristic",
        ),
        llm_callback: Optional[callable] = None,
    ) -> List[TestCase]:
        """Generate *count* test cases using the given *strategies*.

        Args:
            cpp_file: Path to the C++ source file.
            count: Target number of test cases.  Actual count may be
                slightly lower if some strategies produce fewer cases.
            strategies: Enabled strategies — any subset of
                ``("random", "boundary", "edge", "heuristic", "llm")``.
            llm_callback: Callable ``(prompt: str) -> str``
                for LLM-assisted generation.  Required when ``"llm"`` is
                in *strategies*.

        Returns:
            List of ``(input_text, expected_output_or_None)`` pairs.
            Expected output is always ``None`` (generated tests don't
            have reference outputs — the C++ program itself serves as
            the oracle at validation time).
        """
        try:
            with open(cpp_file, encoding="utf-8") as fh:
                cpp_source = fh.read()
        except (OSError, UnicodeDecodeError):
            return []

        cases: list[TestCase] = []

        # Analyse source once for heuristics
        structure = _analyse_input_structure(cpp_source)

        # Distribute cases across strategies
        active = [s for s in strategies if s != "llm"]
        if not active and "llm" not in strategies:
            return cases

        per_strategy = max(1, count // max(len(active), 1))
        remainder = count - per_strategy * len(active)

        for i, strategy in enumerate(active):
            budget = per_strategy + (1 if i < remainder else 0)
            if strategy == "random":
                cases.extend(self._generate_random(budget, structure))
            elif strategy == "boundary":
                cases.extend(self._generate_boundary(budget, structure))
            elif strategy == "edge":
                cases.extend(self._generate_edge(budget, structure))
            elif strategy == "heuristic":
                cases.extend(self._generate_heuristic(budget, structure))

        # LLM-assisted generation (if enabled and callback provided)
        if "llm" in strategies and llm_callback is not None:
            llm_budget = max(1, count // 4)
            cases.extend(self._generate_llm(llm_budget, cpp_source, llm_callback))

        # Filter out invalid cases and trim
        cases = _filter_valid(cases, _MAX_INPUT_SIZE)
        self._rng.shuffle(cases)
        return cases[:count]

    # ------------------------------------------------------------------
    # Strategy: random
    # ------------------------------------------------------------------

    def _generate_random(
        self, count: int, structure: _InputStructure
    ) -> List[TestCase]:
        """Generate uniformly random integer inputs respecting structure."""
        cases: list[TestCase] = []
        for _ in range(count):
            inp = structure.generate_input(self._rng, "random")
            cases.append((inp, None))
        return cases

    # ------------------------------------------------------------------
    # Strategy: boundary
    # ------------------------------------------------------------------

    # Typical boundary values for integer inputs
    _BOUNDARY_VALUES: tuple[int, ...] = (
        0, 1, -1,
        2**31 - 1,          # INT32_MAX
        -(2**31),           # INT32_MIN
        2**63 - 1,          # INT64_MAX
        -(2**63),           # INT64_MIN
        255, -255,
        65535, -65535,
        1000000, -1000000,
    )

    def _generate_boundary(
        self, count: int, structure: _InputStructure
    ) -> List[TestCase]:
        """Generate inputs at or near integer boundary values."""
        cases: list[TestCase] = []
        values = list(self._BOUNDARY_VALUES)
        self._rng.shuffle(values)

        for i in range(count):
            if structure.has_structure:
                inp = structure.generate_input(
                    self._rng, "boundary",
                    boundary_override=values[i % len(values)],
                )
            else:
                if structure.scalar_vars:
                    parts = [str(values[i % len(values)]) for _ in structure.scalar_vars]
                    inp = "\n".join(parts) + "\n"
                else:
                    inp = f"{values[i % len(values)]}\n"
            cases.append((inp, None))

        return cases[:count]

    # ------------------------------------------------------------------
    # Strategy: edge
    # ------------------------------------------------------------------

    _EDGE_INPUTS: tuple[str, ...] = (
        "",             # empty input
        "\n",           # just newline
        "0\n",          # zero
        "-1\n",         # negative one
        "999999999\n",  # large positive
        "-999999999\n", # large negative
        "0 0\n",        # multiple zeros
        "1 2 3 4 5\n",  # multiple values
        " \n",          # whitespace only
    )

    def _generate_edge(
        self, count: int, structure: _InputStructure
    ) -> List[TestCase]:
        """Generate edge-case / corner-case inputs."""
        cases: list[TestCase] = []
        edge_list = list(self._EDGE_INPUTS)

        for i in range(count):
            # When we have structure, use it for most cases but
            # occasionally inject raw edge inputs
            if structure.has_structure and i % 3 != 0:
                inp = structure.generate_input(self._rng, "edge")
            else:
                inp = edge_list[i % len(edge_list)]
            cases.append((inp, None))

        return cases[:count]

    # ------------------------------------------------------------------
    # Strategy: heuristic
    # ------------------------------------------------------------------

    def _generate_heuristic(
        self,
        count: int,
        structure: _InputStructure,
    ) -> List[TestCase]:
        """Generate type-aware inputs using the analysed input structure.

        When a rich structure is detected (arrays, nested loops, etc.)
        the generated inputs respect that structure.  Falls back to
        simple type-aware generation when no structure is found.
        """
        cases: list[TestCase] = []

        if structure.has_structure:
            # Use the structured generator
            for _ in range(count):
                inp = structure.generate_input(self._rng, "heuristic")
                cases.append((inp, None))
            return cases

        # Fallback: simple scalar type-aware generation
        if not structure.scalar_vars:
            return self._generate_random(count, structure)

        for _ in range(count):
            parts: list[str] = []
            for var in structure.scalar_vars:
                parts.append(_generate_scalar_value(var.var_type, self._rng))
            cases.append(("\n".join(parts) + "\n", None))

        return cases

    # ------------------------------------------------------------------
    # Strategy: LLM-assisted
    # ------------------------------------------------------------------

    def _generate_llm(
        self,
        count: int,
        cpp_source: str,
        llm_callback: callable,
    ) -> List[TestCase]:
        """Ask the LLM to generate interesting test inputs.

        Args:
            count: Target number of test cases.
            cpp_source: Full C++ source code.
            llm_callback: Callable that sends a prompt to the LLM and
                returns the response text.

        Returns:
            List of ``(input, None)`` pairs parsed from the LLM response.
        """
        cases: list[TestCase] = []

        prompt = f"""\
Analyse the following C++ program and suggest {count} test inputs that would
thoroughly exercise its logic (including edge cases and boundary conditions).

For each test case, provide ONLY the raw input that would be fed to stdin,
with one test per line.  Use "---" as a separator between test cases.

IMPORTANT: Return ONLY the test inputs — no explanations, no markdown.

C++ Program:
{cpp_source}"""

        try:
            response = llm_callback(prompt)
            raw_inputs = response.split("---")
            for raw in raw_inputs:
                inp = raw.strip()
                if inp:
                    cases.append(
                        (inp + "\n" if not inp.endswith("\n") else inp, None)
                    )
        except Exception:
            # LLM generation is best-effort; silently fall back
            pass

        return cases[:count]


# ======================================================================
# Input structure analysis (v2.1 — detects arrays, nested loops, etc.)
# ======================================================================

class _InputVar:
    """Represents a single variable read from stdin."""

    def __init__(self, var_type: str, name: str) -> None:
        self.var_type: str = var_type
        self.name: str = name


class _ArrayRead:
    """Represents an array being populated via ``cin >> arr[i]`` in a loop."""

    def __init__(
        self,
        name: str,
        var_type: str,
        count_var: str | None = None,
        count_const: int | None = None,
    ) -> None:
        self.name: str = name
        self.var_type: str = var_type
        self.count_var: str | None = count_var   # e.g. "n"
        self.count_const: int | None = count_const  # e.g. 100


class _InputStructure:
    """Describes the input structure of a C++ program.

    Captures scalar variables, single arrays, and 2D (nested) array
    reads so that the test generator can produce structurally valid
    inputs.
    """

    def __init__(self) -> None:
        self.scalar_vars: list[_InputVar] = []
        self.arrays: list[_ArrayRead] = []
        self.nested_arrays: list[tuple[_ArrayRead, _ArrayRead]] = []
        """(outer_array, inner_array) pairs for 2D array patterns."""

    @property
    def has_structure(self) -> bool:
        """True when arrays or nested arrays were detected."""
        return bool(self.arrays or self.nested_arrays)

    def generate_input(
        self,
        rng: random.Random,
        mode: str = "heuristic",
        boundary_override: int | None = None,
    ) -> str:
        """Generate a structurally valid input string.

        Args:
            rng: Random instance for reproducibility.
            mode: ``"random"``, ``"boundary"``, ``"edge"``, or
                ``"heuristic"``.
            boundary_override: When set, use this value for all
                boundary-mode scalar reads.

        Returns:
            Newline-delimited input string.
        """
        lines: list[str] = []

        # First: emit scalar variables in declaration order
        for var in self.scalar_vars:
            val = _generate_scalar_value(var.var_type, rng, mode, boundary_override)
            lines.append(val)

        # Second: emit array elements
        for arr in self.arrays:
            count = _resolve_count(arr, rng)
            for _ in range(count):
                val = _generate_scalar_value(arr.var_type, rng, mode, boundary_override)
                lines.append(val)

        # Third: emit 2D array elements (row-major)
        for outer, inner in self.nested_arrays:
            outer_count = _resolve_count(outer, rng)
            inner_count = _resolve_count(inner, rng)
            for _ in range(outer_count):
                for _ in range(inner_count):
                    val = _generate_scalar_value(
                        inner.var_type, rng, mode, boundary_override,
                    )
                    lines.append(val)

        if not lines:
            # No structure — generate a single random value
            val = _generate_scalar_value("int", rng, mode, boundary_override)
            lines.append(val)

        return "\n".join(lines) + "\n"


# ======================================================================
# C++ source analysis
# ======================================================================

def _analyse_input_structure(cpp_source: str) -> _InputStructure:
    """Analyse a C++ source file and return its input structure.

    Detects:
    * Scalar variable declarations and ``cin >> x`` reads.
    * 1D array reads: ``for(...) cin >> a[i]``.
    * 2D array reads: nested loops with ``cin >> a[i][j]``.

    Args:
        cpp_source: Full C++ source code as a string.

    Returns:
        An :class:`_InputStructure` describing the detected patterns.
    """
    structure = _InputStructure()
    source = _strip_comments(cpp_source)

    # -- Step 1: find type declarations ------------------------------------
    declared: dict[str, str] = {}
    decl_pattern = re.compile(
        r'\b(int|long\s+long|long|short|float|double|char|string)'
        r'\s+([a-zA-Z_]\w*(?:\s*\[[^\]]*\])?\s*(?:,|;|=|$))'
    )
    for m in decl_pattern.finditer(source):
        vtype = m.group(1).replace(" ", "")
        rest = m.group(2)
        # Extract the name (strip array brackets, commas, etc.)
        name_match = re.match(r'([a-zA-Z_]\w*)', rest.strip())
        if name_match:
            name = name_match.group(1)
            declared[name] = vtype

    # -- Step 2: find scalar cin >> reads -----------------------------------
    cin_scalar_pattern = re.compile(
        r'cin\s*>>\s*([a-zA-Z_]\w*(?:\s*>>\s*[a-zA-Z_]\w*)*)\s*;'
    )
    for m in cin_scalar_pattern.finditer(source):
        names = [n.strip() for n in m.group(1).split(">>")]
        for name in names:
            vtype = declared.get(name, "int")
            # Don't add duplicates
            if name not in {v.name for v in structure.scalar_vars}:
                structure.scalar_vars.append(_InputVar(vtype, name))

    # -- Step 3: find 1D array reads ---------------------------------------
    # Pattern: for(...) { ... cin >> arr[i]; }
    # Or:     while(...) { ... cin >> arr[i]; }
    array_read_pattern = re.compile(
        r'cin\s*>>\s*([a-zA-Z_]\w*)\s*\['
    )
    # Find arrays with index patterns like a[i], a[index], etc.
    for m in re.finditer(
        r'cin\s*>>\s*([a-zA-Z_]\w*)\s*\[\s*[a-zA-Z_]\w*\s*\]',
        source,
    ):
        arr_name = m.group(1)
        if arr_name not in {a.name for a in structure.arrays}:
            vtype = declared.get(arr_name, "int")
            # Try to find the loop bound that controls this cin
            count_var = _find_loop_bound_for(source, m.start())
            structure.arrays.append(
                _ArrayRead(arr_name, vtype, count_var=count_var, count_const=None)
            )

    # -- Step 4: find 2D array reads (nested loops) ------------------------
    # Pattern: for(i) for(j) cin >> a[i][j]
    nested_reads = list(re.finditer(
        r'cin\s*>>\s*([a-zA-Z_]\w*)\s*\[\s*[a-zA-Z_]\w*\s*\]\s*\[\s*[a-zA-Z_]\w*\s*\]',
        source,
    ))
    if nested_reads:
        for m in nested_reads:
            arr_name = m.group(1)
            vtype = declared.get(arr_name, "int")
            if arr_name not in {a.name for a in structure.arrays}:
                # Treat as a 2D array — use reasonable defaults
                structure.nested_arrays.append((
                    _ArrayRead(f"{arr_name}_rows", vtype, count_const=3),
                    _ArrayRead(f"{arr_name}_cols", vtype, count_const=3),
                ))

    return structure


def _find_loop_bound_for(source: str, position: int) -> str | None:
    """Heuristically find the loop bound variable for a cin at *position*.

    Searches backwards from *position* for a ``for(... <bound; ...)``
    pattern and returns the bound variable name.

    Args:
        source: C++ source code.
        position: Character offset of the ``cin`` statement.

    Returns:
        Bound variable name (e.g. ``"n"``) or ``None``.
    """
    before = source[:position]
    # Find the innermost for-loop that precedes this position
    for_pattern = re.compile(
        r'for\s*\([^;]*;\s*([a-zA-Z_]\w*)\s*<\s*([a-zA-Z_]\w*)\s*;[^)]*\)',
    )
    matches = list(for_pattern.finditer(before))
    if matches:
        last = matches[-1]
        return last.group(2)  # the bound variable
    return None


def _resolve_count(arr: _ArrayRead, rng: random.Random) -> int:
    """Resolve the number of elements for an array read.

    Args:
        arr: The array read descriptor.
        rng: Random instance.

    Returns:
        Integer count (1 – _MAX_ARRAY_SIZE).
    """
    if arr.count_const is not None:
        return min(arr.count_const, _MAX_ARRAY_SIZE)
    # Variable bound — use a small random size
    return rng.randint(1, _MAX_ARRAY_SIZE)


def _generate_scalar_value(
    var_type: str,
    rng: random.Random,
    mode: str = "heuristic",
    boundary_override: int | None = None,
) -> str:
    """Generate a single scalar value for a variable of *var_type*.

    Args:
        var_type: C++ type name (``"int"``, ``"float"``, ``"string"``, etc.).
        rng: Random instance.
        mode: Generation mode — ``"random"``, ``"boundary"``, ``"edge"``,
            or ``"heuristic"``.
        boundary_override: When set, use this integer value directly.

    Returns:
        String representation of the value.
    """
    int_types = {"int", "long", "longlong", "short"}
    float_types = {"float", "double"}

    if var_type in int_types:
        if boundary_override is not None:
            return str(boundary_override)
        if mode == "boundary":
            return str(rng.choice([
                0, 1, -1, 2**31 - 1, -(2**31), 255, -255, 1000000, -1000000,
            ]))
        if mode == "edge":
            return rng.choice(["0", "-1", "999999999", "-999999999"])
        if mode == "random":
            return str(rng.randint(-10000, 10000))
        # heuristic: mix of strategies
        choice = rng.choice(["boundary", "zero", "small", "random"])
        if choice == "boundary":
            return str(rng.choice([0, 1, -1, 2**31 - 1, -(2**31), 255, 1000000]))
        if choice == "zero":
            return "0"
        if choice == "small":
            return str(rng.randint(-100, 100))
        return str(rng.randint(-10000, 10000))

    if var_type in float_types:
        return f"{rng.uniform(-1000.0, 1000.0):.6f}"

    if var_type == "string":
        return f"s{rng.randint(0, 1000)}"

    if var_type == "char":
        return rng.choice("abcdefghijklmnopqrstuvwxyz")

    # Fallback
    return str(rng.randint(-1000, 1000))


def _strip_comments(source: str) -> str:
    """Remove C++ single-line and block comments.

    Args:
        source: C++ source code.

    Returns:
        Source with comments replaced by spaces (line count preserved).
    """
    # Remove block comments
    source = re.sub(r'/\*.*?\*/', ' ', source, flags=re.DOTALL)
    # Remove line comments
    source = re.sub(r'//[^\n]*', ' ', source)
    return source


def _filter_valid(cases: list[TestCase], max_size: int) -> list[TestCase]:
    """Filter out invalid or oversized test cases.

    Args:
        cases: Raw generated test cases.
        max_size: Maximum allowed input size in characters.

    Returns:
        Filtered list of cases.
    """
    valid: list[TestCase] = []
    for inp, exp in cases:
        # Must be a string and not excessively large
        if not isinstance(inp, str):
            continue
        if len(inp) > max_size:
            continue
        valid.append((inp, exp))
    return valid
