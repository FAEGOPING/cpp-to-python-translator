"""
figures.py — Publication-Quality Figure Generation (v5.2)
=============================================================

Generates all figures directly from experiment CSV data via
:mod:`statistics`.  Every figure reads from the single source of
truth — no hardcoded paths or duplicate data loading.

V5.2 improvements:
    - Repository parsing uses source_mapping.csv (real repo names)
    - Algorithm category classification from OriginalPath metadata
    - Repair gain capped at total_programs (no impossible values)
    - Error types grouped to reduce "Other" proportion
    - Small error categories merged into "Others" for readability
    - Consistent styling: colour palette, font, legend placement
    - Aggregation verified: counts never exceed totals

Output directory: ``dataset_manager/reports/figures/``

Generated figures (PNG + PDF, 300 dpi):
    - translation_success.png/.pdf       — compile / runtime / functional rates
    - compile_success.png/.pdf           — Python compile pass/fail
    - repair_gain.png/.pdf               — pipeline progression (if repair data)
    - error_category_distribution.png/.pdf — error type breakdown
    - repair_distribution.png/.pdf       — repair rounds histogram (if repair)
    - dataset_distribution.png/.pdf      — programs per dataset source
    - repository_distribution.png/.pdf   — top repositories
    - loc_histogram.png/.pdf             — LOC distribution
    - filter_distribution.png/.pdf       — program classification
    - repository_success.png/.pdf        — success rate per repository
    - category_success.png/.pdf          — success rate per category
    - loc_success.png/.pdf               — success rate by LOC range
    - compile_error_distribution.png/.pdf — compile error categories

Usage::

    python figures.py                    # latest experiment
    python figures.py --exp-dir PATH     # specific experiment
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

_DEPS = os.path.dirname(os.path.abspath(__file__))
if _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from dataset_manager.utils import (
    FIGURES_DIR, REPORTS_DIR, BENCHMARK_DIR, Logger, read_csv, timestamp,
)
from statistics import load_stats, ExperimentStats, _safe_float, _safe_int, _safe_bool

# ============================================================================
# Consistent styling — dissertation-quality
# ============================================================================

# ---- Dissertation colour palette (consistent across all figures) ----
_PALETTE = {
    "compile":    "#4C72B0",   # blue
    "runtime":    "#55A868",   # green
    "functional": "#64B35A",   # light green
    "repair":     "#DD8452",   # orange
    "failure":    "#C44E52",   # red
    "neutral":    "#999999",   # grey
    "dark":       "#2C3E50",   # near-black (annotations)
}
_COLORS_10 = ["#4C72B0", "#55A868", "#64B35A", "#DD8452", "#C44E52",
              "#999999", "#8B8B8B", "#7A7A7A", "#696969", "#585858"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 16, "axes.labelsize": 14,
    "figure.titlesize": 18, "legend.fontsize": 11,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save_fig(name: str) -> str:
    """Save figure as PNG + PDF at 300 dpi."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=300, pad_inches=0.2)
    p = os.path.join(FIGURES_DIR, f"{name}.png")
    plt.close()
    return p


def _warn(logger: Logger, name: str, reason: str) -> Optional[str]:
    logger.warn(f"  {name}: skipped ({reason})")
    return None


# ============================================================================
# Repository & category helpers
# ============================================================================

# Cache for source_mapping data (loaded once per figure generation session)
_SOURCE_MAP: dict[str, dict] = {}
"""Program filename → source mapping row (lazy-loaded from CSV)."""


def _load_source_map() -> dict[str, dict]:
    """Load source_mapping.csv and index by program filename.

    Returns a dict mapping ``program_NNNNNN.cpp`` → dict with keys
    ``OriginalPath``, ``Repository``, ``Category``, ``GitHubURL``.
    """
    global _SOURCE_MAP
    if _SOURCE_MAP:
        return _SOURCE_MAP
    path = os.path.join(REPORTS_DIR, "source_mapping.csv")
    for r in read_csv(path):
        pid = r.get("ProgramID", "")
        fname = pid + ".cpp" if not pid.endswith(".cpp") else pid
        _SOURCE_MAP[fname] = r
    return _SOURCE_MAP


def _repo_for_program(program_name: str) -> str:
    """Return the human-readable repository name for a program."""
    sm = _load_source_map()
    row = sm.get(program_name, {})
    repo = row.get("Repository", "")
    if repo:
        return repo
    # Fallback: try OriginalPath
    path = row.get("OriginalPath", program_name)
    parts = path.split("/")
    if len(parts) >= 2:
        return parts[1]
    return "Unknown"


def _classify_algorithm_category(program_name: str) -> str:
    """Classify a program into an algorithm category from its OriginalPath.

    Uses directory path segments from the source mapping to determine
    the algorithm domain. Falls back to filename-keyword matching when
    the path provides insufficient information.
    """
    sm = _load_source_map()
    row = sm.get(program_name, {})
    path = row.get("OriginalPath", program_name).lower()

    # Category keywords — ordered by specificity (graph before greedy, etc.)
    _CATEGORIES: list[tuple[str, list[str]]] = [
        ("Dynamic Programming",  ["dp", "dynamic_programming", "dynamic programming",
                                   "knapsack", "lcs", "lis", "edit_distance",
                                   "matrix_chain", "coin_change", "fibonacci"]),
        ("Graph",                ["graph", "dijkstra", "bfs", "dfs", "mst",
                                   "topological", "euler", "floyd", "bellman",
                                   "kruskal", "prim", "tarjan", "kosaraju",
                                   "shortest_path", "minimum_spanning", "flow",
                                   "bipartite", "strongly_connected"]),
        ("Tree",                 ["tree algorithm", "tree_algorithm", "binary_tree",
                                   "bst", "trie", "avl", "segment_tree", "fenwick",
                                   "lca", "subtree", "centroid"]),
        ("Greedy",               ["greedy", "huffman", "activity_selection",
                                   "fractional_knapsack"]),
        ("Math",                 ["math", "prime", "gcd", "lcm", "modular",
                                   "factorial", "number_theory", "combinatorics",
                                   "probability", "numerical", "matrix", "polynomial",
                                   "arithmetic", "algebra", "geometry",
                                   "mathematics", "mathematical"]),
        ("Sorting",              ["sort", "merge_sort", "quick_sort", "heap_sort",
                                   "bubble", "insertion_sort", "selection_sort",
                                   "counting_sort", "radix", "sorting"]),
        ("String",               ["string", "kmp", "z_algorithm", "suffix",
                                   "trie", "aho_corasick", "palindrome",
                                   "manacher", "hashing", "rabin_karp",
                                   "substring", "pattern"]),
        ("Data Structures",      ["data_structure", "stack", "queue", "linked_list",
                                   "deque", "priority_queue", "hash", "heap",
                                   "array", "list", "set", "map"]),
        ("Bit Manipulation",     ["bit_manipulation", "bitmask", "xor",
                                   "bitset", "bitwise", "bit_operation"]),
        ("Backtracking",         ["backtrack", "n_queen", "sudoku", "permutation",
                                   "recursion and backtracking"]),
        ("Search",               ["binary_search", "ternary_search", "bisection",
                                   "searching", "search"]),
        ("Simulation",           ["simulation", "simulate"]),
        ("Geometry",             ["geometry", "convex_hull", "closest_pair",
                                   "point", "line_intersection"]),
    ]

    # Score each category by keyword matches in the path
    scores: dict[str, int] = {}
    for cat_name, keywords in _CATEGORIES:
        score = 0
        for kw in keywords:
            if kw in path:
                score += 1
                # Bonus for directory-level match
                if f"/{kw}/" in path or f"/{kw}_" in path or path.endswith(f"/{kw}"):
                    score += 2
        if score > 0:
            scores[cat_name] = score

    if scores:
        return max(scores, key=scores.get)

    return "Other"


# ============================================================================
# Error classification helpers
# ============================================================================

def _normalise_error_type(raw_type: str) -> str:
    """Map a raw error type string to a clean, dissertation-friendly label.

    Collapses variants (e.g. ``"Timeout"`` and ``"Timeout (10s)"``)
    into a single canonical name.
    """
    t = raw_type.strip()
    # Direct mappings
    _MAP: dict[str, str] = {
        "SyntaxError":       "Syntax Error",
        "IndentationError":  "Indentation Error",
        "NameError":         "Name Error",
        "TypeError":         "Type Error",
        "ValueError":        "Value Error",
        "IndexError":        "Index Error",
        "KeyError":          "Key Error",
        "AttributeError":    "Attribute Error",
        "ImportError":       "Import Error",
        "ModuleNotFoundError":"Module Not Found",
        "ZeroDivisionError": "Zero Division",
        "RecursionError":    "Recursion Error",
        "RuntimeError":      "Runtime Error",
        "FileNotFoundError": "File Not Found",
        "OSError":           "OS Error",
        "MemoryError":       "Memory Error",
        "OverflowError":     "Overflow Error",
        "UnboundLocalError": "Unbound Local",
        "StopIteration":     "Stop Iteration",
        "AssertionError":    "Assertion Error",
        "EOFError":          "EOF Error",
        "TabError":          "Tab Error",
        "FunctionalMismatch":"Functional Mismatch",
        "MaxRoundsExceeded": "Max Rounds Exceeded",
    }
    if t in _MAP:
        return _MAP[t]

    # Partial matches for timeout variants
    if "timeout" in t.lower() or "timed out" in t.lower():
        return "Timeout"
    if "connection" in t.lower():
        return "Connection Error"
    if "c++_execution" in t.lower() or "cppexecution" in t.lower():
        return "C++ Execution Failed"

    # Unrecognised → keep as-is but clean up
    return t.replace("_", " ").strip().title()


def _normalise_error_category(raw_cat: str) -> str:
    """Map a raw error category to a clean dissertation label."""
    m = {
        "syntax":     "Syntax",
        "runtime":    "Runtime",
        "semantic":   "Semantic",
        "timeout":    "Timeout",
        "unknown":    "Unknown",
        "none":       "None",
        "":           "Unknown",
    }
    return m.get(raw_cat.strip().lower(), raw_cat.strip().title() or "Unknown")


# ============================================================================
# Experiment-data-driven figures
# ============================================================================

def translation_success(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Bar chart: compile / runtime / functional success rates from experiment CSV."""
    if stats.is_empty:
        return _warn(logger, "translation_success", "no experiment data")

    p = stats.total_programs
    stages = ["Compile", "Runtime", "Functional"]
    values = [stats.compile_pass, stats.runtime_pass, stats.functional_pass]
    rates = [stats.compile_success_rate, stats.runtime_success_rate,
             stats.functional_success_rate]

    _, ax = plt.subplots(figsize=(8, 6))
    colors = [_PALETTE["compile"], _PALETTE["runtime"], _PALETTE["functional"]]
    bars = ax.bar(stages, rates, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Translation Success Across Validation Stages\n({p} Programs)",
                 fontweight="bold", pad=12)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, min(110, max(rates) * 1.20 + 5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))

    for bar, v, rate in zip(bars, values, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v}/{p}\n({rate:.1f}%)",
                ha="center", fontsize=13, fontweight="bold", color=_PALETTE["dark"])

    _save_fig("translation_success")
    logger.info(f"  translation_success: compile={stats.compile_pass} "
                f"runtime={stats.runtime_pass} functional={stats.functional_pass}")
    return os.path.join(FIGURES_DIR, "translation_success.png")


def compile_pass_rate(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Pie chart: Python compile pass vs fail."""
    if stats.is_empty:
        return _warn(logger, "compile_success", "no experiment data")

    passed, failed = stats.compile_pass, stats.compile_fail
    _, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        [passed, failed],
        labels=[f"Pass ({passed})", f"Fail ({failed})"],
        autopct="%1.1f%%",
        colors=[_PALETTE["functional"], _PALETTE["failure"]],
        startangle=90,
        explode=(0, 0.05),
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in texts: t.set_fontsize(12)
    for at in autotexts: at.set_fontsize(14); at.set_fontweight("bold")
    ax.set_title(f"Translation Compile Validation\n({stats.total_programs} Programs)",
                 fontweight="bold", pad=12)
    _save_fig("compile_success")
    logger.info(f"  compile_success: {passed}/{passed + failed} "
                f"({stats.compile_success_rate:.1f}%)")
    return os.path.join(FIGURES_DIR, "compile_success.png")


def repair_gain(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Pipeline progression with vs without repair, or simplified
    validation stages when repair provides no additional gain."""
    if stats.is_empty:
        return _warn(logger, "repair_gain", "no experiment data")
    if not stats.has_repair_data:
        logger.info("  repair_gain: skipped (no repair data)")
        return None

    p = stats.total_programs

    # Count repaired programs properly
    repaired_pass = min(stats.repair_success_gain, p)
    compile_initial = sum(
        1 for r in stats.program_stats
        if _safe_bool(r.get("InitialCompilePass"))
    )
    after_repair = min(compile_initial + repaired_pass, p)
    functional = stats.functional_pass

    # If repair provided NO benefit, simplify to the three basic stages
    if after_repair == compile_initial:
        stages = ["Compile\nSuccess", "Runtime\nSuccess", "Functional\nSuccess"]
        values = [stats.compile_pass, stats.runtime_pass, stats.functional_pass]
        colors = [_PALETTE["compile"], _PALETTE["runtime"], _PALETTE["functional"]]
        fig_title = f"Translation Pipeline Results\n({p} Programs)"
    else:
        stages = ["Initial\nCompile", "After Repair\n(Compile Fixed)", "Final\nFunctional"]
        values = [compile_initial, after_repair, functional]
        colors = [_PALETTE["compile"], _PALETTE["repair"], _PALETTE["functional"]]
        fig_title = f"Translation Pipeline with Repair\n({p} Programs)"

    _, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(stages, values, color=colors, width=0.5, edgecolor="white", linewidth=0.5)
    ax.set_title(fig_title, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Programs")
    ax.set_ylim(0, p * 1.15)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + p * 0.02,
                f"{val}  ({val / max(p, 1) * 100:.1f}%)",
                ha="center", fontsize=13, fontweight="bold", color=_PALETTE["dark"])

    _save_fig("repair_gain")
    logger.info(f"  repair_gain: gain={stats.repair_success_gain}")
    return os.path.join(FIGURES_DIR, "repair_gain.png")


def error_distribution(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Horizontal bar: normalised error types from experiment detail CSV.

    Groups rare errors (≤2% or ≤2 occurrences) into "Other" for readability.
    """
    if stats.is_empty or not stats.error_counts:
        logger.info("  error_distribution: skipped (no errors)")
        return None

    # Normalise error type names
    norm_counts: dict[str, int] = defaultdict(int)
    for raw_type, count in stats.error_counts.items():
        clean = _normalise_error_type(raw_type)
        norm_counts[clean] += count

    # Sort descending
    sorted_items = sorted(norm_counts.items(), key=lambda x: -x[1])
    total_err = sum(v for _, v in sorted_items)

    # Merge rare categories (≤2 or ≤2%) into "Other"
    threshold = max(2, int(total_err * 0.02))
    top: list[tuple[str, int]] = []
    other: int = 0
    for label, count in sorted_items:
        if count > threshold:
            top.append((label, count))
        else:
            other += count
    if other > 0:
        top.append(("Other", other))

    labels, values = zip(*reversed(top))

    _, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)))
    norm = plt.Normalize(0, max(values) if max(values) > 0 else 1)
    cmap = plt.cm.Reds
    colors_list = [cmap(0.35 + 0.65 * norm(v)) for v in values]

    bars = ax.barh(labels, values, color=colors_list, edgecolor="white", linewidth=0.5,
                   height=0.7)
    ax.set_title(f"Error Distribution — {total_err} Total Errors", fontweight="bold", pad=12)
    ax.set_xlabel("Occurrences")
    for bar, val in zip(bars, values):
        pct = val / max(total_err, 1) * 100
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val}  ({pct:.1f}%)", va="center", fontsize=12, color=_PALETTE["dark"])

    _save_fig("error_category_distribution")
    logger.info(f"  error_distribution: {len(top)} types, {total_err} total errors")
    return os.path.join(FIGURES_DIR, "error_category_distribution.png")


def compile_error_distribution_figure(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Horizontal bar: error category distribution from experiment CSV.

    Groups rare categories into "Other" and sorts by frequency.
    """
    if stats.is_empty or not stats.error_categories:
        logger.info("  compile_error_distribution: skipped (no error categories)")
        return None

    # Normalise category names
    norm_cats: dict[str, int] = defaultdict(int)
    for raw_cat, count in stats.error_categories.items():
        clean = _normalise_error_category(raw_cat)
        norm_cats[clean] += count

    sorted_items = sorted(norm_cats.items(), key=lambda x: -x[1])
    total_cat = sum(v for _, v in sorted_items)

    # Merge categories with ≤3 occurrences into "Other"
    small: int = 0
    top: list[tuple[str, int]] = []
    for label, count in sorted_items:
        if count > 3:
            top.append((label, count))
        else:
            small += count
    if small > 0:
        top.append(("Other", small))

    labels, values = zip(*reversed(top))

    _, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)))
    cmap = plt.cm.Oranges
    norm = plt.Normalize(0, max(values) if max(values) > 0 else 1)
    colors_list = [cmap(0.35 + 0.65 * norm(v)) for v in values]

    bars = ax.barh(labels, values, color=colors_list, edgecolor="white", linewidth=0.5,
                   height=0.7)
    ax.set_title(f"Error Category Distribution — {total_cat} Total", fontweight="bold", pad=12)
    ax.set_xlabel("Occurrences")
    for bar, val in zip(bars, values):
        pct = val / max(total_cat, 1) * 100
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val}  ({pct:.1f}%)", va="center", fontsize=12, color=_PALETTE["dark"])

    _save_fig("compile_error_distribution")
    logger.info(f"  compile_error_distribution: {len(top)} categories, {total_cat} total")
    return os.path.join(FIGURES_DIR, "compile_error_distribution.png")


def repair_distribution(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Repair outcome categories: no repair needed, successful after N
    repairs, or failed after maximum rounds."""
    if stats.is_empty:
        return _warn(logger, "repair_distribution", "no experiment data")
    if not stats.has_repair_data:
        logger.info("  repair_distribution: skipped (no repair data)")
        return None

    # Categorise programs by repair outcome from program_stats
    no_repair_needed = 0
    success_1 = 0
    success_2 = 0
    success_3 = 0
    failed_max = 0

    for r in stats.program_stats:
        rr = _safe_int(r.get("RepairRounds"))
        func_pass = _safe_bool(r.get("FunctionalPass"))
        if rr == 0:
            no_repair_needed += 1
        elif func_pass:
            if rr == 1:
                success_1 += 1
            elif rr == 2:
                success_2 += 1
            else:
                success_3 += 1
        else:
            failed_max += 1

    categories = [
        ("No Repair\nNeeded",          no_repair_needed, _PALETTE["functional"]),
        ("Successful\nafter 1 Repair",  success_1,        _PALETTE["compile"]),
        ("Successful\nafter 2 Repairs", success_2,        _PALETTE["runtime"]),
        ("Successful\nafter 3 Repairs", success_3,        _PALETTE["repair"]),
        ("Failed after\nMax Repairs",   failed_max,       _PALETTE["failure"]),
    ]
    # Drop empty categories
    categories = [(l, v, c) for l, v, c in categories if v > 0]

    total = sum(v for _, v, _ in categories)
    if total == 0:
        return _warn(logger, "repair_distribution", "no repair data")

    labels, values, colors = zip(*categories)

    _, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.7)))
    bars = ax.barh(list(reversed(labels)), list(reversed(values)),
                   color=list(reversed(colors)), edgecolor="white",
                   linewidth=0.5, height=0.65)
    ax.set_title(f"Repair Outcomes — {stats.total_programs} Programs",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Number of Programs")

    for bar, val in zip(bars, reversed(values)):
        pct = val / max(total, 1) * 100
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val}  ({pct:.1f}%)", va="center", fontsize=12,
                fontweight="bold", color=_PALETTE["dark"])

    _save_fig("repair_distribution")
    logger.info(f"  repair_distribution: {total} programs categorised")
    return os.path.join(FIGURES_DIR, "repair_distribution.png")


# ============================================================================
# Repository & category success figures
# ============================================================================

def repository_success(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Bar chart: success rate per repository (from source_mapping.csv)."""
    if stats.is_empty:
        return _warn(logger, "repository_success", "no experiment data")

    repo_results: dict[str, list[bool]] = defaultdict(list)
    for r in stats.program_stats:
        name = r.get("Program", "")
        repo = _repo_for_program(name)
        repo_results[repo].append(_safe_bool(r.get("FunctionalPass")))

    if not repo_results:
        return _warn(logger, "repository_success", "no repository data")

    rows = []
    for repo, results in repo_results.items():
        rows.append({
            "repository": repo,
            "total": len(results),
            "passed": sum(results),
            "rate": sum(results) / max(len(results), 1) * 100,
        })
    rows.sort(key=lambda r: -r["total"])

    # Only include repos with ≥ 20 programs for meaningful comparison
    MIN_SAMPLES = 20
    valid_rows = [r for r in rows if r["total"] >= MIN_SAMPLES]
    if len(valid_rows) < 3:
        logger.info(f"  repository_success: only {len(valid_rows)} repos with "
                     f"≥{MIN_SAMPLES} samples — skipping figure")
        return None

    labels = [r["repository"] for r in valid_rows]
    rates = [r["rate"] for r in valid_rows]
    totals = [r["total"] for r in valid_rows]

    _, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    colors = [_PALETTE["functional"] if rate >= 50 else _PALETTE["failure"]
              for rate in rates]
    bars = ax.barh(labels, rates, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Translation Success by Repository\n(≥{MIN_SAMPLES} programs, {stats.total_programs} total)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Functional Success Rate (%)")
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))

    for bar, rate, total in zip(bars, rates, totals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%  (n={total})", va="center", fontsize=12,
                color=_PALETTE["dark"])

    _save_fig("repository_success")
    logger.info(f"  repository_success: {len(valid_rows)} repos (≥{MIN_SAMPLES} samples)")
    return os.path.join(FIGURES_DIR, "repository_success.png")


def category_success(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Bar chart: success rate per algorithm category (from OriginalPath metadata)."""
    if stats.is_empty:
        return _warn(logger, "category_success", "no experiment data")

    cat_results: dict[str, list[bool]] = defaultdict(list)
    for r in stats.program_stats:
        name = r.get("Program", "")
        cat = _classify_algorithm_category(name)
        passed = _safe_bool(r.get("FunctionalPass"))
        cat_results[cat].append(passed)

    if not cat_results:
        return _warn(logger, "category_success", "no category data")

    rows = []
    for cat, results in cat_results.items():
        rows.append({
            "category": cat,
            "total": len(results),
            "passed": sum(results),
            "rate": sum(results) / max(len(results), 1) * 100,
        })
    rows.sort(key=lambda r: -r["total"])

    labels = [r["category"] for r in rows]
    rates = [r["rate"] for r in rows]
    totals = [r["total"] for r in rows]

    _, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)))
    colors = [_PALETTE["functional"] if rate >= 50 else _PALETTE["failure"] for rate in rates]
    bars = ax.barh(labels, rates, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Functional Success Rate by Algorithm Category — {stats.total_programs} Programs",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Success Rate (%)")
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))

    for bar, rate, total in zip(bars, rates, totals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%  (n={total})", va="center", fontsize=12,
                color=_PALETTE["dark"])

    _save_fig("category_success")
    logger.info(f"  category_success: {len(rows)} categories")
    return os.path.join(FIGURES_DIR, "category_success.png")


# ============================================================================
# Dataset-driven figures (from dataset_manager/reports/*.csv)
# ============================================================================

def dataset_distribution(logger: Logger) -> Optional[str]:
    """Bar chart: programs per dataset source (from source_mapping.csv Repository)."""
    sm = _load_source_map()
    if not sm:
        return _warn(logger, "dataset_distribution", "source_mapping.csv")

    repo_counts = Counter(row.get("Repository", "Unknown") for row in sm.values())
    top_items = [(r, c) for r, c in repo_counts.most_common(15) if r != "Unknown"]
    if not top_items:
        top_items = repo_counts.most_common(10)
    labels, values = zip(*top_items)

    _, ax = plt.subplots(figsize=(10, 6))
    colors_list = _COLORS_10[:len(labels)]
    bars = ax.bar(labels, values, color=colors_list, edgecolor="white", linewidth=0.5, width=0.6)
    ax.set_title("Dataset Distribution — Programs per Source", fontweight="bold", pad=12)
    ax.set_ylabel("Number of C++ Programs")
    plt.xticks(rotation=25, ha="right", fontsize=12)
    ax.bar_label(bars, fmt="%d", fontsize=12, padding=2)
    _save_fig("dataset_distribution")
    logger.info(f"  dataset_distribution: {len(labels)} sources, {sum(values)} programs")
    return os.path.join(FIGURES_DIR, "dataset_distribution.png")


def repository_distribution(logger: Logger) -> Optional[str]:
    """Horizontal bar: programs per repository (top 15, from source_mapping)."""
    sm = _load_source_map()
    if not sm:
        return _warn(logger, "repository_distribution", "source_mapping.csv")

    counts = Counter(row.get("Repository", "Unknown") for row in sm.values())
    top = [(r, c) for r, c in counts.most_common(15) if r != "Unknown"]
    if not top:
        top = counts.most_common(15)
    labels, values = zip(*reversed(top))

    _, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.4)))
    norm = plt.Normalize(0, len(labels) - 1 if len(labels) > 1 else 1)
    colors_list = [plt.cm.Blues(0.4 + 0.5 * norm(i)) for i in range(len(labels))]
    bars = ax.barh(labels, values, color=colors_list, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_title("Repository Distribution — Top by Program Count", fontweight="bold", pad=12)
    ax.set_xlabel("Number of C++ Programs")
    ax.bar_label(bars, fmt="%d", fontsize=12, padding=2)
    _save_fig("repository_distribution")
    logger.info(f"  repository_distribution: top {len(top)} of {len(counts)} repos")
    return os.path.join(FIGURES_DIR, "repository_distribution.png")


def loc_histogram(logger: Logger) -> Optional[str]:
    """Histogram: lines of code distribution."""
    rows = read_csv(os.path.join(REPORTS_DIR, "metadata.csv"))
    if not rows:
        return _warn(logger, "loc_histogram", "metadata.csv")

    locs = [_safe_int(r.get("CodeLines", "0")) for r in rows
            if _safe_int(r.get("CodeLines", "0")) > 0]
    if not locs:
        return _warn(logger, "loc_histogram", "no LOC data")
    # Trim outliers (99th percentile)
    cutoff = sorted(locs)[int(len(locs) * 0.99)]
    locs = [l for l in locs if l <= cutoff]

    _, ax = plt.subplots(figsize=(10, 6))
    ax.hist(locs, bins=50, color=_PALETTE["compile"], edgecolor="white",
            alpha=0.85, linewidth=0.3)
    ax.set_title("Lines of Code (LOC) Distribution", fontweight="bold", pad=12)
    ax.set_xlabel("Lines of Code")
    ax.set_ylabel("Frequency")
    if locs:
        mean_loc = sum(locs) / len(locs)
        ax.axvline(mean_loc, color=_PALETTE["failure"], linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_loc:.0f} LOC")
        ax.legend(frameon=False)
    _save_fig("loc_histogram")
    logger.info(f"  loc_histogram: {len(locs)} programs")
    return os.path.join(FIGURES_DIR, "loc_histogram.png")


def filter_distribution(logger: Logger) -> Optional[str]:
    """Pie chart: program classification (executable/library/test/dependency)."""
    rows = read_csv(os.path.join(REPORTS_DIR, "filter_report.csv"))
    if not rows:
        return _warn(logger, "filter_distribution", "filter_report.csv")

    data: dict[str, int] = {}
    for r in rows:
        m, v = r.get("Metric", ""), r.get("Value", "0").replace("%", "")
        try:
            if "ExecutablePrograms" in m: data["Executable"] = int(v)
            elif "Library" in m and "Remove" in m: data["Library"] = int(v)
            elif "Test" in m and "Remove" in m: data["Tests"] = int(v)
            elif "Dependency" in m and "Remove" in m: data["Dependency"] = int(v)
        except (ValueError, TypeError): pass
    if not data:
        return _warn(logger, "filter_distribution", "data extraction")

    labels, values = zip(*data.items())
    _, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values, labels=[f"{l} ({v})" for l, v in zip(labels, values)],
        autopct="%1.1f%%",
        colors=[_PALETTE["functional"], _PALETTE["failure"], _PALETTE["repair"], _PALETTE["compile"]],
        startangle=90, explode=tuple(0.03 for _ in labels),
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    for t in texts: t.set_fontsize(11)
    for at in autotexts: at.set_fontsize(12); at.set_fontweight("bold")
    ax.set_title("Program Classification — Filtering Breakdown", fontweight="bold", pad=12)
    _save_fig("filter_distribution")
    logger.info(f"  filter_distribution: {sum(values)} files")
    return os.path.join(FIGURES_DIR, "filter_distribution.png")


def repair_effectiveness(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Grouped bar: compile / runtime / functional success with and without repair.

    Shows the contribution of the repair mechanism by comparing the
    initial (before repair) vs final (after repair) success rates at
    each validation stage.
    """
    if stats.is_empty:
        return _warn(logger, "repair_effectiveness", "no experiment data")
    if not stats.has_repair_data:
        logger.info("  repair_effectiveness: skipped (no repair data)")
        return None

    p = stats.total_programs

    # Before repair: use initial compile pass, then final runtime/functional
    # from programs that never needed repair (RepairRounds=0)
    initial_compile = sum(
        1 for r in stats.program_stats
        if _safe_bool(r.get("InitialCompilePass"))
    )
    # After repair: final compile, runtime, functional
    final_compile = stats.compile_pass
    final_runtime = stats.runtime_pass
    final_functional = stats.functional_pass

    stages = ["Compile", "Runtime", "Functional"]
    before_vals = [initial_compile, stats.runtime_pass, stats.functional_pass]
    after_vals = [final_compile, final_runtime, final_functional]

    x = np.arange(len(stages))
    width = 0.35

    _, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width / 2, before_vals, width,
                   label="Initial (Before Repair)",
                   color=_PALETTE["neutral"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, after_vals, width,
                   label="Final (After Repair)",
                   color=_PALETTE["repair"], edgecolor="white", linewidth=0.5)

    ax.set_title(f"Effect of Repair on Translation Success\n({p} Programs)",
                 fontweight="bold", pad=12)
    ax.set_ylabel("Number of Programs")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylim(0, p * 1.15)
    ax.legend(frameon=True, loc="lower right", fontsize=12)

    for bar, val in zip(bars1, before_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + p * 0.02,
                f"{val}", ha="center", fontsize=13, fontweight="bold",
                color=_PALETTE["neutral"])
    for bar, val in zip(bars2, after_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + p * 0.02,
                f"{val}", ha="center", fontsize=13, fontweight="bold",
                color=_PALETTE["repair"])

    _save_fig("repair_effectiveness")
    logger.info(f"  repair_effectiveness: initial_compile={initial_compile} "
                f"final_compile={final_compile}")
    return os.path.join(FIGURES_DIR, "repair_effectiveness.png")


def loc_success(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Bar chart: functional success rate by LOC range.

    Cross-references experiment data with metadata.csv via
    source_mapping.csv to determine LOC for each program.
    """
    if stats.is_empty:
        return _warn(logger, "loc_success", "no experiment data")

    # Build: program_NNNNNN.cpp → LOC via source_mapping
    # metadata.csv has:    File=algorithms/.../foo.cpp, CodeLines=N
    # source_mapping has:  ProgramID=program_000001, OriginalPath=algorithms/.../foo.cpp
    sm = _load_source_map()

    # Build: OriginalPath → CodeLines
    meta_rows = read_csv(os.path.join(REPORTS_DIR, "metadata.csv"))
    path_to_loc: dict[str, int] = {}
    for r in meta_rows:
        fpath = r.get("File", r.get("Filename", ""))
        loc = _safe_int(r.get("CodeLines", "0"))
        if fpath and loc > 0:
            path_to_loc[fpath] = loc

    # Build: program_NNNNNN.cpp → LOC by cross-referencing
    prog_to_loc: dict[str, int] = {}
    for prog_name, row in sm.items():
        orig = row.get("OriginalPath", "")
        if orig and orig in path_to_loc:
            prog_to_loc[prog_name] = path_to_loc[orig]
        else:
            # Fallback: search for partial match
            for pp, loc in path_to_loc.items():
                if prog_name.replace(".cpp", "") in pp or pp.endswith(prog_name):
                    prog_to_loc[prog_name] = loc
                    break

    # Bin programs by LOC
    _BINS = [
        (0, 50,   "0–50"),
        (51, 100, "51–100"),
        (101, 150, "101–150"),
        (151, 200, "151–200"),
        (201, 99999, "201+"),
    ]
    bin_data: dict[str, list[bool]] = defaultdict(list)

    for r in stats.program_stats:
        name = r.get("Program", "")
        loc = prog_to_loc.get(name, 0)
        passed = _safe_bool(r.get("FunctionalPass"))
        matched = False
        for lo, hi, label in _BINS:
            if lo <= loc <= hi:
                bin_data[label].append(passed)
                matched = True
                break
        if not matched:
            bin_data["No LOC\nData"].append(passed)

    rows = []
    for _, _, label in _BINS:
        results = bin_data.get(label, [])
        if results:
            rows.append((label, results))
    # Also include unmatched if any
    unmatched = bin_data.get("No LOC\nData", [])
    if len(unmatched) > stats.total_programs * 0.30:
        logger.warn(f"  loc_success: {len(unmatched)}/{stats.total_programs} "
                     f"programs could not be matched to LOC data")
    if unmatched:
        rows.append(("No LOC\nData", unmatched))

    if not rows:
        return _warn(logger, "loc_success", "LOC/program mapping")

    labels = [r[0] for r in rows]
    rates = [sum(r[1]) / max(len(r[1]), 1) * 100 for r in rows]
    sizes = [len(r[1]) for r in rows]

    _, ax = plt.subplots(figsize=(10, 6))
    colors_list = [
        _PALETTE["functional"] if rate >= 50 else _PALETTE["failure"]
        for rate in rates
    ]
    bars = ax.bar(labels, rates, color=colors_list, edgecolor="white",
                  linewidth=0.5, width=0.6)
    ax.set_title(f"Translation Success by Program Size (LOC)\n({stats.total_programs} Programs)",
                 fontweight="bold", pad=12)
    ax.set_ylabel("Functional Success Rate (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    for bar, rate, sz in zip(bars, rates, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{rate:.1f}%\n(n={sz})", ha="center", fontsize=13,
                fontweight="bold", color=_PALETTE["dark"])
    _save_fig("loc_success")
    logger.info(f"  loc_success: {len(rows)} LOC ranges")
    return os.path.join(FIGURES_DIR, "loc_success.png")


# ============================================================================
# All figures registry
# ============================================================================

_EXPERIMENT_FIGURES: list[tuple[str, Any]] = [
    ("translation_success", translation_success),
    ("compile_pie", compile_pass_rate),
    ("repair_gain", repair_gain),
    ("repair_effectiveness", repair_effectiveness),
    ("error_distribution", error_distribution),
    ("compile_error_distribution", compile_error_distribution_figure),
    ("repair_distribution", repair_distribution),
    ("repository_success", repository_success),
    ("category_success", category_success),
    ("loc_success", loc_success),
]

_DATASET_FIGURES: list[tuple[str, Any]] = [
    ("dataset_distribution", dataset_distribution),
    ("repository_distribution", repository_distribution),
    ("loc_histogram", loc_histogram),
    ("filter_distribution", filter_distribution),
]


def generate_all(logger: Logger | None = None, exp_dir: str | None = None) -> List[str]:
    """Generate all figures from the latest (or specified) experiment.

    Args:
        logger: Optional :class:`Logger`.
        exp_dir: Optional experiment directory path.

    Returns:
        List of paths to generated figure files.
    """
    if logger is None:
        logger = Logger("figures")

    logger.info("Loading experiment statistics …")
    stats = load_stats(exp_dir, logger)
    if not stats.is_empty:
        logger.info(f"  {stats.total_programs} programs, "
                    f"compile={stats.compile_success_rate:.1f}%, "
                    f"functional={stats.functional_success_rate:.1f}%")

    generated: list[str] = []

    # Experiment data figures
    for name, func in _EXPERIMENT_FIGURES:
        try:
            p = func(stats, logger)
            if p: generated.append(p)
        except Exception as exc:
            logger.warn(f"  {name}: {exc}")

    # Dataset figures
    for name, func in _DATASET_FIGURES:
        try:
            p = func(logger)
            if p: generated.append(p)
        except Exception as exc:
            logger.warn(f"  {name}: {exc}")

    logger.info(f"Generated {len(generated)} figures → {FIGURES_DIR}")
    return generated


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: generate all figures."""
    exp_dir = None
    args = argv if argv is not None else sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--exp-dir" and i + 1 < len(args):
            exp_dir = args[i + 1]; i += 2
        else: i += 1
    logger = Logger("figures")
    paths = generate_all(logger, exp_dir)
    print(f"\n{'=' * 60}")
    print(f"Figures generated: {len(paths)}")
    for p in paths: print(f"  {p}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
