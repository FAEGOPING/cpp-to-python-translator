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

# Colour palette: muted, accessible, consistent across figures
_PALETTE = {
    "blue":     "#4472C4",
    "orange":   "#ED7D31",
    "green":    "#70AD47",
    "red":      "#E74C3C",
    "purple":   "#9B59B6",
    "teal":     "#1ABC9C",
    "yellow":   "#F1C40F",
    "grey":     "#95A5A6",
    "dark":     "#2C3E50",
}
_COLORS_10 = ["#4472C4", "#ED7D31", "#70AD47", "#E74C3C", "#9B59B6",
              "#1ABC9C", "#F1C40F", "#E67E22", "#3498DB", "#2ECC71"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12,
    "figure.titlesize": 16, "legend.fontsize": 9,
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
    colors = [_PALETTE["blue"], _PALETTE["orange"], _PALETTE["green"]]
    bars = ax.bar(stages, rates, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Translation Success Rate — {p} Programs", fontweight="bold", pad=12)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, min(110, max(rates) * 1.20 + 5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))

    for bar, v, rate in zip(bars, values, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v}/{p}\n({rate:.1f}%)",
                ha="center", fontsize=11, fontweight="bold", color=_PALETTE["dark"])

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
        colors=[_PALETTE["green"], _PALETTE["red"]],
        startangle=90,
        explode=(0, 0.05),
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in texts: t.set_fontsize(12)
    for at in autotexts: at.set_fontsize(14); at.set_fontweight("bold")
    ax.set_title(f"Python Compile Validation — {stats.total_programs} Programs",
                 fontweight="bold", pad=12)
    _save_fig("compile_success")
    logger.info(f"  compile_success: {passed}/{passed + failed} "
                f"({stats.compile_success_rate:.1f}%)")
    return os.path.join(FIGURES_DIR, "compile_success.png")


def repair_gain(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Pipeline progression: initial compile → after repair → final functional."""
    if stats.is_empty:
        return _warn(logger, "repair_gain", "no experiment data")
    if not stats.has_repair_data:
        logger.info("  repair_gain: skipped (no repair data)")
        return None

    p = stats.total_programs
    # Count repaired programs properly
    repaired_pass = min(stats.repair_success_gain, p)

    # Three stages: initial compile, after repair (compile fixed), final functional
    stages = ["Initial Compile", "After Repair\n(Compile Fixed)", "Final Functional"]
    compile_initial = sum(
        1 for r in stats.program_stats
        if _safe_bool(r.get("InitialCompilePass"))
    )
    after_repair = min(compile_initial + repaired_pass, p)
    functional = stats.functional_pass

    values = [compile_initial, after_repair, functional]
    colors = [_PALETTE["blue"], _PALETTE["orange"], _PALETTE["green"]]

    _, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(stages, values, color=colors, width=0.5, edgecolor="white", linewidth=0.5)
    ax.set_title("Pipeline Stage Progression with Repair", fontweight="bold", pad=12)
    ax.set_ylabel("Number of Programs")
    ax.set_ylim(0, p * 1.15)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + p * 0.02,
                f"{val}  ({val / max(p, 1) * 100:.1f}%)",
                ha="center", fontsize=11, fontweight="bold", color=_PALETTE["dark"])

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
                f"{val}  ({pct:.1f}%)", va="center", fontsize=9, color=_PALETTE["dark"])

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
                f"{val}  ({pct:.1f}%)", va="center", fontsize=9, color=_PALETTE["dark"])

    _save_fig("compile_error_distribution")
    logger.info(f"  compile_error_distribution: {len(top)} categories, {total_cat} total")
    return os.path.join(FIGURES_DIR, "compile_error_distribution.png")


def repair_distribution(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Histogram: repair rounds per program.

    Skips the figure when fewer than 3 programs needed repair (insufficient
    data for a meaningful distribution).
    """
    if stats.is_empty:
        return _warn(logger, "repair_distribution", "no experiment data")
    if not stats.has_repair_data:
        logger.info("  repair_distribution: skipped (no repair data)")
        return None

    rr_dist = stats.repair_round_distribution
    if not rr_dist:
        return _warn(logger, "repair_distribution", "no repair round data")

    # If insufficient repair samples (all zero rounds), skip
    repaired = sum(c for rnd, c in rr_dist.items() if rnd > 0)
    if repaired < 3:
        logger.info(f"  repair_distribution: insufficient repair samples "
                     f"({repaired} programs needed repair)")
        return None

    # Build bars for rounds 0..max
    max_rr = max(rr_dist.keys())
    x_labels = [str(i) for i in range(max_rr + 1)]
    values = [rr_dist.get(i, 0) for i in range(max_rr + 1)]

    _, ax = plt.subplots(figsize=(10, 6))
    colors = [_PALETTE["grey"] if i == 0 else _PALETTE["orange"]
              for i in range(max_rr + 1)]
    bars = ax.bar(x_labels, values, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.6)
    ax.set_title(f"Repair Rounds per Program — {stats.total_programs} Programs",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Repair Rounds")
    ax.set_ylabel("Number of Programs")

    # Annotate bars with counts
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    str(val), ha="center", fontsize=10, fontweight="bold",
                    color=_PALETTE["dark"])

    _save_fig("repair_distribution")
    logger.info(f"  repair_distribution: avg={stats.avg_repair_rounds:.2f}, "
                f"max={max_rr}, {repaired} repaired")
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

    labels = [r["repository"] for r in rows]
    rates = [r["rate"] for r in rows]
    totals = [r["total"] for r in rows]

    _, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)))
    colors = [_PALETTE["green"] if rate >= 50 else _PALETTE["red"] for rate in rates]
    bars = ax.barh(labels, rates, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Functional Success Rate by Repository — {stats.total_programs} Programs",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Success Rate (%)")
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))

    for bar, rate, total in zip(bars, rates, totals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%  (n={total})", va="center", fontsize=9,
                color=_PALETTE["dark"])

    _save_fig("repository_success")
    logger.info(f"  repository_success: {len(rows)} repos")
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
    colors = [_PALETTE["green"] if rate >= 50 else _PALETTE["red"] for rate in rates]
    bars = ax.barh(labels, rates, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Functional Success Rate by Algorithm Category — {stats.total_programs} Programs",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Success Rate (%)")
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))

    for bar, rate, total in zip(bars, rates, totals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%  (n={total})", va="center", fontsize=9,
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
    plt.xticks(rotation=25, ha="right", fontsize=9)
    ax.bar_label(bars, fmt="%d", fontsize=9, padding=2)
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
    colors_list = [plt.cm.viridis(norm(i)) for i in range(len(labels))]
    bars = ax.barh(labels, values, color=colors_list, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_title("Repository Distribution — Top by Program Count", fontweight="bold", pad=12)
    ax.set_xlabel("Number of C++ Programs")
    ax.bar_label(bars, fmt="%d", fontsize=9, padding=2)
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
    ax.hist(locs, bins=50, color=_PALETTE["blue"], edgecolor="white",
            alpha=0.85, linewidth=0.3)
    ax.set_title("Lines of Code (LOC) Distribution", fontweight="bold", pad=12)
    ax.set_xlabel("Lines of Code")
    ax.set_ylabel("Frequency")
    if locs:
        mean_loc = sum(locs) / len(locs)
        ax.axvline(mean_loc, color=_PALETTE["red"], linestyle="--", linewidth=1.5,
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
        colors=[_PALETTE["green"], _PALETTE["red"], _PALETTE["orange"], _PALETTE["blue"]],
        startangle=90, explode=tuple(0.03 for _ in labels),
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    for t in texts: t.set_fontsize(11)
    for at in autotexts: at.set_fontsize(12); at.set_fontweight("bold")
    ax.set_title("Program Classification — Filtering Breakdown", fontweight="bold", pad=12)
    _save_fig("filter_distribution")
    logger.info(f"  filter_distribution: {sum(values)} files")
    return os.path.join(FIGURES_DIR, "filter_distribution.png")


def loc_success(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Scatter/bar: success rate by LOC range from experiment + metadata."""
    if stats.is_empty:
        return _warn(logger, "loc_success", "no experiment data")

    # Load LOC from metadata
    meta_rows = read_csv(os.path.join(REPORTS_DIR, "metadata.csv"))
    loc_map: dict[str, int] = {}
    for r in meta_rows:
        fname = r.get("Filename", r.get("ProgramID", ""))
        if fname:
            loc_map[fname] = _safe_int(r.get("CodeLines", "0"))

    # Bin programs by LOC
    bins = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 99999)]
    bin_labels = ["0–50", "51–100", "101–200", "201–400", "401–800", "800+"]
    bin_data: dict[str, list[bool]] = defaultdict(list)

    for r in stats.program_stats:
        name = r.get("Program", "")
        loc = loc_map.get(name, 0)
        passed = _safe_bool(r.get("FunctionalPass"))
        for (lo, hi), label in zip(bins, bin_labels):
            if lo <= loc <= hi:
                bin_data[label].append(passed)
                break

    labels = []
    rates = []
    sizes = []
    for label in bin_labels:
        results = bin_data.get(label, [])
        if results:
            labels.append(label)
            rates.append(sum(results) / max(len(results), 1) * 100)
            sizes.append(len(results))

    if not labels:
        return _warn(logger, "loc_success", "LOC/program mapping")

    _, ax = plt.subplots(figsize=(10, 6))
    colors_list = [
        _PALETTE["green"] if r >= 50 else _PALETTE["red"] for r in rates
    ]
    bars = ax.bar(labels, rates, color=colors_list, edgecolor="white",
                  linewidth=0.5, width=0.6)
    ax.set_title("Functional Success Rate by LOC Range", fontweight="bold", pad=12)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    for bar, rate, sz in zip(bars, rates, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{rate:.1f}%\n(n={sz})", ha="center", fontsize=10,
                fontweight="bold", color=_PALETTE["dark"])
    _save_fig("loc_success")
    logger.info(f"  loc_success: {len(labels)} LOC ranges")
    return os.path.join(FIGURES_DIR, "loc_success.png")


# ============================================================================
# All figures registry
# ============================================================================

_EXPERIMENT_FIGURES: list[tuple[str, Any]] = [
    ("translation_success", translation_success),
    ("compile_pie", compile_pass_rate),
    ("repair_gain", repair_gain),
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
