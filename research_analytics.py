"""
research_analytics.py — Dissertation-Quality Research Analytics
=================================================================

Reads all existing CSV reports from ``dataset_manager/reports/`` and
produces publication-quality analyses, figures, LaTeX tables, and an
enhanced research report suitable for direct inclusion in an MSc
dissertation.

**Does NOT modify** any existing module, pipeline stage, CSV format,
or public API.  All outputs are additive — new files only.

Generated outputs (under ``reports/``):

    CSV analyses:
        repository_analysis.csv
        category_analysis.csv
        loc_analysis.csv
        compile_error_summary.csv
        compile_error_examples.csv
        repair_analysis.csv

    LaTeX tables:
        latex/table_repository.tex
        latex/table_category.tex
        latex/table_loc.tex
        latex/table_compile.tex
        latex/table_repair.tex

    Figures (PNG + PDF, 300 dpi):
        figures/repository_success.png/.pdf
        figures/category_success.png/.pdf
        figures/loc_success.png/.pdf
        figures/compile_error_distribution.png/.pdf
        figures/repair_gain.png/.pdf

    Configuration:
        experiment_configuration.json

    Enhanced research report:
        report.md  (overwrites — regenerated with full analysis)

Usage::

    python research_analytics.py
"""

from __future__ import annotations

import json
import os
import platform
import re
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

from dataset_manager.utils import (
    REPORTS_DIR, FIGURES_DIR, LOGS_DIR, BENCHMARK_DIR, RAW_CPP_DIR,
    Logger, read_csv, write_csv, timestamp, get_compiler, memory_usage_mb,
)

# ============================================================================
# Styling — consistent publication quality
# ============================================================================

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.3,
})

LATEX_DIR = os.path.join(REPORTS_DIR, "latex")

def _save_fig(name: str) -> str:
    """Save current matplotlib figure as PNG + PDF at 300 dpi."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=300)
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    plt.close()
    return path


def _safe_float(v: Any, default: float = 0.0) -> float:
    try: return float(v)
    except: return default


def _safe_int(v: Any, default: int = 0) -> int:
    try: return int(float(v))
    except: return default


# ============================================================================
# Data loading — all sources read once
# ============================================================================

class DataBundle:
    """In-memory cache of all available CSV reports."""

    def __init__(self) -> None:
        self.compile  = read_csv(os.path.join(REPORTS_DIR, "compile_report.csv"))
        self.metadata = read_csv(os.path.join(REPORTS_DIR, "metadata.csv"))
        self.progtype = read_csv(os.path.join(REPORTS_DIR, "program_type.csv"))
        self.sources  = read_csv(os.path.join(REPORTS_DIR, "source_mapping.csv"))
        self.repostat = read_csv(os.path.join(REPORTS_DIR, "repository_statistics.csv"))
        self.filter   = read_csv(os.path.join(REPORTS_DIR, "filter_report.csv"))

        # Build lookup indexes
        self.compile_map: dict[str, dict] = {}
        for r in self.compile:
            self.compile_map[r.get("File", r.get("RelativePath", ""))] = r

        self.meta_map: dict[str, dict] = {}
        for r in self.metadata:
            self.meta_map[r.get("File", "")] = r

        # Repository → list of files
        self.repo_files: dict[str, list[str]] = defaultdict(list)
        for r in self.compile:
            f = r.get("File", r.get("RelativePath", ""))
            repo = f.split("/")[1] if "/" in f else "unknown"
            self.repo_files[repo].append(f)

        # Source mapping
        self.source_map: dict[str, dict] = {}
        if self.sources:
            for r in self.sources:
                pid = r.get("ProgramID", "")
                self.source_map[pid] = r


# ============================================================================
# PART 1 — Repository-Level Analysis
# ============================================================================

def _analyse_repositories(db: DataBundle, logger: Logger) -> List[dict]:
    """Compute per-repository statistics."""
    results: list[dict] = []
    repos = sorted(db.repo_files.keys())

    for repo in repos:
        files = db.repo_files[repo]
        n = len(files)
        compile_pass = sum(1 for f in files
                           if db.compile_map.get(f, {}).get("Status") == "PASS")
        compile_times = [_safe_float(db.compile_map.get(f, {}).get("CompileTimeSeconds", 0))
                         for f in files if db.compile_map.get(f, {}).get("Status") == "PASS"]

        # LOC from metadata
        locs: list[int] = []
        for f in files:
            meta = db.meta_map.get(f, {})
            loc = _safe_int(meta.get("CodeLines", 0))
            if loc > 0:
                locs.append(loc)

        results.append({
            "repository": repo,
            "total_programs": n,
            "average_loc": round(sum(locs) / max(len(locs), 1), 1),
            "compile_pass": compile_pass,
            "compile_rate": round(compile_pass / max(n, 1) * 100, 1),
            "compile_fail": n - compile_pass,
            "average_compile_time": round(sum(compile_times) / max(len(compile_times), 1), 3),
        })
    return results


def _write_repository_analysis(rows: List[dict]) -> str:
    """Write repository_analysis.csv."""
    path = os.path.join(REPORTS_DIR, "repository_analysis.csv")
    h = ["Repository", "TotalPrograms", "AverageLOC", "CompilePass",
         "CompileRate", "CompileFail", "AverageCompileTime"]
    key_map = {"Repository": "repository", "TotalPrograms": "total_programs",
               "AverageLOC": "average_loc", "CompilePass": "compile_pass",
               "CompileRate": "compile_rate", "CompileFail": "compile_fail",
               "AverageCompileTime": "average_compile_time"}
    write_csv(path, h, [[r[key_map[k]] for k in h] for r in rows])
    return path


def _fig_repository_success(rows: List[dict]) -> str:
    """Horizontal bar chart of compile success per repository."""
    if not rows:
        return ""
    labels = [r["repository"] for r in rows]
    rates  = [r["compile_rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if r >= 50 else "#e74c3c" for r in rates]
    bars = ax.barh(labels, rates, color=colors)
    ax.set_title("Compile Success Rate by Repository")
    ax.set_xlabel("Success Rate (%)"); ax.set_xlim(0, 105)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
    return _save_fig("repository_success")


def _latex_table_repository(rows: List[dict]) -> str:
    """Write LaTeX table for repository analysis."""
    path = os.path.join(LATEX_DIR, "table_repository.tex")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Compile Success by Repository}",
        r"\label{tab:repo-success}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Repository & Programs & Avg.\ LOC & Compile Pass & Rate (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            rf"{r['repository']} & {r['total_programs']} & {r['average_loc']:.0f} & "
            rf"{r['compile_pass']} & {r['compile_rate']:.1f} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")
    return path


# ============================================================================
# PART 2 — Algorithm Category Analysis
# ============================================================================

_CPP_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Graph": ["graph", "dijkstra", "bfs", "dfs", "mst", "topological", "euler",
               "floyd", "bellman", "kruskal", "prim", "tarjan", "kosaraju"],
    "Tree": ["tree", "bst", "trie", "avl", "segment", "fenwick", "lca",
              "binary_tree", "heap", "red_black"],
    "Dynamic Programming": ["dp", "dynamic", "knapsack", "lcs", "lis",
                             "edit_distance", "matrix_chain", "coin_change"],
    "Greedy": ["greedy", "huffman", "activity_selection", "fractional_knapsack"],
    "Math": ["math", "prime", "gcd", "lcm", "modular", "factorial", "fibonacci",
              "number_theory", "combinatorics", "probability", "matrix_exponentiation",
              "fft", "numerical"],
    "Binary Search": ["binary_search", "ternary_search", "bisection"],
    "Sorting": ["sort", "merge_sort", "quick_sort", "heap_sort", "bubble",
                 "insertion_sort", "selection_sort", "counting_sort", "radix"],
    "String": ["string", "kmp", "z_algorithm", "suffix", "trie", "aho_corasick",
                "palindrome", "manacher", "hashing", "rabin_karp"],
    "Backtracking": ["backtrack", "n_queen", "sudoku", "permutation"],
    "Geometry": ["geometry", "convex_hull", "closest_pair", "point", "line_intersection"],
    "Bit Manipulation": ["bit", "bitmask", "xor", "bitset"],
    "Data Structures": ["stack", "queue", "linked_list", "deque", "priority_queue",
                         "hash", "map", "set", "list", "array"],
    "Simulation": ["simulation", "simulate"],
}

def _classify_category(file_path: str) -> str:
    """Classify a C++ file into an algorithm category by path + name."""
    lower = file_path.lower()
    scores: dict[str, int] = defaultdict(int)
    for cat, keywords in _CPP_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[cat] += 1
    if scores:
        return max(scores, key=lambda k: scores[k])
    return "Other"


def _analyse_categories(db: DataBundle) -> List[dict]:
    """Compute per-category statistics."""
    cat_files: dict[str, list[str]] = defaultdict(list)
    for r in db.compile:
        f = r.get("File", r.get("RelativePath", ""))
        cat = _classify_category(f)
        cat_files[cat].append(f)

    results: list[dict] = []
    for cat in sorted(cat_files.keys()):
        files = cat_files[cat]
        n = len(files)
        compile_pass = sum(1 for f in files
                           if db.compile_map.get(f, {}).get("Status") == "PASS")
        locs = [_safe_int(db.meta_map.get(f, {}).get("CodeLines", 0))
                for f in files]
        avg_loc = round(sum(locs) / max(len(locs), 1), 1)
        results.append({
            "category": cat, "programs": n, "average_loc": avg_loc,
            "compile_pass": compile_pass,
            "compile_rate": round(compile_pass / max(n, 1) * 100, 1),
        })
    return results


def _write_category_analysis(rows: List[dict]) -> str:
    path = os.path.join(REPORTS_DIR, "category_analysis.csv")
    h = ["Category", "Programs", "AverageLOC", "CompilePass", "CompileRate"]
    km = {"Category": "category", "Programs": "programs",
          "AverageLOC": "average_loc", "CompilePass": "compile_pass",
          "CompileRate": "compile_rate"}
    write_csv(path, h, [[r[km[k]] for k in h] for r in rows])
    return path


def _fig_category_success(rows: List[dict]) -> str:
    if not rows:
        return ""
    top = sorted(rows, key=lambda r: r["programs"], reverse=True)[:12]
    labels = [r["category"] for r in top]
    rates  = [r["compile_rate"] for r in top]
    counts = [r["programs"] for r in top]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.barh(labels, rates, color=plt.cm.Set3(range(len(labels))))
    ax1.set_title("Compile Success Rate by Algorithm Category")
    ax1.set_xlabel("Compile Success Rate (%)"); ax1.set_xlim(0, 105)
    ax1.bar_label(bars, fmt="%.1f%%", fontsize=8)

    ax2 = ax1.twiny()
    ax2.plot(counts, range(len(labels)), "o-", color="black", markersize=4,
             alpha=0.5, label="Program count")
    ax2.set_xlabel("Number of Programs")
    ax2.legend(loc="lower right")
    return _save_fig("category_success")


def _latex_table_category(rows: List[dict]) -> str:
    path = os.path.join(LATEX_DIR, "table_category.tex")
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Compile Success by Algorithm Category}",
        r"\label{tab:category-success}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Category & Programs & Avg.\ LOC & Compile Rate (\%) \\", r"\midrule",
    ]
    for r in sorted(rows, key=lambda r: r["programs"], reverse=True)[:15]:
        lines.append(
            rf"{r['category']} & {r['programs']} & {r['average_loc']:.0f} & "
            rf"{r['compile_rate']:.1f} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")
    return path


# ============================================================================
# PART 3 — LOC Analysis
# ============================================================================

_LOC_BINS = [
    ("0-50",    0,  50),
    ("51-100",  51, 100),
    ("101-200", 101, 200),
    ("201-500", 201, 500),
    ("500+",    501, 999_999),
]

def _analyse_loc(db: DataBundle) -> List[dict]:
    """Group programs by LOC bins and compute success rates."""
    bins: dict[str, dict] = {b[0]: {"total": 0, "pass": 0, "times": [], "locs": []}
                              for b in _LOC_BINS}

    for r in db.compile:
        f = r.get("File", r.get("RelativePath", ""))
        meta = db.meta_map.get(f, {})
        loc = _safe_int(meta.get("CodeLines", 0))
        status = r.get("Status", "")
        ct = _safe_float(r.get("CompileTimeSeconds", 0))

        for label, lo, hi in _LOC_BINS:
            if lo <= loc <= hi:
                bins[label]["total"] += 1
                if status == "PASS":
                    bins[label]["pass"] += 1
                    if ct > 0:
                        bins[label]["times"].append(ct)
                bins[label]["locs"].append(loc)
                break

    results: list[dict] = []
    for label, _, _ in _LOC_BINS:
        b = bins[label]
        n = b["total"]
        results.append({
            "loc_range": label, "total_programs": n,
            "compile_pass": b["pass"],
            "compile_rate": round(b["pass"] / max(n, 1) * 100, 1),
            "average_compile_time": round(sum(b["times"]) / max(len(b["times"]), 1), 3),
            "average_loc": round(sum(b["locs"]) / max(len(b["locs"]), 1), 1),
        })
    return results


def _write_loc_analysis(rows: List[dict]) -> str:
    path = os.path.join(REPORTS_DIR, "loc_analysis.csv")
    h = ["LOCRange", "TotalPrograms", "CompilePass", "CompileRate",
         "AverageCompileTime", "AverageLOC"]
    km = {"LOCRange": "loc_range", "TotalPrograms": "total_programs",
          "CompilePass": "compile_pass", "CompileRate": "compile_rate",
          "AverageCompileTime": "average_compile_time", "AverageLOC": "average_loc"}
    write_csv(path, h, [[r[km[k]] for k in h] for r in rows])
    return path


def _fig_loc_success(rows: List[dict]) -> str:
    if not rows:
        return ""
    labels = [r["loc_range"] for r in rows]
    rates  = [r["compile_rate"] for r in rows]
    counts = [r["total_programs"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, rates, color=plt.cm.viridis([0.2, 0.4, 0.6, 0.75, 0.9]))
    ax.set_title("Compile Success Rate by Program Size (LOC)")
    ax.set_ylabel("Success Rate (%)"); ax.set_ylim(0, 105)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=10)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"n={cnt}", ha="center", fontsize=8, color="gray")
    return _save_fig("loc_success")


def _latex_table_loc(rows: List[dict]) -> str:
    path = os.path.join(LATEX_DIR, "table_loc.tex")
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Compile Success by Program Size (LOC)}",
        r"\label{tab:loc-success}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"LOC Range & Programs & Compile Pass & Rate (\%) & Avg.\ Time (s) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            rf"{r['loc_range']} & {r['total_programs']} & {r['compile_pass']} & "
            rf"{r['compile_rate']:.1f} & {r['average_compile_time']:.3f} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")
    return path


# ============================================================================
# PART 4 — Compile Error Classification
# ============================================================================

_ERROR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Missing Header (bits/stdc++.h)", re.compile(r"bits/stdc\+\+\.h.*No such file", re.I)),
    ("Missing Header (PBDS)", re.compile(r"(ext/pb_ds|pb_ds).*No such file", re.I)),
    ("Missing Header (other)", re.compile(r"fatal error:.*No such file", re.I)),
    ("Undefined Reference", re.compile(r"undefined reference|Undefined symbols", re.I)),
    ("Link Error", re.compile(r"ld:.*error|linker command failed", re.I)),
    ("Multiple Main", re.compile(r"multiple definition.*main|duplicate symbol.*main", re.I)),
    ("Template Error", re.compile(r"template.*error|no matching function", re.I)),
    ("C++20 Feature", re.compile(r"warning:.*C\+\+20|c\+\+2a", re.I)),
    ("Platform Dependency", re.compile(r"#error.*not supported|This platform", re.I)),
    ("Syntax Error", re.compile(r"error: expected|error: unknown type", re.I)),
]


def _classify_compile_error(stderr: str) -> str:
    """Classify a compile error into a known category."""
    if not stderr:
        return "Unknown"
    for label, pat in _ERROR_PATTERNS:
        if pat.search(stderr):
            return label
    return "Other"


def _analyse_compile_errors(db: DataBundle) -> Tuple[List[dict], List[dict]]:
    """Classify compile errors and produce summary + example rows."""
    counter: Counter = Counter()
    examples: dict[str, str] = {}

    for r in db.compile:
        if r.get("Status") != "FAIL":
            continue
        stderr = r.get("Stderr", "")
        cat = _classify_compile_error(stderr)
        counter[cat] += 1
        if cat not in examples and stderr:
            examples[cat] = stderr[:250]

    total = sum(counter.values())
    summary = [{"category": cat, "count": cnt,
                "percentage": round(cnt / max(total, 1) * 100, 1)}
               for cat, cnt in counter.most_common()]

    example_rows = [{"category": cat, "count": counter[cat],
                     "example": examples.get(cat, "")[:250]}
                    for cat, _ in counter.most_common(12)]

    return summary, example_rows


def _write_compile_error_report(summary: List[dict], examples: List[dict]) -> str:
    summary_path = os.path.join(REPORTS_DIR, "compile_error_summary.csv")
    examples_path = os.path.join(REPORTS_DIR, "compile_error_examples.csv")

    write_csv(summary_path,
              ["Category", "Count", "Percentage"],
              [[r["category"], r["count"], r["percentage"]] for r in summary])

    write_csv(examples_path,
              ["Category", "Count", "ExampleStderr"],
              [[r["category"], r["count"], r["example"]] for r in examples])

    # Write a Markdown error report
    md_path = os.path.join(REPORTS_DIR, "compile_error_report.md")
    lines = ["# Compile Error Analysis", "",
             f"Total failures: {sum(r['count'] for r in summary)}", ""]
    lines += ["| Category | Count | Percentage |", "|----------|-------|------------|"]
    for r in summary:
        lines.append(f"| {r['category']} | {r['count']} | {r['percentage']:.1f}% |")
    lines.append("")
    lines.append("## Representative Examples")
    for r in examples[:8]:
        lines.append(f"\n### {r['category']} ({r['count']} occurrences)\n")
        lines.append("```")
        lines.append(r["example"][:300])
        lines.append("```")
    with open(md_path, "w") as f: f.write("\n".join(lines) + "\n")
    return summary_path


def _fig_compile_error_distribution(summary: List[dict]) -> str:
    if not summary:
        return ""
    top = summary[:10]
    labels = [r["category"] for r in top]
    values = [r["count"] for r in top]

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.cm.Reds([0.3 + 0.7 * i / len(labels) for i in range(len(labels))])
    bars = ax.barh(labels, values, color=colors)
    ax.set_title("Compile Error Distribution")
    ax.set_xlabel("Number of Failures")
    ax.bar_label(bars, fmt="%d", fontsize=9)
    ax.invert_yaxis()
    return _save_fig("compile_error_distribution")


def _latex_table_compile(summary: List[dict]) -> str:
    path = os.path.join(LATEX_DIR, "table_compile.tex")
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Compile Error Classification}",
        r"\label{tab:compile-errors}",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Error Category & Count & Rate (\%) \\", r"\midrule",
    ]
    for r in summary[:12]:
        lines.append(rf"{r['category']} & {r['count']} & {r['percentage']:.1f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")
    return path


# ============================================================================
# PART 5 — Repair Gain Analysis (uses compile pass/fail as proxy)
# ============================================================================

def _analyse_repair(db: DataBundle) -> List[dict]:
    """Analyse compile success as a proxy for repair gain.

    If translation results exist (summary_results.csv from run.py),
    those are incorporated.  Otherwise, compile pass rate is used
    as the baseline metric.
    """
    proj_root = os.path.dirname(os.path.abspath(__file__))
    trans_csv = os.path.join(proj_root, "summary_results.csv")
    trans_rows = read_csv(trans_csv)

    total = len(db.compile)
    compile_pass = sum(1 for r in db.compile if r.get("Status") == "PASS")
    compile_fail = total - compile_pass

    if trans_rows:
        # Translation results exist — use real repair data
        rr_list = [_safe_int(r.get("RepairRounds", "0")) for r in trans_rows]
        avg_repair = sum(rr_list) / max(len(rr_list), 1) if rr_list else 0
        max_repair = max(rr_list) if rr_list else 0
        func_ok = sum(1 for r in trans_rows if r.get("FunctionalPass", "").lower() == "true")
        trans_ok = sum(1 for r in trans_rows if r.get("FinalCompilePass", "").lower() == "true")
    else:
        # Use compile pass as proxy
        avg_repair = 0
        max_repair = 0
        func_ok = 0
        trans_ok = compile_pass

    return [{
        "total_files": total,
        "initial_compile_pass": compile_pass,
        "initial_compile_rate": round(compile_pass / max(total, 1) * 100, 1),
        "final_compile_pass": trans_ok,
        "functional_pass": func_ok,
        "average_repair_rounds": round(avg_repair, 2),
        "maximum_repair_rounds": max_repair,
        "improvement": round((trans_ok - compile_pass) / max(total, 1) * 100, 1),
    }]


def _write_repair_analysis(data: List[dict]) -> str:
    path = os.path.join(REPORTS_DIR, "repair_analysis.csv")
    d = data[0]
    h = ["Metric", "Value"]
    rows = [
        ["TotalFiles", d["total_files"]],
        ["InitialCompilePass", d["initial_compile_pass"]],
        ["InitialCompileRate", f"{d['initial_compile_rate']}%"],
        ["FinalCompilePass", d["final_compile_pass"]],
        ["FunctionalPass", d["functional_pass"]],
        ["AverageRepairRounds", d["average_repair_rounds"]],
        ["MaximumRepairRounds", d["maximum_repair_rounds"]],
        ["ImprovementOverInitial", f"{d['improvement']}%"],
    ]
    write_csv(path, h, rows)
    return path


def _fig_repair_gain(data: List[dict]) -> str:
    d = data[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    stages = ["Raw Files", "Compile Pass", "After Repair\n(if available)"]
    values = [d["total_files"], d["initial_compile_pass"], d["final_compile_pass"]]
    colors = ["#3498db", "#f39c12", "#2ecc71"]

    bars = ax.bar(stages, values, color=colors)
    ax.set_title("Pipeline Stage Progression")
    ax.set_ylabel("Number of Programs")
    ax.bar_label(bars, fmt="%d", fontsize=12, fontweight="bold")

    # Add percentage annotations
    total = d["total_files"]
    for bar, val, color in zip(bars, values, colors):
        pct = val / max(total, 1) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"({pct:.1f}%)", ha="center", fontsize=10, color=color)
    return _save_fig("repair_gain")


def _latex_table_repair(data: List[dict]) -> str:
    path = os.path.join(LATEX_DIR, "table_repair.tex")
    d = data[0]
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Pipeline Stage Progression}",
        r"\label{tab:repair-gain}",
        r"\begin{tabular}{lr}", r"\toprule",
        r"Metric & Value \\", r"\midrule",
        rf"Total Files & {d['total_files']} \\",
        rf"Initial Compile Pass & {d['initial_compile_pass']} ({d['initial_compile_rate']}\%) \\",
        rf"After Repair & {d['final_compile_pass']} \\",
        rf"Improvement & {d['improvement']}\% \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")
    return path


# ============================================================================
# PART 8 — Experiment Configuration
# ============================================================================

def _generate_experiment_config(db: DataBundle) -> str:
    """Write experiment_configuration.json."""
    import subprocess

    # Git commit
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:
        git_hash = "unknown"

    # Compiler info
    try:
        compiler = get_compiler()
    except Exception:
        compiler = {"name": "unknown", "version": "unknown", "executable": "unknown"}

    config = {
        "experiment_timestamp": timestamp(),
        "git_commit": git_hash,
        "compiler": compiler,
        "operating_system": platform.platform(),
        "python_version": sys.version,
        "dataset_size": len(db.compile),
        "benchmark_programs": len(db.sources) if db.sources else 0,
        "repositories_scanned": len(db.repostat),
    }

    path = os.path.join(REPORTS_DIR, "experiment_configuration.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return path


# ============================================================================
# PART 9 — Enhanced Research Report (dissertation-ready)
# ============================================================================

def _generate_enhanced_report(
    db: DataBundle,
    repo_rows: List[dict],
    cat_rows: List[dict],
    loc_rows: List[dict],
    err_summary: List[dict],
    repair_data: List[dict],
    fig_paths: List[str],
    logger: Logger,
) -> str:
    """Generate a comprehensive dissertation-ready Markdown report."""
    compiler = get_compiler()
    total = len(db.compile)
    compile_pass = sum(1 for r in db.compile if r.get("Status") == "PASS")

    lines: list[str] = []
    sep = "=" * 70

    # Title
    lines.append(sep)
    lines.append("AUTOMATED C++ → PYTHON TRANSLATION")
    lines.append("RESEARCH EXPERIMENT REPORT")
    lines.append(sep)
    lines += [
        "", f"**Generated:** {timestamp()}",
        f"**Experiment:** LLM-Based C++ → Python Translation with Iterative Repair",
        f"**Platform:** Research-Grade C++ → Python Translation & Evaluation Framework",
        "",
    ]

    # Experiment Configuration
    lines += ["-" * 70, "## Experiment Configuration", "-" * 70, ""]
    lines += [
        f"| Setting | Value |",
        f"|---------|-------|",
        f"| Compiler | {compiler['name']} |",
        f"| Compiler Executable | `{compiler['executable']}` |",
        f"| C++ Standard | C++{compiler['standard'].replace('c++', '')} |",
        f"| Operating System | {platform.platform()} |",
        f"| Python | {sys.version.split()[0]} |",
        f"| Timestamp | {timestamp()} |",
        "",
    ]

    # Dataset Overview
    lines += ["-" * 70, "## 1. Dataset Overview", "-" * 70, ""]
    repo_count = len(db.repostat)
    lines += [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Repositories | {repo_count} |",
        f"| Total C++ files extracted | {len(db.progtype)} |",
        f"| Executable programs (filtered) | {total} |",
        f"| Compiled successfully | {compile_pass} ({compile_pass / max(total, 1) * 100:.1f}%) |",
        f"| Benchmark dataset | {len(db.sources)} programs |",
        "",
    ]

    # Dataset Sources
    if db.repostat:
        lines += ["### Repository Sources", ""]
        lines += ["| Repository | C++ Files | Total LOC |", "|------------|-----------|-----------|"]
        for r in db.repostat:
            name = r.get("repository_name", r.get("RepositoryName", ""))
            src = r.get("dataset_source", r.get("DatasetSource", ""))
            cpp = r.get("cpp_files", r.get("CppFiles", "0"))
            loc = r.get("total_loc", r.get("TotalLOC", "0"))
            if name:
                lines.append(f"| {src}/{name} | {cpp} | {loc} |")
        lines.append("")

    # Repository Analysis
    if repo_rows:
        lines += ["-" * 70, "## 2. Repository-Level Analysis", "-" * 70, ""]
        lines += [
            "The following table presents per-repository compile success rates.",
            "",
            "| Repository | Programs | Avg. LOC | Compile Pass | Rate |",
            "|------------|----------|----------|--------------|------|",
        ]
        for r in repo_rows:
            lines.append(
                f"| {r['repository']} | {r['total_programs']} | {r['average_loc']:.0f} | "
                f"{r['compile_pass']} | {r['compile_rate']:.1f}% |"
            )
        lines.append("")
        if os.path.isfile(os.path.join(FIGURES_DIR, "repository_success.png")):
            lines.append("![Repository Success Rates](figures/repository_success.png)")
            lines.append("")

    # Category Analysis
    if cat_rows:
        lines += ["-" * 70, "## 3. Algorithm Category Analysis", "-" * 70, ""]
        lines += [
            "Programs were automatically classified by algorithm category using "
            "directory names and keywords.",
            "",
            "| Category | Programs | Avg. LOC | Compile Pass | Rate |",
            "|----------|----------|----------|--------------|------|",
        ]
        for r in sorted(cat_rows, key=lambda r: r["programs"], reverse=True)[:15]:
            lines.append(
                f"| {r['category']} | {r['programs']} | {r['average_loc']:.0f} | "
                f"{r['compile_pass']} | {r['compile_rate']:.1f}% |"
            )
        lines.append("")
        if os.path.isfile(os.path.join(FIGURES_DIR, "category_success.png")):
            lines.append("![Category Success Rates](figures/category_success.png)")
            lines.append("")

    # LOC Analysis
    if loc_rows:
        lines += ["-" * 70, "## 4. Program Size Analysis", "-" * 70, ""]
        lines += [
            "Programs were grouped by lines of code to analyse the relationship "
            "between program size and compile success.",
            "",
            "| LOC Range | Programs | Compile Pass | Rate | Avg. Compile Time |",
            "|-----------|----------|--------------|------|-------------------|",
        ]
        for r in loc_rows:
            lines.append(
                f"| {r['loc_range']} | {r['total_programs']} | {r['compile_pass']} | "
                f"{r['compile_rate']:.1f}% | {r['average_compile_time']:.3f}s |"
            )
        lines.append("")
        if os.path.isfile(os.path.join(FIGURES_DIR, "loc_success.png")):
            lines.append("![LOC Success Rates](figures/loc_success.png)")
            lines.append("")

    # Compile Error Analysis
    if err_summary:
        lines += ["-" * 70, "## 5. Compile Error Analysis", "-" * 70, ""]
        total_errs = sum(r["count"] for r in err_summary)
        lines += [
            f"A total of **{total_errs}** compilation failures were analysed.",
            "",
            "| Error Category | Count | Percentage |",
            "|----------------|-------|------------|",
        ]
        for r in err_summary[:10]:
            lines.append(f"| {r['category']} | {r['count']} | {r['percentage']:.1f}% |")
        lines.append("")
        if os.path.isfile(os.path.join(FIGURES_DIR, "compile_error_distribution.png")):
            lines.append("![Error Distribution](figures/compile_error_distribution.png)")
            lines.append("")

    # Repair Analysis
    lines += ["-" * 70, "## 6. Pipeline Progression", "-" * 70, ""]
    d = repair_data[0]
    lines += [
        "The following table shows program survival at each stage of the pipeline.",
        "",
        f"| Stage | Programs | Rate |",
        f"|-------|----------|------|",
        f"| Raw files | {d['total_files']} | 100.0% |",
        f"| Compile pass | {d['initial_compile_pass']} | {d['initial_compile_rate']}% |",
    ]
    if d["final_compile_pass"] > d["initial_compile_pass"]:
        lines.append(f"| After repair | {d['final_compile_pass']} | +{d['improvement']}% |")
    lines.append("")
    if os.path.isfile(os.path.join(FIGURES_DIR, "repair_gain.png")):
        lines.append("![Pipeline Progression](figures/repair_gain.png)")
        lines.append("")

    # Figures
    lines += ["-" * 70, "## 7. Generated Figures", "-" * 70, ""]
    for p in fig_paths:
        name = os.path.basename(p).replace(".png", "").replace("_", " ").title()
        rel = f"figures/{os.path.basename(p)}"
        lines.append(f"- **{name}**: `{rel}`")
    lines.append("")

    # Main Findings
    lines += ["-" * 70, "## 8. Main Findings", "-" * 70, ""]
    comp_rate = compile_pass / max(total, 1) * 100
    top_repo = repo_rows[0]["repository"] if repo_rows else "unknown"
    top_cat = max(cat_rows, key=lambda r: r["programs"])["category"] if cat_rows else "unknown"

    findings = [
        f"1. **Compile success rate:** {comp_rate:.1f}% of {total} executable C++ programs "
        f"compiled successfully with {compiler['name']} ({compiler['executable']}).",
        f"2. **Repository variation:** compile rates vary across {len(repo_rows)} repositories, "
        f"ranging from {min(r['compile_rate'] for r in repo_rows):.1f}% to "
        f"{max(r['compile_rate'] for r in repo_rows):.1f}%.",
        f"3. **Program size:** larger programs (500+ LOC) show "
        f"{'lower' if loc_rows and loc_rows[-1]['compile_rate'] < loc_rows[0]['compile_rate'] else 'comparable'} "
        f"compile rates compared to small programs (0-50 LOC).",
        f"4. **Error distribution:** the most common compile error category is "
        f"'{err_summary[0]['category']}' ({err_summary[0]['count']} occurrences, "
        f"{err_summary[0]['percentage']:.1f}%).",
    ]
    for f in findings:
        lines.append(f)
    lines.append("")

    # Threats to Validity
    lines += ["-" * 70, "## 9. Threats to Validity", "-" * 70, ""]
    lines += [
        "1. **Repository bias:** the dataset is drawn from public GitHub repositories, "
        "which may not represent the full distribution of C++ code in practice.",
        "2. **Compiler compatibility:** programs requiring external libraries, build "
        "systems (CMake, Make), or specific compiler versions may fail despite "
        "being valid C++.",
        "3. **Category classification:** algorithm categories are inferred from "
        "directory names and keywords, which may misclassify some programs.",
        "4. **Competitive programming style:** many programs use non-standard "
        "headers (`<bits/stdc++.h>`) or compiler-specific extensions that affect "
        "portability.",
        "",
    ]

    # Future Work
    lines += ["-" * 70, "## 10. Future Work", "-" * 70, ""]
    lines += [
        "1. Expand the dataset with additional repositories from AtCoder, LeetCode, "
        "and other competitive programming platforms.",
        "2. Integrate LLM-based translation results for end-to-end evaluation.",
        "3. Implement fuzzing-based differential testing for translation validation.",
        "4. Explore fine-tuning strategies for domain-specific C++ → Python translation.",
        "5. Investigate the impact of coding style and comment density on translation quality.",
        "",
    ]

    lines.append(sep)
    lines.append("*Report automatically generated by the Research Analytics module.*")
    lines.append(sep)

    path = os.path.join(REPORTS_DIR, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ============================================================================
# Main orchestrator
# ============================================================================

def main() -> None:
    """Run the complete research analytics pipeline."""
    logger = Logger("research_analytics")
    logger.info("Starting research analytics …")

    # Load all data
    logger.info("Loading data …")
    db = DataBundle()
    logger.info(f"  compile report: {len(db.compile)} rows")
    logger.info(f"  metadata:       {len(db.metadata)} rows")
    logger.info(f"  source mapping: {len(db.sources)} rows")
    logger.info(f"  repo statistics: {len(db.repostat)} repositories")

    all_fig_paths: list[str] = []

    # PART 1 — Repository analysis
    logger.info("Part 1: Repository-level analysis …")
    repo_rows = _analyse_repositories(db, logger)
    _write_repository_analysis(repo_rows)
    fp = _fig_repository_success(repo_rows)
    if fp: all_fig_paths.append(fp)
    _latex_table_repository(repo_rows)
    logger.info(f"  {len(repo_rows)} repositories analysed")

    # PART 2 — Category analysis
    logger.info("Part 2: Algorithm category analysis …")
    cat_rows = _analyse_categories(db)
    _write_category_analysis(cat_rows)
    fp = _fig_category_success(cat_rows)
    if fp: all_fig_paths.append(fp)
    _latex_table_category(cat_rows)
    logger.info(f"  {len(cat_rows)} categories analysed")

    # PART 3 — LOC analysis
    logger.info("Part 3: LOC analysis …")
    loc_rows = _analyse_loc(db)
    _write_loc_analysis(loc_rows)
    fp = _fig_loc_success(loc_rows)
    if fp: all_fig_paths.append(fp)
    _latex_table_loc(loc_rows)
    logger.info(f"  {len(loc_rows)} LOC ranges analysed")

    # PART 4 — Compile error classification
    logger.info("Part 4: Compile error classification …")
    err_summary, err_examples = _analyse_compile_errors(db)
    _write_compile_error_report(err_summary, err_examples)
    fp = _fig_compile_error_distribution(err_summary)
    if fp: all_fig_paths.append(fp)
    _latex_table_compile(err_summary)
    logger.info(f"  {len(err_summary)} error categories, {sum(r['count'] for r in err_summary)} total failures")

    # PART 5 — Repair gain analysis
    logger.info("Part 5: Repair gain analysis …")
    repair_data = _analyse_repair(db)
    _write_repair_analysis(repair_data)
    fp = _fig_repair_gain(repair_data)
    if fp: all_fig_paths.append(fp)
    _latex_table_repair(repair_data)
    logger.info(f"  repair analysis complete")

    # PART 8 — Experiment configuration
    logger.info("Part 8: Experiment configuration …")
    cfg_path = _generate_experiment_config(db)
    logger.info(f"  {cfg_path}")

    # PART 9 — Enhanced report
    logger.info("Part 9: Generating enhanced research report …")
    report_path = _generate_enhanced_report(
        db, repo_rows, cat_rows, loc_rows, err_summary, repair_data,
        all_fig_paths, logger,
    )
    logger.info(f"  {report_path}")

    # Summary
    logger.info("")
    logger.info(f"Research analytics complete.")
    logger.info(f"  Figures: {len(all_fig_paths)}")
    logger.info(f"  LaTeX tables: 5 in {LATEX_DIR}/")
    logger.info(f"  CSV analyses: 6 in {REPORTS_DIR}/")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Configuration: {cfg_path}")

    for p in all_fig_paths:
        logger.info(f"    {p}")

    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
