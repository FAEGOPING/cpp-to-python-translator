"""
figures.py — Publication-Quality Figure Generation
=====================================================

Generates all figures directly from experiment CSV data via
:mod:`statistics`.  Every figure reads from the single source of
truth — no hardcoded paths or duplicate data loading.

Output directory: ``reports/figures/``

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

from dataset_manager.utils import (
    FIGURES_DIR, REPORTS_DIR, Logger, read_csv, timestamp,
)
from statistics import load_stats, ExperimentStats, _safe_float, _safe_int, _safe_bool

# ============================================================================
# Styling
# ============================================================================

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12,
    "figure.titlesize": 16, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.3,
})


def _save_fig(name: str) -> str:
    """Save figure as PNG + PDF at 300 dpi."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=300)
    p = os.path.join(FIGURES_DIR, f"{name}.png")
    plt.close()
    return p


# ============================================================================
# Experiment-data-driven figures
# ============================================================================

def translation_success(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Bar chart: compile / runtime / functional success rates from experiment CSV."""
    if stats.is_empty:
        logger.warn("No experiment data — skipping translation_success")
        return None

    stages = ["Compile", "Runtime", "Functional"]
    values = [stats.compile_pass, stats.runtime_pass, stats.functional_pass]
    rates = [stats.compile_success_rate, stats.runtime_success_rate,
             stats.functional_success_rate]

    _, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(stages, rates, color=["#3498db", "#f39c12", "#2ecc71"])
    ax.set_title(f"Translation Success Rate — {stats.total_programs} programs")
    ax.set_ylabel("Success Rate (%)"); ax.set_ylim(0, 105)
    for bar, v, rate in zip(bars, values, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v}/{stats.total_programs} ({rate:.1f}%)",
                ha="center", fontsize=12, fontweight="bold")
    _save_fig("translation_success")
    logger.info(f"  translation_success: compile={stats.compile_pass} runtime={stats.runtime_pass} functional={stats.functional_pass}")
    return os.path.join(FIGURES_DIR, "translation_success.png")


def compile_pass_rate(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Pie chart: Python compile pass vs fail."""
    if stats.is_empty:
        logger.warn("No experiment data — skipping compile_success")
        return None

    passed, failed = stats.compile_pass, stats.compile_fail
    _, ax = plt.subplots(figsize=(7, 7))
    wedges, _, autotexts = ax.pie(
        [passed, failed],
        labels=[f"PASS ({passed})", f"FAIL ({failed})"],
        autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"], startangle=90,
    )
    for at in autotexts: at.set_fontsize(13)
    ax.set_title(f"Python Compile Validation — {stats.total_programs} programs")
    _save_fig("compile_success")
    logger.info(f"  compile_success: {passed}/{passed + failed} ({stats.compile_success_rate:.1f}%)")
    return os.path.join(FIGURES_DIR, "compile_success.png")


def repair_gain(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Pipeline progression: raw → compile pass → after repair."""
    if stats.is_empty:
        logger.warn("No experiment data — skipping repair_gain")
        return None
    if not stats.has_repair_data:
        logger.info("  repair_gain: skipped (no repair data)")
        return None

    stages = ["Total Programs", "Compile Pass", "After Repair"]
    values = [stats.total_programs, stats.compile_pass,
              stats.compile_pass + stats.repair_success_gain]
    colors = ["#3498db", "#f39c12", "#2ecc71"]

    _, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(stages, values, color=colors)
    ax.set_title("Pipeline Stage Progression with Repair")
    ax.set_ylabel("Number of Programs")
    ax.bar_label(bars, fmt="%d", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stats.total_programs * 0.01,
                f"({val / max(stats.total_programs, 1) * 100:.1f}%)",
                ha="center", fontsize=10, color="gray")
    _save_fig("repair_gain")
    logger.info(f"  repair_gain: gain={stats.repair_success_gain}")
    return os.path.join(FIGURES_DIR, "repair_gain.png")


def error_distribution(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Horizontal bar: error types from experiment detail CSV."""
    if stats.is_empty or not stats.error_counts:
        logger.info("  error_distribution: skipped (no errors)")
        return None

    top = sorted(stats.error_counts.items(), key=lambda x: -x[1])[:12]
    labels, values = zip(*reversed(top))

    _, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, values, color=plt.cm.Reds(
        [0.3 + 0.7 * i / len(labels) for i in range(len(labels))]))
    ax.set_title("Error Category Distribution")
    ax.set_xlabel("Occurrences")
    ax.bar_label(bars, fmt="%d", fontsize=8)
    _save_fig("error_category_distribution")
    logger.info(f"  error_distribution: {len(top)} error types")
    return os.path.join(FIGURES_DIR, "error_category_distribution.png")


def repair_distribution(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Histogram: repair rounds per program."""
    if stats.is_empty:
        return None
    if not stats.has_repair_data:
        logger.info("  repair_distribution: skipped (no repair data)")
        return None

    repair_rounds = [_safe_int(r.get("RepairRounds"))
                     for r in stats.program_stats
                     if _safe_int(r.get("RepairRounds")) > 0]
    if not repair_rounds:
        logger.info("  repair_distribution: skipped (all needed 0 rounds)")
        return None

    _, ax = plt.subplots(figsize=(10, 6))
    max_rr = max(repair_rounds)
    bins = range(1, max_rr + 2)
    ax.hist(repair_rounds, bins=bins, color="orange", edgecolor="white",
            alpha=0.8, align="left")
    ax.set_title("Repair Iterations per Program")
    ax.set_xlabel("Number of Repair Rounds"); ax.set_ylabel("Number of Programs")
    ax.set_xticks(range(1, max_rr + 1))
    _save_fig("repair_distribution")
    logger.info(f"  repair_distribution: avg={stats.avg_repair_rounds:.2f}")
    return os.path.join(FIGURES_DIR, "repair_distribution.png")


# ============================================================================
# Dataset-driven figures (from dataset_manager/reports/*.csv)
# ============================================================================

def dataset_distribution(logger: Logger) -> Optional[str]:
    """Bar chart: programs per dataset source (from metadata.csv)."""
    rows = read_csv(os.path.join(REPORTS_DIR, "metadata.csv"))
    if not rows: return _warn(logger, "dataset_distribution", "metadata.csv")
    counts = Counter(r.get("Category", "unknown") for r in rows)
    labels, values = zip(*sorted(counts.items(), key=lambda x: -x[1]))
    _, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=plt.cm.Set3(range(len(labels))))
    ax.set_title("Dataset Distribution — Programs per Source")
    ax.set_xlabel("Dataset Source"); ax.set_ylabel("Number of C++ Programs")
    ax.bar_label(bars, fmt="%d", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    _save_fig("dataset_distribution")
    logger.info(f"  dataset_distribution: {len(labels)} sources, {sum(values)} programs")
    return os.path.join(FIGURES_DIR, "dataset_distribution.png")


def repository_distribution(logger: Logger) -> Optional[str]:
    """Horizontal bar: programs per repository (top 15)."""
    rows = read_csv(os.path.join(REPORTS_DIR, "metadata.csv"))
    if not rows: return _warn(logger, "repository_distribution", "metadata.csv")
    counts = Counter(r.get("Repository", "unknown") for r in rows)
    top = counts.most_common(15)
    labels, values = zip(*reversed(top))
    _, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, values, color=plt.cm.viridis([i / len(labels) for i in range(len(labels))]))
    ax.set_title("Repository Distribution — Top 15 by Program Count")
    ax.set_xlabel("Number of C++ Programs")
    ax.bar_label(bars, fmt="%d", fontsize=8)
    _save_fig("repository_distribution")
    logger.info(f"  repository_distribution: top {len(top)} of {len(counts)} repos")
    return os.path.join(FIGURES_DIR, "repository_distribution.png")


def loc_histogram(logger: Logger) -> Optional[str]:
    """Histogram: lines of code distribution."""
    rows = read_csv(os.path.join(REPORTS_DIR, "metadata.csv"))
    if not rows: return _warn(logger, "loc_histogram", "metadata.csv")
    locs = [_safe_int(r.get("CodeLines", "0")) for r in rows if _safe_int(r.get("CodeLines", "0")) > 0]
    if locs:
        cutoff = sorted(locs)[int(len(locs) * 0.99)]
        locs = [l for l in locs if l <= cutoff]
    _, ax = plt.subplots(figsize=(10, 6))
    ax.hist(locs, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_title("Lines of Code (LOC) Distribution")
    ax.set_xlabel("Lines of Code"); ax.set_ylabel("Frequency")
    if locs:
        ax.axvline(sum(locs) / len(locs), color="red", linestyle="--",
                   label=f"Mean: {sum(locs) / len(locs):.0f}")
    ax.legend()
    _save_fig("loc_histogram")
    logger.info(f"  loc_histogram: {len(locs)} programs")
    return os.path.join(FIGURES_DIR, "loc_histogram.png")


def filter_distribution(logger: Logger) -> Optional[str]:
    """Pie chart: program classification."""
    rows = read_csv(os.path.join(REPORTS_DIR, "filter_report.csv"))
    if not rows: return _warn(logger, "filter_distribution", "filter_report.csv")
    data: dict[str, int] = {}
    for r in rows:
        m, v = r.get("Metric", ""), r.get("Value", "0").replace("%", "")
        try:
            if "ExecutablePrograms" in m: data["Executable"] = int(v)
            elif "Library" in m and "Remove" in m: data["Library"] = int(v)
            elif "Test" in m and "Remove" in m: data["Tests"] = int(v)
            elif "Dependency" in m and "Remove" in m: data["Dependency"] = int(v)
        except (ValueError, TypeError): pass
    if not data: return _warn(logger, "filter_distribution", "data extraction")
    labels, values = zip(*data.items())
    _, ax = plt.subplots(figsize=(8, 8))
    wedges, _, autotexts = ax.pie(
        values, labels=[f"{l} ({v})" for l, v in zip(labels, values)],
        autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c", "#f39c12", "#3498db"],
        startangle=90,
    )
    for at in autotexts: at.set_fontsize(11)
    ax.set_title("Program Classification — Filtering Breakdown")
    _save_fig("filter_distribution")
    logger.info(f"  filter_distribution: {sum(values)} files")
    return os.path.join(FIGURES_DIR, "filter_distribution.png")


# Dispatched figures that depend on experiment data + dataset data
def repository_success(stats: ExperimentStats, logger: Logger) -> Optional[str]:
    """Bar chart: success rate per repository."""
    if stats.is_empty: return _warn(logger, "repository_success", "experiment data")
    repo_pass: dict[str, list[bool]] = defaultdict(list)
    for r in stats.program_stats:
        name = r.get("Program", "")
        repo = name.split("_")[0] if "_" in name else "unknown"
        repo_pass[repo].append(_safe_bool(r.get("FunctionalPass")))
    if not repo_pass: return None
    rows = []
    for repo, results in repo_pass.items():
        rows.append({"repository": repo, "total": len(results),
                     "passed": sum(results), "rate": sum(results) / len(results) * 100})
    rows.sort(key=lambda r: -r["rate"])
    labels, rates = [r["repository"] for r in rows], [r["rate"] for r in rows]
    _, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if r >= 50 else "#e74c3c" for r in rates]
    bars = ax.barh(labels, rates, color=colors)
    ax.set_title("Success Rate by Repository (from experiment)")
    ax.set_xlabel("Success Rate (%)"); ax.set_xlim(0, 105)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
    _save_fig("repository_success")
    logger.info(f"  repository_success: {len(rows)} repos")
    return os.path.join(FIGURES_DIR, "repository_success.png")


# ============================================================================
# All figures registry
# ============================================================================

_EXPERIMENT_FIGURES = [
    ("translation_success", translation_success),
    ("compile_pie", compile_pass_rate),
    ("repair_gain", repair_gain),
    ("error_distribution", error_distribution),
    ("repair_distribution", repair_distribution),
    ("repository_success", repository_success),
]

_DATASET_FIGURES = [
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


def _warn(logger: Logger, name: str, reason: str) -> None:
    logger.warn(f"  {name}: skipped (no {reason})")


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
