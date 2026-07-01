"""
figures.py — Automatic Publication-Quality Figure Generation
==============================================================

Generates publication-quality figures (PNG + PDF) for dissertation
inclusion, from all available experiment and dataset CSV reports.

Output directory: ``reports/figures/``

Generated figures:
    - dataset_distribution.png/.pdf   — programs per dataset source
    - repository_distribution.png/.pdf — programs per repository
    - loc_histogram.png/.pdf          — LOC distribution
    - compile_success.png/.pdf        — compile success rate
    - translation_success.png/.pdf    — translation success rate
    - repair_success.png/.pdf         — repair iteration distribution
    - translation_time.png/.pdf       — translation time distribution
    - error_category_distribution.png/.pdf — error types

Usage::

    python figures.py
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

# Ensure project root
_DEPS = os.path.dirname(os.path.abspath(__file__))
if _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

from dataset_manager.utils import (
    FIGURES_DIR,
    REPORTS_DIR,
    read_csv,
    Logger,
    timestamp,
)

# ============================================================================
# Styling — publication quality
# ============================================================================

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _save_figure(name: str) -> None:
    """Save current figure as both PNG and PDF.

    Args:
        name: Base filename (without extension).
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        plt.savefig(path, bbox_inches="tight", dpi=300)


# ============================================================================
# Individual figures
# ============================================================================

def dataset_distribution(logger: Logger) -> Optional[str]:
    """Bar chart: number of programs per dataset source.

    Args:
        logger: :class:`Logger` instance.

    Returns:
        Path to the saved figure, or ``None`` on failure.
    """
    meta_path = os.path.join(REPORTS_DIR, "metadata.csv")
    rows = read_csv(meta_path)
    if not rows:
        logger.warn("No metadata.csv — skipping dataset_distribution")
        return None

    counts = Counter(r.get("Category", "unknown") for r in rows)
    labels, values = zip(*sorted(counts.items(), key=lambda x: -x[1]))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=plt.cm.Set3(range(len(labels))))
    ax.set_title("Dataset Distribution — Programs per Source")
    ax.set_xlabel("Dataset Source")
    ax.set_ylabel("Number of C++ Programs")
    ax.bar_label(bars, fmt="%d", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    _save_figure("dataset_distribution")
    plt.close()
    logger.info(f"  dataset_distribution: {len(labels)} sources, {sum(values)} programs")
    return os.path.join(FIGURES_DIR, "dataset_distribution.png")


def repository_distribution(logger: Logger) -> Optional[str]:
    """Horizontal bar chart: programs per repository (top 15).

    Args:
        logger: :class:`Logger` instance.

    Returns:
        Path to the saved figure.
    """
    meta_path = os.path.join(REPORTS_DIR, "metadata.csv")
    rows = read_csv(meta_path)
    if not rows:
        logger.warn("No metadata.csv — skipping repository_distribution")
        return None

    counts = Counter(r.get("Repository", "unknown") for r in rows)
    top = counts.most_common(15)
    labels, values = zip(*reversed(top))

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, values, color=plt.cm.viridis(
        [i / len(labels) for i in range(len(labels))]
    ))
    ax.set_title("Repository Distribution — Top 15 by Program Count")
    ax.set_xlabel("Number of C++ Programs")
    ax.bar_label(bars, fmt="%d", fontsize=8)
    _save_figure("repository_distribution")
    plt.close()
    logger.info(f"  repository_distribution: top {len(top)} of {len(counts)} repos")
    return os.path.join(FIGURES_DIR, "repository_distribution.png")


def loc_histogram(logger: Logger) -> Optional[str]:
    """Histogram of lines-of-code distribution.

    Args:
        logger: :class:`Logger` instance.

    Returns:
        Path to the saved figure.
    """
    meta_path = os.path.join(REPORTS_DIR, "metadata.csv")
    rows = read_csv(meta_path)
    if not rows:
        logger.warn("No metadata.csv — skipping loc_histogram")
        return None

    locs: list[int] = []
    for r in rows:
        try:
            locs.append(int(r.get("CodeLines", "0")))
        except (ValueError, TypeError):
            locs.append(0)

    # Filter outliers (99th percentile)
    if locs:
        locs_sorted = sorted(locs)
        cutoff = locs_sorted[int(len(locs_sorted) * 0.99)]
        locs = [l for l in locs if l <= cutoff]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(locs, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_title("Lines of Code (LOC) Distribution")
    ax.set_xlabel("Lines of Code")
    ax.set_ylabel("Frequency")
    ax.axvline(sum(locs) / max(len(locs), 1), color="red", linestyle="--",
               label=f"Mean: {sum(locs) / max(len(locs), 1):.0f}")
    ax.legend()
    _save_figure("loc_histogram")
    plt.close()
    logger.info(f"  loc_histogram: {len(locs)} programs")
    return os.path.join(FIGURES_DIR, "loc_histogram.png")


def compile_success_rate(logger: Logger) -> Optional[str]:
    """Pie chart: compile pass vs fail.

    Args:
        logger: :class:`Logger` instance.

    Returns:
        Path to the saved figure.
    """
    compile_path = os.path.join(REPORTS_DIR, "compile_report.csv")
    rows = read_csv(compile_path)
    if not rows:
        logger.warn("No compile_report.csv — skipping compile_success")
        return None

    passed = sum(1 for r in rows if r.get("Status", "") == "PASS")
    failed = len(rows) - passed

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["#2ecc71", "#e74c3c"]
    wedges, texts, autotexts = ax.pie(
        [passed, failed],
        labels=[f"PASS ({passed})", f"FAIL ({failed})"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(13)
    ax.set_title(f"Compile Validation — {passed + failed} programs")
    _save_figure("compile_success")
    plt.close()
    logger.info(f"  compile_success: {passed}/{passed + failed} ({passed / max(passed + failed, 1) * 100:.1f}%)")
    return os.path.join(FIGURES_DIR, "compile_success.png")


def translation_success_rate(logger: Logger) -> Optional[str]:
    """Bar chart: translation success by stage (compile / runtime / functional).

    Args:
        logger: :class:`Logger` instance.

    Returns:
        Path to the saved figure.
    """
    summary_path = os.path.join(os.path.dirname(REPORTS_DIR), "summary_results.csv")
    rows = read_csv(summary_path)
    if not rows:
        # Try the experiment_summary.csv
        exp_path = os.path.join(REPORTS_DIR, "experiment_summary.csv")
        rows = read_csv(exp_path)

    if not rows:
        logger.warn("No summary results — skipping translation_success")
        return None

    total = len(rows)
    compile_ok = sum(1 for r in rows if r.get("FinalCompilePass", "").lower() == "true")
    runtime_ok = sum(1 for r in rows if r.get("RuntimePass", "").lower() == "true")
    functional_ok = sum(1 for r in rows if r.get("FunctionalPass", "").lower() == "true")

    stages = ["Compile", "Runtime", "Functional"]
    values = [compile_ok, runtime_ok, functional_ok]
    rates = [v / max(total, 1) * 100 for v in values]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(stages, rates, color=["#3498db", "#f39c12", "#2ecc71"])
    ax.set_title(f"Translation Success Rate — {total} programs")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", fontsize=12, fontweight="bold")
    _save_figure("translation_success")
    plt.close()
    logger.info(f"  translation_success: compile={compile_ok} runtime={runtime_ok} functional={functional_ok}")
    return os.path.join(FIGURES_DIR, "translation_success.png")


def repair_distribution(logger: Logger) -> Optional[str]:
    """Histogram: repair rounds per program.

    Args:
        logger: :class:`Logger` instance.

    Returns:
        Path to the saved figure.
    """
    summary_path = os.path.join(os.path.dirname(REPORTS_DIR), "summary_results.csv")
    rows = read_csv(summary_path)
    if not rows:
        logger.warn("No summary results — skipping repair_distribution")
        return None

    repair_rounds: list[int] = []
    for r in rows:
        try:
            repair_rounds.append(int(float(r.get("RepairRounds", "0"))))
        except (ValueError, TypeError):
            repair_rounds.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    max_rr = max(repair_rounds) if repair_rounds else 0
    bins = range(0, max_rr + 2)
    ax.hist(repair_rounds, bins=bins, color="orange", edgecolor="white",
            alpha=0.8, align="left")
    ax.set_title("Repair Iterations per Program")
    ax.set_xlabel("Number of Repair Rounds")
    ax.set_ylabel("Number of Programs")
    ax.set_xticks(range(0, max_rr + 1))
    _save_figure("repair_distribution")
    plt.close()
    logger.info(f"  repair_distribution: avg={sum(repair_rounds) / max(len(repair_rounds), 1):.2f}")
    return os.path.join(FIGURES_DIR, "repair_distribution.png")


def error_category_distribution(logger: Logger) -> Optional[str]:
    """Horizontal bar chart: error types encountered.

    Args:
        logger: :class:`Logger` instance.

    Returns:
        Path to the saved figure.
    """
    detail_path = os.path.join(os.path.dirname(REPORTS_DIR), "experiment_results.csv")
    rows = read_csv(detail_path)
    if not rows:
        logger.warn("No experiment results — skipping error_distribution")
        return None

    error_types = [r.get("ErrorType", "Unknown") for r in rows
                   if r.get("ErrorType", "None") != "None"]
    counts = Counter(error_types)
    top = counts.most_common(12)

    if not top:
        logger.warn("No errors found — skipping error_distribution")
        return None

    labels, values = zip(*reversed(top))
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, values, color=plt.cm.Reds(
        [0.3 + 0.7 * i / len(labels) for i in range(len(labels))]
    ))
    ax.set_title("Error Category Distribution")
    ax.set_xlabel("Occurrences")
    ax.bar_label(bars, fmt="%d", fontsize=8)
    _save_figure("error_category_distribution")
    plt.close()
    logger.info(f"  error_distribution: {len(top)} error types")
    return os.path.join(FIGURES_DIR, "error_category_distribution.png")


# ============================================================================
# Generate all figures
# ============================================================================

def filter_distribution(logger: Logger) -> Optional[str]:
    """Pie chart: program classification breakdown (executable/library/test/dep).

    Args:
        logger: :class:`Logger` instance.

    Returns:
        Path to the saved figure.
    """
    filter_path = os.path.join(REPORTS_DIR, "filter_report.csv")
    rows = read_csv(filter_path)
    if not rows:
        logger.warn("No filter_report.csv — skipping filter_distribution")
        return None

    data: dict[str, int] = {}
    for r in rows:
        metric = r.get("Metric", "")
        val = r.get("Value", "0").replace("%", "")
        try:
            if "ExecutablePrograms" in metric:
                data["Executable"] = int(val)
            elif "Library" in metric and "Remove" in metric:
                data["Library"] = int(val)
            elif "Test" in metric and "Remove" in metric:
                data["Tests"] = int(val)
            elif "Dependency" in metric and "Remove" in metric:
                data["Dependency"] = int(val)
        except (ValueError, TypeError):
            pass

    if not data:
        logger.warn("No filter data extracted")
        return None

    labels, values = zip(*data.items())
    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db"]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values, labels=[f"{l} ({v})" for l, v in zip(labels, values)],
        autopct="%1.1f%%", colors=colors, startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax.set_title("Program Classification — Filtering Breakdown")
    _save_figure("filter_distribution")
    plt.close()
    logger.info(f"  filter_distribution: {sum(values)} files across {len(labels)} categories")
    return os.path.join(FIGURES_DIR, "filter_distribution.png")


_ALL_FIGURES = [
    ("dataset_distribution", dataset_distribution),
    ("repository_distribution", repository_distribution),
    ("loc_histogram", loc_histogram),
    ("filter_distribution", filter_distribution),
    ("compile_success_rate", compile_success_rate),
    ("translation_success_rate", translation_success_rate),
    ("repair_distribution", repair_distribution),
    ("error_category_distribution", error_category_distribution),
]


def generate_all(logger: Logger | None = None) -> List[str]:
    """Generate all available figures.

    Args:
        logger: Optional :class:`Logger` instance.

    Returns:
        List of paths to successfully generated figures.
    """
    if logger is None:
        logger = Logger("figures")

    logger.info("Generating publication-quality figures …")
    generated: list[str] = []

    for name, func in _ALL_FIGURES:
        try:
            path = func(logger)
            if path:
                generated.append(path)
                logger.count(f"figure_{name}", 1)
        except Exception as exc:
            logger.warn(f"Figure '{name}' failed: {exc}")

    logger.info(f"Figures generated: {len(generated)}/{len(_ALL_FIGURES)}")
    logger.info(f"Output directory: {FIGURES_DIR}")
    return generated


# ============================================================================
# CLI
# ============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """Generate all publication-quality figures."""
    import sys as _sys
    logger = Logger("figures")
    paths = generate_all(logger)

    print(f"\n{'=' * 60}")
    print(f"Figures generated: {len(paths)}")
    for p in paths:
        print(f"  {p}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
