"""
report_generator.py — Automatic Research Report Generator
===========================================================

Generates ``report.md`` from real experiment data via :mod:`statistics`.
All metrics are computed directly from ``experiment_results.csv`` and
``summary_results.csv`` — no hardcoded paths or placeholder values.

The report is suitable for direct inclusion in a Master's dissertation.

Usage::

    python report_generator.py
    python report_generator.py --exp-dir experiment_results/2026-07-02_10-00-00
    python report_generator.py --output reports/my_report.md
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

_DEPS = os.path.dirname(os.path.abspath(__file__))
if _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

from dataset_manager.utils import (
    REPORTS_DIR, BENCHMARK_DIR, FIGURES_DIR, Logger, read_csv, timestamp, get_compiler,
)
from statistics import load_stats, stats_summary, ExperimentStats, _safe_float, _safe_int, _safe_bool


def _warn_missing(log: Logger, item: str) -> None:
    log.warn(f"  {item}: not available")


def generate_report(
    output_path: str | None = None,
    logger: Logger | None = None,
    exp_dir: str | None = None,
) -> str:
    """Generate a comprehensive Markdown research report.

    Args:
        output_path: Where to write the report.  Defaults to
            ``reports/report.md``.
        logger: Optional :class:`Logger`.
        exp_dir: Optional experiment directory.

    Returns:
        Path to the generated report.
    """
    if logger is None:
        logger = Logger("report_generator")
    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, "report.md")

    logger.info("Loading experiment data …")
    stats = load_stats(exp_dir, logger)

    sep = "=" * 70
    lines: list[str] = []

    # ==================================================================
    # Title
    # ==================================================================
    lines += [
        sep,
        "AUTOMATED C++ → PYTHON TRANSLATION",
        "RESEARCH EXPERIMENT REPORT",
        sep, "",
        f"**Generated:** {timestamp()}",
        f"**Experiment:** LLM-Based C++ → Python Translation with Iterative Repair",
        f"**Platform:** C++ → Python Translation Research Platform v3.0",
        "",
    ]

    # ==================================================================
    # Experiment Information
    # ==================================================================
    lines += ["-" * 70, "## 1. Experiment Information", "-" * 70, ""]

    lines += ["| Setting | Value |", "|---------|-------|"]

    if stats.metadata:
        m = stats.metadata
        lines.append(f"| Experiment ID | {m.get('experiment_id', 'N/A')} |")
        lines.append(f"| Timestamp | {m.get('timestamp', 'N/A')} |")
        lines.append(f"| Git Commit | `{m.get('git_commit', 'N/A')[:12]}...` |")

    try:
        c = get_compiler()
        lines += [
            f"| Compiler | {c['name']} |",
            f"| Compiler Executable | `{c['executable']}` |",
            f"| C++ Standard | C++{c['standard'].replace('c++', '')} |",
        ]
    except Exception:
        pass

    if stats.metadata:
        lines.append(f"| OS | {stats.metadata.get('operating_system', 'N/A')} |")
        lines.append(f"| Python | {stats.metadata.get('python_version', 'N/A')} |")
        cli = stats.metadata.get("cli_arguments", {})
        lines.append(f"| Repair Enabled | {cli.get('repair', False)} |")
        lines.append(f"| Runtime Verification | {cli.get('runtime', False)} |")
        lines.append(f"| Max Repair Rounds | {cli.get('max_repair_rounds', 'default')} |")
    lines.append("")

    # ==================================================================
    # Dataset Summary
    # ==================================================================
    repo_stats = read_csv(os.path.join(REPORTS_DIR, "repository_statistics.csv"))
    metadata = read_csv(os.path.join(REPORTS_DIR, "metadata.csv"))
    filter_rep = read_csv(os.path.join(REPORTS_DIR, "filter_report.csv"))

    total_cpp = sum(_safe_int(r.get("cpp_files", r.get("CppFiles", "0"))) for r in repo_stats)
    total_loc = sum(_safe_int(r.get("total_loc", r.get("TotalLOC", "0"))) for r in repo_stats)

    lines += ["-" * 70, "## 2. Dataset Summary", "-" * 70, ""]
    lines += ["| Metric | Value |", "|--------|-------|"]
    lines.append(f"| Repositories | {len(repo_stats)} |")
    lines.append(f"| Total C++ files (raw) | {total_cpp} |")
    lines.append(f"| Total LOC | {total_loc:,} |")

    if metadata:
        locs = [_safe_int(r.get("CodeLines", 0)) for r in metadata if _safe_int(r.get("CodeLines", 0)) > 0]
        if locs:
            lines.append(f"| Average LOC per program | {sum(locs) / len(locs):.1f} |")
            lines.append(f"| Maximum LOC | {max(locs)} |")

    # Repository table
    if repo_stats:
        lines += ["", "### Repository Sources", ""]
        lines += ["| Repository | C++ Files | Total LOC |", "|------------|-----------|-----------|"]
        for r in repo_stats:
            name = r.get("repository_name", r.get("RepositoryName", ""))
            src = r.get("dataset_source", r.get("DatasetSource", ""))
            cpp = r.get("cpp_files", r.get("CppFiles", "0"))
            loc = r.get("total_loc", r.get("TotalLOC", "0"))
            if name or src:
                lines.append(f"| {src}/{name} | {cpp} | {loc} |")
    lines.append("")

    # Filtering
    if filter_rep:
        def _fm(key: str) -> str:
            for r in filter_rep:
                if key in r.get("Metric", ""): return r.get("Value", "N/A")
            return "N/A"
        exec_p = _fm("ExecutablePrograms")
        lib_p = _fm("Remove_Library")
        test_p = _fm("Remove_Test")
        dep_p = _fm("Remove_Dependency")
        if exec_p != "N/A":
            lines += ["### Dataset Filtering", ""]
            lines += ["| Category | Count |", "|----------|-------|"]
            lines.append(f"| ✅ Executable programs | {exec_p} |")
            lines.append(f"| ❌ Library files removed | {lib_p} |")
            lines.append(f"| ❌ Test files removed | {test_p} |")
            lines.append(f"| ❌ Dependency files removed | {dep_p} |")
            lines.append("")
    lines.append("")

    # ==================================================================
    # Translation Results (from experiment CSVs)
    # ==================================================================
    if not stats.is_empty:
        p = stats.total_programs
        lines += ["-" * 70, "## 3. Translation Results", "-" * 70, ""]
        lines += ["| Metric | Count | Rate |", "|--------|-------|------|"]
        lines.append(f"| Programs | {p} | — |")
        lines.append(f"| Compile Success | {stats.compile_pass}/{p} | **{stats.compile_success_rate:.1f}%** |")
        lines.append(f"| Runtime Success | {stats.runtime_pass}/{p} | **{stats.runtime_success_rate:.1f}%** |")
        lines.append(f"| Functional Equivalence | {stats.functional_pass}/{p} | **{stats.functional_success_rate:.1f}%** |")
        lines.append(f"| Overall Success | {stats.overall_success}/{p} | **{stats.overall_success_rate:.1f}%** |")
        lines.append("")

        # Timing
        if stats.has_timing_data:
            lines += ["### Performance", ""]
            lines += ["| Metric | Value |", "|--------|-------|"]
            lines.append(f"| Average Translation Time | {stats.avg_translation_time:.2f}s |")
            lines.append(f"| Average Compile Time | {stats.avg_compile_time:.4f}s |")
            lines.append(f"| Average Runtime | {stats.avg_runtime_time:.4f}s |")
            lines.append(f"| Average Validation Time | {stats.avg_validation_time:.4f}s |")
            lines.append(f"| Average Total Time | {stats.avg_total_time:.2f}s |")
            lines.append("")

        # Token Statistics & Cost
        if stats.total_prompt_tokens > 0 or stats.total_tokens > 0:
            try:
                from token_tracker import estimate_cost
                cost = estimate_cost(stats.total_prompt_tokens,
                                     stats.total_completion_tokens)
                cost_str = f"${cost:.4f}"
            except Exception:
                cost_str = "N/A"
            lines += ["### Token Usage & Cost", ""]
            lines += ["| Metric | Value |", "|--------|-------|"]
            lines.append(f"| Total Prompt Tokens | {stats.total_prompt_tokens:,} |")
            lines.append(f"| Total Completion Tokens | {stats.total_completion_tokens:,} |")
            lines.append(f"| Total Tokens | {stats.total_tokens:,} |")
            lines.append(f"| Avg Prompt/Program | {stats.avg_prompt_tokens_per_program:.0f} |")
            lines.append(f"| Avg Completion/Program | {stats.avg_completion_tokens_per_program:.0f} |")
            lines.append(f"| Estimated API Cost | {cost_str} |")
            lines.append("")

        # Repair
        if stats.has_repair_data:
            lines += ["### Repair Results", ""]
            lines += ["| Metric | Value |", "|--------|-------|"]
            lines.append(f"| Programs Repaired | {stats.programs_repaired} |")
            lines.append(f"| Average Repair Rounds | {stats.avg_repair_rounds:.2f} |")
            lines.append(f"| Maximum Repair Rounds | {stats.max_repair_rounds} |")
            lines.append(f"| Repair Success Gain | +{stats.repair_success_gain} programs |")
            lines.append("")
        else:
            lines += ["### Repair", "", "*No repair was performed in this experiment.*", ""]
    else:
        lines += ["-" * 70, "## 3. Translation Results", "-" * 70, ""]
        lines += ["*No translation experiment data found. Run `python experiment_runner.py --limit N --repair` first.*", ""]

    # ==================================================================
    # Error Analysis
    # ==================================================================
    if not stats.is_empty and stats.error_counts:
        lines += ["-" * 70, "## 4. Error Analysis", "-" * 70, ""]
        total_errs = sum(stats.error_counts.values())
        lines += [f"**{total_errs}** errors recorded across all programs.", ""]
        lines += ["| Error Type | Count | Percentage |", "|------------|-------|------------|"]
        for err, cnt in sorted(stats.error_counts.items(), key=lambda x: -x[1])[:12]:
            lines.append(f"| {err} | {cnt} | {cnt / max(total_errs, 1) * 100:.1f}% |")
        lines.append("")
    elif stats.is_empty:
        lines += ["-" * 70, "## 4. Error Analysis", "-" * 70, ""]
        lines += ["*No experiment data available.*", ""]
    else:
        lines += ["-" * 70, "## 4. Error Analysis", "-" * 70, ""]
        lines += [f"✅ **All {stats.total_programs} programs passed without errors.**", ""]

    # ==================================================================
    # Figures
    # ==================================================================
    lines += ["-" * 70, "## 5. Generated Figures", "-" * 70, ""]
    fig_dir = FIGURES_DIR
    if os.path.isdir(fig_dir):
        pngs = sorted(f for f in os.listdir(fig_dir) if f.endswith(".png"))
        if pngs:
            for f in pngs:
                name = f.replace(".png", "").replace("_", " ").title()
                lines.append(f"### {name}")
                lines.append(f"![{name}](figures/{f})")
                lines.append("")
        else:
            lines += ["*No figures generated. Run `python figures.py` first.*", ""]
    lines.append("")

    # ==================================================================
    # Main Findings
    # ==================================================================
    lines += ["-" * 70, "## 6. Main Findings", "-" * 70, ""]

    if not stats.is_empty:
        p = stats.total_programs
        findings = []
        if stats.functional_success_rate >= 90:
            findings.append(
                f"1. **High translation success:** {stats.functional_success_rate:.1f}% of "
                f"{p} programs achieved functional equivalence with the original C++ code."
            )
        elif stats.functional_success_rate > 0:
            findings.append(
                f"1. **Translation success:** {stats.functional_success_rate:.1f}% ({stats.functional_pass}/{p}) "
                f"of programs achieved functional equivalence."
            )
        else:
            findings.append(f"1. **{p} programs** were translated from C++ to Python via LLM.")

        if stats.compile_fail > 0:
            findings.append(
                f"2. **Compile failures:** {stats.compile_fail}/{p} programs ({stats.compile_success_rate:.1f}%) "
                f"failed Python compilation, indicating syntax or import issues in the generated code."
            )
        else:
            findings.append(f"2. **All {p} programs** compiled successfully in Python.")

        if stats.has_repair_data and stats.repair_success_gain > 0:
            findings.append(
                f"3. **Repair effectiveness:** iterative repair recovered "
                f"{stats.repair_success_gain} programs, improving overall success."
            )
        elif stats.has_repair_data:
            findings.append(f"3. **Repair:** {stats.avg_repair_rounds:.1f} average repair rounds per program.")
    else:
        findings = ["*No experiment data available for analysis.*"]

    for f in findings:
        lines.append(f)
    lines.append("")

    # ==================================================================
    # Threats to Validity
    # ==================================================================
    lines += ["-" * 70, "## 7. Threats to Validity", "-" * 70, ""]
    lines += [
        "1. **Repository bias:** the dataset is drawn from public competitive programming "
        "repositories, which may not represent general-purpose C++ code.",
        "2. **LLM variability:** translation quality may vary across different LLM models, "
        "temperatures, and API versions.",
        "3. **Test coverage:** functional equivalence is verified against provided test "
        "inputs only; edge cases may reveal undetected discrepancies.",
        "4. **Compiler version:** results are specific to the compiler and version used.",
        "",
    ]

    lines += ["-" * 70, "## 8. Future Work", "-" * 70, ""]
    lines += [
        "1. Expand the dataset with additional repositories and algorithm categories.",
        "2. Implement fuzzing-based test generation for stronger functional validation.",
        "3. Compare multiple LLM models for translation quality.",
        "4. Investigate fine-tuning strategies for domain-specific translation.",
        "",
    ]

    lines.append(sep)
    lines.append("*Report automatically generated by the C++ → Python Translation Research Platform.*")
    lines.append(sep)

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report + "\n")

    logger.info(f"Report written: {output_path}  ({len(report):,} chars)")
    return output_path


def main(argv: Optional[List[str]] = None) -> None:
    logger = Logger("report_generator")
    output = None
    exp_dir = None
    args = argv if argv is not None else sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            output = args[i + 1]; i += 2
        elif args[i] == "--exp-dir" and i + 1 < len(args):
            exp_dir = args[i + 1]; i += 2
        else: i += 1
    path = generate_report(output, logger, exp_dir)
    print(f"\nReport generated: {path}")


if __name__ == "__main__":
    main()
