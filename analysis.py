"""
analysis.py — Automatic Experiment Analysis
============================================

Reads ``experiment_results.csv`` and ``summary_results.csv`` from the
current project root and produces:

* ``analysis_report.csv`` — machine-readable aggregate statistics.
* ``analysis_report.txt`` — human-readable report suitable for direct
  inclusion in a dissertation chapter.

Usage::

    python3 analysis.py                          # uses default paths
    python3 analysis.py --output-dir ./reports   # custom output dir
    python3 analysis.py --csv experiment_results.csv --summary summary_results.csv

The analysis is **non-destructive** — it never modifies the source CSVs.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class ProgramStats:
    """Aggregate statistics for a single program."""
    program: str
    initial_compile_pass: bool = False
    final_compile_pass: bool = False
    runtime_pass: bool = False
    functional_pass: bool = False
    repair_rounds: int = 0
    total_time: float = 0.0
    error_types: List[str] = field(default_factory=list)
    round_count: int = 0
    translation_times: List[float] = field(default_factory=list)
    compile_times: List[float] = field(default_factory=list)
    runtime_times: List[float] = field(default_factory=list)
    validation_times: List[float] = field(default_factory=list)
    repair_times: List[float] = field(default_factory=list)
    test_counts: List[int] = field(default_factory=list)
    passed_test_counts: List[int] = field(default_factory=list)


@dataclass
class AggregateStats:
    """Experiment-wide aggregate statistics."""
    total_programs: int = 0
    compile_success_rate: float = 0.0
    runtime_success_rate: float = 0.0
    functional_success_rate: float = 0.0
    overall_success_rate: float = 0.0
    avg_translation_time: float = 0.0
    avg_validation_time: float = 0.0
    avg_runtime: float = 0.0
    avg_repair_count: float = 0.0
    repair_distribution: Dict[int, int] = field(default_factory=dict)
    error_distribution: Dict[str, int] = field(default_factory=dict)
    avg_generated_test_count: float = 0.0


# ======================================================================
# CSV readers
# ======================================================================

def _read_detail_csv(path: str) -> List[dict]:
    """Read ``experiment_results.csv`` into a list of dicts."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_summary_csv(path: str) -> List[dict]:
    """Read ``summary_results.csv`` into a list of dicts."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ======================================================================
# Analysis logic
# ======================================================================

def _parse_bool(val: str) -> bool:
    """Parse a CSV boolean string."""
    return val.strip().lower() in ("true", "1", "yes")


def _parse_float(val: str, default: float = 0.0) -> float:
    """Safely parse a float from CSV."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _parse_int(val: str, default: int = 0) -> int:
    """Safely parse an int from CSV."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def analyse(
    detail_csv: str,
    summary_csv: str,
) -> tuple[AggregateStats, List[ProgramStats]]:
    """Run the full analysis and return aggregate + per-program statistics.

    Args:
        detail_csv: Path to ``experiment_results.csv``.
        summary_csv: Path to ``summary_results.csv``.

    Returns:
        ``(aggregate, per_program_list)``.
    """
    detail_rows = _read_detail_csv(detail_csv)
    summary_rows = _read_summary_csv(summary_csv)

    # Build per-program stats from summary
    program_map: dict[str, ProgramStats] = {}
    for row in summary_rows:
        name = row.get("Program", "unknown")
        ps = ProgramStats(
            program=name,
            initial_compile_pass=_parse_bool(row.get("InitialCompilePass", "False")),
            final_compile_pass=_parse_bool(row.get("FinalCompilePass", "False")),
            runtime_pass=_parse_bool(row.get("RuntimePass", "False")),
            functional_pass=_parse_bool(row.get("FunctionalPass", "False")),
            repair_rounds=_parse_int(row.get("RepairRounds", "0")),
            total_time=_parse_float(row.get("TotalTime", "0")),
        )
        program_map[name] = ps

    # Enrich from detail rows
    error_counter: Counter = Counter()
    repair_counts: list[int] = []

    for row in detail_rows:
        name = row.get("Program", "unknown")
        ps = program_map.get(name)
        if ps is None:
            ps = ProgramStats(program=name)
            program_map[name] = ps

        err = row.get("ErrorType", "Unknown")
        if err and err != "None":
            error_counter[err] += 1
            ps.error_types.append(err)

        ps.round_count += 1

        # Collect timing columns if present (extended logging)
        for col, target in [
            ("TranslationTime", ps.translation_times),
            ("CompileTime", ps.compile_times),
            ("RuntimeTime", ps.runtime_times),
            ("ValidationTime", ps.validation_times),
            ("RepairTime", ps.repair_times),
        ]:
            if col in row:
                val = _parse_float(row[col])
                if val > 0:
                    target.append(val)

        # Test count columns if present
        if "GeneratedTestCount" in row:
            ps.test_counts.append(_parse_int(row["GeneratedTestCount"]))
        if "PassedTestCount" in row:
            ps.passed_test_counts.append(_parse_int(row["PassedTestCount"]))

    # Collect repair rounds
    for ps in program_map.values():
        if ps.repair_rounds > 0 or ps.round_count > 0:
            repair_counts.append(ps.repair_rounds)

    per_program = list(program_map.values())

    # ---- Compute aggregate stats -------------------------------------------
    agg = AggregateStats()
    agg.total_programs = len(per_program)

    if agg.total_programs > 0:
        agg.compile_success_rate = (
            sum(1 for p in per_program if p.final_compile_pass) / agg.total_programs
        )
        agg.runtime_success_rate = (
            sum(1 for p in per_program if p.runtime_pass) / agg.total_programs
        )
        agg.functional_success_rate = (
            sum(1 for p in per_program if p.functional_pass) / agg.total_programs
        )
        agg.overall_success_rate = (
            sum(
                1
                for p in per_program
                if p.final_compile_pass and p.runtime_pass and p.functional_pass
            )
            / agg.total_programs
        )

    all_times = [p.total_time for p in per_program if p.total_time > 0]
    if all_times:
        agg.avg_runtime = sum(all_times) / len(all_times)

    if repair_counts:
        agg.avg_repair_count = sum(repair_counts) / len(repair_counts)

    agg.repair_distribution = dict(Counter(repair_counts))
    agg.error_distribution = dict(error_counter)

    all_trans_times: list[float] = []
    all_val_times: list[float] = []
    for ps in per_program:
        all_trans_times.extend(ps.translation_times)
        all_val_times.extend(ps.validation_times)
    if all_trans_times:
        agg.avg_translation_time = sum(all_trans_times) / len(all_trans_times)
    if all_val_times:
        agg.avg_validation_time = sum(all_val_times) / len(all_val_times)

    all_test_counts: list[int] = []
    for ps in per_program:
        all_test_counts.extend(ps.test_counts)
    if all_test_counts:
        agg.avg_generated_test_count = sum(all_test_counts) / len(all_test_counts)

    return agg, per_program


# ======================================================================
# Report writers
# ======================================================================

def _write_csv_report(agg: AggregateStats, per_program: List[ProgramStats], path: str) -> None:
    """Write ``analysis_report.csv``."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)

        # Section 1: aggregate
        w.writerow(["Metric", "Value"])
        w.writerow(["TotalPrograms", agg.total_programs])
        w.writerow(["CompileSuccessRate", f"{agg.compile_success_rate:.4f}"])
        w.writerow(["RuntimeSuccessRate", f"{agg.runtime_success_rate:.4f}"])
        w.writerow(["FunctionalSuccessRate", f"{agg.functional_success_rate:.4f}"])
        w.writerow(["OverallTranslationSuccessRate", f"{agg.overall_success_rate:.4f}"])
        w.writerow(["AverageTranslationTime_s", f"{agg.avg_translation_time:.3f}"])
        w.writerow(["AverageValidationTime_s", f"{agg.avg_validation_time:.3f}"])
        w.writerow(["AverageTotalTime_s", f"{agg.avg_runtime:.3f}"])
        w.writerow(["AverageRepairCount", f"{agg.avg_repair_count:.2f}"])
        w.writerow(["AverageGeneratedTestCount", f"{agg.avg_generated_test_count:.1f}"])
        w.writerow([])

        # Section 2: error distribution
        w.writerow(["ErrorType", "Count"])
        for err, cnt in sorted(agg.error_distribution.items(), key=lambda x: -x[1]):
            w.writerow([err, cnt])
        w.writerow([])

        # Section 3: repair distribution
        w.writerow(["RepairRounds", "ProgramCount"])
        for rounds, cnt in sorted(agg.repair_distribution.items()):
            w.writerow([rounds, cnt])
        w.writerow([])

        # Section 4: per-program
        w.writerow([
            "Program", "InitialCompile", "FinalCompile", "Runtime",
            "Functional", "RepairRounds", "TotalTime_s",
        ])
        for ps in per_program:
            w.writerow([
                ps.program,
                ps.initial_compile_pass,
                ps.final_compile_pass,
                ps.runtime_pass,
                ps.functional_pass,
                ps.repair_rounds,
                f"{ps.total_time:.2f}",
            ])


def _write_txt_report(agg: AggregateStats, per_program: List[ProgramStats], path: str) -> None:
    """Write ``analysis_report.txt``."""
    lines: list[str] = []
    sep = "=" * 68

    lines.append(sep)
    lines.append("AUTOMATIC EXPERIMENT ANALYSIS REPORT")
    lines.append("C++ → Python Translation & Evaluation Framework")
    lines.append(sep)
    lines.append("")
    lines.append(f"Generated: {_timestamp()}")
    lines.append(f"Programs evaluated: {agg.total_programs}")
    lines.append("")

    # -- Success rates -------------------------------------------------------
    lines.append("-" * 68)
    lines.append("SUCCESS RATES")
    lines.append("-" * 68)
    lines.append(f"  Compile Success Rate:          {agg.compile_success_rate:.2%}")
    lines.append(f"  Runtime Success Rate:          {agg.runtime_success_rate:.2%}")
    lines.append(f"  Functional Success Rate:       {agg.functional_success_rate:.2%}")
    lines.append(f"  Overall Translation Success:   {agg.overall_success_rate:.2%}")
    lines.append("")

    # -- Timing --------------------------------------------------------------
    lines.append("-" * 68)
    lines.append("TIMING (averages)")
    lines.append("-" * 68)
    lines.append(f"  Average Translation Time:      {agg.avg_translation_time:.3f}s")
    lines.append(f"  Average Validation Time:       {agg.avg_validation_time:.3f}s")
    lines.append(f"  Average Total Experiment Time: {agg.avg_runtime:.3f}s")
    lines.append("")

    # -- Repair --------------------------------------------------------------
    lines.append("-" * 68)
    lines.append("REPAIR STATISTICS")
    lines.append("-" * 68)
    lines.append(f"  Average Repair Rounds:         {agg.avg_repair_count:.2f}")
    lines.append("  Repair Round Distribution:")
    for rounds, cnt in sorted(agg.repair_distribution.items()):
        pct = cnt / agg.total_programs * 100 if agg.total_programs else 0
        bar = "█" * int(pct / 5) if pct > 0 else ""
        lines.append(f"    {rounds} round(s): {cnt} program(s) ({pct:.1f}%) {bar}")
    lines.append("")

    # -- Error distribution --------------------------------------------------
    lines.append("-" * 68)
    lines.append("ERROR TYPE DISTRIBUTION")
    lines.append("-" * 68)
    if agg.error_distribution:
        total_errors = sum(agg.error_distribution.values())
        for err, cnt in sorted(agg.error_distribution.items(), key=lambda x: -x[1]):
            pct = cnt / total_errors * 100 if total_errors else 0
            bar = "█" * int(pct / 5) if pct > 0 else ""
            lines.append(f"  {err:30s} {cnt:4d} ({pct:5.1f}%) {bar}")
    else:
        lines.append("  (no errors recorded)")
    lines.append("")

    # -- Per-program ---------------------------------------------------------
    lines.append("-" * 68)
    lines.append("PER-PROGRAM SUMMARY")
    lines.append("-" * 68)
    for ps in sorted(per_program, key=lambda p: p.program):
        status = "✅" if (ps.final_compile_pass and ps.runtime_pass and ps.functional_pass) else "❌"
        lines.append(
            f"  {status} {ps.program:35s} "
            f"compile={ps.final_compile_pass}  runtime={ps.runtime_pass}  "
            f"functional={ps.functional_pass}  repairs={ps.repair_rounds}  "
            f"time={ps.total_time:.1f}s"
        )
    lines.append("")

    # -- Test generation -----------------------------------------------------
    if agg.avg_generated_test_count > 0:
        lines.append("-" * 68)
        lines.append("TEST GENERATION")
        lines.append("-" * 68)
        lines.append(f"  Average Generated Test Count:  {agg.avg_generated_test_count:.1f}")
        lines.append("")

    lines.append(sep)
    lines.append("END OF REPORT")
    lines.append(sep)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ======================================================================
# Utility
# ======================================================================

def _timestamp() -> str:
    """Return a human-readable UTC timestamp."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ======================================================================
# Entry point
# ======================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for the analysis module.

    Supports ``--csv``, ``--summary``, and ``--output-dir`` flags.
    """
    args = argv if argv is not None else sys.argv[1:]

    project_root = os.path.dirname(os.path.abspath(__file__))
    detail_csv = os.path.join(project_root, "experiment_results.csv")
    summary_csv = os.path.join(project_root, "summary_results.csv")
    output_dir = project_root

    # Minimal CLI parsing (no argparse dependency for portability)
    i = 0
    while i < len(args):
        if args[i] == "--csv" and i + 1 < len(args):
            detail_csv = args[i + 1]; i += 2
        elif args[i] == "--summary" and i + 1 < len(args):
            summary_csv = args[i + 1]; i += 2
        elif args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]; i += 2
        else:
            i += 1

    if not os.path.isfile(detail_csv) and not os.path.isfile(summary_csv):
        print(
            "No experiment CSVs found.  Run the translation pipeline "
            "(python3 run.py) first to generate data.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading: {detail_csv}")
    print(f"Reading: {summary_csv}")

    agg, per_program = analyse(detail_csv, summary_csv)

    csv_path = os.path.join(output_dir, "analysis_report.csv")
    txt_path = os.path.join(output_dir, "analysis_report.txt")

    _write_csv_report(agg, per_program, csv_path)
    _write_txt_report(agg, per_program, txt_path)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {txt_path}")
    print(f"\n  Programs: {agg.total_programs}")
    print(f"  Compile success:  {agg.compile_success_rate:.2%}")
    print(f"  Runtime success:  {agg.runtime_success_rate:.2%}")
    print(f"  Functional success: {agg.functional_success_rate:.2%}")
    print(f"  Overall success:  {agg.overall_success_rate:.2%}")
    print(f"  Avg repair count: {agg.avg_repair_count:.2f}")
    print(f"  Avg total time:   {agg.avg_runtime:.2f}s")


if __name__ == "__main__":
    main()
