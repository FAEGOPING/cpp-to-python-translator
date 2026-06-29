"""
experiment_runner.py — Automated Translation Experiment Runner
===============================================================

Connects the Dataset Manager with the Translation Framework to
automate large-scale C++ → Python translation experiments.

Workflow:
    Benchmark Dataset → Translation → Compile Check → Repair →
    Differential Validation → Statistics → CSV Results

Usage::

    python experiment_runner.py                        # all programs
    python experiment_runner.py --limit 10             # first 10 programs
    python experiment_runner.py --programs 1,5,10,42   # specific programs
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional

# Ensure project root is on path
_DEPS = os.path.dirname(os.path.abspath(__file__))
if _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

from config import Config, DEFAULT_CONFIG
from run import (
    set_config, get_config,
    process_program,
    load_test_cases,
)
from run import _cleanup_cpp_cache
from dataset_manager.utils import (
    BENCHMARK_DIR,
    REPORTS_DIR,
    Logger,
    write_csv,
    read_csv,
    timestamp,
)


def _parse_args(argv: Optional[List[str]] = None) -> tuple[int, List[int]]:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        ``(limit, specific_program_ids)``.
    """
    args = argv if argv is not None else sys.argv[1:]
    limit: int = 0  # 0 = all
    programs: list[int] = []

    i = 0
    while i < len(args):
        if args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif args[i] == "--programs" and i + 1 < len(args):
            programs = [int(x.strip()) for x in args[i + 1].split(",") if x.strip()]
            i += 2
        else:
            i += 1

    return limit, programs


def _prepare_samples_dir(cfg: Config, program_files: List[str]) -> str:
    """Copy benchmark programs into the samples directory temporarily.

    Creates ``.in`` files with empty input for programs that have no
    test cases, so the translation pipeline can run without manual
    test setup.

    Args:
        cfg: Active :class:`Config`.
        program_files: List of benchmark program filenames.

    Returns:
        Path to the samples directory.
    """
    import shutil
    samples = cfg.samples_dir

    # Ensure samples dir exists
    os.makedirs(samples, exist_ok=True)

    # Clean previous experiment programs (keep original example.*)
    for f in os.listdir(samples):
        if f.startswith("program_") and (f.endswith(".cpp") or f.endswith(".in")):
            os.remove(os.path.join(samples, f))

    # Copy programs
    for fname in program_files:
        src = os.path.join(BENCHMARK_DIR, fname)
        dst = os.path.join(samples, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    return samples


def _collect_benchmark_programs(limit: int, specific: List[int]) -> List[str]:
    """Collect benchmark program filenames to run.

    Args:
        limit: Max number of programs (0 = all).
        specific: Specific program IDs (overrides limit).

    Returns:
        Sorted list of ``program_NNNNNN.cpp`` filenames.
    """
    all_files = sorted(
        f for f in os.listdir(BENCHMARK_DIR)
        if f.startswith("program_") and f.endswith(".cpp")
    )

    if specific:
        result = []
        for pid in specific:
            fname = f"program_{pid:06d}.cpp"
            if fname in all_files:
                result.append(fname)
        return result

    if limit and limit > 0:
        return all_files[:limit]

    return all_files


def _generate_experiment_summary(logger: Logger) -> List[List]:
    """Generate a high-level experiment summary from CSV results.

    Args:
        logger: :class:`Logger` instance.

    Returns:
        List of rows for the summary CSV.
    """
    # Read translation framework's summary CSV
    summary_path = get_config().summary_csv
    detail_path = get_config().csv_file

    summary_rows = read_csv(summary_path)
    detail_rows = read_csv(detail_path)

    total = len(summary_rows)
    compile_pass = sum(1 for r in summary_rows if r.get("FunctionalPass", "").lower() == "true")
    runtime_pass = sum(1 for r in summary_rows if r.get("RuntimePass", "").lower() == "true")
    functional_pass = sum(1 for r in summary_rows if r.get("FunctionalPass", "").lower() == "true")

    repair_rounds = []
    total_times = []
    for r in summary_rows:
        rr = r.get("RepairRounds", "0")
        tt = r.get("TotalTime", "0")
        try:
            repair_rounds.append(float(rr))
        except (ValueError, TypeError):
            pass
        try:
            total_times.append(float(tt))
        except (ValueError, TypeError):
            pass

    avg_repair = sum(repair_rounds) / max(len(repair_rounds), 1)
    avg_time = sum(total_times) / max(len(total_times), 1)

    # Repository stats (if available)
    from dataset_manager.utils import REPORTS_DIR as DM_REPORTS
    repo_stats_path = os.path.join(DM_REPORTS, "repository_statistics.csv")
    repo_rows = read_csv(repo_stats_path)
    n_repos = len(repo_rows)

    # Benchmark stats
    bench_meta = os.path.join(BENCHMARK_DIR, "metadata.csv")
    bench_rows = read_csv(bench_meta)
    n_bench = len(bench_rows)

    return [
        ["TotalRepositories", str(n_repos)],
        ["BenchmarkPrograms", str(n_bench)],
        ["ProgramsTranslated", str(total)],
        ["CompileSuccessRate", f"{compile_pass / max(total, 1) * 100:.2f}%"],
        ["RuntimeSuccessRate", f"{runtime_pass / max(total, 1) * 100:.2f}%"],
        ["FunctionalSuccessRate", f"{functional_pass / max(total, 1) * 100:.2f}%"],
        ["AverageRepairRounds", f"{avg_repair:.2f}"],
        ["AverageTranslationTimeSeconds", f"{avg_time:.2f}"],
    ]


# ============================================================================
# Main entry point
# ============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """Run the translation experiment on benchmark dataset programs.

    Args:
        argv: CLI argument list (defaults to ``sys.argv[1:]``).
    """
    limit, specific = _parse_args(argv)
    logger = Logger("experiment_runner")

    # Verify benchmark dataset exists
    bench_files = _collect_benchmark_programs(limit, specific)
    if not bench_files:
        logger.error("No benchmark programs found. Run build_dataset first.")
        sys.exit(1)

    logger.info(f"Experiment Runner")
    logger.info(f"  Benchmark programs: {len(bench_files)}")
    logger.info(f"  Limit:              {limit or 'all'}")
    logger.info(f"  Started:            {timestamp()}")

    # Configure translation framework to use benchmark programs
    cfg = get_config()
    _prepare_samples_dir(cfg, bench_files)

    logger.info(f"  Samples dir:        {cfg.samples_dir}")
    logger.info(f"  Translated dir:     {cfg.translated_dir}")
    logger.info(f"  Max repair rounds:  {cfg.max_repair_rounds}")

    # Run the translation pipeline
    logger.info("Starting translation experiments …")
    t0 = time.time()

    try:
        from run import main as run_main
        run_main()
    except Exception as exc:
        logger.error(f"Translation pipeline error: {exc}")
    finally:
        _cleanup_cpp_cache()

    elapsed = time.time() - t0
    logger.info(f"Experiment completed in {elapsed:.1f}s")
    logger.count("total_time_seconds", int(elapsed))
    logger.count("programs_processed", len(bench_files))

    # Generate experiment summary
    logger.info("Generating experiment summary …")
    summary_rows = _generate_experiment_summary(logger)
    summary_path = os.path.join(REPORTS_DIR, "experiment_summary.csv")
    write_csv(summary_path, ["Metric", "Value"], summary_rows)
    logger.info(f"Summary written: {summary_path}")

    print(f"\n{logger.summary()}")


if __name__ == "__main__":
    main()
