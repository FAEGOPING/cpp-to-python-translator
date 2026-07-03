"""
paper_archive.py — Automatic Paper Archive System
===================================================

Automatically organises every completed experiment into a paper-ready
directory structure under ``paper_results/EXP_<limit>/``.

The archive is created **after** report generation and is designed to
never interrupt the experiment pipeline — all file operations are
wrapped in individual try/except blocks, and missing files produce
warnings rather than errors.

Usage::

    from paper_archive import archive_paper_results

    archive_paper_results(run_dir, limit, logger)

Directory layout created::

    paper_results/
        EXP_<limit>/
            csv/            — experiment CSV files
            reports/        — generated Markdown report
            config/         — experiment configuration
            figures/        — publication-quality figures
            translated/     — translated Python programs
            README.txt      — experiment summary
"""

from __future__ import annotations

import os
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
PAPER_ROOT = PROJECT_ROOT / "paper_results"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiment_results"
TRANSLATED_DIR = PROJECT_ROOT / "translated"
FIGURES_SRC = PROJECT_ROOT / "dataset_manager" / "reports" / "figures"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def archive_paper_results(
    run_dir: str,
    limit: str,
    logger: Optional[object] = None,
) -> Optional[Path]:
    """Create a paper-ready archive of an experiment.

    Copies CSVs, reports, config, figures, and translated programs into
    ``paper_results/EXP_<limit>/`` and generates a ``README.txt`` summary.

    Args:
        run_dir: Absolute path to the experiment output directory
                 (e.g. ``experiment_results/2026-07-02_15-40-26/``).
        limit: The ``--limit`` value passed to the experiment runner
               (e.g. ``"20"``, ``"100"``, ``"all"``).
        logger: Optional Logger instance for structured output.
                If ``None``, messages are printed to stdout.

    Returns:
        Path to the archive directory, or ``None`` if the archive could
        not be created.
    """
    # ---- helpers ----
    def _log(msg: str) -> None:
        if logger and hasattr(logger, "info"):
            logger.info(msg)
        else:
            print(msg)

    def _warn(msg: str) -> None:
        if logger and hasattr(logger, "warn"):
            logger.warn(msg)
        else:
            print(f"  [WARN] {msg}")

    # ---- determine archive name ----
    limit_normalised = str(limit).strip().lower()
    if limit_normalised in ("all", "none", ""):
        archive_name = "EXP_ALL"
    else:
        archive_name = f"EXP_{limit}"

    archive_dir = PAPER_ROOT / archive_name

    _log("")
    _log("-" * 70)
    _log("PAPER ARCHIVE — Creating paper-ready experiment archive")
    _log("-" * 70)
    _log(f"  Archive:  {archive_dir}")

    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _warn(f"Could not create archive directory: {exc}")
        return None

    # ---- subdirectories ----
    subdirs = ["csv", "reports", "config", "figures", "translated"]
    for sub in subdirs:
        try:
            (archive_dir / sub).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _warn(f"Could not create {sub}/: {exc}")

    src_run = Path(run_dir)

    # ---- copy CSVs ----
    csv_src = src_run / "csv"
    csv_dst = archive_dir / "csv"
    csv_count = 0
    if csv_src.is_dir():
        for f in csv_src.glob("*.csv"):
            try:
                shutil.copy2(f, csv_dst / f.name)
                csv_count += 1
            except (OSError, shutil.SameFileError) as exc:
                _warn(f"Could not copy {f.name}: {exc}")
    if csv_count == 0:
        _warn("No CSV files found to archive")
    else:
        _log(f"  Copied {csv_count} CSV file(s)")

    # ---- copy report ----
    report_src = src_run / "reports" / "report.md"
    report_dst = archive_dir / "reports" / "report.md"
    if report_src.is_file():
        try:
            shutil.copy2(report_src, report_dst)
            _log("  Copied report.md")
        except (OSError, shutil.SameFileError) as exc:
            _warn(f"Could not copy report.md: {exc}")
    else:
        _warn("report.md not found — skipping")

    # ---- copy config ----
    config_src = src_run / "config"
    config_dst = archive_dir / "config"
    config_count = 0
    if config_src.is_dir():
        for f in config_src.iterdir():
            if f.is_file():
                try:
                    shutil.copy2(f, config_dst / f.name)
                    config_count += 1
                except (OSError, shutil.SameFileError) as exc:
                    _warn(f"Could not copy {f.name}: {exc}")
    if config_count == 0:
        _warn("No config files found to archive")
    else:
        _log(f"  Copied {config_count} config file(s)")

    # ---- copy figures ----
    fig_dst = archive_dir / "figures"
    fig_count = 0
    if FIGURES_SRC.is_dir():
        for f in FIGURES_SRC.iterdir():
            if f.is_file():
                try:
                    shutil.copy2(f, fig_dst / f.name)
                    fig_count += 1
                except (OSError, shutil.SameFileError) as exc:
                    _warn(f"Could not copy figure {f.name}: {exc}")
    if fig_count == 0:
        _warn("No figures found to archive")
    else:
        _log(f"  Copied {fig_count} figure(s)")

    # ---- copy translated programs ----
    trans_dst = archive_dir / "translated"
    py_count = 0
    if TRANSLATED_DIR.is_dir():
        for f in TRANSLATED_DIR.glob("*.py"):
            try:
                shutil.copy2(f, trans_dst / f.name)
                py_count += 1
            except (OSError, shutil.SameFileError) as exc:
                _warn(f"Could not copy {f.name}: {exc}")
    if py_count == 0:
        _warn("No translated Python files found to archive")
    else:
        _log(f"  Copied {py_count} translated Python file(s)")

    # ---- generate README ----
    _log("  Generating README.txt …")
    try:
        _generate_readme(archive_dir, run_dir, limit)
        _log("  README.txt written")
    except Exception as exc:
        _warn(f"Could not generate README.txt: {exc}")

    # ---- final banner ----
    _print_summary(archive_name)

    return archive_dir


# ---------------------------------------------------------------------------
# README generation
# ---------------------------------------------------------------------------

def _generate_readme(archive_dir: Path, run_dir: str, limit: str) -> None:
    """Generate ``README.txt`` inside the archive directory.

    Pulls experiment metadata from the config JSON and statistics from
    :func:`statistics.load_stats`.  Missing values are rendered as ``N/A``.
    """
    readme_path = archive_dir / "README.txt"

    # --- load experiment config ---
    config = {}
    config_path = Path(run_dir) / "config" / "experiment_configuration.json"
    if config_path.is_file():
        import json
        try:
            with open(config_path) as fh:
                config = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass

    # --- load statistics ---
    stats = None
    try:
        from statistics import load_stats
        stats = load_stats(run_dir)
    except Exception:
        pass

    # --- extract values ---
    exp_id = config.get("experiment_id", os.path.basename(run_dir))
    timestamp = config.get("timestamp", "N/A")

    # Split timestamp into date/time if possible
    if timestamp != "N/A" and " " in timestamp:
        date_str, time_str = timestamp.split(" ", 1)
    else:
        date_str = timestamp
        time_str = "N/A"

    dataset_size = _nvl(config.get("selection", {}).get("total_available"))
    final_count = _nvl(config.get("selection", {}).get("final_count"))
    limit_val = limit if str(limit).strip().lower() != "all" else "ALL"
    repair_enabled = _nvl(config.get("cli_arguments", {}).get("repair"))
    compiler_name = "N/A"
    compiler_info = config.get("compiler", {})
    if isinstance(compiler_info, dict):
        compiler_name = compiler_info.get("name", "N/A")
    python_ver = config.get("python_version", "N/A")
    git_commit = config.get("git_commit", "N/A")
    if git_commit != "N/A" and len(git_commit) > 8:
        git_commit = git_commit[:8]  # short hash

    # Statistics fields (with fallback)
    compile_rate = _fmt_pct(_safe_attr(stats, "compile_success_rate"))
    runtime_rate = _fmt_pct(_safe_attr(stats, "runtime_success_rate"))
    functional_rate = _fmt_pct(_safe_attr(stats, "functional_success_rate"))
    avg_trans = _fmt_sec(_safe_attr(stats, "avg_translation_time"))
    avg_compile = _fmt_sec(_safe_attr(stats, "avg_compile_time"))
    avg_runtime = _fmt_sec(_safe_attr(stats, "avg_runtime_time"))
    avg_valid = _fmt_sec(_safe_attr(stats, "avg_validation_time"))
    avg_repair_time = _fmt_sec(_safe_attr(stats, "avg_repair_time"))
    avg_repair_rounds = _nvl(_safe_attr(stats, "avg_repair_rounds"))

    # ---- write ----
    lines = textwrap.dedent(f"""\
    ================================================================================
    EXPERIMENT ARCHIVE — {archive_dir.name}
    ================================================================================

    Experiment ID:         {exp_id}
    Date:                  {date_str}
    Time:                  {time_str}
    Dataset Size:          {dataset_size}
    Limit:                 {limit_val}
    Programs Run:          {final_count}

    Compile Success Rate:  {compile_rate}
    Runtime Success Rate:  {runtime_rate}
    Functional Success Rate:{functional_rate}

    Repair Enabled:        {repair_enabled}

    Average Translation Time:  {avg_trans}
    Average Compile Time:      {avg_compile}
    Average Runtime:           {avg_runtime}
    Average Validation Time:   {avg_valid}
    Average Repair Time:       {avg_repair_time}
    Average Repair Rounds:     {avg_repair_rounds}

    Compiler:              {compiler_name}
    Python Version:        {python_ver}
    Git Commit:            {git_commit}
    Output Directory:      paper_results/{archive_dir.name}/

    ================================================================================
    Generated by paper_archive.py on {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
    ================================================================================
    """)

    with open(readme_path, "w") as fh:
        fh.write(lines)


# ---------------------------------------------------------------------------
# Summary banner
# ---------------------------------------------------------------------------

def _print_summary(archive_name: str) -> None:
    """Print a console summary banner at the end of the experiment."""
    banner = textwrap.dedent(f"""\

    {'=' * 70}
    Experiment finished successfully.

    Paper archive created:

        paper_results/{archive_name}/

    Archive contains:

        CSV             — experiment results data
        Reports         — generated Markdown report
        Configuration   — experiment parameters
        Figures         — publication-quality figures
        Translated      — translated Python programs
        README          — experiment summary

    {'=' * 70}
    """)
    print(banner)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_attr(obj: object, attr: str) -> object:
    """Return *obj.attr* or ``None`` if the attribute does not exist."""
    if obj is None:
        return None
    return getattr(obj, attr, None)


def _nvl(value: object) -> str:
    """Return *value* as a string, or ``"N/A"`` if ``None``."""
    if value is None:
        return "N/A"
    return str(value)


def _fmt_pct(value: object) -> str:
    """Format a float as a percentage string, or ``"N/A"``."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.1f}%"
    except (ValueError, TypeError):
        return "N/A"


def _fmt_sec(value: object) -> str:
    """Format a float as seconds with one decimal, or ``"N/A"``."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.1f}s"
    except (ValueError, TypeError):
        return "N/A"
