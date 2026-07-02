"""
experiment_runner.py — Configurable Research Experiment Runner
================================================================

The single entry point for all C++ → Python translation experiments.
Every experiment is controlled via CLI arguments or YAML configuration
files — no Python source editing required.

Usage::

    python experiment_runner.py --limit 20
    python experiment_runner.py --limit 20 --repair --runtime
    python experiment_runner.py --repository algorithms --limit 50
    python experiment_runner.py --category graph --limit 100
    python experiment_runner.py --min-loc 50 --max-loc 200 --repair
    python experiment_runner.py --random 100 --seed 42
    python experiment_runner.py --config experiments/pilot.yaml
    python experiment_runner.py --resume

Every run creates a timestamped directory under
``experiment_results/<YYYY-MM-DD_HH-MM-SS>/`` containing CSVs, figures,
reports, logs, and a copy of the experiment configuration.

Existing modules (benchmark.py, run.py, dataset_manager/) are NOT
modified — the experiment runner only orchestrates them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random as _random
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on path
_DEPS = os.path.dirname(os.path.abspath(__file__))
if _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

from config import Config, DEFAULT_CONFIG
from run import set_config as _set_run_config, get_config as _get_run_config
from dataset_manager.utils import (
    BENCHMARK_DIR, REPORTS_DIR, FIGURES_DIR, DATASETS_DIR,
    Logger, read_csv, write_csv, timestamp as _ts, get_compiler,
)

# ============================================================================
# Directory constants
# ============================================================================

EXPERIMENT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "experiment_results")
"""All experiment outputs are stored under this directory."""


def _now_dirname() -> str:
    """Timestamped directory name for this experiment run."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


# ============================================================================
# CLI — argparse
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with full help text and examples."""
    p = argparse.ArgumentParser(
        prog="experiment_runner",
        description="Configurable C++ → Python Translation Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # 20-program pilot study
              python experiment_runner.py --limit 20

              # 20-program pilot with repair and runtime verification
              python experiment_runner.py --limit 20 --repair --runtime

              # All programs from a single repository
              python experiment_runner.py --repository algorithms --limit all

              # 50 programs from cses
              python experiment_runner.py --repository cses --limit 50

              # 100 graph-algorithm programs
              python experiment_runner.py --category graph --limit 100

              # Medium-sized programs with repair
              python experiment_runner.py --min-loc 50 --max-loc 200 --repair

              # Reproducible random sample
              python experiment_runner.py --random 100 --seed 42

              # Full benchmark (all programs, repair, runtime)
              python experiment_runner.py --limit all --repair --runtime

              # Using a YAML configuration file
              python experiment_runner.py --config experiments/pilot.yaml

              # Resume an interrupted experiment
              python experiment_runner.py --resume
            """),
    )

    # -- General ------------------------------------------------------------
    general = p.add_argument_group("General")
    general.add_argument("--limit", type=str, default="all",
                         help="Number of programs to run (e.g. 20, 'all')")
    general.add_argument("--config", type=str, default=None,
                         help="Path to YAML/JSON configuration file")
    general.add_argument("--verbose", action="store_true",
                         help="Enable detailed per-program logging")
    general.add_argument("--resume", action="store_true",
                         help="Resume from the last unfinished experiment")
    general.add_argument("--output-dir", type=str, default=None,
                         help="Custom output directory (overrides timestamped dir)")

    # -- Selection ----------------------------------------------------------
    selection = p.add_argument_group("Program Selection")
    selection.add_argument("--repository", type=str, action="append", default=None,
                           help="Filter by repository (repeatable)")
    selection.add_argument("--category", type=str, action="append", default=None,
                           help="Filter by algorithm category (repeatable)")
    selection.add_argument("--min-loc", type=int, default=None,
                           help="Minimum lines of code")
    selection.add_argument("--max-loc", type=int, default=None,
                           help="Maximum lines of code")
    selection.add_argument("--random", dest="random_n", type=int, default=None,
                           help="Randomly sample N programs")
    selection.add_argument("--seed", type=int, default=42,
                           help="Random seed for reproducible sampling")

    # -- Experiment features ------------------------------------------------
    features = p.add_argument_group("Experiment Features")
    features.add_argument("--repair", action="store_true",
                          help="Enable iterative LLM repair")
    features.add_argument("--no-repair", action="store_true",
                          help="Disable repair (for baseline experiments)")
    features.add_argument("--runtime", action="store_true",
                          help="Enable C++ vs Python runtime output comparison")
    features.add_argument("--max-repair-rounds", type=int, default=None,
                          help="Maximum repair iterations (default: 5)")

    # -- Reports -------------------------------------------------------------
    reports = p.add_argument_group("Reports & Output")
    reports.add_argument("--no-figures", action="store_true",
                         help="Skip automatic figure generation")
    reports.add_argument("--no-report", action="store_true",
                         help="Skip automatic report generation")

    return p


# ============================================================================
# YAML config support (falls back to JSON if PyYAML unavailable)
# ============================================================================

def _load_config_file(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON configuration file.

    Args:
        path: Path to ``.yaml``, ``.yml``, or ``.json`` file.

    Returns:
        Merged config dict.
    """
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except ImportError:
            sys.exit("PyYAML is required for YAML config files. "
                     "Install with: pip install pyyaml")
    elif path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    else:
        # Try YAML first, then JSON
        try:
            import yaml
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    return {}


def _apply_yaml_config(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Overlay YAML config values onto argparse namespace.

    CLI arguments always take precedence over YAML values.
    Only applies a YAML key when the corresponding CLI arg is at
    its default (not explicitly passed).

    Args:
        args: Parsed argparse namespace (mutated in place).
        config: Dict loaded from YAML/JSON file.
    """
    # Dataset section
    ds = config.get("dataset", {})
    _set_if_default(args, "repository", ds.get("repository"))
    _set_if_default(args, "limit", ds.get("limit"))
    if ds.get("random") and args.random_n is None:
        args.random_n = ds.get("random_n", ds.get("count", 100))
    if ds.get("seed") and args.seed == 42:
        args.seed = ds.get("seed")

    # Filters section
    flt = config.get("filters", {})
    _set_if_default(args, "min_loc", flt.get("min_loc"))
    _set_if_default(args, "max_loc", flt.get("max_loc"))
    _set_if_default(args, "category", flt.get("category"))
    _set_if_default(args, "repository", flt.get("repository"))

    # Translation section
    trans = config.get("translation", {})
    if trans.get("enabled") is False:
        pass  # translation always runs; skip-translation handled by benchmark stage

    # Repair section
    repair = config.get("repair", {})
    if repair.get("enabled") is True and not args.repair:
        args.repair = True
    if repair.get("enabled") is False and not args.no_repair:
        args.no_repair = True
    _set_if_default(args, "max_repair_rounds", repair.get("max_iterations"))

    # Runtime section
    runtime = config.get("runtime", {})
    if runtime.get("enabled") is True and not args.runtime:
        args.runtime = True

    # Report section
    report = config.get("report", {})
    if report.get("generate_figures") is False:
        args.no_figures = True


def _set_if_default(args: argparse.Namespace, attr: str, value: Any) -> None:
    """Set *attr* on *args* only if the YAML *value* is not None
    and the current CLI value is at its default.

    For list-typed args (repository, category), wrap scalar values.
    """
    if value is None:
        return
    current = getattr(args, attr, None)
    if current is not None and current != argparse.SUPPRESS:
        # For lists (--repository / --category), only set from YAML if not
        # already specified via CLI
        if isinstance(current, list) and len(current) > 0:
            return
        # For non-list defaults, only set if CLI hasn't changed them
    if isinstance(value, list):
        setattr(args, attr, value)
    elif attr in ("repository", "category"):
        if isinstance(value, str):
            setattr(args, attr, [value])
        elif isinstance(value, list):
            setattr(args, attr, value)
    else:
        setattr(args, attr, value)


# ============================================================================
# Program selection engine
# ============================================================================

# Algorithm category keywords (same as research_analytics.py)
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "graph": ["graph", "dijkstra", "bfs", "dfs", "mst", "topological", "euler",
              "floyd", "bellman", "kruskal", "prim", "tarjan", "kosaraju"],
    "tree": ["tree", "bst", "trie", "avl", "segment", "fenwick", "lca",
             "binary_tree", "heap"],
    "dp": ["dp", "dynamic", "knapsack", "lcs", "lis", "edit_distance",
           "matrix_chain", "coin_change"],
    "greedy": ["greedy", "huffman", "activity_selection"],
    "math": ["math", "prime", "gcd", "lcm", "modular", "factorial", "fibonacci",
             "number_theory", "combinatorics", "probability", "numerical"],
    "search": ["binary_search", "ternary_search", "bisection", "search"],
    "sorting": ["sort", "merge_sort", "quick_sort", "heap_sort", "bubble",
                "insertion_sort", "selection_sort", "counting_sort", "radix"],
    "string": ["string", "kmp", "z_algorithm", "suffix", "trie", "aho_corasick",
               "palindrome", "manacher", "hashing", "rabin_karp"],
    "backtracking": ["backtrack", "n_queen", "sudoku", "permutation"],
    "geometry": ["geometry", "convex_hull", "closest_pair", "point"],
    "bit": ["bit", "bitmask", "xor", "bitset", "bit_manipulation"],
    "datastructures": ["stack", "queue", "linked_list", "deque",
                       "priority_queue", "hash", "data_structures", "list", "array"],
    "simulation": ["simulation", "simulate"],
}


def _classify_category(file_path: str) -> str:
    """Classify a file into an algorithm category by path keywords.

    Args:
        file_path: Relative file path (e.g. from metadata.csv).

    Returns:
        Category name string.
    """
    lower = file_path.lower()
    scores: dict[str, int] = {}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > 0:
            scores[cat] = score
    if scores:
        return max(scores, key=lambda k: scores[k])
    return "other"


def _select_programs(args: argparse.Namespace, logger: Logger) -> Tuple[List[str], Dict[str, Any]]:
    """Select benchmark programs matching the given filters.

    Applies filters in order: repository → category → LOC → random.
    Each filter reduces the candidate set.

    Args:
        args: Parsed arguments.
        logger: :class:`Logger` instance.

    Returns:
        ``(program_filenames, selection_metadata)`` where
        *selection_metadata* records the filters applied and counts.
    """
    # Load benchmark metadata
    bench_meta_path = os.path.join(BENCHMARK_DIR, "metadata.csv")
    all_meta = read_csv(bench_meta_path)

    # Load source mapping for repository info
    mapping_path = os.path.join(REPORTS_DIR, "source_mapping.csv")
    mapping = read_csv(mapping_path)

    # Build a dict: program filename → metadata
    prog_meta: dict[str, dict] = {}
    for r in all_meta:
        fname = r.get("Filename", r.get("ProgramID", ""))
        if fname and fname.endswith(".cpp"):
            prog_meta[fname] = r

    # Build a dict: program filename → source mapping
    prog_map: dict[str, dict] = {}
    for r in mapping:
        pid = r.get("ProgramID", "") + ".cpp" if not r.get("ProgramID", "").endswith(".cpp") else r.get("ProgramID", "")
        fname = r.get("Filename", pid)
        prog_map[fname] = r

    # Also check files on disk
    disk_files = set(
        f for f in os.listdir(BENCHMARK_DIR)
        if f.startswith("program_") and f.endswith(".cpp")
    )

    # Enrich with repo/category/LOC
    candidates: list[str] = sorted(disk_files)
    meta: Dict[str, Any] = {"total_available": len(candidates)}

    logger.info(f"Total benchmark programs available: {len(candidates)}")

    # --- Filter: repository ---
    if args.repository:
        repos = set(args.repository)
        filtered: list[str] = []
        for f in candidates:
            m = prog_map.get(f, {})
            cat_repo = m.get("Category", m.get("Repository", ""))
            orig = m.get("OriginalPath", f)
            # Try extracting repo from path: category/repo/.../file.cpp
            parts = orig.split("/")
            repo_from_path = parts[1] if len(parts) > 1 else ""
            if cat_repo in repos or repo_from_path in repos:
                filtered.append(f)
            # Also check the metadata's Category/Repository columns
            pm = prog_meta.get(f, {})
            if pm.get("Category", "") in repos or pm.get("Repository", "") in repos:
                if f not in filtered:
                    filtered.append(f)
        candidates = sorted(filtered)
        meta["repository_filter"] = list(repos)
        meta["after_repository"] = len(candidates)
        logger.info(f"  After --repository {repos}: {len(candidates)} programs")

    # --- Filter: category ---
    if args.category:
        cats = set(args.category)
        filtered: list[str] = []
        for f in candidates:
            # Classify by original path
            m = prog_map.get(f, {})
            orig = m.get("OriginalPath", f)
            cat = _classify_category(orig)
            if cat in cats:
                filtered.append(f)
        candidates = sorted(filtered)
        meta["category_filter"] = list(cats)
        meta["after_category"] = len(candidates)
        logger.info(f"  After --category: {len(candidates)} programs (categories: {cats})")

    # --- Filter: LOC ---
    if args.min_loc is not None or args.max_loc is not None:
        filtered: list[str] = []
        for f in candidates:
            pm = prog_meta.get(f, {})
            loc = int(pm.get("CodeLines", pm.get("FileSizeBytes", "0")))
            if loc == 0:
                # Fallback: check filesize
                fpath = os.path.join(BENCHMARK_DIR, f)
                try:
                    loc = os.path.getsize(fpath)
                except OSError:
                    loc = 0
            if args.min_loc is not None and loc < args.min_loc:
                continue
            if args.max_loc is not None and loc > args.max_loc:
                continue
            filtered.append(f)
        candidates = sorted(filtered)
        meta["loc_filter"] = {"min": args.min_loc, "max": args.max_loc}
        meta["after_loc"] = len(candidates)
        logger.info(f"  After --min-loc/--max-loc: {len(candidates)} programs")

    # --- Random sampling ---
    if args.random_n is not None:
        rng = _random.Random(args.seed)
        if args.random_n < len(candidates):
            candidates = sorted(rng.sample(candidates, args.random_n))
            meta["random_sampled"] = True
            meta["random_n"] = len(candidates)
            meta["seed"] = args.seed
            logger.info(f"  After --random {args.random_n} (seed={args.seed}): {len(candidates)} programs")

    # --- Final limit ---
    if args.limit != "all" and args.limit is not None:
        try:
            n = int(args.limit)
            candidates = candidates[:n]
            meta["final_limit"] = n
            logger.info(f"  After --limit {n}: {len(candidates)} programs")
        except (ValueError, TypeError):
            pass  # "all" or invalid

    meta["final_count"] = len(candidates)
    logger.info(f"Final selection: {len(candidates)} programs")

    return candidates, meta


# ============================================================================
# Experiment recording
# ============================================================================

def _create_experiment_dir(args: argparse.Namespace) -> str:
    """Create the timestamped experiment output directory.

    Args:
        args: Parsed arguments.

    Returns:
        Absolute path to the experiment run directory.
    """
    if args.output_dir:
        run_dir = os.path.join(EXPERIMENT_ROOT, args.output_dir)
    elif args.resume:
        # Find the most recent experiment directory
        try:
            dirs = sorted(
                d for d in os.listdir(EXPERIMENT_ROOT)
                if os.path.isdir(os.path.join(EXPERIMENT_ROOT, d))
            )
            if dirs:
                run_dir = os.path.join(EXPERIMENT_ROOT, dirs[-1])
                return run_dir  # don't recreate, reuse
        except OSError:
            pass
        run_dir = os.path.join(EXPERIMENT_ROOT, _now_dirname())
    else:
        run_dir = os.path.join(EXPERIMENT_ROOT, _now_dirname())

    # Create subdirectories
    for sub in ("csv", "reports", "figures", "logs", "config"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    return run_dir


def _record_config(args: argparse.Namespace, run_dir: str,
                   selection_meta: Dict[str, Any]) -> str:
    """Save the experiment configuration for reproducibility.

    Args:
        args: Parsed arguments.
        run_dir: Experiment run directory.
        selection_meta: Selection filter metadata.

    Returns:
        Path to the saved configuration JSON.
    """
    compiler = {}
    try:
        compiler = get_compiler()
    except Exception:
        pass

    # Git hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:
        git_hash = "unknown"

    config = {
        "experiment_id": os.path.basename(run_dir),
        "timestamp": _ts(),
        "git_commit": git_hash,
        "compiler": compiler,
        "operating_system": platform.platform(),
        "python_version": sys.version.split()[0],
        "cli_arguments": {
            "limit": args.limit,
            "repository": args.repository,
            "category": args.category,
            "min_loc": args.min_loc,
            "max_loc": args.max_loc,
            "random_n": args.random_n,
            "seed": args.seed,
            "repair": args.repair,
            "runtime": args.runtime,
        },
        "selection": selection_meta,
    }

    # Add YAML config source
    if args.config:
        config["config_file"] = os.path.abspath(args.config)

    path = os.path.join(run_dir, "config", "experiment_configuration.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    return path


# ============================================================================
# Translation pipeline integration
# ============================================================================

def _prepare_samples_dir(program_files: List[str]) -> str:
    """Copy selected benchmark programs into the samples directory.

    Cleans previous experiment programs first.  The original
    ``example.cpp`` and its test files are preserved.

    Args:
        program_files: List of ``program_NNNNNN.cpp`` filenames.

    Returns:
        Path to the samples directory.
    """
    cfg = _get_run_config()
    samples = cfg.samples_dir
    os.makedirs(samples, exist_ok=True)

    # Clean previous experiment programs (keep original example.*)
    for f in os.listdir(samples):
        if f.startswith("program_") and (f.endswith(".cpp") or f.endswith(".in")):
            try:
                os.remove(os.path.join(samples, f))
            except OSError:
                pass

    # Copy selected programs
    for fname in program_files:
        src = os.path.join(BENCHMARK_DIR, fname)
        dst = os.path.join(samples, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    return samples


def _run_translation_experiment(
    program_files: List[str],
    args: argparse.Namespace,
    run_dir: str,
    logger: Logger,
) -> bool:
    """Run the C++ → Python translation pipeline on selected programs.

    Args:
        program_files: List of program filenames.
        args: Parsed arguments.
        run_dir: Experiment output directory.
        logger: :class:`Logger` instance.

    Returns:
        ``True`` if the translation completed successfully.
    """
    cfg = _get_run_config()

    # Configure repair rounds
    if args.max_repair_rounds is not None:
        cfg.max_repair_rounds = args.max_repair_rounds
    elif args.repair:
        cfg.max_repair_rounds = 5  # default
    elif args.no_repair:
        cfg.max_repair_rounds = 0

    # Redirect experiment CSVs into the experiment directory
    csv_subdir = os.path.join(run_dir, "csv")
    cfg.csv_file = os.path.join(csv_subdir, "experiment_results.csv")
    cfg.summary_csv = os.path.join(csv_subdir, "summary_results.csv")
    cfg.verbose_output = args.verbose

    # Re-apply config so the run module picks up the new paths
    _set_run_config(cfg)

    logger.info(f"  Samples dir:    {cfg.samples_dir}")
    logger.info(f"  Translated dir: {cfg.translated_dir}")
    logger.info(f"  Max repair:     {cfg.max_repair_rounds}")
    logger.info(f"  Results CSV:    {cfg.csv_file}")
    logger.info(f"  Summary CSV:    {cfg.summary_csv}")

    # Copy programs to samples/
    _prepare_samples_dir(program_files)
    logger.info(f"  Programs staged: {len(program_files)}")

    # Run the translation pipeline
    from run import main as run_main
    from run import _cleanup_cpp_cache

    try:
        run_main()
    except Exception as exc:
        logger.error(f"Translation pipeline error: {exc}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        _cleanup_cpp_cache()

    return True


# ============================================================================
# Main entry point
# ============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """Parse args, select programs, run experiment, generate outputs.

    Args:
        argv: CLI argument list (defaults to ``sys.argv[1:]``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- YAML config overlay ----
    if args.config:
        yaml_cfg = _load_config_file(args.config)
        _apply_yaml_config(args, yaml_cfg)

    # ---- Setup ----
    run_dir = _create_experiment_dir(args)
    logger = Logger(f"experiment_{os.path.basename(run_dir)}")

    # Print banner
    logger.info("=" * 70)
    logger.info("C++ → PYTHON TRANSLATION EXPERIMENT RUNNER")
    logger.info("=" * 70)
    logger.info(f"Experiment ID: {os.path.basename(run_dir)}")
    logger.info(f"Output:        {run_dir}")

    # Compiler info
    try:
        c = get_compiler()
        logger.info(f"Compiler:      {c['name']} ({c['executable']})")
    except Exception:
        pass

    logger.info(f"Repair:        {args.repair} (rounds={args.max_repair_rounds or 'default'})")
    logger.info(f"Runtime:       {args.runtime}")
    logger.info(f"Config file:   {args.config or 'none'}")
    logger.info(f"Resume:        {args.resume}")

    # ---- Program selection ----
    if not args.resume:
        logger.info("-" * 70)
        logger.info("Selecting programs …")
        program_files, selection_meta = _select_programs(args, logger)

        if not program_files:
            logger.error("No programs match the selection criteria.")
            sys.exit(1)

        # Record configuration
        cfg_path = _record_config(args, run_dir, selection_meta)
        logger.info(f"Configuration saved: {cfg_path}")
    else:
        # Resume: read from last experiment
        logger.info("Resume mode: using programs from last experiment")
        cfg_path = os.path.join(run_dir, "config", "experiment_configuration.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                prev_cfg = json.load(f)
            prev_count = prev_cfg.get("selection", {}).get("final_count", 0)
            logger.info(f"  Previous experiment had {prev_count} programs")
        # Find programs already in samples/
        samples = _get_run_config().samples_dir
        program_files = sorted(
            f for f in os.listdir(samples)
            if f.startswith("program_") and f.endswith(".cpp")
        )
        if not program_files:
            logger.error("No program files in samples/. Cannot resume.")
            sys.exit(1)
        logger.info(f"  Resuming with {len(program_files)} programs in samples/")
        selection_meta = {"resumed": True, "final_count": len(program_files)}

    # ---- Run translation ----
    logger.info("-" * 70)
    logger.info("Running translation experiment …")
    t0 = time.time()

    ok = _run_translation_experiment(program_files, args, run_dir, logger)

    elapsed = time.time() - t0
    logger.info(f"Translation completed in {elapsed:.1f}s")

    # ---- Post-experiment: copy outputs into experiment directory ----
    logger.info("-" * 70)
    logger.info("Collecting experiment outputs …")

    # Copy translated Python files
    trans_src = _get_run_config().translated_dir
    trans_dest = os.path.join(run_dir, "translated")
    if os.path.isdir(trans_src):
        os.makedirs(trans_dest, exist_ok=True)
        for f in os.listdir(trans_src):
            if f.endswith(".py"):
                shutil.copy2(os.path.join(trans_src, f),
                             os.path.join(trans_dest, f))
        logger.info(f"  Translated files: {trans_dest}")

    # Generate experiment-level analytics using research_analytics if available
    if not args.no_figures:
        logger.info("Generating figures …")
        try:
            import figures as _fig
            _fig.generate_all(logger)
        except Exception as exc:
            logger.warn(f"Figure generation failed: {exc}")

    if not args.no_report:
        logger.info("Generating report …")
        try:
            from report_generator import generate_report
            report_path = os.path.join(run_dir, "reports", "report.md")
            generate_report(report_path, logger)
        except Exception as exc:
            logger.warn(f"Report generation failed: {exc}")

    # ---- Final summary ----
    print(f"\n{'═' * 70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'═' * 70}")
    print(f"  Experiment ID:  {os.path.basename(run_dir)}")
    print(f"  Programs:       {len(program_files)}")
    print(f"  Repair:         {args.repair}")
    print(f"  Runtime:        {args.runtime}")
    print(f"  Duration:       {elapsed:.1f}s")
    print(f"  Output:         {run_dir}/")
    print(f"    csv/          — experiment results")
    print(f"    reports/      — generated reports")
    print(f"    figures/      — generated figures")
    print(f"    logs/         — run logs")
    print(f"    config/       — experiment configuration")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
