"""
benchmark.py — Complete Research Benchmark Pipeline
=====================================================

One-command execution of the entire research workflow:

    Dataset Scan → CPP Extraction → Deduplication → Compile Validation
    → Metadata Generation → Benchmark Dataset → Translation → Repair
    → Statistics → Figures → CSV Reports → Research Report

Usage::

    python benchmark.py                    # full pipeline
    python benchmark.py --limit 10         # 10 programs only
    python benchmark.py --skip-translation # dataset only, no LLM
    python benchmark.py --stage build      # single stage

Version: 2.2
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure project root on path
_DEPS = os.path.dirname(os.path.abspath(__file__))
if _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

from dataset_manager.utils import (
    PROJECT_ROOT,
    RAW_CPP_DIR,
    BENCHMARK_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    Logger,
    timestamp,
    memory_usage_mb,
)

# ============================================================================
# Stage definitions
# ============================================================================

_STAGE_SCAN = (
    "scan", "Scan Repositories",
    "dataset_manager.scan_repositories",
)
_STAGE_EXTRACT = (
    "extract", "Extract CPP Files",
    "dataset_manager.extract_cpp",
)
_STAGE_DEDUP = (
    "dedup", "Deduplicate Files",
    "dataset_manager.deduplicate",
)
_STAGE_COMPILE = (
    "compile", "Compile Validation",
    "dataset_manager.validate_cpp",
)
_STAGE_METADATA = (
    "metadata", "Generate Metadata",
    "dataset_manager.metadata_generator",
)
_STAGE_BUILD = (
    "build", "Build Benchmark Dataset",
    "dataset_manager.build_dataset",
)
_STAGE_MAP = (
    "map", "Source Mapping",
    "dataset_manager.map_sources",
)
_STAGE_TRANSLATE = (
    "translate", "Translation Experiment",
    "experiment_runner",
)
_STAGE_FIGURES = (
    "figures", "Generate Figures",
    "figures",
)
_STAGE_REPORT = (
    "report", "Generate Research Report",
    "report_generator",
)

_ALL_STAGES: List[Tuple[str, str, str]] = [
    _STAGE_SCAN,
    _STAGE_EXTRACT,
    _STAGE_DEDUP,
    _STAGE_COMPILE,
    _STAGE_METADATA,
    _STAGE_BUILD,
    _STAGE_MAP,
    _STAGE_TRANSLATE,
    _STAGE_FIGURES,
    _STAGE_REPORT,
]


# ============================================================================
# Stage runner
# ============================================================================

def _run_stage(
    key: str,
    name: str,
    module: str,
    logger: Logger,
    extra_args: Optional[List[str]] = None,
) -> bool:
    """Run a single pipeline stage.

    Args:
        key: Short stage key (e.g. ``"scan"``).
        name: Human-readable stage name.
        module: Python module path.
        logger: :class:`Logger` instance.
        extra_args: Extra CLI args for the stage.

    Returns:
        ``True`` if the stage completed successfully.
    """
    print(f"\n{'─' * 70}")
    print(f"STAGE: {name}  [{key}]")
    print(f"{'─' * 70}")

    t0 = time.time()
    mem_before = memory_usage_mb()

    try:
        import importlib
        mod = importlib.import_module(module)

        # Pass extra args to modules that support them
        if extra_args and hasattr(mod, 'main'):
            mod.main(extra_args)
        else:
            mod.main()

        elapsed = time.time() - t0
        mem_after = memory_usage_mb()
        logger.info(
            f"Stage '{name}' completed: {elapsed:.1f}s, "
            f"memory: {mem_before:.1f}MB → {mem_after:.1f}MB"
        )
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        logger.error(f"Stage '{name}' FAILED after {elapsed:.1f}s: {exc}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Pipeline runner
# ============================================================================

def _parse_args(argv: Optional[List[str]]) -> Tuple[Optional[str], int, bool]:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        ``(stage_filter, limit, skip_translation)``.
    """
    args = argv if argv is not None else sys.argv[1:]
    stage_filter: str | None = None
    limit: int = 0
    skip_translation: bool = False

    i = 0
    while i < len(args):
        if args[i] == "--stage" and i + 1 < len(args):
            stage_filter = args[i + 1]
            i += 2
        elif args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif args[i] == "--skip-translation":
            skip_translation = True
            i += 1
        else:
            i += 1

    return stage_filter, limit, skip_translation


def run_pipeline(
    stage_filter: Optional[str] = None,
    limit: int = 0,
    skip_translation: bool = False,
) -> Dict[str, bool]:
    """Run the complete benchmark pipeline.

    Args:
        stage_filter: Run only this stage (key).  ``None`` = all stages.
        limit: Max programs for translation (0 = all).
        skip_translation: Omit the LLM translation stage.

    Returns:
        ``{stage_key: success}`` mapping.
    """
    logger = Logger("benchmark")
    results: dict[str, bool] = {}

    logger.info("=" * 70)
    logger.info("RESEARCH BENCHMARK PIPELINE")
    logger.info("C++ → Python Translation & Evaluation Platform")
    logger.info("=" * 70)
    logger.info(f"Started:  {timestamp()}")
    logger.info(f"Limit:    {limit or 'all'}")
    logger.info(f"Translate: {not skip_translation}")

    # Determine which stages to run
    if stage_filter:
        selected = [(k, n, m) for k, n, m in _ALL_STAGES if k == stage_filter]
        if not selected:
            valid = ", ".join(k for k, _, _ in _ALL_STAGES)
            logger.error(f"Unknown stage: '{stage_filter}'. Valid: {valid}")
            sys.exit(1)
        logger.info(f"Running single stage: {stage_filter}")
    else:
        selected = _ALL_STAGES
        logger.info(f"Running full pipeline ({len(_ALL_STAGES)} stages)")

    # Run stages
    passed = 0
    failed = 0

    for key, name, module in selected:
        # Skip translation if requested
        if key == "translate" and skip_translation:
            logger.info(f"Skipping stage: {name} (--skip-translation)")
            continue

        extra: Optional[List[str]] = None
        if key == "translate" and limit > 0:
            extra = ["--limit", str(limit)]

        ok = _run_stage(key, name, module, logger, extra)
        results[key] = ok
        if ok:
            passed += 1
        else:
            failed += 1

    # ---- Final summary ----------------------------------------------------
    total_elapsed = logger.elapsed
    print(f"\n{'═' * 70}")
    print("BENCHMARK PIPELINE COMPLETE")
    print(f"{'═' * 70}")
    print(f"  Stages run:    {len(selected)}")
    print(f"  Passed:        {passed}")
    print(f"  Failed:        {failed}")
    print(f"  Duration:      {total_elapsed:.1f}s")
    print(f"  Finished:      {timestamp()}")
    print(f"  Reports:       {REPORTS_DIR}/")
    print(f"  Figures:       {FIGURES_DIR}/")
    print(f"  Logs:          {LOGS_DIR}/")
    print(f"  Dataset:       {BENCHMARK_DIR}/")
    print(f"{'═' * 70}\n")

    return results


# ============================================================================
# Main entry point
# ============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """Parse args and run the benchmark pipeline.

    Args:
        argv: CLI argument list (defaults to ``sys.argv[1:]``).
    """
    stage_filter, limit, skip_translation = _parse_args(argv)
    run_pipeline(stage_filter, limit, skip_translation)


if __name__ == "__main__":
    main()
