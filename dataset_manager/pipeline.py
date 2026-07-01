"""
dataset_manager/pipeline.py — Full Automation Pipeline
=======================================================

Orchestrates the complete dataset construction workflow from raw
cloned repositories to a benchmark-ready dataset.

Stages:
    1.  Repository Scan      →  ``repository_statistics.csv``
    2.  CPP Extraction       →  ``raw_cpp/``
    3.  Program Filtering    →  ``filter_report.csv``
    4.  Duplicate Removal    →  ``duplicate_report.csv``
    5.  Compile Validation   →  ``compile_report.csv``
    6.  Metadata Generation  →  ``metadata.csv``
    7.  Benchmark Dataset    →  ``benchmark_dataset/``
    8.  Source Mapping       →  ``source_mapping.csv``

Usage::

    python dataset_manager/pipeline.py

    # Or run individual stages:
    python dataset_manager/pipeline.py --stage scan
    python dataset_manager/pipeline.py --stage extract
    python dataset_manager/pipeline.py --stage dedup
    python dataset_manager/pipeline.py --stage compile
    python dataset_manager/pipeline.py --stage metadata
    python dataset_manager/pipeline.py --stage build
"""

from __future__ import annotations
import os as _os
import sys as _sys
_DEPS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _DEPS not in _sys.path:
    _sys.path.insert(0, _DEPS)


import os
import sys
import time
from typing import Optional, List

from dataset_manager.utils import Logger, timestamp


# ============================================================================
# Stage runners
# ============================================================================

def _run_stage(
    name: str,
    module_name: str,
    logger: Logger,
) -> bool:
    """Run a single pipeline stage via its module's ``main()``.

    Args:
        name: Human-readable stage name.
        module_name: Full dotted module path (e.g.
            ``"dataset_manager.scan_repositories"``).
        logger: :class:`Logger` instance.

    Returns:
        ``True`` if the stage completed without exception.
    """
    print(f"\n{'─' * 60}")
    print(f"STAGE: {name}")
    print(f"{'─' * 60}")

    t0 = time.time()
    try:
        import importlib
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - t0
        logger.info(f"Stage '{name}' completed in {elapsed:.1f}s")
        return True
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error(f"Stage '{name}' FAILED after {elapsed:.1f}s: {exc}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main pipeline
# ============================================================================

_STAGES: list[tuple[str, str, str]] = [
    ("scan",      "Scan Repositories",     "dataset_manager.scan_repositories"),
    ("extract",   "Extract CPP Files",     "dataset_manager.extract_cpp"),
    ("filter",    "Filter Programs",       "dataset_manager.filter_programs"),
    ("dedup",     "Deduplicate Files",     "dataset_manager.deduplicate"),
    ("compile",   "Compile Validation",    "dataset_manager.validate_cpp"),
    ("metadata",  "Generate Metadata",     "dataset_manager.metadata_generator"),
    ("build",     "Build Benchmark Dataset", "dataset_manager.build_dataset"),
    ("map",       "Source Mapping",        "dataset_manager.map_sources"),
]


def main(argv: Optional[List[str]] = None) -> None:
    """Run the full dataset construction pipeline.

    Args:
        argv: CLI argument list (defaults to ``sys.argv[1:]``).

    Supports ``--stage <name>`` to run a single stage.
    """
    args = argv if argv is not None else sys.argv[1:]
    logger = Logger("pipeline")

    # Check for --stage filter
    stage_filter: str | None = None
    i = 0
    while i < len(args):
        if args[i] == "--stage" and i + 1 < len(args):
            stage_filter = args[i + 1]
            i += 2
        else:
            i += 1

    logger.info("Dataset Manager Pipeline")
    logger.info(f"Started: {timestamp()}")

    # Determine which stages to run
    if stage_filter:
        selected = [(k, n, m) for k, n, m in _STAGES if k == stage_filter]
        if not selected:
            logger.error(f"Unknown stage: '{stage_filter}'. "
                         f"Valid stages: {', '.join(k for k, _, _ in _STAGES)}")
            sys.exit(1)
        logger.info(f"Running single stage: {stage_filter}")
    else:
        selected = list(_STAGES)
        logger.info(f"Running full pipeline ({len(_STAGES)} stages)")

    # Run stages
    passed = 0
    failed = 0
    for key, name, module in selected:
        ok = _run_stage(name, module, logger)
        if ok:
            passed += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'═' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Stages run:    {len(selected)}")
    print(f"  Passed:        {passed}")
    print(f"  Failed:        {failed}")
    print(f"  Finished:      {timestamp()}")
    print(f"{'═' * 60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
