"""
scheduler.py — Automated Experiment Scheduler (Research Platform v5.0)
=======================================================================

Runs a sequence of experiments defined in a YAML plan file,
automatically generating paper archives for each one.

Usage::

    python3 scheduler.py                          # uses experiment_plan.yaml
    python3 scheduler.py --plan my_plan.yaml      # custom plan
    python3 scheduler.py --workers 4              # override workers for all

Plan file format (``experiment_plan.yaml``)::

    experiments:
      - limit: 100
        repair: false

      - limit: 100
        repair: true

      - limit: 500
        repair: false

      - limit: 500
        repair: true

      - limit: 1000
        repair: false

      - limit: 1000
        repair: true

      - limit: 20
        repair: false
        workers: 4

Each experiment runs sequentially.  The scheduler stops on the first
failure unless ``--continue-on-error`` is specified.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = PROJECT_ROOT / "experiment_plan.yaml"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scheduler",
        description="Automated experiment scheduler for the C++ → Python "
                    "translation research platform.",
    )
    p.add_argument("--plan", type=str, default=str(DEFAULT_PLAN),
                   help="Path to YAML plan file (default: experiment_plan.yaml)")
    p.add_argument("--workers", type=int, default=None,
                   help="Override workers for all experiments")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Continue to next experiment even if one fails")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without executing anything")
    return p


def _load_plan(path: str) -> dict[str, Any]:
    """Load and validate a YAML or JSON experiment plan."""
    if not os.path.isfile(path):
        sys.exit(f"Plan file not found: {path}")

    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except ImportError:
            sys.exit("PyYAML is required for YAML plan files. "
                     "Install with: pip install pyyaml")
    elif path.endswith(".json"):
        import json
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        sys.exit(f"Unsupported plan format: {path}. Use .yaml or .json")

    if not isinstance(data, dict) or "experiments" not in data:
        sys.exit("Plan file must contain an 'experiments' list.")

    experiments = data["experiments"]
    if not isinstance(experiments, list) or len(experiments) == 0:
        sys.exit("Plan file 'experiments' must be a non-empty list.")

    return data


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    plan = _load_plan(args.plan)
    experiments = plan["experiments"]

    total = len(experiments)
    print(f"\n{'═' * 70}")
    print("EXPERIMENT SCHEDULER")
    print(f"{'═' * 70}")
    print(f"  Plan:    {args.plan}")
    print(f"  Total:   {total} experiment(s)")
    print(f"{'═' * 70}\n")

    results: list[dict] = []

    for i, exp in enumerate(experiments, 1):
        exp_name = _exp_label(exp)
        print(f"[{i}/{total}] {exp_name}")
        print("-" * 50)

        if args.dry_run:
            cmd = _build_cmd(exp, args.workers)
            print(f"  Would run:  {' '.join(cmd)}")
            print()
            continue

        t0 = time.time()
        success = True
        try:
            cmd = _build_cmd(exp, args.workers)
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                success = False
                print(f"  ❌ Failed (exit {result.returncode})")
        except KeyboardInterrupt:
            print(f"\n  ⚠️  Interrupted by user")
            results.append({"experiment": exp_name, "status": "interrupted"})
            break
        except Exception as exc:
            success = False
            print(f"  ❌ Error: {exc}")

        elapsed = time.time() - t0
        status = "✅" if success else "❌"
        print(f"  {status}  Completed in {elapsed:.0f}s\n")

        results.append({
            "experiment": exp_name,
            "status": "passed" if success else "failed",
            "elapsed_s": round(elapsed),
        })

        if not success and not args.continue_on_error:
            print("Stopping due to error (use --continue-on-error to override).")
            break

    # Final summary
    passed = sum(1 for r in results if r["status"] == "passed")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n{'═' * 70}")
    print("SCHEDULER COMPLETE")
    print(f"{'═' * 70}")
    print(f"  Total:     {total} experiments")
    print(f"  Run:       {len(results)}")
    print(f"  Passed:    {passed}")
    print(f"  Failed:    {failed}")
    print(f"  Finished:  {_now_utc()}")
    print(f"{'═' * 70}\n")


def _exp_label(exp: dict) -> str:
    """Human-readable label for an experiment entry."""
    limit = exp.get("limit", "all")
    repair = exp.get("repair", False)
    runtime = exp.get("runtime", False)
    extra = ""
    if repair:
        extra += " +repair"
    if runtime:
        extra += " +runtime"
    return f"EXP_{str(limit).upper()}{extra}"


def _build_cmd(exp: dict, global_workers: int | None) -> list[str]:
    """Build a CLI command list for an experiment entry."""
    cmd = ["python3", "experiment_runner.py"]

    limit = exp.get("limit", "all")
    cmd.extend(["--limit", str(limit)])

    if exp.get("repair"):
        cmd.append("--repair")
    if exp.get("runtime"):
        cmd.append("--runtime")

    workers = exp.get("workers", global_workers or 1)
    cmd.extend(["--workers", str(workers)])

    if exp.get("repository"):
        repos = exp["repository"]
        if isinstance(repos, list):
            for r in repos:
                cmd.extend(["--repository", r])
        else:
            cmd.extend(["--repository", str(repos)])

    if exp.get("seed"):
        cmd.extend(["--seed", str(exp["seed"])])

    return cmd


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


if __name__ == "__main__":
    main()
