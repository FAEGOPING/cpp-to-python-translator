"""
config.py — Lightweight Configuration System
=============================================

Provides a :class:`Config` dataclass that centralises all tunable parameters
for the C++ → Python translation and evaluation framework.

Default values reproduce the exact behaviour of the original (pre-config)
system, ensuring full backward compatibility.

Usage::

    from config import Config, DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG                     # use defaults
    cfg = Config(max_repair_rounds=7)        # override selectively

    # Or load from a JSON file:
    cfg = Config.load("experiment_config.json")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Central configuration for the translation & evaluation framework.

    Every field has a default that mirrors the behaviour of the original
    system, so existing experiments continue to run without modification.
    """

    # ---- File system ---------------------------------------------------------
    project_root: str = field(
        default_factory=lambda: os.path.dirname(os.path.abspath(__file__))
    )
    """Root directory of the project (auto-detected)."""

    samples_dir: str = ""
    """Directory containing C++ source files.  Defaults to
    ``<project_root>/samples`` when empty."""

    translated_dir: str = ""
    """Directory for translated Python files.  Defaults to
    ``<project_root>/translated`` when empty."""

    csv_file: str = ""
    """Path to detailed experiment CSV.  Defaults to
    ``<project_root>/experiment_results.csv``."""

    summary_csv: str = ""
    """Path to summary CSV.  Defaults to
    ``<project_root>/summary_results.csv``."""

    # ---- Pipeline control ----------------------------------------------------
    max_repair_rounds: int = 5
    """Maximum number of repair iterations per program."""

    execution_timeout: int = 10
    """Timeout in seconds for subprocess execution (C++ and Python)."""

    # ---- Automatic test generation -------------------------------------------
    auto_test: bool = False
    """When ``True``, automatically generate additional test cases beyond
    those provided manually in ``.in`` / ``.out`` files."""

    generated_cases: int = 50
    """Number of test cases to generate when ``auto_test`` is enabled."""

    test_strategies: tuple[str, ...] = (
        "random",
        "boundary",
        "edge",
        "heuristic",
    )
    """Enabled test-generation strategies.  May include ``"random"``,
    ``"boundary"``, ``"edge"``, ``"heuristic"``, and ``"llm"``."""

    # ---- Validation ----------------------------------------------------------
    validation_strategy: str = "differential"
    """Strategy for functional validation:

    * ``"single"`` — only use the ``.in`` file (original behaviour).
    * ``"differential"`` — run every available test case and require all
      to pass.
    """

    # ---- Prompt strategy -----------------------------------------------------
    prompt_strategy: str = "enhanced"
    """Prompt strategy for translation and repair:

    * ``"basic"`` — original prompt text.
    * ``"enhanced"`` — prompts enriched with error categories, repair
      history, and structured failure context.
    """

    # ---- Caching -------------------------------------------------------------
    enable_caching: bool = True
    """Cache C++ and Python execution results to avoid duplicate work."""

    # ---- Logging -------------------------------------------------------------
    extended_logging: bool = True
    """When ``True``, append additional timing and test-count columns to
    the experiment CSV (fully backward-compatible — old columns remain)."""

    verbose_output: bool = False
    """Print detailed progress information to stdout."""

    # ---- Internal (computed) -------------------------------------------------
    def __post_init__(self) -> None:
        """Resolve derived paths."""
        if not self.samples_dir:
            self.samples_dir = os.path.join(self.project_root, "samples")
        if not self.translated_dir:
            self.translated_dir = os.path.join(self.project_root, "translated")
        if not self.csv_file:
            self.csv_file = os.path.join(self.project_root, "experiment_results.csv")
        if not self.summary_csv:
            self.summary_csv = os.path.join(self.project_root, "summary_results.csv")

    # ---- Serialisation -------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist configuration to a JSON file."""
        data = {k: v for k, v in self.__dict__.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from a JSON file, falling back to defaults
        for any key that is missing."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        # Only pass keys that the dataclass actually accepts
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Module-level default — import this for the standard profile
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = Config()
"""Pre-built :class:`Config` with all defaults (reproduces original behaviour)."""


# ---------------------------------------------------------------------------
# Legacy-compatible globals (used by run.py)
# ---------------------------------------------------------------------------

def _resolve_legacy() -> Config:
    """Return the active config, preferring ``DEFAULT_CONFIG`` but
    respecting the ``PROJECT_CONFIG`` environment variable if set."""
    env_path = os.getenv("PROJECT_CONFIG")
    if env_path and os.path.isfile(env_path):
        return Config.load(env_path)
    return DEFAULT_CONFIG
