"""
dataset_manager — Automated Benchmark Dataset Construction
===========================================================

Provides a complete pipeline for building a research-grade C++
benchmark dataset from cloned GitHub repositories.

Pipeline stages:
    1. Repository Scan
    2. CPP Extraction
    3. SHA-256 Deduplication
    4. Compile Validation (g++)
    5. Metadata Generation
    6. Benchmark Dataset Assembly

All modules can be run independently or orchestrated via
:mod:`dataset_manager.pipeline`.

Version: 1.0
"""

from __future__ import annotations

__version__ = "1.0.0"
