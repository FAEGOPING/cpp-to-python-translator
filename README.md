# C++ → Python Translation Research Platform (v2.2)

## Overview

A complete research platform for automated evaluation of LLM-based
C++ → Python translation at scale.

The system assembles benchmark datasets from public GitHub repositories,
translates C++ programs to Python via LLM, validates correctness through
differential testing, and generates publication-quality figures and reports.

**One command from raw repositories to research results:**

```bash
python benchmark.py --limit 100
```

## Architecture

```
Public GitHub Repositories
      │
      ▼
Dataset Manager  ──→  Clone, Scan, Extract, Dedup, Compile, Metadata
      │
      ▼
Benchmark Dataset  ──→  program_000001.cpp ... program_NNNNNN.cpp
      │
      ▼
Experiment Runner  ──→  Translation Framework (LLM + Repair)
      │
      ▼
Analysis & Figures  ──→  Statistics, Graphs, CSV Reports, Markdown Report
```

## Project Structure

```
project/
├── benchmark.py                 # Full research pipeline (one command)
├── experiment_runner.py         # Connects Dataset Manager + Translation Framework
├── run.py                       # C++ → Python translation + repair (entry point)
├── gpt_api.py                   # LLM API interface (DeepSeek)
├── config.py                    # Configuration system
├── cache.py                     # Execution + compilation caches
├── test_generator.py            # Automatic test case generation
├── differential_testing.py      # Multi-test differential validation
├── analysis.py                  # Experiment analysis + CSV reports
├── figures.py                   # Publication-quality figure generation
├── report_generator.py          # Automatic Markdown research report
├── samples/                     # C++ test programs
├── translated/                  # Generated Python output
├── dataset_manager/
│   ├── __init__.py
│   ├── utils.py                 # Shared utilities (Logger, CSV, SHA256, metrics)
│   ├── repositories.txt         # GitHub repository list
│   ├── clone_repositories.py    # Git clone manager
│   ├── scan_repositories.py     # Repository statistics + metrics
│   ├── extract_cpp.py           # C++ file extractor
│   ├── deduplicate.py           # SHA-256 deduplication
│   ├── validate_cpp.py          # g++ compile validation + diagnostics
│   ├── metadata_generator.py    # Per-file software engineering metrics
│   ├── build_dataset.py         # Benchmark dataset assembler
│   ├── map_sources.py           # Source-to-benchmark mapping
│   ├── pipeline.py              # Dataset construction pipeline
│   ├── raw_cpp/                 # Staging area for extracted C++
│   ├── benchmark_dataset/       # Final benchmark dataset
│   ├── reports/                 # Generated CSVs + figures + report.md
│   │   └── figures/             # PNG + PDF figures
│   ├── logs/                    # Timestamped run logs
│   └── README.md                # Dataset Manager documentation
└── README.md                    # This file
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

```bash
export DEEPSEEK_API_KEY="your_api_key"
```

### One-Command Full Pipeline

```bash
# Full pipeline: dataset → translate → analyze → report
python benchmark.py

# Dataset construction only (no LLM)
python benchmark.py --skip-translation

# First 10 programs only
python benchmark.py --limit 10

# Single stage
python benchmark.py --stage compile
```

### Individual Components

```bash
# Dataset construction pipeline
python dataset_manager/pipeline.py

# Translation experiment
python experiment_runner.py --limit 10

# Single program translation
python run.py

# Analysis
python analysis.py

# Figures
python figures.py

# Research report
python report_generator.py

# Individual dataset stages
python dataset_manager/scan_repositories.py
python dataset_manager/extract_cpp.py
python dataset_manager/deduplicate.py
python dataset_manager/validate_cpp.py
python dataset_manager/metadata_generator.py
python dataset_manager/build_dataset.py
python dataset_manager/map_sources.py
```

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| Scan | `scan_repositories.py` | Repository statistics (owner, size, LOC, commits) |
| Extract | `extract_cpp.py` | Copy .cpp files preserving directory mapping |
| Deduplicate | `deduplicate.py` | SHA-256 content deduplication |
| Compile | `validate_cpp.py` | g++ compile validation + diagnostics |
| Metadata | `metadata_generator.py` | LOC, complexity, STL usage, patterns |
| Build | `build_dataset.py` | Sequential program_NNNNNN.cpp dataset |
| Map | `map_sources.py` | Program ID → original repo/GitHub URL |
| Translate | `run.py` / `experiment_runner.py` | LLM translation + iterative repair |
| Figures | `figures.py` | Publication-quality PNG + PDF figures |
| Report | `report_generator.py` | Automated Markdown research report |

## Output Files

### CSV Reports (`dataset_manager/reports/`)

| File | Contents |
|------|----------|
| `repository_statistics.csv` | Per-repo: owner, URL, size, LOC, commits |
| `extraction_report.csv` | Files copied / skipped / failed |
| `duplicate_report.csv` | All duplicates with SHA-256 |
| `dedup_summary.csv` | Unique vs duplicate ratio |
| `compile_report.csv` | Per-file: status, time, warnings, errors, return code |
| `metadata.csv` | LOC, functions, classes, loops, conditionals, complexity, STL |
| `source_mapping.csv` | Program ID → original repo → GitHub URL |
| `experiment_summary.csv` | High-level experiment metrics |

### Figures (`dataset_manager/reports/figures/`)

| Figure | Description |
|--------|-------------|
| `dataset_distribution.png/.pdf` | Programs per dataset source |
| `repository_distribution.png/.pdf` | Top repositories by program count |
| `loc_histogram.png/.pdf` | Lines of code distribution |
| `compile_success.png/.pdf` | Compile pass/fail pie chart |
| `translation_success.png/.pdf` | Success rate at each pipeline stage |
| `repair_distribution.png/.pdf` | Repair iterations per program |
| `error_category_distribution.png/.pdf` | Error types encountered |

### Logs (`dataset_manager/logs/`)

Timestamped per-module log files with counters, warnings, errors, and timing.

## Key Features

### Translation Framework
- LLM-based C++ → Python translation (DeepSeek-V4-Pro)
- Automated compilation validation (py_compile)
- Runtime execution with timeout
- Differential testing — all test cases must pass
- Iterative self-repair with smart prompt compression
- Error category classification (syntax / runtime / semantic)
- Execution caching (compile once, execute many)

### Dataset Manager
- Automated Git repository cloning
- Per-repository statistical scanning (LOC, complexity, commits)
- SHA-256 content deduplication
- g++ compile validation with full diagnostics
- Software engineering metrics (functions, classes, cyclomatic complexity)
- Source-to-benchmark mapping (full reproducibility)
- Designed for 5,000+ program datasets

### Experiment System
- Fully automated benchmark pipeline
- Experiment statistics and analysis
- Publication-quality figure generation
- Automatic Markdown research report
- Comprehensive logging with per-stage timing and memory usage

## Research Contributions

This platform supports the following research evaluation:

1. **LLM-assisted C++ → Python translation** with compiler feedback
2. **Iterative self-repair** with error-category-aware prompting
3. **Differential testing** as a validation strategy
4. **Automatic test generation** for translation quality assessment
5. **Large-scale automated benchmarking** across diverse repositories
6. **Statistical analysis** of translation and repair effectiveness

## Requirements

- Python 3.10+
- g++ (for C++ compilation and validation)
- DeepSeek API key (for LLM translation)
- Git (for repository cloning)
