# Dataset Manager — Automated Benchmark Dataset Construction

## Overview

The Dataset Manager is a modular pipeline that transforms raw cloned
GitHub repositories of C++ code into a clean, deduplicated,
compile-validated benchmark dataset suitable for LLM translation
experiments.

It is designed to scale to **5,000+ C++ programs** while remaining
fully automated and reproducible.

## Architecture

```
repositories.txt
      │
      ▼
clone_repositories.py  ──→  ~/datasets/<category>/<repo>/
      │
      ▼
scan_repositories.py   ──→  reports/repository_statistics.csv
      │
      ▼
extract_cpp.py         ──→  raw_cpp/<category>/<repo>/.../file.cpp
      │
      ▼
deduplicate.py         ──→  reports/duplicate_report.csv
      │
      ▼
validate_cpp.py        ──→  reports/compile_report.csv
      │
      ▼
metadata_generator.py  ──→  reports/metadata.csv
      │
      ▼
build_dataset.py       ──→  benchmark_dataset/program_NNNNNN.cpp
                            benchmark_dataset/metadata.csv
```

## Directory Structure

```
dataset_manager/
├── __init__.py              # Package init
├── utils.py                 # Shared utilities (logging, CSV, hashing, subprocess)
├── repositories.txt         # GitHub repository list
├── clone_repositories.py    # Git clone manager
├── scan_repositories.py     # Statistical scanner
├── extract_cpp.py           # C++ file extractor
├── deduplicate.py           # SHA-256 duplicate removal
├── validate_cpp.py          # g++ compile validation
├── metadata_generator.py    # Per-file metadata analysis
├── build_dataset.py         # Benchmark dataset assembler
├── pipeline.py              # Full automation orchestrator
├── raw_cpp/                 # Extracted C++ sources (staging)
├── benchmark_dataset/       # Final benchmark dataset
├── reports/                 # Generated CSV reports
├── logs/                    # Pipeline run logs
└── README.md                # This file
```

## Quick Start

### 1. Configure Repositories

Edit `repositories.txt` to list the GitHub repositories you want to
include:

```
algorithms  https://github.com/TheAlgorithms/C-Plus-Plus.git
codeforces  https://github.com/user/Competitive-Programming.git
cses        https://github.com/user/CSES-Solutions.git
```

Format: `<category>  <git-clone-url>`

### 2. Clone Repositories

```bash
python dataset_manager/clone_repositories.py
```

This clones every repository into `~/datasets/<category>/<repo>/`.
Already-cloned repos are skipped.  Generates
`reports/download_report.csv`.

### 3. Run the Full Pipeline

```bash
python dataset_manager/pipeline.py
```

This executes all stages automatically:

| Stage | Module | Output |
|-------|--------|--------|
| 1. Scan | `scan_repositories.py` | `reports/repository_statistics.csv` |
| 2. Extract | `extract_cpp.py` | `raw_cpp/` |
| 3. Dedup | `deduplicate.py` | `reports/duplicate_report.csv` |
| 4. Compile | `validate_cpp.py` | `reports/compile_report.csv` |
| 5. Metadata | `metadata_generator.py` | `reports/metadata.csv` |
| 6. Build | `build_dataset.py` | `benchmark_dataset/` |

### 4. Run Individual Stages

```bash
python dataset_manager/pipeline.py --stage scan
python dataset_manager/pipeline.py --stage extract
python dataset_manager/pipeline.py --stage compile
```

Or run modules directly:

```bash
python dataset_manager/scan_repositories.py
python dataset_manager/extract_cpp.py
python dataset_manager/deduplicate.py
python dataset_manager/validate_cpp.py
python dataset_manager/metadata_generator.py
python dataset_manager/build_dataset.py
```

## Output Files

### `benchmark_dataset/`

The final dataset directory containing:

- `program_000001.cpp` … `program_NNNNNN.cpp` — sequential, zero-padded filenames
- `metadata.csv` — maps each program ID to its original source

### `reports/`

| File | Contents |
|------|----------|
| `download_report.csv` | Clone results (success / exists / 404 / error) |
| `repository_statistics.csv` | Per-category file counts, sizes, depth |
| `extraction_report.csv` | Files copied / skipped / failed |
| `duplicate_report.csv` | Every duplicate file with its original and SHA-256 |
| `dedup_summary.csv` | Unique vs duplicate counts |
| `compile_report.csv` | Per-file compile status, time, error message |
| `metadata.csv` | LOC, function/loop/conditional counts, STL usage, category, repo |

### `logs/`

Timestamped per-module log files with counts, timings, and error details.

## CSV Formats

### compile_report.csv
```
File, Status, CompileTimeSeconds, Error
algorithms/C-Plus-Plus/search/binary_search.cpp, PASS, 0.123,
codeforces/Competitive-Programming/42.cpp, FAIL, 0.089, "error: 'x' was not declared"
```

### metadata.csv
```
File, LOC, Functions, Loops, Conditionals, Includes, STLUsage, FileSizeBytes, Category, Repository, CompileStatus, Author
algorithms/.../binary_search.cpp, 35, 1, 1, 2, 2, "vector, algorithm, iostream", 1024, algorithms, C-Plus-Plus, PASS,
```

### benchmark_dataset/metadata.csv
```
ProgramID, Filename, OriginalSource, Category, Repository, FileSizeBytes
program_000001, program_000001.cpp, algorithms/C-Plus-Plus/search/binary_search.cpp, algorithms, C-Plus-Plus, 1024
```

## Design Principles

1. **Non-destructive** — Original repositories and `raw_cpp/` are never modified
2. **Resumable** — Each stage can be run independently; failed stages can be re-run
3. **Scalable** — Batch processing with progress logging every 500–1000 files
4. **Reproducible** — SHA-256 hashing ensures deterministic deduplication
5. **Modular** — Each `.py` file is a self-contained, independently runnable module

## Integration with Translation Framework

The benchmark dataset produced by this module is designed to feed
directly into the C++ → Python translation framework (`run.py`):

```bash
# Build the dataset
python dataset_manager/pipeline.py

# Copy desired programs to the translation framework's samples/ directory
cp dataset_manager/benchmark_dataset/program_000001.cpp samples/
cp dataset_manager/benchmark_dataset/program_000002.cpp samples/

# Run the translation experiment
python run.py
```

## Scalability Notes

The pipeline is designed to handle **5,000–10,000+ C++ files**:

- SHA-256 deduplication is O(n) with streaming hashing (64KB chunks)
- Compile validation processes files sequentially with progress logging
- Metadata analysis uses pre-compiled regex patterns
- All reports are written incrementally, not held in memory
