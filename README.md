# Compiler-Assisted C++ to Python Translation System (v2.0)

## Overview

This project implements a compiler-assisted closed-loop code translation system that automatically translates C++ programs into Python using a Large Language Model (DeepSeek-V4-Pro).

The system performs:

1. C++ to Python translation
2. Automatic compilation checking
3. Error feedback extraction
4. Iterative repair using LLM feedback
5. Experimental logging and evaluation
6. Multi-test-case differential testing
7. Automatic test case generation
8. Execution result caching
9. Enhanced repair prompts with error categories
10. Automatic experiment analysis & reporting

## Workflow

```
C++ Source Code
      │
      ▼
LLM Translation
      │
      ▼
Compile Check  ←── py_compile validation
      │
      ▼
Runtime Check  ←── subprocess + timeout
      │
      ▼
Differential Testing  ←── C++ vs Python, all test cases
      │
      ▼                    (fails)
LLM Repair  ──────────────────┘
      │
      ▼
Repeat (max N rounds)
      │
      ▼
Experiment Analysis
```

## Features

* Automatic C++ to Python translation
* Compiler-assisted feedback loop
* Multi-round automatic repair
* Multi-test-case support (numbered `.in` / `.out` files)
* Differential testing framework
* Automatic test generation (random, boundary, edge, heuristic, LLM)
* Execution result caching
* Enhanced repair prompts with error categories
* Extended experiment logging (per-phase timing, test counts, success rates)
* Automatic experiment analysis and report generation

## Project Structure

```text
samples/              # C++ test programs & test cases
translated/           # Generated Python programs
run.py                # Main execution script (entry point)
gpt_api.py            # DeepSeek API interface
config.py             # Configuration system
test_generator.py     # Automatic test case generation
differential_testing.py  # Multi-test differential validation
cache.py              # Execution result caching
analysis.py           # Experiment analysis & reporting
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set your DeepSeek API key:

```bash
export DEEPSEEK_API_KEY="your_api_key"
```

Configure the framework (all fields optional — defaults match the original behaviour):

```python
from config import Config
from run import set_config

cfg = Config(
    max_repair_rounds=5,      # max repair iterations
    execution_timeout=10,      # seconds
    auto_test=False,           # enable auto test generation
    generated_cases=50,        # how many tests to generate
    validation_strategy="differential",  # "single" or "differential"
    enable_caching=True,       # avoid duplicate executions
    extended_logging=True,     # additional CSV columns
)
set_config(cfg)
```

## Test Cases

### Mode A — Numbered files (multi-test)
```
samples/example_1.in   →  input
samples/example_1.out  →  expected output
samples/example_2.in
samples/example_2.out
```
The translation must pass **all** test cases.

### Mode B — Single file (legacy, backward compatible)
```
samples/example.in     →  single test input
```
Behaviour identical to v1.0.

## Run

```bash
python3 run.py
```

## Analysis

```bash
python3 analysis.py
```

This generates `analysis_report.csv` and `analysis_report.txt` from the
experiment logs — suitable for direct inclusion in dissertation chapters.

## Output

The system generates:

* translated Python files (`translated/`)
* experiment_results.csv (detailed per-round log)
* summary_results.csv (one row per program)
* analysis_report.csv (aggregate statistics)
* analysis_report.txt (human-readable report)

## Research Goal

Evaluate whether compiler-assisted iterative feedback improves the
correctness of LLM-based code translation compared with single-pass
translation. The v2.0 framework additionally evaluates:

* Differential testing as a validation strategy
* Automatic test generation for translation quality assessment
* Error-category-aware repair prompting
* Execution caching for experimental efficiency
