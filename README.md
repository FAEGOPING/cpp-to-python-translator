# Compiler-Assisted C++ to Python Translation System

## Overview

This project implements a compiler-assisted closed-loop code translation system that automatically translates C++ programs into Python using a Large Language Model (DeepSeek-V4-Pro).

The system performs:

1. C++ to Python translation
2. Automatic compilation checking
3. Error feedback extraction
4. Iterative repair using LLM feedback
5. Experimental logging and evaluation

## Workflow

C++ Source Code

↓

DeepSeek Translation

↓

Python Code

↓

Compilation Check

↓

Error Feedback

↓

Automatic Repair

↓

Repeat Until Success

## Features

* Automatic C++ to Python translation
* Compiler-assisted feedback loop
* Multi-round automatic repair
* Experiment logging
* Performance evaluation metrics

## Project Structure

```text
samples/          # C++ test programs
translated/       # Generated Python programs
run.py            # Main execution script
gpt_api.py        # DeepSeek API interface
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

## Run

```bash
python3 run.py
```

## Output

The system generates:

* translated Python files
* experiment_results.csv
* summary_results.csv

## Research Goal

Evaluate whether compiler-assisted iterative feedback improves the correctness of LLM-based code translation compared with single-pass translation.
