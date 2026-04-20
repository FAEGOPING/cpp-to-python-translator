# C++ to Python Translator (LLM-based Closed-Loop System)

## 📌 Overview

This project implements a **compiler-assisted, closed-loop code translation system** that automatically converts C++ programs into Python using large language models (LLMs).

Unlike traditional one-pass translation approaches, this system introduces an **iterative error feedback mechanism**, where translation results are validated and corrected automatically until they pass compilation or reach a predefined limit.

---

## 🚀 Key Features

* 🔄 **Automatic C++ → Python translation** using LLMs
* 🧪 **Syntax validation** via `py_compile`
* 🔁 **Closed-loop error feedback** for iterative correction
* ⚙️ **Automated repair mechanism** based on error messages
* 📊 Designed for **experimental evaluation and analysis**

---

## 🧠 System Workflow

The system follows a closed-loop pipeline:

1. Translate C++ code into Python using LLM
2. Validate Python code (syntax check)
3. Capture error messages (if any)
4. Send error feedback back to the LLM
5. Generate a corrected version
6. Repeat until success or max iterations

---

## 📁 Project Structure

```text
.
├── run.py              # Main execution script (closed-loop control)
├── gpt_api.py          # LLM API interaction module
├── example.cpp         # Sample input file
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🛠️ Installation

Make sure you have Python 3 installed.

Install dependencies:

```bash
pip install openai
```

---

## 🔑 API Key Setup

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your_api_key"
```

⚠️ **Do NOT hardcode your API key in the source code.**

---

## ▶️ Usage

Run the main script:

```bash
python run.py
```

The system will:

* Translate the C++ code
* Attempt to compile the Python code
* Automatically fix errors (if any)
* Output the final corrected version

---

## 📊 Example Output

```text
Initial Translation:
<generated python code>

❌ Error at round 0:
SyntaxError...

❌ Error at round 1:
...

✅ Success at round 2

Final Python Code:
<corrected code>
```

---

## 🔬 Research Focus

This project explores:

* Execution-guided code translation
* Compiler-assisted feedback loops
* Iterative refinement in LLM-based systems

---

## ⚠️ Limitations

* Focuses primarily on **syntax correctness**, not full semantic equivalence
* Limited handling of complex C++ features (e.g., pointers, templates)
* Depends on external LLM API

---

## 📈 Future Work

* Runtime validation and I/O comparison
* Semantic equivalence checking
* Error classification and statistical analysis
* Multi-model comparison (e.g., OpenAI vs Gemini)

---

## 📌 Project Type

Final Year Project – Computer Science

---

## 👤 Author

FAEGOPING
