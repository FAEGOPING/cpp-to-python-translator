import os
import csv
import time
import subprocess
import tempfile

from gpt_api import call_gpt


# =====================================================
# Configuration
# =====================================================

PROJECT_ROOT = "/Users/tianjabez/Desktop/project"

CSV_FILE = os.path.join(
    PROJECT_ROOT,
    "experiment_results.csv"
)

SUMMARY_CSV = os.path.join(
    PROJECT_ROOT,
    "summary_results.csv"
)

TRANSLATED_DIR = os.path.join(
    PROJECT_ROOT,
    "translated"
)

SAMPLES_DIR = os.path.join(
    PROJECT_ROOT,
    "samples"
)

os.makedirs(TRANSLATED_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)


# =====================================================
# Translation
# =====================================================

def translate_cpp(cpp_code):

    prompt = f"""
You are an expert software engineer.

Translate the following C++ code into correct Python 3.

Requirements:
1. Preserve original functionality.
2. Return ONLY Python code.
3. Do not include explanations.

C++ Code:

{cpp_code}
"""

    return call_gpt(prompt)


# =====================================================
# Repair
# =====================================================

def fix_code(code, error):

    prompt = f"""
The following Python code contains errors.

Python Code:

{code}

Error Message:

{error}

Please repair the code.

Requirements:
1. Preserve original functionality.
2. Fix the errors.
3. Return ONLY corrected Python code.
"""

    return call_gpt(prompt)


# =====================================================
# Compile Check
# =====================================================

def check_compile(code):

    with tempfile.NamedTemporaryFile(
        suffix=".py",
        mode="w",
        delete=False
    ) as f:

        f.write(code)
        file_path = f.name

    result = subprocess.run(
        ["python3", "-m", "py_compile", file_path],
        capture_output=True,
        text=True
    )

    os.remove(file_path)

    if result.returncode != 0:
        return False, result.stderr

    return True, None


# =====================================================
# Save Python Code
# =====================================================

def save_code(program_name, code):

    filename = os.path.join(
        TRANSLATED_DIR,
        program_name.replace(".cpp", ".py")
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(code)


# =====================================================
# Detailed Log
# =====================================================

def log_result(
    program,
    round_num,
    status,
    error_type,
    success,
    elapsed_time
):

    file_exists = os.path.isfile(CSV_FILE)

    with open(
        CSV_FILE,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "Program",
                "Round",
                "Status",
                "ErrorType",
                "Success",
                "TimeSeconds",
                "Model",
                "RepairCount"
            ])

        writer.writerow([
            program,
            round_num,
            status,
            error_type,
            success,
            round(elapsed_time, 2),
            "DeepSeek-V4-Pro",
            round_num
        ])


# =====================================================
# Summary Log
# =====================================================

def log_summary(
    program,
    initial_pass,
    final_pass,
    repair_rounds,
    total_time
):

    file_exists = os.path.isfile(
        SUMMARY_CSV
    )

    with open(
        SUMMARY_CSV,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "Program",
                "InitialPass",
                "FinalPass",
                "RepairRounds",
                "TotalTime"
            ])

        writer.writerow([
            program,
            initial_pass,
            final_pass,
            repair_rounds,
            round(total_time, 2)
        ])


# =====================================================
# Process Program
# =====================================================

def process_program(program_path):

    program_name = os.path.basename(program_path)

    print("\n" + "=" * 60)
    print(f"Processing: {program_name}")
    print("=" * 60)

    with open(
        program_path,
        "r",
        encoding="utf-8"
    ) as f:

        cpp_code = f.read()

    start_time = time.time()

    print("\nGenerating translation...")

    python_code = translate_cpp(cpp_code)

    max_rounds = 5

    initial_pass = False

    for round_num in range(max_rounds):

        success, error = check_compile(
            python_code
        )

        if round_num == 0:
            initial_pass = success

        elapsed = time.time() - start_time

        if success:

            print(
                f"\n✅ Compilation Passed "
                f"(Round {round_num})"
            )

            save_code(
                program_name,
                python_code
            )

            print(
                f"\nSaved to:\n"
                f"{TRANSLATED_DIR}/"
                f"{program_name.replace('.cpp', '.py')}"
            )

            log_result(
                program_name,
                round_num,
                "Success",
                "None",
                True,
                elapsed
            )

            log_summary(
                program_name,
                initial_pass,
                True,
                round_num,
                elapsed
            )

            return

        error_type = "Unknown"

        if error:

            print("\n========== FULL ERROR ==========")
            print(error)
            print("================================")

            known_errors = [
                "SyntaxError",
                "IndentationError",
                "NameError",
                "TypeError",
                "ValueError",
                "IndexError",
                "KeyError",
                "AttributeError",
                "ImportError",
                "ModuleNotFoundError"
            ]

            for err in known_errors:

                if err in error:
                    error_type = err
                    break

        print(f"\n❌ Round {round_num} Failed")
        print(f"Error Type: {error_type}")

        log_result(
            program_name,
            round_num,
            "Repair",
            error_type,
            False,
            elapsed
        )

        python_code = fix_code(
            python_code,
            error
        )

    print("\n❌ Maximum repair rounds reached.")

    save_code(
        program_name,
        python_code
    )

    log_result(
        program_name,
        max_rounds,
        "Failed",
        error_type,
        False,
        elapsed
    )

    log_summary(
        program_name,
        initial_pass,
        False,
        max_rounds,
        elapsed
    )


# =====================================================
# Main
# =====================================================

def main():

    cpp_files = [

        os.path.join(
            SAMPLES_DIR,
            file
        )

        for file in os.listdir(SAMPLES_DIR)

        if file.endswith(".cpp")
    ]

    print(
        f"\nFound {len(cpp_files)} C++ files."
    )

    if len(cpp_files) == 0:

        print(
            f"\nNo C++ files found in:\n{SAMPLES_DIR}"
        )

        return

    for cpp_file in cpp_files:

        process_program(cpp_file)

    print("\nExperiment completed.")

    print(
        f"\nDetailed results:\n{CSV_FILE}"
    )

    print(
        f"\nSummary results:\n{SUMMARY_CSV}"
    )

    print(
        f"\nTranslated files:\n{TRANSLATED_DIR}"
    )


if __name__ == "__main__":
    main()