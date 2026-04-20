import subprocess
import tempfile
from gpt_api import call_gpt


def translate_cpp(cpp_code):
    prompt = f"""
Translate the following C++ code into correct Python 3 code.
Return ONLY Python code.

C++ code:
{cpp_code}
"""
    return call_gpt(prompt)


def fix_code(code, error):
    prompt = f"""
The following Python code has errors.

Code:
{code}

Error:
{error}

Fix the code.
Return ONLY corrected Python code.
"""
    return call_gpt(prompt)


def check_compile(code):
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        file_path = f.name

    result = subprocess.run(
        ["python", "-m", "py_compile", file_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return False, result.stderr
    return True, None


def main():
    with open("example.cpp", "r") as f:
        cpp_code = f.read()

    python_code = translate_cpp(cpp_code)
    print("Initial Translation:\n", python_code)

    max_rounds = 5

    for i in range(max_rounds):
        success, error = check_compile(python_code)

        if success:
            print(f"\n✅ Success at round {i}")
            break

        print(f"\n❌ Error at round {i}:\n", error)

        python_code = fix_code(python_code, error)

    print("\nFinal Python Code:\n")
    print(python_code)


if __name__ == "__main__":
    main()