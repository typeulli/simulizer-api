import subprocess
from pathlib import Path

path_executable = Path(__file__).parent / "bin" / "block2cpp.exe"
if not path_executable.exists():
    raise FileNotFoundError(f"Executable not found at {path_executable}. Please build the project first.")


def cppize(blockjson: str, main_fn_name: str = "main", target: str = "cpp") -> str:
    process = subprocess.Popen(
        [str(path_executable), f"--target={target}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    input_data = f"{main_fn_name}\n{blockjson}"
    stdout, stderr = process.communicate(input=input_data)
    if process.returncode != 0:
        raise RuntimeError(f"block2cpp.exe failed with error: {stderr}")
    return stdout


if __name__ == "__main__":
    import sys

    print(cppize(sys.stdin.read()))
