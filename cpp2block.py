import json
import subprocess
from pathlib import Path
path_executable = Path(__file__).parent / "bin" / "cpp2block.exe"
if not path_executable.exists():
    raise FileNotFoundError(f"Executable not found at {path_executable}. Please build the project first.")
def blocklynize(cpp_code: str) -> str:
    process = subprocess.Popen(
        [str(path_executable)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    line_number = cpp_code.count("\n") + 1
    input_data = f"{line_number}\n{cpp_code}"
    stdout, stderr = process.communicate(input=input_data)
    if process.returncode != 0:
        raise RuntimeError(f"cpp2block.exe failed with error: {stderr}")
    
    data = json.loads(stdout.strip())
    return data