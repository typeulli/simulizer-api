import subprocess
import pathlib
import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

from strip_simstd import strip_simstd

console = Console()
path_here = pathlib.Path(__file__).parent


def embed_as_hpp(input_path: str, output_path: str, var_name: str):
    data = open(input_path, "rb").read()
    chunks = [f"0x{b:02x}" for b in data]
    rows = [", ".join(chunks[i:i+12]) for i in range(0, len(chunks), 12)]
    body = ",\n  ".join(rows)
    hpp = f"unsigned char {var_name}[] = {{\n  {body}\n}};\nunsigned int {var_name}_len = {len(data)};\n"
    open(output_path, "w").write(hpp)


def run_cmd(cmd: list[str]):
    """Run a subprocess and raise with its output if it fails."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(message or f"명령이 종료 코드 {result.returncode}로 실패했습니다.")


def make_bin_dir():
    (path_here / "bin").mkdir(exist_ok=True)


def write_binary_data_cpp():
    path_data_cpp = path_here / "lib/bin/binary_data.cpp"
    path_data_cpp.write_text("""
#include "console.hpp"
#include "index.html.hpp"
#include "console.html.hpp"
""", encoding="utf-8")


def compile_binary_data():
    path_data_cpp = path_here / "lib/bin/binary_data.cpp"
    run_cmd([
        "C:/mingw64/bin/g++.exe",
        "-c", str(path_data_cpp),
        "-std=c++17",
        "-o", str(path_here / "bin/binary_data.o"),
        f"-I{path_here}",
    ])


def build_emcc_runtime():
    """em++ 는 .bat 파일이라 shell=True 로 실행해야 PATH 해소가 됩니다."""
    cmd = (
        "em++ runtime.cpp "
        "-O3 -std=c++17 "
        "-sMAIN_MODULE=2 "
        "-sALLOW_MEMORY_GROWTH=1 "
        "-sALLOW_TABLE_GROWTH=1 "
        "--js-library simulizer_lib.js "
        "-sMODULARIZE=1 -sEXPORT_NAME=createSimulizerRuntime "
        "-sEXPORTED_RUNTIME_METHODS=['loadDynamicLibrary','FS','cwrap','ccall'] "
        "-o bin/runtime.js"
    )
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(message or f"em++ exited with code {result.returncode}")


# (단계 이름, 실행 함수) 목록 — 위에서 아래 순서로 빌드가 진행됩니다.
steps = [
    ("bin 디렉터리 준비", make_bin_dir),
    ("cpp2block.exe 컴파일 (clang++)", lambda: run_cmd([
        "C:/Program Files/LLVM/bin/clang++.exe",
        "cpp2block.cpp",
        "-O3",
        "-o", "bin/cpp2block.exe",
        "-IC:/Program Files/LLVM/include",
        "C:/Program Files/LLVM/lib/libclang.lib",
        "-std=c++17",
    ])),
    ("block2cpp.exe 컴파일 (g++)", lambda: run_cmd([
        "C:/mingw64/bin/g++.exe",
        "block2cpp.cpp",
        "-O3",
        "-o", "bin/block2cpp.exe",
        "-IC:/dev/vcpkg/installed/x64-windows/include",
        "-std=c++17",
        "-static", "-static-libgcc", "-static-libstdc++",
    ])),
    ("리소스 컴파일 (windres)", lambda: run_cmd([
        "C:/mingw64/bin/windres.exe",
        "res/simulizer.rc",
        "-o", "bin/resource.o",
        "--output-format=coff",
    ])),
    ("simstd.hpp 본문 제거 (libclang)", lambda: strip_simstd(
        path_here / "simstd.hpp", path_here / "bin/include/simstd.hpp")),
    ("index.html 임베드", lambda: embed_as_hpp(
        "res/index.html", "lib/bin/index.html.hpp", "_binary_assets_index_html")),
    ("console.html 임베드", lambda: embed_as_hpp(
        "res/console.html", "lib/bin/console.html.hpp", "_binary_assets_console_html")),
    ("binary_data.cpp 생성", write_binary_data_cpp),
    ("binary_data.o 컴파일 (g++)", compile_binary_data),
    ("runtime 모듈 빌드 (em++)", build_emcc_runtime),
]


def main():
    total = len(steps)
    console.rule("[bold cyan]Simulizer 빌드 시작[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("빌드 진행 중", total=total)

        for index, (desc, fn) in enumerate(steps, start=1):
            label = f"[{index}/{total}] {desc}"
            progress.update(task, description=label)
            progress.console.print(f"  [yellow]●[/yellow] {label} ...")

            try:
                fn()
            except Exception as error:
                progress.console.print(f"  [bold red]✗[/bold red] {label}")
                progress.console.print(
                    Panel(str(error), title="[red]오류[/red]", border_style="red")
                )
                console.rule("[bold red]빌드 실패[/bold red]")
                sys.exit(1)

            progress.console.print(f"  [bold green]✓[/bold green] {label}")
            progress.advance(task)

        progress.update(task, description="[bold green]모든 단계 완료[/bold green]")

    console.rule("[bold green]빌드 완료 🎉[/bold green]")


if __name__ == "__main__":
    main()
