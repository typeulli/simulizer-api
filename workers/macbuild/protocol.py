"""
Wire types shared between the macOS build server (routes/macbuild.py) and the
off-box macOS build worker (worker.py, run on a Mac).

The whole compile sandbox is shipped inline as a flat list of files — the
worker reconstructs the directory layout under a temp dir, runs the native
`clang++`, and streams the resulting binary back in base64 chunks (there is no
side HTTP channel; everything rides the WebSocket). Source payloads are small
text; the only sizeable direction is the returned binary, which is chunked so
no single frame is large.
"""

from typing import Optional

from pydantic import BaseModel


class MacBuildFile(BaseModel):
    # `path` is POSIX-relative to the reconstructed build-dir root (e.g.
    # "project/output.cpp", "_system/simstd.hpp", "_data/binary_data.cpp").
    path: str
    content_b64: str  # base64 of the raw file bytes


class MacBuildRequest(BaseModel):
    files: list[MacBuildFile]
    entry_rel: str        # translation unit, relative to build-dir root
    system_rel: str       # staged system-header dir (-I)
    project_rel: str      # materialized project root (-I)
    data_cpp_rel: str     # portable embedded-asset source compiled alongside
    out_name: str = "output"
    # Pre-resolved (server-validated) compiler flags — std/opt are closed enums
    # and each define matches a strict identifier regex, so nothing here is
    # attacker-controlled by the time it reaches the worker's command line.
    std_flag: str         # e.g. "-std=c++17"
    opt_flag: str         # e.g. "-O3"
    define_flags: list[str] = []  # e.g. ["-DSIMULIZER_ALONE"]


class MacBuildResponse(BaseModel):
    # one of: "progress" | "chunk" | "done" | "error"
    type: str = "progress"
    message: Optional[str] = None   # progress label or error detail
    seq: Optional[int] = None       # chunk ordinal (chunks arrive in order)
    data: Optional[str] = None      # base64 of one binary chunk
