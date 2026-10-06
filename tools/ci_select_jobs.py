#!/usr/bin/env python3
"""Decide which CI test jobs a change needs.

Reads changed file paths on stdin and prints ``cpu=true|false`` and
``gpu=true|false`` lines for $GITHUB_OUTPUT. Run from the repository root.

- cpu: the gcc/clang/Intel jobs. Skipped when every file is one those builds
  neither compile nor read.
- gpu: the self-hosted GPU job. Run for the GPU libraries, build and CI files,
  and files the GPU build reaches through #include from lib/core, lib/metadata,
  the GPU libraries, the kotekan executable, or the stages
  config/ci-tests/gpu_batch uses. A .cpp counts when its header is reached.
  Boost tests do not count: the CPU jobs build and run them.
"""

import os
import re
import subprocess
import sys

CPU_SKIP = re.compile(
    r"^(lib/(cuda|hip|opencl|gpu)/|julia/|config/ci-tests/gpu_batch/|docs/)|\.md$"
)
# Compiled into a CPU boost test despite living in lib/cuda.
CPU_NEED = re.compile(r"^lib/cuda/upchannelizeReference\.")

GPU_NEED = re.compile(
    r"^(lib/(cuda|opencl|gpu)/|kotekan/|external/|cmake/|(lib/[^/]+/|lib/)?CMakeLists\.txt$"
    r"|config/ci-tests/(gpu_batch/|run_tests\.sh)|tools/docker/|tools/ci_select_jobs\.py|\.github/)"
)
GPU_ROOTS = re.compile(r"^(lib/(core|metadata|gpu|cuda|opencl)/|kotekan/)")
SOURCE = re.compile(r"\.(c|cpp|h|hpp|cuh|inc)$")


def git_files(*paths):
    return subprocess.check_output(["git", "ls-files", *paths], text=True).split()


def gpu_reached():
    """Return the path stems the GPU tests reach through #include."""
    src = [f for f in git_files("lib", "kotekan", "external") if SOURCE.search(f)]
    text = {f: open(f, errors="ignore").read() for f in src}

    stages = set()
    for f in git_files("config/ci-tests/gpu_batch"):
        stages |= set(re.findall(r"kotekan_stage['\"]?:\s*['\"]?(\w+)", open(f).read()))

    todo = [f for f in src if GPU_ROOTS.match(f)]
    for f in src:
        registered = re.findall(r"REGISTER_KOTEKAN_STAGE\(\s*(\w+)\s*\)", text[f])
        if stages.intersection(registered):
            todo.append(f)

    by_name = {}
    for f in src:
        by_name.setdefault(os.path.basename(f), []).append(f)
    seen = set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for inc in re.findall(r'#\s*include\s+[<"]([^>"]+)[>"]', text[f]):
            todo += by_name.get(os.path.basename(inc), [])
    return {os.path.splitext(f)[0] for f in seen}


def main():
    files = [line.strip() for line in sys.stdin if line.strip()]
    cpu = any(not CPU_SKIP.search(f) or CPU_NEED.search(f) for f in files)
    gpu = any(GPU_NEED.search(f) for f in files)
    if not gpu:
        reached = gpu_reached()
        gpu = any(os.path.splitext(f)[0] in reached for f in files)
    print(f"cpu={str(cpu).lower()}")
    print(f"gpu={str(gpu).lower()}")


if __name__ == "__main__":
    main()
