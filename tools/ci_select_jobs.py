#!/usr/bin/env python3
"""Decide which CI test jobs a change needs.

Reads changed file paths from the arguments or stdin and prints ``cpu=true|false``,
``gpu=true|false`` and ``ptx=true|false`` lines for $GITHUB_OUTPUT.
``--self-test`` checks the cases in CASES instead; the Lint job runs it.

- cpu: the gcc/clang/Intel jobs. Run when any file matches CPU_NEED.
- gpu: the self-hosted GPU job. Run when any file matches GPU_NEED, or is a
  file the GPU build reaches through #include from lib/core, lib/metadata,
  the GPU libraries, the kotekan executable, or the stages
  config/ci-tests/gpu_batch uses. A .cpp counts when its header is reached.
  Also run when a changed source uses float16_t or a GPU conditional: the
  CPU builds compile it, but with another float16_t (see DataType.hpp).
  Boost tests do not count: the CPU jobs build and run them.
- ptx: the ptxas check of the generated kernels. Run when a generated
  kernel or its wrapper changes.
"""

import os
import re
import subprocess
import sys

# A file needs the CPU jobs unless the CPU builds neither compile nor read it.
CPU_NEED = re.compile(
    r"""
    ^(?!                            # anything except:
        lib/(cuda|hip|opencl|gpu)/  #   GPU libraries, built only with CUDA/HIP/OpenCL
      | julia/                      #   kernel generator; its output is in lib/cuda
      | config/ci-tests/gpu_batch/  #   configs only the GPU job runs
      | docs/                       #   built by the separate docs job
      | .*\.md$                     #   markdown anywhere
    )
    | ^lib/cuda/upchannelizeReference\.  # compiled into a CPU boost test
    """,
    re.VERBOSE,
)

# Files that need the GPU job whatever the #include scan finds.
GPU_NEED = re.compile(
    r"""
    ^(
        lib/(cuda|opencl|gpu)/                # GPU libraries, incl. generated kernels loaded at runtime
      | kotekan/                              # the executable and its config renderer
      | external/                             # vendored libraries (ksgpu, n2k, ...)
      | cmake/                                # build options and feature detection
      | (lib/[^/]+/|lib/)?CMakeLists\.txt$    # top-level and lib/ build files
      | config/ci-tests/gpu_batch/            # the GPU test configs
      | config/ci-tests/run_tests\.sh         # their runner
      | tools/docker/                         # the CI images
      | tools/ci_select_jobs\.py              # this script
      | \.github/                             # the CI workflows
    )
    """,
    re.VERBOSE,
)

# Files the ptxas check reads.
PTX_NEED = re.compile(
    r"^(lib/cuda/generated/|tools/check_ptx\.sh|tools/ci_select_jobs\.py|\.github/)"
)

# Source text that compiles differently in the GPU build.
GPU_TEXT = re.compile(r"float16_t|KOTEKAN_FLOAT16|WITH_(CUDA|HIP|OPENCL)|cuda_fp16")

# Where the #include scan starts, besides the stages the GPU test configs use.
GPU_ROOTS = re.compile(r"^(lib/(core|metadata|gpu|cuda|opencl)/|kotekan/)")
SOURCE = re.compile(r"\.(c|cpp|h|hpp|cuh|inc)$")


def git_files(*paths):
    return subprocess.check_output(["git", "ls-files", *paths], text=True).split()


def gpu_reached():
    """Return the path stems the GPU tests reach through #include."""
    src = [f for f in git_files("lib", "kotekan", "external") if SOURCE.search(f)]
    text = {f: open(f, errors="ignore").read() for f in src}

    # Stage types named in the GPU test configs, including .j2 templates.
    stages = set()
    for f in git_files("config/ci-tests/gpu_batch"):
        stages |= set(re.findall(r"kotekan_stage['\"]?:\s*['\"]?(\w+)", open(f).read()))

    # Start from the root directories and the files that register those stages.
    todo = [f for f in src if GPU_ROOTS.match(f)]
    for f in src:
        registered = re.findall(r"REGISTER_KOTEKAN_STAGE\(\s*(\w+)\s*\)", text[f])
        if stages.intersection(registered):
            todo.append(f)

    # Resolve includes by file name: each lib/ library exports its directory as
    # an include path, and a match that is too broad only adds GPU runs.
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
    # Compare stems so a .cpp counts when only its header is included.
    return {os.path.splitext(f)[0] for f in seen}


def select(files):
    """Return (cpu, gpu, ptx) for the changed paths."""
    cpu = any(CPU_NEED.search(f) for f in files)
    ptx = any(PTX_NEED.search(f) for f in files)
    gpu = any(GPU_NEED.search(f) for f in files)
    if not gpu:
        for f in files:
            if SOURCE.search(f) and os.path.exists(f):
                gpu = gpu or bool(GPU_TEXT.search(open(f, errors="ignore").read()))
    # The #include scan reads every source file, so skip it when already decided.
    if not gpu:
        reached = gpu_reached()
        gpu = any(os.path.splitext(f)[0] in reached for f in files)
    return cpu, gpu, ptx


# (changed paths, cpu, gpu, ptx)
CASES = [
    (["lib/stages/AirspyAlign.cpp"], True, False, False),
    (["./lib/stages/setFRB1Phase.cpp"], True, True, False),  # as typed by hand
    (["lib/stages/setFRB1Phase.cpp"], True, True, False),  # stage in gpu_batch
    (["lib/stages/processFeedGains.cpp"], True, True, False),  # uses float16_t
    (["lib/core/Stage.cpp"], True, True, False),  # GPU_ROOTS
    (["lib/cuda/cudaCommand.cpp"], False, True, False),
    (["lib/cuda/upchannelizeReference.cpp"], True, True, False),
    (["lib/cuda/generated/Upchannelizer_chord_U16_K4.ptx"], False, True, True),
    (["config/ci-tests/gpu_batch/verify_cuda_upchan.j2"], False, True, False),
    (["config/ci-tests/cpu_batch/new.yaml"], True, False, False),
    (["docs/sphinx/index.rst", "README.md"], False, False, False),
    (["tools/check_ptx.sh"], True, False, True),
    ([".github/workflows/main.yaml"], True, True, True),
]


def self_test():
    bad = 0
    for files, *want in CASES:
        got = list(select([os.path.relpath(f) for f in files]))
        if got != want:
            print(f"{files}: want cpu,gpu,ptx={want}, got {got}")
            bad += 1
    print(f"{len(CASES) - bad}/{len(CASES)} cases pass")
    return bad == 0


def main():
    os.chdir(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(0 if self_test() else 1)
    lines = sys.argv[1:] or sys.stdin
    files = [os.path.relpath(line.strip()) for line in lines if line.strip()]
    cpu, gpu, ptx = select(files)
    print(f"cpu={str(cpu).lower()}")
    print(f"gpu={str(gpu).lower()}")
    print(f"ptx={str(ptx).lower()}")


if __name__ == "__main__":
    main()
