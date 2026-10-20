#!/usr/bin/env python3
"""Run one X-engine integration test: run kotekan with the test's jinja variables, then check
the written output. See docs/sphinx/dev/dev_integration_tests.rst.

A test module defines either `VARIANTS`, each run and checked on its own, or `RUNS`, which are
all run before one check compares them. A variant can also POST to kotekan's REST server while it
runs: `posts(tel, config, variant)` returns a list of (delay, endpoint, payload), each posted
`delay` seconds after the server answers. If the module defines `MEMORY_INTERVAL`, kotekan's
resident memory and GPU memory are sampled every `MEMORY_INTERVAL` seconds into
`<out_dir>/memory.json` (see `sample_memory`).

Example:
    run.py --tel pathfinder --test timing --kotekan build-release/kotekan/kotekan \\
        --out /scratch/$USER/kotekan-integration-tests
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

import jinja2
import yaml

from kit import CheckFailed

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO, "config", "integration-tests")


def render_yaml(tel, var):
    """The test config rendered the way kotekan renders it, as YAML text."""
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(CONFIG_DIR))
    return env.get_template(f"{tel}.j2").render(var)


def render_config(tel, var):
    """Render the test config and return the top-level values. Only literal values are usable;
    kotekan evaluates arithmetic expressions itself."""
    return yaml.safe_load(render_yaml(tel, var))


def rest_url(args, endpoint):
    return f"http://localhost:{args.port}/{endpoint.lstrip('/')}"


def post_all(args, proc, posts):
    """Wait for kotekan's REST server, then make the POSTs `posts` (see the module docstring)
    while kotekan runs."""
    while proc.poll() is None:
        try:
            urllib.request.urlopen(rest_url(args, "status"), timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    t_up = time.time()
    for delay, endpoint, payload in sorted(posts, key=lambda post: post[0]):
        time.sleep(max(t_up + delay - time.time(), 0))
        if proc.poll() is not None:
            return
        request = urllib.request.Request(
            rest_url(args, endpoint),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            print(f"POST {endpoint} {json.dumps(payload)}: {response.status}")


def gpu_memory_mib(pid):
    """The GPU memory (MiB) that nvidia-smi reports for process `pid`, or None."""
    try:
        lines = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return None
    used = [int(m) for p, m in (line.split(",") for line in lines) if int(p) == pid]
    return sum(used) if used else None


def sample_memory(proc, interval, samples):
    """Append [seconds since start, VmRSS (KiB), GPU memory (MiB) or None] for kotekan to
    `samples` every `interval` seconds while it runs."""
    t0 = time.time()
    while proc.poll() is None:
        try:
            with open(f"/proc/{proc.pid}/status") as status:
                rss = [
                    int(line.split()[1]) for line in status if line.startswith("VmRSS:")
                ]
        except OSError:
            break
        if rss:
            samples.append([time.time() - t0, rss[0], gpu_memory_mib(proc.pid)])
        time.sleep(interval)


def run_kotekan(args, var, log_path, posts, memory_interval=None):
    # kotekan gets the rendered config as a file: the variables can exceed the 128 KiB limit
    # of a command line argument (T7's 4096 manual FRB2 beams for CHORD)
    config_path = os.path.splitext(log_path)[0] + ".yaml"
    with open(config_path, "w") as f:
        f.write(render_yaml(args.tel, var))
    cmd = [
        os.path.abspath(args.kotekan),
        "--bind-address",
        f"localhost:{args.port}",
        "--config",
        config_path,
    ]
    with open(log_path, "w") as log:
        t0 = time.time()
        with subprocess.Popen(
            cmd, cwd=args.cwd, stdout=log, stderr=subprocess.STDOUT
        ) as proc:
            samples = []
            if memory_interval:
                threading.Thread(
                    target=sample_memory,
                    args=(proc, memory_interval, samples),
                    daemon=True,
                ).start()
            try:
                post_all(args, proc, posts)
                code = proc.wait(timeout=max(t0 + args.timeout - time.time(), 0))
            except subprocess.TimeoutExpired:
                proc.kill()
                code = "timeout"
            except OSError as error:
                proc.kill()
                code = f"POST failed ({error})"
        return code, time.time() - t0, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tel", required=True, choices=["pathfinder", "chord", "chime"]
    )
    parser.add_argument("--test", required=True)
    parser.add_argument(
        "--variant", action="append", help="run only this variant (repeatable)"
    )
    parser.add_argument("--kotekan", required=True, help="the kotekan binary")
    parser.add_argument(
        "--cwd", default=REPO, help="working directory for kotekan (default: the repo)"
    )
    parser.add_argument(
        "--out", required=True, help="scratch directory for test output"
    )
    parser.add_argument("--port", type=int, default=12048)
    parser.add_argument("--timeout", type=float, default=1800, help="seconds")
    parser.add_argument(
        "--keep", action="store_true", help="keep the output after a pass"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="re-check existing output"
    )
    args = parser.parse_args()

    test = importlib.import_module(f"check_{args.test}")
    if hasattr(test, "RUNS"):
        if args.variant:
            parser.error(f"--variant: check_{args.test} compares all of its runs")
        return 0 if run_compared(args, test) else 1
    variants = args.variant or test.VARIANTS
    failed = 0
    for variant in variants:
        failed += not run_variant(args, test, variant)
    return 1 if failed else 0


def prepare(args, test, name):
    """The output directory, jinja variables and rendered config of variant or run `name`."""
    out_dir = os.path.join(os.path.abspath(args.out), f"{args.tel}_{args.test}_{name}")
    config = render_config(args.tel, {})
    var = test.variables(args.tel, config, name)
    var["out_dir"] = out_dir
    return out_dir, var, render_config(args.tel, var)


def run(args, label, out_dir, var, posts=(), memory_interval=None):
    """Run kotekan into `out_dir`, unless --check-only; return whether it succeeded."""
    if args.check_only:
        return True
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    log_path = out_dir + ".log"
    code, seconds, samples = run_kotekan(args, var, log_path, posts, memory_interval)
    print(f"kotekan exited with {code} after {seconds:.0f} s (log: {log_path})")
    if memory_interval:
        with open(os.path.join(out_dir, "memory.json"), "w") as f:
            json.dump(samples, f)
    if code != 0:
        print(f"FAIL {label}: kotekan exit code {code}")
        return False
    return True


def report(args, label, check, check_args, out_dirs):
    """Call `check(*check_args)` and print its result; after a pass delete `out_dirs` unless
    --keep. Return whether the check passed."""
    try:
        for line in check(*check_args):
            print("  " + line)
    except CheckFailed as error:
        print(f"FAIL {label}: {error}")
        return False
    print(f"PASS {label}")
    if not args.keep:
        for out_dir in out_dirs:
            shutil.rmtree(out_dir)
    return True


def run_variant(args, test, variant):
    """Run and check one variant of a test; return whether it passed."""
    label = f"{args.tel} {args.test} {variant}"
    out_dir, var, config = prepare(args, test, variant)
    posts = test.posts(args.tel, config, variant) if hasattr(test, "posts") else []
    if not run(
        args, label, out_dir, var, posts, getattr(test, "MEMORY_INTERVAL", None)
    ):
        return False
    return report(args, label, test.check, (out_dir, var, config), [out_dir])


def run_compared(args, test):
    """Run every run in `test.RUNS`, then check them together with `test.check(runs)`, where
    `runs` maps each run name to its output directory, jinja variables and config."""
    label = f"{args.tel} {args.test}"
    runs = {name: prepare(args, test, name) for name in test.RUNS}
    for name, (out_dir, var, config) in runs.items():
        if not run(args, f"{label} {name}", out_dir, var):
            return False
    out_dirs = [out_dir for out_dir, var, config in runs.values()]
    return report(args, label, test.check, (runs,), out_dirs)


if __name__ == "__main__":
    sys.exit(main())
