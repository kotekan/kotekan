#!/usr/bin/env python3
"""Run one X-engine integration test: run kotekan with the test's jinja variables, then check
the written output. See docs/sphinx/dev/dev_integration_tests.rst.

A test module defines either `VARIANTS`, each run and checked on its own, or `RUNS`, which are
all run before one check compares them.

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
import time

import jinja2
import yaml

from kit import CheckFailed

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO, "config", "integration-tests")


def render_config(tel, var):
    """Render the test config the way kotekan does and return the top-level values. Only
    literal values are usable; kotekan evaluates arithmetic expressions itself."""
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(CONFIG_DIR))
    return yaml.safe_load(env.get_template(f"{tel}.j2").render(var))


def run_kotekan(args, var, log_path):
    cmd = [
        os.path.abspath(args.kotekan),
        "--bind-address",
        f"localhost:{args.port}",
        "--config",
        os.path.join(CONFIG_DIR, f"{args.tel}.j2"),
        "--variables",
        json.dumps(var),
    ]
    with open(log_path, "w") as log:
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=args.cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        return code, time.time() - t0


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


def run(args, label, out_dir, var):
    """Run kotekan into `out_dir`, unless --check-only; return whether it succeeded."""
    if args.check_only:
        return True
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    log_path = out_dir + ".log"
    code, seconds = run_kotekan(args, var, log_path)
    print(f"kotekan exited with {code} after {seconds:.0f} s (log: {log_path})")
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
    if not run(args, label, out_dir, var):
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
