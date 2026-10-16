#!/usr/bin/env python3
"""Run one X-engine integration test: run kotekan with the test's jinja variables, then check
the written output. See docs/sphinx/dev/dev_integration_tests.rst.

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
    out_dir = os.path.join(os.path.abspath(args.out), f"{args.tel}_{args.test}")
    config = render_config(args.tel, {})
    var = test.variables(args.tel, config)
    var["out_dir"] = out_dir
    config = render_config(args.tel, var)

    log_path = out_dir + ".log"
    if not args.check_only:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        code, seconds = run_kotekan(args, var, log_path)
        print(f"kotekan exited with {code} after {seconds:.0f} s (log: {log_path})")
        if code != 0:
            print(f"FAIL {args.tel} {args.test}: kotekan exit code {code}")
            return 1

    try:
        for line in test.check(out_dir, var, config):
            print("  " + line)
    except CheckFailed as error:
        print(f"FAIL {args.tel} {args.test}: {error}")
        return 1
    print(f"PASS {args.tel} {args.test}")
    if not args.keep:
        shutil.rmtree(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
