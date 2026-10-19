#!/usr/bin/env python3
"""nb_run.py's own check: with --cwd, the outputs go back into the notebook that was read.

    ~/gnss/venv/bin/python scripts/gnss/site/test_nb_run.py

nb_run.py imports matplotlib, so this runs under an interpreter that has it (venv, not venv-ft).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_relative_notebook_with_cwd():
    d = tempfile.mkdtemp(prefix="nb_run_")
    try:
        a, b = os.path.join(d, "A"), os.path.join(d, "B")
        os.makedirs(os.path.join(a, "sub"))
        os.makedirs(b)
        nb = os.path.join(a, "sub", "nb.ipynb")
        with open(nb, "w") as fh:
            json.dump(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "source": ["1+1"],
                            "metadata": {},
                            "outputs": [],
                            "execution_count": None,
                        }
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                },
                fh,
            )
        p = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "nb_run.py"),
                "sub/nb.ipynb",
                "--cwd",
                "../B",
            ],
            cwd=a,
            capture_output=True,
            text=True,
        )
        outs = json.load(open(nb))["cells"][0]["outputs"]
        check(
            p.returncode == 0
            and [o.get("data", {}).get("text/plain") for o in outs] == [["2"]],
            "sub/nb.ipynb run from A with --cwd ../B carries its execute_result (rc %d%s)"
            % (
                p.returncode,
                ": " + p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "",
            ),
        )
        check(
            not os.path.exists(os.path.join(b, "sub")),
            "and nothing is written under the --cwd directory",
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception as e:
            check(False, "%s raised %s: %s" % (t.__name__, type(e).__name__, e))
    if _fails:
        print("\n%d FAILED" % len(_fails))
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
