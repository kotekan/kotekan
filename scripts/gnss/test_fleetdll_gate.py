#!/usr/bin/env python3
"""fleetdll_gate.py's own checks that need no C++ build: what a failing run leaves behind.

    python3 scripts/gnss/test_fleetdll_gate.py

The C++ arm is a stub: one that prints an empty answer, so the comparison fails on every PRN,
and one that exits non-zero, so the gate aborts.
"""
import os
import re
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


def _run_gate(stub_body):
    """(returncode, the directory the gate says it kept or None) for a stub C++ arm."""
    stub_dir = tempfile.mkdtemp(prefix="fleetdll_stub-")
    try:
        stub = os.path.join(stub_dir, "fleetdll")
        with open(stub, "w") as fh:
            fh.write("#!%s\n%s" % (sys.executable, stub_body))
        os.chmod(stub, 0o755)
        r = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "fleetdll_gate.py"),
                "--no-build",
                "--exe",
                stub,
            ],
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(stub_dir)
    m = re.search(r"^kept in (\S+)$", r.stdout, re.M)
    return r.returncode, (m.group(1) if m else None)


def _check_kept(rc, kept, what):
    check(rc == 1, "%s: exits 1 (rc %d)" % (what, rc))
    check(
        kept is not None and os.path.isfile(os.path.join(kept, "frames.bin")),
        "%s: names the directory it kept, with the fixture in it (%r)" % (what, kept),
    )
    # Remove only what the gate made: a temp dir of its own prefix.
    if kept and os.path.basename(kept).startswith("fleetdll_gate-"):
        shutil.rmtree(kept, ignore_errors=True)


def test_a_failing_run_keeps_its_fixture_and_says_where():
    rc, kept = _run_gate(
        "import json\nprint(json.dumps({'frames': 0, 'late_frames': 0, 'chains': {}}))\n"
    )
    _check_kept(rc, kept, "a wrong C++ answer")


def test_an_aborted_run_keeps_its_fixture_and_says_where():
    rc, kept = _run_gate("import sys\nsys.exit('stub refuses')\n")
    _check_kept(rc, kept, "a C++ arm that exits non-zero")


def main():
    print("fleetdll_gate.py checks")
    test_a_failing_run_keeps_its_fixture_and_says_where()
    test_an_aborted_run_keeps_its_fixture_and_says_where()
    if _fails:
        print("FAILED (%d)" % len(_fails))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
