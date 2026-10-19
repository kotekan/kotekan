#!/usr/bin/env python3
"""gen_fleet.py's own checks: the generator's per-node notes reach the operator, and --check
exits 1 only for a config that differs (2 when nothing could be compared).

    python3 scripts/gnss/test_gen_fleet.py

No generator runs here: main() is driven with subprocess.run replaced, against a throwaway
manifest, a one-line template as the base and a preview directory.
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_fleet  # noqa: E402

NOTE = (
    "  note: GPS_L2C_CM has no covering channel on cx44 (carrier 1227.60 MHz, mainlobe "
    "1.023 MHz) -- chain SKIPPED on this node\n"
)

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def manifest(d, base):
    path = os.path.join(d, "m.yaml")
    with open(path, "w") as fh:
        fh.write("base: %s\nnodes:\n  cx44: {}\n" % base)
    return path


def test_relay_notes():
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        gen_fleet._relay_notes("cx44", "  node          cx44\n" + NOTE + NOTE)
    check(
        err.getvalue() == "cx44: " + NOTE.strip() + "\n",
        "a SKIPPED note is relayed once with the node's name, the summary is not",
    )
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        gen_fleet._relay_notes("cx44", "  WARNING   the EOP table above already ends\n")
    check(
        err.getvalue() == "cx44: WARNING   the EOP table above already ends\n",
        "a WARNING is relayed",
    )
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        gen_fleet._relay_notes("cx44", "  node          cx44\n  rest port     12048\n")
    check(err.getvalue() == "", "a run with no SKIPPED or WARNING relays nothing")


def test_check_relays_notes():
    """--check of one node whose generator succeeds with a note on stderr."""
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.j2")
        with open(base, "w") as fh:
            fh.write("a: 1\n")
        with open(os.path.join(d, "chord_gnss_cx44.yaml"), "w") as fh:
            fh.write("a: 1\n")
        argv = ["gen_fleet.py", manifest(d, base), "--check", "--preview-dir", d]

        def fake_run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout="a: 1\n", stderr=NOTE)

        real_run, real_argv = subprocess.run, sys.argv
        subprocess.run, sys.argv = fake_run, argv
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = gen_fleet.main()
        finally:
            subprocess.run, sys.argv = real_run, real_argv
    check(rc is None and "ok " in out.getvalue(), "the node checks ok")
    check(
        "cx44: note: GPS_L2C_CM" in err.getvalue(),
        "--check relays the generator's SKIPPED note",
    )


def run_check(base):
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.run(
            [sys.executable, os.path.join(HERE, "gen_fleet.py"), manifest(d, base)]
            + ["--check", "--preview-dir", d],
            capture_output=True,
            text=True,
        )
    return p.returncode, p.stderr


def test_missing_base_exits_2():
    rc, err = run_check("/nonexistent/base.j2")
    check(rc == 2, "--check with a missing base exits 2 (got %d)" % rc)
    check("does not exist" in err, "and says the base is missing")


def test_crash_exits_2():
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "broken.j2")
        with open(base, "w") as fh:
            fh.write("{% not a tag %}\n")
        rc, err = run_check(base)
    check(rc == 2, "--check whose stock render raises exits 2 (got %d)" % rc)
    check("Traceback" in err, "and prints the traceback")


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
