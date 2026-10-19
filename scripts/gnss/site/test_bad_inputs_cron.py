#!/usr/bin/env python3
"""bad_inputs_cron.py's own check: a failed copy from choco keeps the staged list in use.

    python3 scripts/gnss/site/test_bad_inputs_cron.py

Nothing reaches choco or a node: stage() runs with subprocess.run replaced and the staged files
pointed at a throwaway directory.
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bad_inputs_cron  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def fake_run(fail_on):
    """subprocess.run for stage(): the ssh stat reports a new state.json; scp copies until the
    fail_on-th call (1-based), which fails."""
    calls = []

    def run(cmd, **kw):
        if cmd[0] == "ssh":
            return subprocess.CompletedProcess(cmd, 0, stdout=b"200 2\n", stderr=b"")
        calls.append(cmd)
        if len(calls) == fail_on:
            raise subprocess.CalledProcessError(1, cmd)
        with open(cmd[-1], "w") as fh:
            fh.write("new\n")
        return subprocess.CompletedProcess(cmd, 0)

    return run


def staged(d, have):
    paths = [os.path.join(d, n) for n in ("state.json", "bffs.yaml", "state.json.meta")]
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
    if have:
        for p, text in zip(paths, ("old\n", "old\n", "100 1\n")):
            with open(p, "w") as fh:
                fh.write(text)
    (
        bad_inputs_cron.STAGE,
        bad_inputs_cron.STAGE_CONF,
        bad_inputs_cron.STAGE_META,
    ) = paths
    return paths


def read(p):
    return open(p).read() if os.path.exists(p) else None


def test_scp_failure_keeps_the_staged_list():
    d = tempfile.mkdtemp(prefix="bad_inputs_")
    real_run = bad_inputs_cron.subprocess.run
    try:
        for fail_on, which in ((1, "state.json"), (2, "bffs.yaml")):
            paths = staged(d, True)
            bad_inputs_cron.subprocess.run = fake_run(fail_on)
            try:
                note = bad_inputs_cron.stage()
            except Exception as e:
                note = "raised %s" % type(e).__name__
            check(
                note is not None and note.startswith("scp failed"),
                "a failed copy of %s with a staged list is a note, not a raise (%r)"
                % (which, note),
            )
            check(
                [read(p) for p in paths] == ["old\n", "old\n", "100 1\n"],
                "and the staged pair and its meta are left as they were",
            )
        paths = staged(d, False)
        bad_inputs_cron.subprocess.run = fake_run(1)
        try:
            bad_inputs_cron.stage()
            raised = False
        except subprocess.CalledProcessError:
            raised = True
        check(
            raised, "with nothing staged the failure still raises (main() logs NO LIST)"
        )
        paths = staged(d, True)
        bad_inputs_cron.subprocess.run = fake_run(0)
        note = bad_inputs_cron.stage()
        check(
            note == "staged bffs state (200 2)"
            and [read(p) for p in paths] == ["new\n", "new\n", "200 2\n"],
            "both copies succeeding replace the pair and the meta (%r)" % note,
        )
    finally:
        bad_inputs_cron.subprocess.run = real_run
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
