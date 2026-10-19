#!/usr/bin/env python3
"""The generator's telem-windows check: the broker's telem-windows (config/gnss_chains_chord.yaml),
the gather's taps_win (FLEET_TRIM_TAPS_WINDOWS) and the combiners' record_export must agree.

    python3 config/test_gen_telem_windows.py

check_telem_windows passes on the committed chains file and refuses a copy whose telem-windows
differs, in `common:` or in one chain's block, or is missing (the broker then uses its own
default).
"""
import contextlib
import io
import os
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_chord_gnss_config as g  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def refused(chains, d):
    """SystemExit text from check_telem_windows on `chains` written to a file, or None."""
    path = os.path.join(d, "chains.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(chains, fh)
    try:
        g.check_telem_windows(path)
    except SystemExit as e:
        return str(e.code)
    return None


def main():
    try:
        g.check_telem_windows()
        check(True, "the committed chains file agrees")
    except SystemExit as e:
        check(False, "the committed chains file agrees (%s)" % e.code)

    with open(g.CHAINS_FILE) as fh:
        real = yaml.safe_load(fh)
    with tempfile.TemporaryDirectory() as d:
        msg = refused(real, d)
        check(msg is None, "a copy of the committed chains file agrees")

        bad = dict(real, common=dict(real["common"], **{"telem-windows": 16}))
        msg = refused(bad, d)
        check(
            msg is not None
            and "common: 16" in msg
            and "(32," in msg
            and "(128 " in msg,
            "common telem-windows 16 is refused, naming all three values (%s)" % msg,
        )

        chain = sorted(real["chains"])[0]
        bad = dict(real, chains=dict(real["chains"]))
        bad["chains"][chain] = dict(
            real["chains"][chain] or {}, **{"telem-windows": 16}
        )
        msg = refused(bad, d)
        check(
            msg is not None and "%s: 16" % chain in msg,
            "a chain overriding telem-windows to 16 is refused",
        )

        common = dict(real["common"])
        common.pop("telem-windows")
        msg = refused(dict(real, common=common), d)
        check(msg is not None, "a chains file without telem-windows is refused")

    rx = g.FLEET_TRIM_TAPS_WINDOWS * g.TELEM_RECORDS_PER_FRAME
    check(rx == 128, "record_export is %d records" % rx)

    # main runs the check: with the gather's depth at 16 the committed chains file disagrees.
    saved = g.FLEET_TRIM_TAPS_WINDOWS, sys.argv
    g.FLEET_TRIM_TAPS_WINDOWS = 16
    base = os.path.join(HERE, "base", "live_config_20260730.json")
    sys.argv = [g.__file__, "--base", base, "--node", "cx19", "--gather-instance"]
    msg = None
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            g.main()
    except SystemExit as e:
        msg = str(e.code)
    finally:
        g.FLEET_TRIM_TAPS_WINDOWS, sys.argv = saved
    check(
        msg is not None and "telem-windows" in msg,
        "the generator refuses to run when they disagree (%s)" % msg,
    )

    if _fails:
        print("\n%d FAILED" % len(_fails))
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
