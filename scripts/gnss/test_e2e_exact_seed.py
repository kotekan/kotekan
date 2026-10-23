#!/usr/bin/env python3
"""e2e's seed is exact on its own sky, so its VERDICT measures the chain and not the harness.

    python3 scripts/gnss/test_e2e_exact_seed.py [--exe scripts/gnss/e2e]

Needs a built e2e (site/build_tool.sh e2e) and a GPU. With --skip-search the seed is the truth,
and the sky's Doppler ramps at the rate the seed carries (--seed-dop-rate), so every record's
phase error is rounding. Each record must have |phase err| under TOL_CHIPS, and the commanded
replica must correlate like the truth one (P/P_true) and sit on the peak (disc): a sky that does
not ramp as its truth says fails one of the three.

At the live record cadence the prompt's per-record carrier step ([4a]) is checked too. The sky's
carrier phase is continuous while each record's replica runs at that record's Doppler from
absolute sample 0, so the step is rate * gap * (t_abs + gap/2) cycles, t_abs being a record's
reference time, and it changes by only rate * gap^2 from record to record. A sky whose carrier
jumps between records moves the mean or raises the sd. The deep fold ([4]) must then resolve the
phase residual about a linear ramp, whatever the size of the mean step, and find it under
TOL_RES.

Without the build or a GPU it exits 77 (environment absent), or fails when GNSS_REQUIRE_GPU is
set.
"""
import argparse
import glob
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(
    r"^\s+(\d+)\s+\S+\s+\S+\s+(\S+)\s+\S+\s+\|\s+(\S+)\s+(\S+)\s+(\S+)\s*$"
)
VERDICT = re.compile(r"^VERDICT: worst \|error\| over (\d+) records = (\S+) chips")
STEP = re.compile(r"\[4a\] RAW\s+per-record step: mean\s+(\S+) cyc\s+sd (\S+) cyc")
RES = re.compile(r"per-record phase residual about a linear ramp: (\S+) rad rms")
FFT, FS = 16384, 3.2e9  # e2e's hop length (2 x spectrum_length) and sample rate
TOL_CHIPS = 0.01
MIN_P_RATIO = 0.999
TOL_DISC = 0.01
TOL_STEP = 0.002  # cycles, on the step's mean and sd
TOL_RES = 0.01  # rad, the deep fold's residual about a linear ramp

CASES = [
    ("default rate, records at 27-57 s", []),
    (
        "rate -0.9 Hz/s, records at 300-330 s",
        ["--seed-dop-rate", "-0.9", "--age-s", "300"],
    ),
    ("rising Doppler from -1500 Hz", ["--dop", "-1500", "--seed-dop-rate", "0.5"]),
    ("constant Doppler", ["--seed-dop-rate", "0"]),
]
LIVE = (
    "default rate, 12 records at the live cadence",
    ["--hop0", "114436200145", "--age-s", "27", "--seed-dop-rate", "-0.36893"]
    + ["--rec-gap-s", "0.0104858", "--nrec", "12"],
)

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def opt(args, name, default):
    return float(args[args.index(name) + 1]) if name in args else default


def wrap1(x):
    """x cycles wrapped into [-0.5, 0.5)."""
    return x - math.floor(x + 0.5)


def expected_step(args):
    """The mean per-record carrier step, cycles, that LIVE's continuous sky gives."""
    hop_s = FFT / FS
    rate, n = opt(args, "--seed-dop-rate", None), int(opt(args, "--nrec", None))
    age = opt(args, "--age-s", None)
    hop = int(opt(args, "--hop0", None)) + math.floor(age / hop_s + 0.5)
    gap = math.floor(opt(args, "--rec-gap-s", None) / hop_s + 0.5) * hop_s
    t0 = (hop * FFT + FFT - 1) / FS
    return wrap1(rate * gap * (t0 + 0.5 * (n - 1) * gap))


def run(exe, args):
    r = subprocess.run([exe, "--skip-search"] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            "e2e failed (%d): %s" % (r.returncode, (r.stdout + r.stderr)[-400:])
        )
    return r.stdout


def check_case(exe, name, args, step=False):
    out = run(exe, args)
    rows = [m for m in map(ROW.match, out.splitlines()) if m]
    nrec = int(opt(args, "--nrec", 4))
    check(len(rows) == nrec, "%s: %d record rows (want %d)" % (name, len(rows), nrec))
    for m in rows:
        err, pr, disc = float(m.group(2)), float(m.group(3)), float(m.group(5))
        check(
            abs(err) < TOL_CHIPS and pr >= MIN_P_RATIO and abs(disc) < TOL_DISC,
            "%s: record %s err %+.3f chips, P/P_true %.4f, disc %+.4f"
            % (name, m.group(1), err, pr, disc),
        )
    v = [m for m in map(VERDICT.match, out.splitlines()) if m]
    check(
        len(v) == 1 and float(v[0].group(2)) < TOL_CHIPS and "CHAIN CLOSES" in out,
        "%s: worst error %s chips closes" % (name, v[0].group(2) if v else "missing"),
    )
    if step:
        s, want = STEP.search(out), expected_step(args)
        mean, sd = (float(s.group(1)), float(s.group(2))) if s else (math.nan, math.nan)
        check(
            abs(wrap1(mean - want)) < TOL_STEP and sd < TOL_STEP,
            "%s: carrier step mean %+.5f cycles (want %+.5f), sd %.5f"
            % (name, mean, want, sd),
        )
        r = RES.search(out)
        check(
            r is not None and float(r.group(1)) < TOL_RES,
            "%s: deep-fold residual %s"
            % (name, r.group(1) + " rad" if r else "unresolved or missing"),
        )


def absent(exe):
    """Why this host cannot run e2e (no build, or no GPU for its despread), or None."""
    if not os.path.exists(exe):
        return "no %s -- build it with site/build_tool.sh e2e" % exe
    if not glob.glob("/dev/nvidia[0-9]*"):
        return "no GPU (/dev/nvidia*) for e2e's despread"
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--exe", default=os.path.join(HERE, "e2e"))
    a = ap.parse_args()
    # 77 is "environment absent" (automake's skip status): it says nothing about the chain.
    # GNSS_REQUIRE_GPU (the test runner sets it on a GPU host) makes it a failure.
    why = absent(a.exe)
    if why and os.environ.get("GNSS_REQUIRE_GPU"):
        raise SystemExit("%s, and GNSS_REQUIRE_GPU is set" % why)
    if why:
        print("skip  %s" % why)
        return 77
    print("e2e exact-seed checks (%s)" % a.exe)
    for name, args in CASES:
        check_case(a.exe, name, args)
    check_case(a.exe, LIVE[0], LIVE[1], step=True)
    if _fails:
        print("FAILED (%d)" % len(_fails))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
