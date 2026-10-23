#!/usr/bin/env python3
"""How far an SBAS GEO's broadcast state vector drifts with age under each propagation model of
gnss_ephemeris.sbas_state ("twobody", "constaccel"), measured against the later records of the
same satellite in the rapid BRDC: the next record's position is the truth (MT9's position LSB
is 0.08 m, the records come every 256 s).

Every S record of the chosen PRNs is propagated with both models to every later record of the
same PRN within --max-age-min; the 3-D position error is binned by age and the count, median
and max printed per bin and model, with the age at which each model's median (in --fine-min
bins) first crosses half an L5 chip (146.5 m, the tracker's pull-in). Read-only on the files.
Run it off the production VM:

    ssh cf06 'nice -n 19 ~/gnss/venv/bin/python <repo>/python/scripts/gnss/sbas_eph_check.py'

Defaults: PRNs 131, 133, 135 (the WAAS GEOs seen from DRAO) from the rapid BRDCs of
2026-10-21/22 in ~/.cache/kotekan_gps; pass other files (RINEX 3 nav, .gz allowed) to redo it.
"""
import argparse
import math
import os
import statistics
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_ephemeris as ge  # noqa: E402

MODELS = ("twobody", "constaccel")
BINS_MIN = ((0, 15), (15, 30), (30, 60), (60, 90), (90, 120), (120, 240))
HALF_CHIP_M = 0.5 * ge.C_LIGHT / 1.023e6  # L5 (and SBAS L5): 146.5 m
DEFAULT_FILES = tuple(
    os.path.expanduser("~/.cache/kotekan_gps/BRDC00WRD_R_2026%d0000_01D_MN.rnx.gz" % doy)
    for doy in (294, 295)
)


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def pair_errors(recs, max_age_s):
    """[(age_s, prn, {model: error_m})] over every (earlier, later) record pair within max_age_s.
    Records sharing an epoch (the same message from two stations) are not paired."""
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the 2 h warning: ages past it are the point here
        for (sysc, prn), rs in sorted(recs.items()):
            if sysc != "S":
                continue
            for i, a in enumerate(rs):
                for b in rs[i + 1 :]:
                    age = b["toe_gpst"] - a["toe_gpst"]
                    if age <= 0.0:
                        continue
                    if age > max_age_s:
                        break
                    err = {m: _dist(ge.sbas_state(a, b["toe_gpst"], model=m)[0], b["pos"]) for m in MODELS}
                    rows.append((age, prn, err))
    return rows


def bin_stats(rows, lo_min, hi_min, model):
    v = [r[2][model] for r in rows if lo_min * 60.0 <= r[0] < hi_min * 60.0]
    if not v:
        return 0, float("nan"), float("nan")
    return len(v), statistics.median(v), max(v)


def crossing_min(rows, model, fine_min):
    """Age (min) at which the median error in fine_min bins first reaches HALF_CHIP_M, by linear
    interpolation between bin centres; None if it never does within the data."""
    prev = None
    k = 0
    while True:
        lo, hi = k * fine_min, (k + 1) * fine_min
        n, med, _ = bin_stats(rows, lo, hi, model)
        if n == 0 and prev is None and not any(r[0] >= lo * 60.0 for r in rows):
            return None
        if n:
            centre = 0.5 * (lo + hi)
            if med >= HALF_CHIP_M:
                if prev is None:
                    return centre
                c0, m0 = prev
                return c0 + (centre - c0) * (HALF_CHIP_M - m0) / (med - m0)
            prev = (centre, med)
        if not any(r[0] >= hi * 60.0 for r in rows):
            return None
        k += 1


def print_table(rows, title):
    print(title)
    head = "  age (min)   " + "".join("| %-10s n   median    max  " % m for m in MODELS)
    print(head)
    for lo, hi in BINS_MIN:
        line = "  %3d-%-3d     " % (lo, hi)
        for m in MODELS:
            n, med, mx = bin_stats(rows, lo, hi, m)
            line += "| %-10s %4d %7.1f %7.1f " % ("", n, med, mx)
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", default=list(DEFAULT_FILES), help="RINEX 3 nav files")
    ap.add_argument("--prn", default="131,133,135", help="SBAS PRNs (RINEX S number + 100)")
    ap.add_argument("--max-age-min", type=float, default=240.0)
    ap.add_argument("--fine-min", type=float, default=10.0, help="bin width for the crossing")
    ap.add_argument("--per-prn", action="store_true", help="also print the table per PRN")
    args = ap.parse_args(argv)
    prns = {int(p) for p in args.prn.split(",")}
    recs = {k: v for k, v in ge.parse_rinex_nav(args.files).items() if k[0] == "S" and k[1] in prns}
    for f in args.files:
        print("file:", f)
    for k in sorted(recs):
        toes = [e["toe_gpst"] for e in recs[k]]
        print(
            "PRN %d: %d records, %.1f h span, cadence median %.0f s"
            % (k[1], len(toes), (toes[-1] - toes[0]) / 3600.0, statistics.median(b - a for a, b in zip(toes, toes[1:])))
        )
    rows = pair_errors(recs, args.max_age_min * 60.0)
    print("%d record pairs within %.0f min; errors in metres (3-D position against the later record)" % (len(rows), args.max_age_min))
    print_table(rows, "all PRNs:")
    for m in MODELS:
        c = crossing_min(rows, m, args.fine_min)
        print("  %-10s median crosses 0.5 L5 chip (%.1f m) at %s" % (m, HALF_CHIP_M, "%.0f min" % c if c is not None else "never within the data"))
    if args.per_prn:
        for prn in sorted(prns):
            sub = [r for r in rows if r[1] == prn]
            if sub:
                print_table(sub, "PRN %d:" % prn)
                for m in MODELS:
                    c = crossing_min(sub, m, args.fine_min)
                    print("  %-10s crosses 0.5 chip at %s" % (m, "%.0f min" % c if c is not None else "never"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
