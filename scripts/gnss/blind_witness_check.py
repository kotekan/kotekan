#!/usr/bin/env python3
"""Task #47: is the served C/N0 backed by a prompt tap that is actually on the signal?

WHY THIS EXISTS. cn0_coh is computed from deep_snr, and the deep fold RE-SEARCHES rate and
phase -- so it re-finds the satellite no matter where the prompt tap was commanded. A row can
therefore serve 41 dB-Hz while E/P/L sit on pure noise, and dll_disc reads ~0 in that state
because a discriminator built from E ~= L ~= noise is zero. That combination (high C/N0, zero
discriminator, low scatter) is exactly what an operator reads as "the array looks great".

This script separates the two populations and checks that the prompt_lock flag lands between
them. Run it against banked observables:

    blind_witness_check.py <jsonl> [<jsonl> ...]
    blind_witness_check.py --window 2026-08-12T15:20:48 2026-08-12T15:45:00 <jsonl> ...

With --window it reports the flag's behaviour inside vs outside that interval, which is how the
2026-08-12 15:20-15:45 blind window was characterised. Exit 1 if the populations are NOT
separated, so this can serve as a gate rather than a report.
"""
import argparse
import datetime as dt
import json
import math
import statistics as st
import sys

# The witnesses, and why each alone is insufficient -- see publish.py's PROMPT LOCK block.
#   s4_raw   ABSOLUTE, per-satellite, no population needed; fooled by strong scintillation.
#   present  immune to scintillation; RELATIVE, so a uniformly strong fleet also reads low.
S4_BAR = 0.90


def load(paths, lo=None, hi=None):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                t = d.get("t")
                if t is None or (lo is not None and not (lo <= t < hi)):
                    continue
                rows.append(d)
    return rows


def blind(d):
    """The witnesses as measured OFFLINE from a row, independent of any flag the broker set --
    so this script can validate the flag rather than merely echo it."""
    s4 = d.get("s4_raw")
    inc = d.get("cn0_inc_dbhz")
    if s4 is None or inc is None:
        return None
    return s4 >= S4_BAR and inc < 28.0


def describe(rows, label):
    """Split on the offline witnesses and report what the SERVED number was doing in each."""
    hi = [d for d in rows if (d.get("cn0_coh_dbhz") or 0) >= 30.0]
    b = [d for d in hi if blind(d) is True]
    g = [d for d in hi if blind(d) is False]
    print("\n=== %s ===" % label)
    print("  bright-looking samples (served cn0_coh >= 30): %d" % len(hi))
    if not hi:
        return None
    print("  of those, BLIND by the offline witnesses:      %d  (%.1f%%)"
          % (len(b), 100.0 * len(b) / len(hi)))
    for name, pop in (("BLIND", b), ("tracking", g)):
        if len(pop) < 20:
            continue
        def med(k):
            v = [d[k] for d in pop if d.get(k) is not None]
            return st.median(v) if v else float("nan")
        disc = [abs(d["dll_disc"]) for d in pop if d.get("dll_disc") is not None]
        print("    %-9s n=%-6d served_cn0 %5.1f | s4_raw %5.2f  cn0_inc %5.1f  |disc| %5.3f"
              % (name, len(pop), med("cn0_coh_dbhz"), med("s4_raw"),
                 med("cn0_inc_dbhz"), st.median(disc) if disc else float("nan")))
    # Does the broker's own flag agree with the offline witnesses? Only meaningful once the
    # broker publishing prompt_lock has been deployed; older rows carry None.
    have = [d for d in hi if d.get("prompt_lock") is not None]
    if have:
        agree = sum(1 for d in have if (d["prompt_lock"] is False) == (blind(d) is True))
        print("  broker prompt_lock agrees with the offline witnesses on %d/%d (%.1f%%)"
              % (agree, len(have), 100.0 * agree / len(have)))
    else:
        print("  broker prompt_lock: not present in these rows (pre-#47 broker)")
    return b, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--window", nargs=2, metavar=("FROM", "TO"),
                    help="ISO UTC interval to characterise separately (e.g. a blind episode)")
    a = ap.parse_args()

    if a.window:
        def ts(s):
            return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp()
        lo, hi = ts(a.window[0]), ts(a.window[1])
        allr = load(a.paths)
        inside = [d for d in allr if lo <= d.get("t", 0) < hi]
        outside = [d for d in allr if not (lo <= d.get("t", 0) < hi)]
        r1 = describe(inside, "INSIDE the window")
        r2 = describe(outside, "OUTSIDE the window")
        if r1 and r2:
            fi = len(r1[0]) / max(1, len(r1[0]) + len(r1[1]))
            fo = len(r2[0]) / max(1, len(r2[0]) + len(r2[1]))
            print("\n  blind fraction inside %.1f%% vs outside %.1f%%" % (100 * fi, 100 * fo))
            # A window that is genuinely a blind episode must be dominated by it, and the rest
            # of the record must not be -- otherwise the witnesses are not discriminating and
            # nothing downstream should trust them.
            if fi < 0.5 or fo > 0.5:
                print("  FAIL: the witnesses do not separate this window from the rest.")
                return 1
            print("  PASS: the witnesses separate the window from the rest.")
        return 0

    describe(load(a.paths), "all samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
