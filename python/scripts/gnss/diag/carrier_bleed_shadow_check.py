#!/usr/bin/env python3
"""Offline validation of the f_ref trim-bleed SHADOW trigger against the LIVE broker CAR logs.

Applies the exact trigger logic from gps_distributed_broker.py's shadow (converged + standing,
recency-windowed) to the per-PRN trim time series parsed out of the running node's broker logs.
Zero disruption -- reads logs only. Confirms the trigger fires on the real standing L2C trims and
not spuriously, before the shadow is ever deployed."""
import re
import sys

# trigger params (must match the broker defaults)
BLEED_HZ = 2.0
STABLE_EMITS = 8
STABLE_HZ = 0.4
MAX_SLOPE = 0.02   # Hz/s -- flat-trim gate (rejects a still-drifting trim)
RECENCY_S = 90.0
LOG_GAP_S = 60.0


def _slope(win):
    """Least-squares slope (Hz/s) of trim vs time over the window [(t, trim), ...]."""
    n = len(win)
    if n < 2:
        return 0.0
    tb = sum(t for t, _ in win) / n
    vb = sum(v for _, v in win) / n
    den = sum((t - tb) ** 2 for t, _ in win)
    return sum((t - tb) * (v - vb) for t, v in win) / den if den > 0 else 0.0

CAR_RE = re.compile(r"PRN (\d+) resid ([-+][\d.]+) Hz trim ([-+][\d.]+)")
TS_RE = re.compile(r"\[broker (\d+):(\d+):(\d+\.\d+)\]")


def parse(path):
    series = {}  # prn -> [(t, trim)]
    for line in open(path, errors="ignore"):
        if "CAR:" not in line:
            continue
        m = TS_RE.search(line)
        if not m:
            continue
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        for prn, _resid, trim in CAR_RE.findall(line.split("CAR:", 1)[1]):
            series.setdefault(int(prn), []).append((t, float(trim)))
    return series


def scan(series, use_slope=True):
    """Replay the broker's shadow trigger over a PRN's trim series; return list of fire events
    (t, trim, spread, slope). With use_slope=False the flat-slope gate is dropped -- lets the
    caller compare how many candidates the slope gate REJECTS (the still-drifting trims)."""
    fires = []
    bh = []
    last_log = -1e9
    for t, trim in series:
        bh.append((t, trim))
        bh = bh[-STABLE_EMITS:]
        vals = [v for _, v in bh]
        sl = _slope(bh)
        if (len(bh) >= STABLE_EMITS and t - bh[0][0] < RECENCY_S
                and abs(trim) >= BLEED_HZ and max(vals) - min(vals) <= STABLE_HZ
                and (not use_slope or abs(sl) <= MAX_SLOPE)
                and t - last_log >= LOG_GAP_S):
            last_log = t
            fires.append((t, trim, max(vals) - min(vals), sl))
    return fires


def main():
    for path in sys.argv[1:]:
        band = path.split("gps_")[-1].split("_broker")[0].split(".log")[0]
        series = parse(path)
        if not series:
            print("%-6s %s: no CAR lines" % (band, path))
            continue
        print("=== %s (%s) : %d PRNs with CAR history ===" % (band, path, len(series)))
        print("    (gate: |trim|>=%.1f, >=%d emits, spread<=%.2f, |slope|<=%.3f Hz/s)"
              % (BLEED_HZ, STABLE_EMITS, STABLE_HZ, MAX_SLOPE))
        n_flat = n_noslope = 0
        for prn in sorted(series):
            s = series[prn]
            trims = [v for _, v in s]
            fires = scan(s, use_slope=True)             # full gate (with flat-slope)
            fires_ns = scan(s, use_slope=False)         # gate WITHOUT the slope requirement
            span = max(trims) - min(trims)
            flag = ""
            if fires:
                n_flat += 1
                flag = "  <-- FLAT candidate x%d (trim ~%+.2f, |slope|~%.3f)" % (
                    len(fires), sum(f[1] for f in fires) / len(fires),
                    sum(abs(f[3]) for f in fires) / len(fires))
            elif fires_ns:
                # would have fired on spread alone, but the slope gate REJECTED it (drifting)
                n_noslope += 1
                flag = "  ..  drifting, slope-REJECTED x%d (|slope|~%.3f Hz/s)" % (
                    len(fires_ns), sum(abs(f[3]) for f in fires_ns) / len(fires_ns))
            print("  PRN %2d: %3d emits, trim [%+.2f, %+.2f] (span %.2f), last %+.2f%s"
                  % (prn, len(s), min(trims), max(trims), span, trims[-1], flag))
        print("  -> %d PRNs FLAT (bleed), %d rejected by the slope gate (were drifting)\n"
              % (n_flat, n_noslope))


if __name__ == "__main__":
    main()
