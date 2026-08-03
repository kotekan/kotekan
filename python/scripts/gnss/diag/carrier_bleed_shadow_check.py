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
STABLE_EMITS = 5
STABLE_HZ = 0.6
RECENCY_S = 30.0
LOG_GAP_S = 60.0

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


def scan(series):
    """Replay the broker's shadow trigger over a PRN's trim series; return list of fire events."""
    fires = []
    bh = []
    last_log = -1e9
    for t, trim in series:
        bh.append((t, trim))
        bh = bh[-STABLE_EMITS:]
        vals = [v for _, v in bh]
        if (len(bh) >= STABLE_EMITS and t - bh[0][0] < RECENCY_S
                and abs(trim) >= BLEED_HZ and max(vals) - min(vals) <= STABLE_HZ
                and t - last_log >= LOG_GAP_S):
            last_log = t
            fires.append((t, trim, max(vals) - min(vals)))
    return fires


def main():
    for path in sys.argv[1:]:
        band = path.split("gps_")[-1].split("_broker")[0].split(".log")[0]
        series = parse(path)
        if not series:
            print("%-6s %s: no CAR lines" % (band, path))
            continue
        print("=== %s (%s) : %d PRNs with CAR history ===" % (band, path, len(series)))
        n_cand = 0
        for prn in sorted(series):
            s = series[prn]
            trims = [v for _, v in s]
            fires = scan(s)
            span = max(trims) - min(trims)
            flag = ""
            if fires:
                n_cand += 1
                flag = "  <-- BLEED CANDIDATE x%d (trim ~%+.2f)" % (
                    len(fires), sum(f[1] for f in fires) / len(fires))
            print("  PRN %2d: %3d emits, trim range [%+.2f, %+.2f] (span %.2f), "
                  "last %+.2f%s" % (prn, len(s), min(trims), max(trims), span,
                                    trims[-1], flag))
        print("  -> %d/%d PRNs would fire the bleed trigger\n" % (n_cand, len(series)))


if __name__ == "__main__":
    main()
