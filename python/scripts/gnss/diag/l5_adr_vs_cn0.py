#!/usr/bin/env python3
"""Per-arc ADR scatter (m, quadratic detrend) vs per-arc C/N0, to read the NOISE LAW.
  slope of log10(scatter) vs cn0_dbhz:  ~ -0.05/dB = 1/sqrt(SNR) plain-product floor
                                        ~ -0.10/dB = 1/SNR   squaring-fold
  python3 l5_adr_vs_cn0.py <obs.jsonl> <band> [min_el]
"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/lwlab/airspy_gps/kotekan/python/scripts/gnss")
import gnss_dtec_live as dt

FREQ = {"l5": 1176.45e6, "l2c": 1227.60e6, "e5a": 1176.45e6, "b2a": 1176.45e6,
        "l1": 1575.42e6, "e1": 1575.42e6, "b1c": 1575.42e6}


def main():
    path, band = sys.argv[1], sys.argv[2].lower()
    min_el = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    lam = dt.C_L / FREQ[band]
    # loader keeping cn0 + el
    per = {}
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("adr_cycles") is None or d.get("cn0_coh_dbhz") is None:
            continue
        if (d.get("el") or -90) < min_el:
            continue
        per.setdefault(d["prn"], []).append(
            (d["t"], d["adr_cycles"], d.get("trim_cycles") or 0.0,
             d.get("adr_arc"), d["cn0_coh_dbhz"]))
    for v in per.values():
        v.sort()
    arcs = []  # (scatter_m, cn0)
    for prn, rows in per.items():
        # split on combiner arc change
        pieces, cur, prev = [], [], None
        for r in rows:
            if prev is not None and r[3] != prev and cur:
                pieces.append(cur); cur = []
            cur.append(r); prev = r[3]
        if cur:
            pieces.append(cur)
        for piece in pieces:
            if len(piece) < 20:
                continue
            t = [r[0] for r in piece]
            y = [r[1] * lam for r in piece]  # raw ADR meters
            cn0 = np.median([r[4] for r in piece])
            sc = dt.arc_scatter(t, y, 10.0, 60.0, 2)
            for s, n, span in sc:
                arcs.append((s, cn0))
    if len(arcs) < 4:
        print("  too few arcs (%d)" % len(arcs)); return
    arcs.sort(key=lambda a: a[1])
    s = np.array([a[0] for a in arcs]); c = np.array([a[1] for a in arcs])
    slope, intc = np.polyfit(c, np.log10(s), 1)
    print("  %d arcs, cn0 %.0f-%.0f dB-Hz" % (len(arcs), c.min(), c.max()))
    # bin by cn0 into terciles
    for lo, hi, lbl in [(0, 33, "low "), (33, 66, "mid "), (66, 100, "high")]:
        idx = [i for i in range(len(arcs))
               if np.percentile(c, lo) <= c[i] <= np.percentile(c, hi)]
        if idx:
            print("    %s C/N0 %.0f dB-Hz: scatter med %.3f m (%d arcs)"
                  % (lbl, np.median(c[idx]), np.median(s[idx]), len(idx)))
    print("  >> log10(scatter) vs C/N0 SLOPE = %+.3f /dB  (%.3f/dB=1/SNR^2 fold, "
          "%.3f/dB=1/sqrt(SNR) plain)" % (slope, -0.10, -0.05))


if __name__ == "__main__":
    main()
