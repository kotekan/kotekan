#!/usr/bin/env python3
"""How the DTV stage's source-0 response changes with time separation, and at which lags.

D(dt) = median over fit pairs of ||g(t+dt) - g(t)||^2 / ||g(t)||^2 per channel, then the median per station; the
fit noise is D at the shortest separation (one refit, ~5 min) / 2 per fit. Per lag: the share of the first->last
change ||g_last - g_first||^2 at each lag, against each lag's share of ||g_first||^2.

    dtv_fit_structure.py --from 0300 --to 1300
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/gnss/gnss-broker/scripts/gnss"))
import dtv_state  # noqa: E402

STATES = os.path.expanduser("~/gnss/builds/dtv-live/states")
NODES = ["cx19", "cx27", "cx42", "cx43", "cx44", "cx47", "cx51", "cx52"]
EDGE = {2714, 2715, 2743, 2744, 2898, 2928}
SEP_MIN = [5, 10, 20, 40, 60, 120, 240, 480, 560]


def tsec(s):
    return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="t0", default="030000")
    ap.add_argument("--to", dest="t1", default="130000")
    a = ap.parse_args()
    lo, hi = tsec(a.t0.ljust(6, "0")), tsec(a.t1.ljust(6, "0"))
    D = {24: {m: [] for m in SEP_MIN}, 30: {m: [] for m in SEP_MIN}}
    lagchange = {24: [], 30: []}
    lagbase = {24: [], 30: []}
    lag_min = None
    for node in NODES:
        ser = {}
        for p in glob.glob(os.path.join(STATES, node, "st_g[01]_*_r0.bin")):
            m = re.search(r"_(\d{6})_r0\.bin$", p)
            if not m or not lo <= tsec(m.group(1)) <= hi:
                continue
            st = dtv_state.read_state(p)
            lag_min = st["lag_min"]
            for fid, r in st["recs"].items():
                ser.setdefault(int(fid), []).append((tsec(m.group(1)), r["g"][0].copy()))
        for fid, v in ser.items():
            if fid in EDGE:
                continue
            v.sort(key=lambda x: x[0])
            t = np.array([x[0] for x in v])
            G = np.array([x[1] for x in v])
            P = np.sum(np.abs(G) ** 2, axis=(1, 2))
            stn = 24 if fid < 2800 else 30
            for msep in SEP_MIN:
                vals = []
                for i in range(len(t)):
                    j = np.searchsorted(t, t[i] + 60 * msep - 120)
                    if j < len(t) and abs(t[j] - t[i] - 60 * msep) <= 150:
                        vals.append(np.sum(np.abs(G[j] - G[i]) ** 2) / P[i])
                if vals:
                    D[stn][msep].append(np.median(vals))
            dg = np.sum(np.abs(G[-1] - G[0]) ** 2, axis=1)
            lagchange[stn].append(dg / dg.sum())
            b = np.sum(np.abs(G[0]) ** 2, axis=1)
            lagbase[stn].append(b / b.sum())
    for stn in (24, 30):
        print("ATSC %d: D(dt) median over channels (p10-p90)" % stn)
        for msep in SEP_MIN:
            v = np.array(D[stn][msep])
            if len(v):
                print("  %4d min  %.3f  (%.3f-%.3f)  n=%d" % (msep, np.median(v), np.percentile(v, 10), np.percentile(v, 90), len(v)))
        lc = np.median(np.array(lagchange[stn]), axis=0)
        lb = np.median(np.array(lagbase[stn]), axis=0)
        order = np.argsort(lc)[::-1][:10]
        print("  first->last change by lag (share of change | share of g): " +
              " ".join("%+d:%.2f|%.2f" % (lag_min + j, lc[j], lb[j]) for j in order))
        far = np.abs(np.arange(len(lc)) + lag_min) > 6
        print("  lags |d|>6 hops: %.2f of the change, %.2f of g" % (lc[far].sum(), lb[far].sum()))


if __name__ == "__main__":
    main()
