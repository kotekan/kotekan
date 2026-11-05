#!/usr/bin/env python3
"""The DTV stage's fits over a night: every state file the refit loop wrote, per node, GPU, channel and source.

Per (node, freq_id, source), in time order: |w - w_first| (the weights are fixed between eigen starts, so g is
comparable across fits), the relative change between consecutive fits ||g_t - g_{t-1}||^2 / ||g_{t-1}||^2, the
coherence |<g_t, g_first>| / (||g_t|| ||g_first||), ||g_t||^2, and the lag profile sum_inputs |g[lag]|^2. From the
node logs: power left before each refit per channel, and each fit's explained fraction.

    dtv_fit_history.py --day 20261103 --from 0250 --to 1300 --out /var/tmp/kvand-vis/dtv_fits_20261103.npz
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
LOGS = "/var/tmp/kvand-dtvlive/log"  # on cf06
NODES = ["cx19", "cx27", "cx42", "cx43", "cx44", "cx47", "cx51", "cx52"]
NAME = re.compile(r"st_g(\d)_(\d{4}|\d{6})_([rs])(\d)\.bin$")


def tsec(hhmmss):
    s = hhmmss if len(hhmmss) == 6 else hhmmss + "00"
    return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="t0", default="0250")
    ap.add_argument("--to", dest="t1", default="1300")
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", default=LOGS)
    a = ap.parse_args()
    lo, hi = tsec(a.t0), tsec(a.t1)
    rows = []  # (node, fid, source, t, kind, dw, g)
    for node in NODES:
        for p in sorted(glob.glob(os.path.join(STATES, node, "st_g*.bin"))):
            m = NAME.search(p)
            if not m:
                continue
            t = tsec(m.group(2))
            if not lo <= t <= hi:
                continue
            kind, k = m.group(3), int(m.group(4))
            st = dtv_state.read_state(p)
            for fid, r in st["recs"].items():
                # a state file carries every source; only the one this file fitted (or both, at an eigen start) is new
                srcs = [k] if kind == "r" else [k]
                for s in srcs:
                    if np.any(r["g"][s] != 0):
                        rows.append((node, int(fid), s, t, kind, r["w"][s].copy(), r["g"][s].copy()))
    # per (node, fid, source): the fits in time order, de-duplicated (a source-1 fit file repeats source 0's g)
    series = {}
    for node, fid, s, t, kind, w, g in rows:
        series.setdefault((node, fid, s), []).append((t, kind, w, g))
    out = {}
    summ = []
    for key, v in sorted(series.items()):
        v.sort(key=lambda x: x[0])
        t = np.array([x[0] for x in v])
        W = np.array([x[2] for x in v])
        G = np.array([x[3] for x in v])  # [n, n_lag, NI]
        n = len(t)
        g0 = G[0].ravel()
        dw = np.array([np.linalg.norm(W[i] - W[0]) / max(np.linalg.norm(W[0]), 1e-30) for i in range(n)])
        pw = np.array([np.sum(np.abs(G[i]) ** 2) for i in range(n)])
        coh = np.array([np.abs(np.vdot(g0, G[i].ravel())) / max(np.linalg.norm(g0) * np.linalg.norm(G[i]), 1e-30)
                        for i in range(n)])
        step = np.array([np.nan] + [np.sum(np.abs(G[i] - G[i - 1]) ** 2) / max(np.sum(np.abs(G[i - 1]) ** 2), 1e-30)
                                    for i in range(1, n)])
        prof = np.sum(np.abs(G) ** 2, axis=2)  # [n, n_lag]
        node, fid, s = key
        tag = "%s_%d_s%d" % key
        out[tag + "_t"] = t
        out[tag + "_dw"] = dw
        out[tag + "_pw"] = pw
        out[tag + "_coh"] = coh
        out[tag + "_step"] = step
        out[tag + "_prof"] = prof
        summ.append((node, fid, s, n, float(np.nanmax(dw)), float(np.nanmedian(step)), float(np.nanpercentile(step, 90)),
                     float(coh[-1]), float(coh.min()), float(pw[-1] / pw[0])))
    # power left before each refit, and explained fractions, from the node logs
    pl = re.compile(r"^(cx\d+) (\d\d):(\d\d):(\d\d)Z before the refit of \S+: power left (.*)$")
    for node in NODES:
        path = os.path.join(a.logs, node + ".log")
        if not os.path.exists(path):
            continue
        ts, vals = [], {}
        for line in open(path, errors="replace"):
            m = pl.match(line.strip())
            if not m:
                continue
            t = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
            if not lo <= t <= hi:
                continue
            ts.append(t)
            for kv in m.group(5).split():
                f, x = kv.split(":")
                vals.setdefault(int(f), []).append((t, float(x)))
        for f, v in vals.items():
            out["%s_%d_left" % (node, f)] = np.array(v)
    np.savez(a.out, **out)
    print("node fid src fits  max|dw|  step_med step_p90  coh_last coh_min  pw_last/first")
    for r in summ:
        print("%s %d %d %4d  %.1e  %.3f    %.3f    %.3f    %.3f   %.2f" % r)
    print("saved", a.out)


if __name__ == "__main__":
    main()
