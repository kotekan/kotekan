#!/usr/bin/env python3
"""What changes in the DTV stage's source-0 response over a night: per-input factors or the paths.

For each (node, freq_id): g_first and g_last (source 0, same weights). The per-input factor
a[i] = sum_lag conj(g_first[lag, i]) g_last[lag, i] / sum_lag |g_first[lag, i]|^2 is what an analog gain/phase
drift of input i would produce (the reference s = w.x scales by one common factor). Reports the share of the
change ||g_last - g_first||^2 that per-input factors explain, and how well the factors agree between channels
of one station, between the two stations, and between nodes (same 128 inputs everywhere).

    dtv_fit_drift.py --first 0257 --last 1259
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


def g0_at(node, hhmm_lo, hhmm_hi, last):
    """freq_id -> source-0 g from the first (or last) state file in [lo, hi] (HHMM) of each GPU."""
    out = {}
    for gpu in (0, 1):
        fs = []
        for p in glob.glob(os.path.join(STATES, node, "st_g%d_*.bin" % gpu)):
            m = re.search(r"st_g\d_(\d{4})(\d\d)?_([rs])0\.bin$", p)
            if m and hhmm_lo <= m.group(1) <= hhmm_hi:
                fs.append((m.group(1) + (m.group(2) or "00"), p))
        if not fs:
            continue
        p = sorted(fs)[-1 if last else 0][1]
        for fid, r in dtv_state.read_state(p)["recs"].items():
            out[int(fid)] = (r["g"][0].copy(), r["w"][0].copy())
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--first", default="0257")
    ap.add_argument("--last", default="1259")
    a = ap.parse_args()
    fac = {}  # (node, fid) -> (a[i], weight[i])
    print("node fid  change  per-input share   |a|-1 rms  arg(a) rms deg")
    for node in NODES:
        A = g0_at(node, a.first, a.first, last=False)
        B = g0_at(node, a.last[:2] + "00", a.last, last=True)
        for fid in sorted(set(A) & set(B)):
            (ga, wa), (gb, wb) = A[fid], B[fid]
            if not np.allclose(wa, wb):
                print("%s %d: weights differ, skipped" % (node, fid))
                continue
            p0 = np.sum(np.abs(ga) ** 2, axis=0)  # per input
            ai = np.sum(np.conj(ga) * gb, axis=0) / np.maximum(p0, 1e-30)
            ch = np.sum(np.abs(gb - ga) ** 2)
            resid = np.sum(np.abs(gb - ai[None, :] * ga) ** 2)
            keep = p0 > 0.05 * np.median(p0[p0 > 0])
            wgt = p0 * keep
            mag = np.sqrt(np.sum(wgt * (np.abs(ai) - 1) ** 2) / wgt.sum())
            ph = np.degrees(np.sqrt(np.sum(wgt * np.angle(ai) ** 2) / wgt.sum()))
            fac[(node, fid)] = (ai, wgt)
            print(
                "%s %d  %.3f   %.2f            %.3f      %.1f"
                % (node, fid, ch / np.sum(np.abs(ga) ** 2), 1 - resid / ch, mag, ph)
            )

    def agree(k1, k2):
        a1, w1 = fac[k1]
        a2, w2 = fac[k2]
        w = np.sqrt(w1 * w2)
        # remove each factor set's mean phase (the reference's common factor differs per channel)
        a1 = a1 * np.exp(-1j * np.angle(np.sum(w * a1)))
        a2 = a2 * np.exp(-1j * np.angle(np.sum(w * a2)))
        num = np.abs(np.sum(w * (a1 - 1) * np.conj(a2 - 1)))
        den = np.sqrt(np.sum(w * np.abs(a1 - 1) ** 2) * np.sum(w * np.abs(a2 - 1) ** 2))
        return num / max(den, 1e-30)

    keys = [k for k in fac if k[1] not in EDGE]
    st = lambda f: 24 if f < 2800 else 30  # noqa: E731
    same_node_same_st, same_node_cross, cross_node_same_st, cross_node_cross = (
        [],
        [],
        [],
        [],
    )
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            c = agree(k1, k2)
            if k1[0] == k2[0]:
                (
                    same_node_same_st if st(k1[1]) == st(k2[1]) else same_node_cross
                ).append(c)
            else:
                (
                    cross_node_same_st if st(k1[1]) == st(k2[1]) else cross_node_cross
                ).append(c)
    print(
        "\ncorrelation of the per-input drift (a - 1) between channels (median, p10-p90):"
    )
    for name, v in (
        ("same node, same station", same_node_same_st),
        ("same node, other station", same_node_cross),
        ("other node, same station", cross_node_same_st),
        ("other node, other station", cross_node_cross),
    ):
        v = np.array(v)
        print(
            "  %-26s n=%4d  %.2f  (%.2f-%.2f)"
            % (name, len(v), np.median(v), np.percentile(v, 10), np.percentile(v, 90))
        )


if __name__ == "__main__":
    main()
