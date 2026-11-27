#!/usr/bin/env python3
"""Measure an F-engine start's pol_step from cudaDtvSubtract saves made either side of it.

    dtv_polstep.py FRAME0_INT [--before DIR] [--after DIR] [--until UTC]

The step table (dtv_fold.STEPS) gives each input's whole-sample step within its pol half; pol_step, the step of
inputs 64-127 as a whole against inputs 0-63, is DTV's to measure. Across the start, with w unchanged, the rolling
fit turns g: g_after / g_before = exp(+i 2 pi f n_i Ts) / c, c one constant per channel and source. With the inputs'
steps taken out, the two halves differ by 2 pi f h Ts. Each ATSC channel's first source (the station; static,
470-610 MHz) gives one such phase, and h is the integer step that fits them best (score: the mean cosine of the residuals; the aliases
sit near +-6 samples, where 6 steps come to about one turn). The later sources are left out: on the 11-26 start most of
their g did not turn with the inputs, and they pull h towards 0.

Pairs: per node directory and GPU, the newest save in BEFORE made before the start (FRAME0_INT is its unix time) and
the newest in AFTER made AFTER_S or more after it (and before --until); a source counts when its w is the same in both.
A GPU is left out when taking the steps out makes its inputs less coherent: its fit was frozen across the start, so g
did not turn. When no input moved this cannot be told, and a frozen GPU reads 0.

Prints h, its score and the runner-up, and exits 0 only when the fit is clear (MIN_SCORE, MIN_MARGIN). It does not
write the table: fengine_steps.py pol-step FRAME0_INT N.
"""
import argparse
import calendar
import glob
import os
import re
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnss")
)
import dtv_fold  # noqa: E402
import dtv_state  # noqa: E402

STATES = os.environ.get("DTV_STATES", "/home/kvand/gnss/builds/dtv-live/states-rev2")
AFTER_S = 180  # the rolling fit's slowest tau is 8 s; a save this long after the start has refitted g
F_LO, F_HI = 470e6, 610e6  # the ATSC channels
MAX_H = 16
MIN_SCORE = 0.9
MIN_MARGIN = 0.05
NI = dtv_fold.NI


def pick(d, g, lo, hi):
    """The newest state file of GPU g in d with lo <= mtime < hi, leaving out folded copies."""
    fs = [
        f
        for f in glob.glob(os.path.join(d, "*g%d_*.bin" % g))
        if not re.search(r"_f\d+\.bin$", f) and lo <= os.path.getmtime(f) < hi
    ]
    return max(fs, key=os.path.getmtime) if fs else None


def ratios(before, after):
    """[(freq Hz, per-input sum over lags of g_after conj(g_before), NaN where either is dark)] of the ATSC channels'
    first source, where its w is the same in both."""
    a, b = dtv_state.read_state(before)["recs"], dtv_state.read_state(after)["recs"]
    out = []
    for fid in sorted(set(a) & set(b)):
        f = dtv_fold.fid_hz(fid)
        if not F_LO <= f <= F_HI:
            continue
        wa, wb, ga, gb = a[fid]["w"][0], b[fid]["w"][0], a[fid]["g"][0], b[fid]["g"][0]
        if not wa.any() or ga.shape != gb.shape or not np.allclose(wa, wb):
            continue
        q = (gb * np.conj(ga)).sum(axis=0).astype(complex)
        m = np.abs(q)
        if not (m > 0).sum() > NI // 2:
            continue
        q[m < 0.1 * np.median(m[m > 0])] = np.nan
        out.append((f, q))
    return out


def fit(bands, n, ts):
    """(h, score per h in -MAX_H..MAX_H, residual phases at h, half coherences) from the bands and input steps n."""
    hs = np.arange(-MAX_H, MAX_H + 1)
    turn, d, coh = [], [], []
    for f, q in bands:
        t = 2 * np.pi * f * ts
        u = q * np.exp(-1j * n * t)
        ok = np.isfinite(u)
        lo, hi = ok.copy(), ok.copy()
        lo[NI // 2 :] = hi[: NI // 2] = False
        if lo.sum() < 4 or hi.sum() < 4:
            continue
        a, b = u[lo].sum(), u[hi].sum()
        turn.append(t)
        d.append(np.angle(b * np.conj(a)))
        coh.append((abs(a) / np.abs(u[lo]).sum(), abs(b) / np.abs(u[hi]).sum()))
    turn, d = np.array(turn), np.array(d)
    if not len(d):
        raise SystemExit("no ATSC channel with the same w on both sides")
    score = np.cos(d[None, :] - hs[:, None] * turn[None, :]).mean(axis=1)
    h = int(hs[np.argmax(score)])
    return (
        h,
        dict(zip(hs.tolist(), score)),
        np.angle(np.exp(1j * (d - h * turn))),
        np.array(coh),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("frame0", type=int)
    ap.add_argument("--before", default=STATES)
    ap.add_argument("--after", default=STATES)
    ap.add_argument(
        "--until", help="UTC, e.g. 2026-11-26T01:00Z: after-saves made before it"
    )
    a = ap.parse_args()
    until = (
        calendar.timegm(time.strptime(a.until.rstrip("Z"), "%Y-%m-%dT%H:%M"))
        if a.until
        else time.time()
    )
    doc = dtv_fold.load_steps()
    ts = doc.get("adc_sample_ns", 0.3125) * 1e-9
    row = [r for r in doc["rows"] if r["frame0_int"] == a.frame0]
    if not row or row[0].get("steps") is None:
        raise SystemExit(
            "the start %d has no measured steps in %s" % (a.frame0, dtv_fold.STEPS)
        )
    n = dtv_fold.steps_between(doc["rows"], row[0]["prev_frame0_int"], a.frame0, False)
    n0 = np.where(np.isfinite(n), 0.0, np.nan)
    moved = np.nansum(n != 0)
    print("table steps: %d inputs read, %d moved" % (np.isfinite(n).sum(), moved))
    nodes = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.after, "cx*")))
    bands = []
    for node in nodes:
        for g in (0, 1):
            p = pick(os.path.join(a.before, node), g, 0, a.frame0)
            q = pick(os.path.join(a.after, node), g, a.frame0 + AFTER_S, until)
            if not p or not q:
                continue
            r = ratios(p, q)
            if not r:
                print(
                    "%s g%d: no ATSC channel with the same w in %s and %s"
                    % (node, g, p, q)
                )
                continue
            # a GPU whose fit was frozen (a satellite pass) has not turned g for the moved inputs
            hk, sk, _, coh = fit(r, n, ts)
            c1, c0 = np.median(coh), np.median(fit(r, n0, ts)[3])
            use = c1 > c0 or not moved
            print(
                "%s g%d: %d ATSC channels, %s -> %s; half coherence %.3f with the steps, %.3f without; %+d (%.3f)%s"
                % (
                    node,
                    g,
                    len(r),
                    os.path.basename(p),
                    os.path.basename(q),
                    c1,
                    c0,
                    hk,
                    sk[hk],
                    "" if use else "; left out: g not refitted since the start",
                )
            )
            if use:
                bands += r
    if not bands:
        raise SystemExit("no pairs of saves either side of %d" % a.frame0)
    h, score, res, coh = fit(bands, n, ts)
    second = max((s, k) for k, s in score.items() if k != h)
    clear = score[h] >= MIN_SCORE and score[h] - second[0] >= MIN_MARGIN
    print(
        "pol_step %+d: score %.3f, next %+d at %.3f; residual %+.0f deg, rms %.0f deg; %d channels; "
        "coherence within the halves %.3f / %.3f (median); %s"
        % (
            h,
            score[h],
            second[1],
            second[0],
            np.degrees(np.angle(np.exp(1j * res).mean())),
            np.degrees(np.sqrt((res ** 2).mean())),
            len(res),
            np.median(coh[:, 0]),
            np.median(coh[:, 1]),
            "clear" if clear else "NOT clear",
        )
    )
    print(
        "scores: "
        + " ".join("%+d:%.2f" % (k, s) for k, s in score.items() if abs(k) <= 8)
    )
    sys.exit(0 if clear else 1)


if __name__ == "__main__":
    main()
