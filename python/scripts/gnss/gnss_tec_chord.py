#!/usr/bin/env python3
"""Geometry-free slant TEC on CHORD, paired at EXACT fleet-ADR grid hops.

    gnss_tec_chord.py --since-h 7 --out /tmp/tec_chord
    gnss_tec_chord.py --pair gal_e5a:gal_e6 --pair gps_l5:gps_l2c --since-h 7 --out /tmp/tec

Emits <out>_series.npz with EXACTLY the columns gnss_tec_movie.py reads
(t, sys, prn, arc, tecu, az, el), plus <out>_arcs.json and <out>_tec.png.

WHY THIS IS NOT `gnss_tec.py --chord`. That tool is the airspy prototype and almost none of
its machinery survives the move:
  * IT INTERPOLATES band B onto band A's row epochs. On CHORD that is fatal, not approximate:
    `adr_cycles` is one instance's accumulator stamped by record COUNT, so the epoch it
    implies is wrong by ~101 ms rms, and a full-carrier ADR turns that into c*dt = 3e7 m.
    Here the two bands are read at the SAME INTEGER HOP and nothing is interpolated.
  * IT REMOVES THE COMMON MODE IN RATE, because three dongles meant three clocks and an
    inter-band drift of ~264 TECU/s. CHORD has ONE F-engine clock for every band, so the
    inter-band term is a CONSTANT (a hardware delay) and dies with the arc mean. Removing it
    in rate here would subtract real ionosphere.
  * ITS ARC RULES bridge coherence_s dropouts. `fadr_arc` is already the certified-continuous
    arc: it ends only where an increment could not be accounted for.
So the prototype is left alone and this is the CHORD producer. Everything downstream
(gnss_tec_movie.py, the IONEX leveling) is shared.

⚠️ THE PAIRING KEY IS `fadr_g_hop`, NOT `fadr_hop`. Both are exact, but a chain's records land
on multiples of ITS OWN record length, so two chains' `fadr_hop` values are generally different
hops. `fadr_g_hop` is the fleet grid (fleetadr.GRID_HOPS = 96*2048 ~ 1.0066 s), a hop every
chain lands on -- the only epochs at which two bands pair with no interpolation at all.

⚠️ WHAT THIS IS AND IS NOT. Output is RELATIVE slant TEC: each joint arc has its own mean
removed, so it carries the arc's SHAPE, not an absolute. The two constants it cannot see are
the satellite DCB and the receiver's inter-band delay for that pair. Absolute TEC needs
leveling onto the code-gf arc mean plus those DCBs -- `gnss_tec_movie.py --ionex` levels a
whole frame against a CODE/IONEX map instead, which is the honest way to get a number.
"""
import argparse
import collections
import json
import math
import os
import sys

C = 299792458.0
K = 40.308e16            # m^3/s^2/TECU
GRID_HOPS = 96 * 2048    # fleetadr.GRID_HOPS

# Carrier centres, Hz -- these MUST match the SignalDef the broker runs (obs_up.sh's table).
FREQ = {"gps_l5": 1176.45e6, "gps_l2c": 1227.60e6,
        "gal_e5a": 1176.45e6, "gal_e5b": 1207.14e6, "gal_e6": 1278.75e6,
        "bds_b2a": 1176.45e6, "bds_b2b": 1207.14e6, "bds_b3i": 1268.52e6}
SYSOF = {"gps": "G", "gal": "E", "bds": "C"}
# Default pairs: the widest frequency lever each constellation has, so the m/TECU factor is
# large and the phase noise divides down. E5a x E6 = 0.0447 m/TECU, B2a x B3I 0.0407,
# L5 x L2C 0.0237. Pooling all three is what fills the sky for a movie; each pair carries its
# own receiver constant, which the per-arc mean removes.
DEFAULT_PAIRS = ["gal_e5a:gal_e6", "bds_b2a:bds_b3i", "gps_l5:gps_l2c"]


def load(path, since_s, cn0_min):
    """{prn: {g_hop: (g_dop_cycles, fadr_arc, t, az, el)}} for one band."""
    out = collections.defaultdict(dict)
    n = kept = 0
    with open(path) as fh:
        for line in fh:
            if len(line) < 50:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            n += 1
            if d["t"] < since_s:
                continue
            gh, gc = d.get("fadr_g_hop"), d.get("fadr_g_dop_cycles")
            if not gh or gc is None:
                continue
            # C/N0 gate on the COHERENT estimator: a dead PRN still exports a fleet ADR (it is
            # built from commanded Doppler plus a residual that is pure noise when nothing is
            # locked), and it looks like a 20 TECU/min "ionosphere". Measured 09-10: E13 at
            # -7 dB-Hz gave 24 TECU rms against 0.09 for the same arc length at 31 dB-Hz.
            cn = d.get("cn0_kcoh_dbhz")
            if cn is None or cn < cn0_min:
                continue
            out[d["prn"]][gh] = (gc, d.get("fadr_arc"), d["t"], d.get("az"), d.get("el"))
            kept += 1
    return out, n, kept


def joint_arcs(A, B, prn, max_gap_hops):
    """Runs of common grid hops over which BOTH bands' fadr_arc is unchanged."""
    hops = sorted(set(A[prn]) & set(B[prn]))
    segs, cur = [], []
    for h in hops:
        if cur:
            p = cur[-1]
            if (A[prn][h][1] != A[prn][p][1] or B[prn][h][1] != B[prn][p][1]
                    or h - p > max_gap_hops):
                segs.append(cur)
                cur = []
        cur.append(h)
    if cur:
        segs.append(cur)
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", default="/home/kvand/gnss/fixtures/obs")
    ap.add_argument("--day", default=None, help="YYYYMMDD (default: today UTC)")
    ap.add_argument("--pair", action="append", default=[],
                    help="A_BAND:B_BAND, repeatable (default: %s)" % ", ".join(DEFAULT_PAIRS))
    ap.add_argument("--since-h", type=float, default=6.0)
    ap.add_argument("--cn0-min", type=float, default=25.0,
                    help="coherent C/N0 gate, BOTH bands; a dead PRN's ADR is a noise walk "
                         "that reads as tens of TECU")
    ap.add_argument("--min-arc-s", type=float, default=300.0)
    ap.add_argument("--max-gap-s", type=float, default=30.0,
                    help="a hole longer than this ends a joint arc even if fadr_arc held")
    ap.add_argument("--out", default="/tmp/tec_chord")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    import time
    day = args.day or time.strftime("%Y%m%d", time.gmtime())
    since = time.time() - args.since_h * 3600.0
    pairs = [p.split(":") for p in (args.pair or DEFAULT_PAIRS)]
    max_gap_hops = int(args.max_gap_s * GRID_HOPS / 1.006632)

    bands = sorted({b for p in pairs for b in p})
    data = {}
    for b in bands:
        path = os.path.join(args.obs_dir, "%s_%s.jsonl" % (b, day))
        if not os.path.exists(path):
            sys.exit("no such observables file: %s" % path)
        data[b], n, kept = load(path, since, args.cn0_min)
        print("%-9s %8d rows read, %7d kept (C/N0 >= %.0f), %2d sats"
              % (b, n, kept, args.cn0_min, len(data[b])))

    rows, summary = [], []
    for a, b in pairs:
        fa, fb = FREQ[a], FREQ[b]
        la, lb = C / fa, C / fb
        mpt = K * (1.0 / fa ** 2 - 1.0 / fb ** 2)      # metres of gf per TECU
        sysid = SYSOF[a.split("_")[0]]
        A, B = data[a], data[b]
        nseg = 0
        for prn in sorted(set(A) & set(B)):
            for k, seg in enumerate(joint_arcs(A, B, prn, max_gap_hops)):
                span = A[prn][seg[-1]][2] - A[prn][seg[0]][2]
                if span < args.min_arc_s or len(seg) < 30:
                    continue
                # ⚠️ SIGN, DERIVED NOT GUESSED (this was wrong for one commit). The obs
                # writer builds carr_resid_m = -fadr_dop_cycles*lam - range_m and expects it
                # to sit near zero, so lam*adr = -rho. The ionosphere ADVANCES the carrier,
                # so the measured carrier range is rho - I with I = 40.308*TEC/f^2 > 0, i.e.
                #     lam*adr = -(rho - I) = -rho + I
                # and the geometry cancels in the difference, leaving
                #     la*adr_a - lb*adr_b = I_a - I_b = mpt*TEC     with mpt > 0 for fa < fb.
                # No minus. The check that this is right is that the fitted slope against the
                # obliquity is POSITIVE (it is the vertical TEC): see --check-obliquity.
                gf = [(la * A[prn][h][0] - lb * B[prn][h][0]) / mpt for h in seg]
                m = sum(gf) / len(gf)
                gf = [x - m for x in gf]               # RELATIVE: the arc constant is unknowable
                sd = math.sqrt(sum(x * x for x in gf) / len(gf))
                # scatter about a 31-point local mean = the measurement noise, not the arc shape
                res = []
                for i in range(len(gf)):
                    lo, hi = max(0, i - 15), min(len(gf), i + 16)
                    res.append(gf[i] - sum(gf[lo:hi]) / (hi - lo))
                noise = math.sqrt(sum(x * x for x in res) / len(res))
                for h, x in zip(seg, gf):
                    _, _, t_, az_, el_ = A[prn][h]
                    rows.append((t_, sysid, prn, nseg, x,
                                 float("nan") if az_ is None else az_,
                                 float("nan") if el_ is None else el_))
                summary.append({"pair": "%s x %s" % (a, b), "sys": sysid, "prn": prn,
                                "arc": nseg, "span_s": round(span, 1), "n": len(seg),
                                "m_per_TECU": round(mpt, 4),
                                "range_TECU": round(max(gf) - min(gf), 3),
                                "rms_TECU": round(sd, 3), "noise_TECU": round(noise, 4)})
                nseg += 1
        print("%-9s x %-9s %.4f m/TECU  %3d joint arc(s) >= %.0f s"
              % (a, b, mpt, nseg, args.min_arc_s))

    if not rows:
        sys.exit("no joint arcs -- try --since-h larger or --cn0-min lower")
    json.dump(summary, open(args.out + "_arcs.json", "w"), indent=1)
    import numpy as np
    rows.sort(key=lambda r: r[0])
    np.savez_compressed(
        args.out + "_series.npz",
        t=np.array([r[0] for r in rows]),
        sys=np.array([r[1] for r in rows]),
        prn=np.array([r[2] for r in rows], dtype=np.int16),
        arc=np.array([r[3] for r in rows], dtype=np.int32),
        tecu=np.array([r[4] for r in rows]),
        az=np.array([r[5] for r in rows]),
        el=np.array([r[6] for r in rows]))
    noises = sorted(s["noise_TECU"] for s in summary)
    print("\nwrote %s_series.npz  (%d epochs, %d arcs, %d sats)"
          % (args.out, len(rows), len(summary), len({(s["sys"], s["prn"]) for s in summary})))
    print("per-arc measurement noise (1 s): median %.3f TECU, best %.3f, worst %.3f"
          % (noises[len(noises) // 2], noises[0], noises[-1]))

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 6))
        t0 = rows[0][0]
        key = collections.defaultdict(list)
        for t_, s_, p_, k_, x_, _, _ in rows:
            key[(s_, p_, k_)].append(((t_ - t0) / 60.0, x_))
        for (s_, p_, _), v in sorted(key.items()):
            ax.plot([q[0] for q in v], [q[1] for q in v], ".", ms=1)
        ax.set_xlabel("minutes since %s UTC"
                      % time.strftime("%H:%M", time.gmtime(t0)))
        ax.set_ylabel("relative slant TEC (TECU, arc mean removed)")
        ax.set_title("CHORD geometry-free TEC -- fleet ADR paired at exact grid hops "
                     "(%d arcs, %d sats)"
                     % (len(summary), len({(s['sys'], s['prn']) for s in summary})))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out + "_tec.png", dpi=110)
        print("wrote", args.out + "_tec.png")


if __name__ == "__main__":
    main()
