#!/usr/bin/env python3
"""CODE-MINUS-CARRIER (CMC): the deterministic multipath metric, from data we already log.

    CMC = P - lambda*Phi   [metres]

Both observables track the SAME range, so geometry, satellite and receiver clocks, and the
troposphere cancel EXACTLY -- no model, no fit, nothing to get wrong. What survives is

    CMC = 2*I + M_code - M_carrier + noise + (an arc-constant ambiguity)

and since code multipath is METRES while carrier multipath is CENTIMETRES, and the ionosphere
is smooth on minutes, a HIGH-PASSED CMC is a clean, model-free measurement of code multipath:
the very thing the beam map has been averaging blind (2026-07-14: the same sky cell cost GPS
12.8 dB in the incoherent estimator and Galileo 12.6 dB in the coherent one -- one physical
reflector, two instrumental disguises).

Both inputs are already in the observables record (gnss_observables.py): cp_chips and
adr_cycles. This costs nothing but arithmetic.

CAVEATS THIS TOOL HANDLES (each one bit us at least once):
  * cp_chips is MOD the code period -> unwrap it.
  * cp_chips is a CURRENCY, not a physical length: it steps by t_abs*f_chip*ddop/f_c chips
    (~21 chips = 6 km!) whenever the broker re-anchors the seed. Those steps are detected and
    each becomes a new SEGMENT -- never smoothed across.
  * adr_cycles resets to 0 at every arc break (a new integer ambiguity), so an arc boundary is
    also a segment boundary.
  * The remaining arc-constant is removed per segment, which is exactly what makes CMC a
    RELATIVE (high-passed) multipath measure -- absolute code bias is not observable here.

Usage:
    python3 gnss_cmc.py                            # all three chains, summary + per-sat
    python3 gnss_cmc.py --obs /tmp/gpswipe/obs_gal_e1.jsonl --band GAL_E1C
"""
import argparse
import json
import math
import os
import sys

import numpy as np

C = 299792458.0
DEFAULT_OBS = {
    "G": ("/tmp/gpswipe/obs_gps_l1.jsonl", 1023.0),
    "E": ("/tmp/gpswipe/obs_gal_e1.jsonl", 4092.0),
    "C": ("/tmp/gpswipe/obs_bds_b1c.jsonl", 10230.0),
}


def load(path):
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("cp_chips") is None or r.get("adr_cycles") is None:
            continue
        rows.append(r)
    return rows


def cmc_series(rows, code_len, chip_rate=1.023e6, carrier_hz=1575.42e6,
               jump_chips=1.0, min_seg=20, hp_s=120.0, code_doppler_sign=1.0):
    """-> list of (prn, t[], cmc_hp[], el[], az[]) segments, CMC high-passed within each.

    A SEGMENT is a run with no arc break and no code-phase currency step: the only stretch
    over which CMC's ambiguity is genuinely constant."""
    lam = C / carrier_hz
    chip_m = C / chip_rate
    out = []
    by = {}
    for r in rows:
        by.setdefault(r["prn"], []).append(r)
    for prn, rs in by.items():
        rs.sort(key=lambda r: r["t"])
        t = np.array([r["t"] for r in rs])
        cp = np.array([r["cp_chips"] for r in rs], float)
        adr = np.array([r["adr_cycles"] for r in rs], float)
        arc = np.array([r.get("adr_arc") or 0 for r in rs])
        el = np.array([r.get("el") if r.get("el") is not None else np.nan for r in rs], float)
        az = np.array([r.get("az") if r.get("az") is not None else np.nan for r in rs], float)

        # ---- RECONSTRUCT THE PHYSICAL CODE PHASE. cp_chips is NOT a range: it is the seed
        # CURRENCY (cp0), back-referenced to absolute sample 0 through the Doppler, so its
        # geometry is already removed -- it drifts at the receiver clock offset (~0.07 chips/s)
        # while the carrier drifts at the range rate (~0.67). Differencing them as-is cancels
        # nothing and gives CMC in KILOMETRES (measured, first attempt). Undo the
        # back-reference, exactly as gnss_deadreckon_check does:
        #     cp_phys = cp0 + t_abs * f_chip * (1 + sign*dop/f_carrier)     (mod L)
        # The NOMINAL term (t_abs*f_chip) is invisible here: an emit interval is an exact
        # whole number of code periods, so it contributes exactly 0 mod L between samples --
        # it is a constant, and CMC removes constants anyway. What matters is the Doppler
        # term, which is what makes the code track the same range the carrier does.
        # t_abs needs only to be approximate: an error in the capture epoch enters as a SLOW
        # trend (it scales with the Doppler), and the high-pass below eats it.
        dop = np.array([r.get("doppler_hz") or 0.0 for r in rs], float)
        t_abs = t - t.min()
        cp_phys = cp + t_abs * chip_rate * code_doppler_sign * dop / carrier_hz
        d = np.diff(cp_phys)
        d -= code_len * np.round(d / code_len)      # unwrap across the code-period wrap
        cp_u = np.concatenate([[cp_phys[0]], cp_phys[0] + np.cumsum(d)])
        d_seed = np.diff(cp)                        # currency steps live in the SEED value
        d_seed -= code_len * np.round(d_seed / code_len)

        # segment boundaries: arc break, time gap, or a code-phase CURRENCY STEP
        brk = np.zeros(len(rs), bool)
        brk[1:] |= np.diff(arc) != 0
        brk[1:] |= np.diff(t) > 5.0
        brk[1:] |= np.abs(d_seed) > jump_chips   # broker re-anchor: a currency step, not range
        idx = np.where(brk)[0]
        segs = np.split(np.arange(len(rs)), idx)
        for sg in segs:
            if len(sg) < min_seg:
                continue
            code_m = cp_u[sg] * chip_m
            carr_m = adr[sg] * lam
            cmc = code_m - carr_m
            cmc -= np.mean(cmc)                 # the arc ambiguity: not observable
            # high-pass: subtract a slow (ionosphere + residual geometry) trend
            ts = t[sg] - t[sg][0]
            win = max(3, int(hp_s / max(np.median(np.diff(ts)), 1e-3)) | 1)
            if len(cmc) > win:
                k = np.ones(win) / win
                trend = np.convolve(cmc, k, mode="same")
                edge = win // 2
                trend[:edge] = trend[edge]
                trend[-edge:] = trend[-edge - 1]
                cmc = cmc - trend
            out.append((prn, t[sg], cmc, el[sg], az[sg]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", default=None, help="observables JSONL (default: all three)")
    ap.add_argument("--sys", default=None, choices=("G", "E", "C"))
    ap.add_argument("--code-len", type=float, default=None)
    ap.add_argument("--hp-s", type=float, default=120.0,
                    help="high-pass window (s): longer keeps more of the fringe, shorter "
                         "rejects more ionosphere. Multipath fringes at these elevations run "
                         "minutes, so 120 s keeps them and kills the iono trend.")
    ap.add_argument("--min-prn-c", type=int, default=19,
                    help="BeiDou capability floor (BDS-2 broadcasts no B1C)")
    args = ap.parse_args()

    chains = ([(args.sys, args.obs, args.code_len)] if args.obs
              else [(k, v[0], v[1]) for k, v in DEFAULT_OBS.items()])
    for tag, path, clen in chains:
        if not os.path.exists(path):
            continue
        rows = load(path)
        if tag == "C":
            rows = [r for r in rows if r["prn"] >= args.min_prn_c]
        if not rows:
            print("%s: no rows with both observables" % tag)
            continue
        segs = cmc_series(rows, clen, hp_s=args.hp_s)
        allc = np.concatenate([s[2] for s in segs]) if segs else np.array([])
        if not len(allc):
            print("%s: no usable segments" % tag)
            continue
        print("\n=== %s: %d rows, %d segments, CMC rms = %.2f m" % (
            tag, len(rows), len(segs), np.std(allc)))
        # CMC vs elevation -- multipath is elevation-structured, thermal noise is not
        el = np.concatenate([s[3] for s in segs])
        m = np.isfinite(el)
        print("   elevation   n      CMC rms (m)")
        for lo in range(10, 90, 10):
            k = m & (el >= lo) & (el < lo + 10)
            if k.sum() > 50:
                print("     %2d-%2d  %6d      %5.2f" % (lo, lo + 10, k.sum(), np.std(allc[k])))


if __name__ == "__main__":
    sys.exit(main())
