#!/usr/bin/env python3
"""dTEC drift decomposition: slopes of EVERY code/carrier combination, per joint arc.

THE QUESTION (buglist "dTEC absolute per-sat drift"): each sat's geometry-free carrier TEC
swings ~40-60 TECU over a transit while the GIM says the real ionosphere is flat ~12 TECU.
Which instrument term carries the swing?

THE METHOD (unblocked 2026-07-31 by the REC_CP export-currency fix): with the code observable
now valid, every clock/LO/iono decomposition is measurable. Per satellite, per joint L1xL2C
arc, fit the drift slope of each combination:

    carr_gf = carr_resid_a - carr_resid_b     LO_ab difference - iono_gf     (the dTEC input)
    code_gf = code_resid_a - code_resid_b     clk_ab difference + iono_gf    (absolute, noisy)
    D       = carr_gf - code_gf               (LO_ab - clk_ab) - 2*iono_gf   (code-leveled)
    CMC_a   = code_resid_a - carr_resid_a     (clk_a - LO_a) + 2*I_a + mp_a  (single-band)
    CMC_b   = code_resid_b - carr_resid_b     (clk_b - LO_b) + 2*I_b + mp_b

Clock/LO terms are COMMON across sats; iono/multipath/loop-wander are PER-SAT. So for each
combination the table reports the across-sat MEDIAN slope (the common mode: LO offsets, clock
drifts) and each sat's RESIDUAL slope after removing it (the science + the suspect). On a
GIM-flat day a per-sat residual D-slope >> ~5 TECU/hr equivalent is INSTRUMENTAL -- and its
correlation against the differential carrier-loop residual / sigma_phi tests the "loop wander
integrates into ADR" hypothesis directly.

code_resid_m is wrapped mod the code length (+-L/2, e.g. +-150 km at L1) -- unwrapped here
along each arc (2 s cadence makes the unwrap unambiguous by 5 orders of magnitude).

Usage (defaults = live L1 x L2C GPS):
    python3 dtec_code_level.py [--a ...l1.jsonl --b ...l2c.jsonl --sys G]
                               [--since-min 720] [--min-arc-s 600] [--smooth-s 300]
Reads only observables jsonl -- safe against a live soak.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gnss_tec import load, arcs, C, K


def interp(arc_rows, t, key):
    """Linear interp of row[key] onto epoch t inside arc_rows (time-sorted); None off-ends
    or when either bracket lacks the field."""
    if t < arc_rows[0]["t"] or t > arc_rows[-1]["t"]:
        return None
    lo, hi = 0, len(arc_rows) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if arc_rows[mid]["t"] <= t:
            lo = mid
        else:
            hi = mid
    a, b = arc_rows[lo], arc_rows[hi]
    va, vb = a.get(key), b.get(key)
    if va is None or vb is None:
        return None
    if b["t"] == a["t"]:
        return va
    w = (t - a["t"]) / (b["t"] - a["t"])
    return va * (1 - w) + vb * w


def unwrap(vals, period_m):
    """Unwrap a wrapped (+-period/2) series along time; None-tolerant (holds the offset
    across gaps -- arcs are already gap-limited upstream)."""
    out, off, prev = [], 0.0, None
    for v in vals:
        if v is None:
            out.append(None)
            continue
        if prev is not None:
            d = (v + off) - prev
            if d > period_m / 2:
                off -= period_m * round(d / period_m)
            elif d < -period_m / 2:
                off += period_m * round(-d / period_m) * -1.0
        out.append(v + off)
        prev = out[-1]
    return out


def linfit(ts, vs):
    """LS slope over (t, v) with None dropped; (slope, rms, n) or None."""
    pts = [(t, v) for t, v in zip(ts, vs) if v is not None]
    if len(pts) < 10:
        return None
    n = len(pts)
    mt = sum(p[0] for p in pts) / n
    mv = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mt) ** 2 for p in pts)
    if sxx <= 0:
        return None
    sl = sum((p[0] - mt) * (p[1] - mv) for p in pts) / sxx
    rms = math.sqrt(sum((p[1] - (mv + sl * (p[0] - mt))) ** 2 for p in pts) / n)
    return sl, rms, n


def pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy), sxy / sxx, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="/tmp/gpswipe/obs_gps_l1.jsonl")
    ap.add_argument("--b", default="/tmp/gps_l2c_gpu/obs_gps_l2c.jsonl")
    ap.add_argument("--sys", default="G")
    ap.add_argument("--since-min", type=float, default=720.0)
    ap.add_argument("--max-gap-s", type=float, default=10.0)
    ap.add_argument("--min-arc-s", type=float, default=600.0)
    ap.add_argument("--min-sig", type=float, default=20.0)
    ap.add_argument("--subtract-trim", action="store_true",
                    help="A/B THE OPEN QUESTION (docs/adr_trim_subtraction.md vs the code as "
                         "built): subtract trim_cycles from adr_cycles before forming carr_gf. "
                         "The doc (07-19) says the broker's ctrim integrates straight into ADR "
                         "and predicts arc drifts of exactly the observed magnitude; but the "
                         "combiner now accumulates dcmd - dres (commanded MINUS the measured "
                         "arg(A) residual, GnssCoherentCombiner.cpp:491), which absorbs a "
                         "CONVERGED trim on its own -- so subtracting could be a DOUBLE "
                         "correction. Run both ways: whichever shrinks the per-sat residual "
                         "D-slope is the right answer, and that settles it empirically.")
    ap.add_argument("--json-out", default=None, help="optional per-arc results jsonl")
    args = ap.parse_args()

    ra = load(args.a, args.since_min * 60, args.sys, args.min_sig)
    rb = load(args.b, args.since_min * 60, args.sys, args.min_sig)
    common = sorted(set(ra) & set(rb))
    if not common:
        sys.exit("no common satellites with ADR in the window")
    r0a, r0b = ra[common[0]][0], rb[common[0]][0]
    fa, fb = r0a["carrier_hz"], r0b["carrier_hz"]
    fac = K * (1.0 / fa**2 - 1.0 / fb**2)  # m of gf per TECU (signed)
    # wrap period per band: L chips * chip length (m)
    per_a = r0a["code_len"] * C / r0a["chip_rate_hz"]
    per_b = r0b["code_len"] * C / r0b["chip_rate_hz"]
    print(f"bands {fa/1e6:.2f} / {fb/1e6:.2f} MHz: {fac:+.4f} m/TECU; "
          f"code wrap {per_a/1e3:.0f} / {per_b/1e3:.0f} km; sats {common}")

    per_arc = []
    for prn in common:
        for aa in arcs(ra[prn], args.max_gap_s):
            for bb in arcs(rb[prn], args.max_gap_s):
                lo = max(aa[0]["t"], bb[0]["t"])
                hi = min(aa[-1]["t"], bb[-1]["t"])
                if hi - lo < args.min_arc_s:
                    continue
                rows_a = [r for r in aa if lo <= r["t"] <= hi]
                ts = [r["t"] for r in rows_a]
                # carr_resid_m = -adr*lam - range. Removing the trim's contribution means
                # ADDING BACK trim_cycles*lam with the same sign the ADR entered with.
                def carr_a_of(r):
                    v = r.get("carr_resid_m")
                    if v is None or not args.subtract_trim:
                        return v
                    tc = r.get("trim_cycles")
                    return v if tc is None else v + tc * (C / fa)
                # unwrap each band's code residual along the arc, then difference
                code_a = unwrap([r.get("code_resid_m") for r in rows_a], per_a)
                code_b = unwrap([interp(bb, t, "code_resid_m") for t in ts], per_b)
                carr_a = [carr_a_of(r) for r in rows_a]
                carr_b = [interp(bb, t, "carr_resid_m") for t in ts]
                if args.subtract_trim:
                    tb = [interp(bb, t, "trim_cycles") for t in ts]
                    carr_b = [None if (v is None or c is None) else v + c * (C / fb)
                              for v, c in zip(carr_b, tb)]
                series = {
                    "carr_gf": [None if (x is None or y is None) else x - y
                                for x, y in zip(carr_a, carr_b)],
                    "code_gf": [None if (x is None or y is None) else x - y
                                for x, y in zip(code_a, code_b)],
                    "cmc_a": [None if (x is None or y is None) else x - y
                              for x, y in zip(code_a, carr_a)],
                    "cmc_b": [None if (x is None or y is None) else x - y
                              for x, y in zip(code_b, carr_b)],
                }
                series["D"] = [None if (x is None or y is None) else x - y
                               for x, y in zip(series["carr_gf"], series["code_gf"])]
                fits = {k: linfit(ts, v) for k, v in series.items()}
                if fits["carr_gf"] is None or fits["D"] is None:
                    continue
                # drivers, per-arc means
                def mean_of(vals):
                    xs = [v for v in vals if v is not None
                          and not (isinstance(v, float) and math.isnan(v))]
                    return (sum(xs) / len(xs)) if xs else None
                lam_a, lam_b = C / fa, C / fb
                drv_resid = mean_of(
                    [None if (r.get("carrier_hz_resid") is None or hzb is None)
                     else lam_a * r["carrier_hz_resid"] - lam_b * hzb
                     for r, hzb in zip(rows_a,
                                       [interp(bb, t, "carrier_hz_resid") for t in ts])])
                drv_sphi = mean_of(
                    [None if (r.get("sigma_phi") is None or spb is None)
                     else r["sigma_phi"] + spb
                     for r, spb in zip(rows_a,
                                       [interp(bb, t, "sigma_phi") for t in ts])])
                mean_el = mean_of([r.get("el") for r in rows_a])
                per_arc.append({
                    "prn": prn, "t0": ts[0], "dur_s": ts[-1] - ts[0], "n": len(ts),
                    "el": mean_el,
                    "slopes_mps": {k: (f[0] if f else None) for k, f in fits.items()},
                    "rms_m": {k: (f[1] if f else None) for k, f in fits.items()},
                    "drv_resid_mps": drv_resid, "drv_sigma_phi": drv_sphi,
                })

    if not per_arc:
        sys.exit("no joint arcs long enough (need more soak time?)")

    keys = ["carr_gf", "code_gf", "D", "cmc_a", "cmc_b"]
    # Across-sat median slope per combination = the common mode (clock/LO); per-sat residual
    # slope after removing it = the science + the suspect.
    med = {}
    for k in keys:
        sl = sorted(a["slopes_mps"][k] for a in per_arc if a["slopes_mps"][k] is not None)
        med[k] = sl[len(sl) // 2] if sl else None

    print(f"\ncommon-mode (across-sat median) slopes, m/hr:")
    print("   " + "  ".join(f"{k}={med[k]*3600:+9.2f}" if med[k] is not None else f"{k}=None"
                            for k in keys))
    print(f"\nper-sat RESIDUAL slopes (median removed), m/hr "
          f"[D in appTECU/hr = D/(-2*fac)]:")
    print(f"{'PRN':>4} {'dur_min':>8} {'el':>5} " +
          "".join(f"{k:>11}" for k in keys) +
          f" {'appTECU/hr':>11} {'Drms_m':>7} {'resid_mm/s':>11} {'sphi':>6}")
    for a in sorted(per_arc, key=lambda x: -abs((x["slopes_mps"]["D"] or 0)
                                                - (med["D"] or 0))):
        cells = []
        for k in keys:
            s = a["slopes_mps"][k]
            cells.append(f"{(s - med[k])*3600:>11.2f}"
                         if (s is not None and med[k] is not None) else f"{'--':>11}")
        d_res = ((a["slopes_mps"]["D"] or 0) - (med["D"] or 0))
        tecu = d_res / (-2.0 * fac) * 3600.0
        print(f"{a['prn']:>4} {a['dur_s']/60:>8.1f} "
              f"{(a['el'] or 0):>5.1f} " + "".join(cells) +
              f" {tecu:>11.1f} {(a['rms_m']['D'] or 0):>7.2f} "
              f"{(a['drv_resid_mps'] or 0)*1000:>11.3f} {(a['drv_sigma_phi'] or 0):>6.3f}")

    # Attribution: per-sat residual D slope vs the differential loop residual / sigma_phi
    dres = [(a["slopes_mps"]["D"] - med["D"]) if (a["slopes_mps"]["D"] is not None
                                                  and med["D"] is not None) else None
            for a in per_arc]
    for label, drv in (("differential carrier residual (m/s)",
                        [a["drv_resid_mps"] for a in per_arc]),
                       ("sigma_phi sum",
                        [a["drv_sigma_phi"] for a in per_arc])):
        pr = pearson(drv, dres)
        if pr:
            r, slope, n = pr
            print(f"\nD residual slope vs {label}: r = {r:+.3f} "
                  f"(slope {slope:+.4g}, {n} arcs)")

    print("\nREADING: carr_gf common mode = inter-band LO ramp (the old ~34 m/s);"
          "\n  code_gf common mode = inter-band SAMPLE-CLOCK ramp (GPSDO-disciplined -> ~0?);"
          "\n  per-sat residual D slope = instrumental carrier drift + true differential iono"
          "\n  (GIM-flat day: >> ~5 appTECU/hr == the dTEC swing, now code-anchored per sat).")
    if args.json_out:
        with open(args.json_out, "w") as f:
            for a in per_arc:
                f.write(json.dumps(a) + "\n")
        print(f"wrote {args.json_out} ({len(per_arc)} arcs)")


if __name__ == "__main__":
    main()
