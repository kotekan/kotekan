#!/usr/bin/env python3
"""Geometry-free relative TEC from two bands' observables logs (arc-segmented carrier).

    python3 gnss_tec.py --a /tmp/gpswipe/obs_gps_l1.jsonl --b /tmp/gps_l5_gpu/obs_gps_l5.jsonl \
                        --out /tmp/tec_l1l5 [--since-min 600] [--sys G]

THE THREE TEC WORK ITEMS, AND WHICH THIS TOOL DOES. The carrier ADR is mm-precise but each lock
arc carries an unknown constant, so step 1 is ARC HANDLING (this tool): segment each band's
series at adr_arc changes (and data gaps), intersect the two bands' arcs per satellite, and form
the geometry-free combination L_gf = lam_a*ADR_a - lam_b*ADR_b on the joint arcs only. Within a
joint arc, VARIATION of L_gf is pure dispersive delay: slant TEC change at carrier precision.
Step 2, LEVELING (absolute TEC), shifts each arc by the arc-mean of (code_gf - carrier_gf) --
BLOCKED until the code observable is fixed (~40 km scatter today). Step 3, DCBs, removes the
constant inter-band hardware delays that survive leveling: satellite DCBs from gnss_dcb.py
(published), the receiver's estimated jointly with TEC (ours spans three dongles + the anchor
offset tau; GPSDO-constant).

THE INTER-BAND RECEIVER CLOCK (our three-dongle wrinkle): each band's ADR rides its own dongle's
clock solve, so L_gf carries a COMMON-MODE (same for every satellite) inter-band clock term --
measured ~0.1 ppm = the two dongles' l-a difference, i.e. kilometers/hour, dwarfing TEC. Two
honest views are produced:
  * per-epoch across-satellite MEDIAN of the arc-relative L_gf = the common mode itself
    (inter-band clock + median TEC drift), reported separately;
  * each satellite MINUS that median = differential slant TEC, the science view. (A
    single-difference vs one reference sat is the textbook form; the median is the same idea
    robust to any one sat's arc break.)

Emits <out>_tec.png (the differential view + the common mode) and <out>_arcs.json (per-arc
stats). Reads only the observables jsonl -- run it against a LIVE soak freely.
"""
import argparse
import json
import math
import os
import sys

C = 299792458.0
K = 40.308e16  # m^3/s^2/TECU: iono group delay = K*TEC/f^2


def load(path, since_s, sysid, min_sig):
    """Rows with a live ADR, newest `since_s` seconds, keyed [prn] -> time-sorted list."""
    rows = {}
    size = os.path.getsize(path)
    # Tail-seek: soak files reach GBs; 400 bytes/row * rate makes ~1 MB/min/sat. Generous slice.
    with open(path) as f:
        f.seek(max(0, size - 512 * 1024 * 1024))
        f.readline()
        tmax = 0.0
        raw = []
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("sys") != sysid or not d.get("adr_records"):
                continue
            # CERTIFIED-COHERENT LOCKS ONLY -- gate on coherence_s > 0, NOT sig. Two reasons,
            # both measured 2026-07-17:
            #  * COASTING sats export a COMMANDED phase + a noise-driven arg(A) random walk
            #    (km/hr); they carry coherence_s = 0 (the deep never cleared its floor), so this
            #    gate excludes them exactly as the old sig gate intended.
            #  * sig = max(deep_snr, amp_snr) includes the INCOHERENT amplitude, which NULLS as
            #    |2f-1| every time the code-period boundary drifts through mid-record (the BOC/
            #    overlay pilots: E1C/B1C/E5a/B2a/L5). At that null sig dips below any threshold
            #    for ~20% of each boundary cycle -- yet the ADR ARC is 100% CONTINUOUS through it
            #    (arc-breaks 0.0/100 vs boundary_f; carrier + segmented-deep sail through). The
            #    old sig gate DROPPED those rows, punching >10 s holes that arcs() then split on,
            #    fracturing hour-long arcs into 2 s stubs -- the entire "certification flicker"
            #    was this metric artifact, not a lock loss. coherence_s > 0 keeps them.
            if not ((d.get("coherence_s") or 0.0) > 0.0):
                continue
            raw.append(d)
            tmax = max(tmax, d["t"])
    for d in raw:
        if d["t"] >= tmax - since_s:
            rows.setdefault(d["prn"], []).append(d)
    for v in rows.values():
        v.sort(key=lambda r: r["t"])
    return rows


def arcs(series, max_gap_s):
    """Split one band's per-sat series into continuous arcs: same adr_arc, no long gaps."""
    out = []
    cur = []
    for r in series:
        if cur and (r["adr_arc"] != cur[-1]["adr_arc"] or r["t"] - cur[-1]["t"] > max_gap_s):
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def interp_adr(arc, t):
    """Linear ADR at time t within an arc (None outside / across a gap)."""
    import bisect
    ts = [r["t"] for r in arc]
    i = bisect.bisect_left(ts, t)
    if i == 0 or i >= len(ts):
        return None
    a, b = arc[i - 1], arc[i]
    if b["t"] - a["t"] > 3.0:
        return None
    w = (t - a["t"]) / (b["t"] - a["t"])
    return a["adr_cycles"] * (1 - w) + b["adr_cycles"] * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="band A observables jsonl (e.g. L1)")
    ap.add_argument("--b", help="band B observables jsonl (e.g. L5 or L2C)")
    ap.add_argument("--sys", default="G")
    ap.add_argument("--pair", action="append", default=[],
                    help="A_PATH:B_PATH:SYS, repeatable. POOLING IS THE POINT: E1/B1C/L1 and "
                         "E5a/B2a/L5 are the same frequency pairs on the same two dongles, so "
                         "the inter-band clock (the common mode, ~34 m/s) is IDENTICAL across "
                         "constellations -- pooling triples the sats constraining it and fills "
                         "the holes where any one constellation has <3 certified sats (measured "
                         "07-17: single-constellation coverage 19%%, and every hole leaks the "
                         "full clock rate = 264 TECU/s into every arc).")
    ap.add_argument("--since-min", type=float, default=720.0)
    ap.add_argument("--max-gap-s", type=float, default=10.0)
    ap.add_argument("--min-arc-s", type=float, default=120.0)
    ap.add_argument("--min-sig", type=float, default=20.0,
                    help="per-row lock-significance gate, BOTH bands (coasting sats export "
                         "model+noise ADR; measured 07-17: phase does NOT survive a >10 s "
                         "sub-sig fade -- rms jump 22 m at 10-20 s, km beyond -- so gated "
                         "gaps are REAL arc breaks)")
    ap.add_argument("--out", default="/tmp/tec")
    ap.add_argument("--dump-series", action="store_true",
                    help="also write <out>_series.npz: per-epoch differential TEC with sky "
                         "position (t, sys, prn, arc, tecu, az, el) -- the TEC-movie input")
    args = ap.parse_args()

    pairs = [tuple(p.split(":")) for p in args.pair]
    if args.a and args.b:
        pairs.append((args.a, args.b, args.sys))
    if not pairs:
        sys.exit("need --pair A:B:SYS (repeatable) or --a/--b/--sys")

    # Joint arcs: for every (arc_a x arc_b) overlap long enough, sample band B onto band A's
    # epochs. Everything downstream is per joint arc -- an arc break in EITHER band ends it.
    sat_arcs = {}  # (sys, prn) -> list of (ts[], gf_m[], el[])
    fa = fb = fac = None
    for path_a, path_b, sysid in pairs:
        ra = load(path_a, args.since_min * 60, sysid, args.min_sig)
        rb = load(path_b, args.since_min * 60, sysid, args.min_sig)
        common = sorted(set(ra) & set(rb))
        if not common:
            print(f"{sysid}: no common satellites with ADR in the window", file=sys.stderr)
            continue
        pfa = ra[common[0]][0]["carrier_hz"]
        pfb = rb[common[0]][0]["carrier_hz"]
        if fa is None:
            fa, fb = pfa, pfb
            fac = K * (1.0 / fa**2 - 1.0 / fb**2)  # m of L_gf per TECU (signed)
        elif (pfa, pfb) != (fa, fb):
            sys.exit(f"{sysid}: bands {pfa/1e6:.2f}/{pfb/1e6:.2f} != pooled "
                     f"{fa/1e6:.2f}/{fb/1e6:.2f} MHz -- pooling requires identical pairs")
        lam_a, lam_b = C / pfa, C / pfb
        print(f"{sysid}: bands {pfa/1e6:.2f} / {pfb/1e6:.2f} MHz: {fac:+.4f} m/TECU; "
              f"sats {common}")
        for prn in common:
            for aa in arcs(ra[prn], args.max_gap_s):
                for bb in arcs(rb[prn], args.max_gap_s):
                    lo = max(aa[0]["t"], bb[0]["t"])
                    hi = min(aa[-1]["t"], bb[-1]["t"])
                    if hi - lo < args.min_arc_s:
                        continue
                    ts, gf, el, geo = [], [], [], []
                    for r in aa:
                        if not (lo <= r["t"] <= hi):
                            continue
                        adr_b = interp_adr(bb, r["t"])
                        if adr_b is None:
                            continue
                        ts.append(r["t"])
                        gf.append(lam_a * r["adr_cycles"] - lam_b * adr_b)
                        el.append(r.get("el"))
                        geo.append((r.get("range_rate_mps"), r.get("range_m"),
                                    r.get("az")))
                    if len(ts) >= 10:
                        sat_arcs.setdefault((sysid, prn), []).append((ts, gf, el, geo))

    if not sat_arcs:
        sys.exit("no joint arcs long enough")

    # Common-mode removal IN RATE, then integrated: the inter-band clock accumulates ~34 m/s,
    # so LEVEL-based medians break the moment arcs start at different times (each sat's
    # arc-relative level has removed a different amount of clock -- the median then represents
    # nobody, and per-sat residuals grow at the full clock rate; observed as "millions of TECU"
    # on the first overnight). Rates are start-time-invariant: per 1 s bin, median across sats
    # of d(gf)/dt = the clock rate (+ median TEC rate, slow); integrate to a common-mode level.
    from collections import defaultdict
    rates = defaultdict(list)
    for prn, arcs_ in sat_arcs.items():
        for ts, gf, _, _ in arcs_:
            for i in range(1, len(ts)):
                dt = ts[i] - ts[i - 1]
                if 0.2 < dt < 3.0:
                    rates[round(ts[i])].append((gf[i] - gf[i - 1]) / dt)
    med_rate = {t: sorted(v)[len(v) // 2] for t, v in rates.items() if len(v) >= 3}
    if not med_rate:
        sys.exit("no epochs with >=3 simultaneous certified sats -- no common mode")

    # THE COMMON MODE MUST BE CONTINUOUS WHEREVER AN ARC IS. It accumulates at the inter-band
    # clock rate (~34 m/s = 264 TECU/s), so any hole the integration skips leaves a permanent
    # level error in every arc that straddles it (measured 07-17: 30 s holes -> the +-1e4 TECU
    # arc drifts). The RATE, unlike the level, is smooth (GPSDO-disciplined frac-N clocks), so
    # holes are bridged by smoothing/interpolating the per-second median rates (Gaussian
    # sigma 10 s, reach 60 s) and integrating over EVERY second. Seconds with no rate bin
    # within 60 s are declared dead: med_at -> None there, which drops those epochs from arcs
    # -- an honest hole, not a silent flat-line.
    # Also: evaluate AT THE EPOCH, never med[round(t)] -- the half-second grid error leaks
    # clock at each PRN's emit-cadence offset (the 07-16 "bimodal, negating" floor, implied
    # dt +-0.3 ms = the cadence offset, NOT a timestamp error; the logged t is us-exact).
    keys = sorted(med_rate)
    t_lo, t_hi = keys[0], keys[-1]
    med = {}
    dead = set()
    acc = 0.0
    ki = 0
    for t in range(t_lo, t_hi + 1):
        while ki < len(keys) - 1 and keys[ki + 1] <= t:
            ki += 1
        # nearby bins (within 60 s), Gaussian-weighted rate
        num = den = 0.0
        j = ki
        while j >= 0 and t - keys[j] <= 60:
            w = math.exp(-((t - keys[j]) / 10.0) ** 2 / 2.0)
            num += w * med_rate[keys[j]]
            den += w
            j -= 1
        j = ki + 1
        while j < len(keys) and keys[j] - t <= 60:
            w = math.exp(-((keys[j] - t) / 10.0) ** 2 / 2.0)
            num += w * med_rate[keys[j]]
            den += w
            j += 1
        if den > 0:
            acc += num / den
            med[t] = acc
        else:
            dead.add(t)
            med[t] = acc  # flat across a dead span; med_at refuses to serve it anyway
    print(f"common-mode: {len(keys)} rate bins over {t_hi - t_lo}s, "
          f"{len(dead)}s dead; sats with joint arcs: {sorted(sat_arcs)}")

    def med_at(t):
        k = math.floor(t)
        if k < t_lo or k + 1 > t_hi or k in dead or (k + 1) in dead:
            return None
        return med[k] + (med[k + 1] - med[k]) * (t - k)

    summary = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
    except ImportError:
        fig = None
    t00 = min(min(ts) for a in sat_arcs.values() for ts, _, _, _ in a)
    long_arcs = []
    series_rows = []
    for (sysid, prn), arcs_ in sorted(sat_arcs.items()):
        for k, (ts, gf, el, geo) in enumerate(arcs_):
            # split at dead common-mode spans: med flat-lined there, so the level on the far
            # side is wrong by the missed clock -- the far side is a NEW arc, honestly
            segs = [[]]
            prev_t = None
            for t, g, gg in zip(ts, gf, geo):
                m = med_at(t)
                if m is None:
                    continue
                if prev_t is not None and any(s in dead
                                              for s in range(math.floor(prev_t),
                                                             math.floor(t) + 1)):
                    segs.append([])
                segs[-1].append((t, g - m, gg))
                prev_t = t
            seg = max(segs, key=len)
            if len(seg) < 10:
                continue
            g0 = seg[0][1]
            rel = [(t, g - g0) for t, g, _ in seg]
            tecu = [r[1] / fac for r in rel]
            span = rel[-1][0] - rel[0][0]
            drift = tecu[-1] - tecu[0]
            mean = sum(tecu) / len(tecu)
            rms = math.sqrt(sum((x - mean) ** 2 for x in tecu) / len(tecu))
            summary.append({"sys": sysid, "prn": prn, "arc": k, "span_s": round(span, 1),
                            "n": len(rel), "dTEC_drift_TECU": round(drift, 3),
                            "scatter_TECU_rms": round(rms, 3),
                            "el_range": [el[0], el[-1]] if el[0] is not None else None})
            if span > 1500:
                long_arcs.append({"sys": sysid, "prn": prn, "arc": k,
                                  "t": [round(r[0], 4) for r in rel],
                                  "rel_m": [round(r[1], 5) for r in rel],
                                  "rr": [g[0] for _, _, g in seg],
                                  "rng": [g[1] for _, _, g in seg]})
            if args.dump_series:
                # az rides geo[2]; el needs realigning to the surviving seg epochs
                el_by_t = {t_: e_ for t_, e_ in zip(ts, el)}
                for (t_, g_, gg_), x in zip(seg, tecu):
                    az_ = gg_[2] if gg_[2] is not None else float("nan")
                    el_ = el_by_t.get(t_)
                    series_rows.append((t_, sysid, prn, k, x, az_,
                                        el_ if el_ is not None else float("nan")))
            if fig:
                ax1.plot([(r[0] - t00) / 60 for r in rel], tecu, ".", ms=2,
                         label=f"{sysid}{prn}" if k == 0 else None)
    if fig:
        mts = sorted(med)
        ax2.plot([(t - t00) / 60 for t in mts], [med[t] for t in mts], "k.", ms=1)
        ax1.set_ylabel("differential slant TEC (TECU, arc-relative)")
        ax1.legend(ncol=6, fontsize=8)
        ax1.set_title(f"geometry-free carrier {fa/1e6:.0f}/{fb/1e6:.0f} MHz -- "
                      f"arc-segmented, common-mode removed")
        ax2.set_ylabel("common mode (m)\n[inter-band clk]")
        ax2.set_xlabel(f"minutes since {t00:.0f}")
        fig.tight_layout()
        fig.savefig(args.out + "_tec.png", dpi=110)
        print("wrote", args.out + "_tec.png")
    json.dump(summary, open(args.out + "_arcs.json", "w"), indent=1)
    print("wrote", args.out + "_arcs.json")
    if long_arcs:
        json.dump(long_arcs, open(args.out + "_long.json", "w"))
        print("wrote", args.out + "_long.json", f"({len(long_arcs)} arcs >1500s)")
    if args.dump_series and series_rows:
        import numpy as _np
        t_ = _np.array([r[0] for r in series_rows])
        sys_ = _np.array([r[1] for r in series_rows])
        _np.savez_compressed(args.out + "_series.npz",
                             t=t_, sys=sys_,
                             prn=_np.array([r[2] for r in series_rows], dtype=_np.int16),
                             arc=_np.array([r[3] for r in series_rows], dtype=_np.int32),
                             tecu=_np.array([r[4] for r in series_rows]),
                             az=_np.array([r[5] for r in series_rows]),
                             el=_np.array([r[6] for r in series_rows]))
        print("wrote", args.out + "_series.npz", f"({len(series_rows)} epochs)")
    for s in summary:
        print(f"  {s['sys']}{s['prn']:2d} arc{s['arc']} {s['span_s']:7.0f}s n={s['n']:5d} "
              f"drift {s['dTEC_drift_TECU']:+7.3f} TECU  scatter {s['scatter_TECU_rms']:.3f} TECU")


if __name__ == "__main__":
    main()
