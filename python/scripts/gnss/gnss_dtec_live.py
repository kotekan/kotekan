#!/usr/bin/env python3
"""Live dTEC checker: per-pair gf arc scatter, raw vs trim-subtracted.

Purpose (Keith, 2026-07-19): keep a live eye on dTEC quality on the way to
triggered-VLBI use, and verify the trim-in-ADR subtraction (docs/
adr_trim_subtraction.md) is correcting things as the timestamped logs accumulate.

For each band pair: build per-sat gf(t) = lam_a*ADR_a - lam_b*ADR_b at EXACT common
epochs (both emit grids are integer stream-seconds -- no interpolation), split into
arcs, detrend each arc linearly, and report the residual scatter in TECU: raw, and
with each band's commanded-trim integral (from the broker-log trim journal,
gnss_trim_journal.py rows) subtracted. The subtraction SIGN per band is calibrated
empirically on the first run (whichever sign reduces total scatter) and printed --
the export convention ("phi enters NEGATED") makes the a-priori sign ambiguous;
v2 (the per-record trim_cycles export) will pin it.

Usage:
  gnss_dtec_live.py --pair G:L1xL2C --since HH:MM
  pairs: G:L1xL2C  G:L1xL5  E:E1xE5a  C:B1CxB2a
"""
import argparse
import json
import statistics
import sys
from datetime import datetime, date

PAIRS = {
    "G:L1xL2C": ("G", "/tmp/gpswipe/obs_gps_l1.jsonl", 1575.42e6,
                 "/tmp/gps_l2c_gpu/obs_gps_l2c.jsonl", 1227.60e6,
                 "/tmp/gps_l1_broker.log", "/tmp/gps_l2c_broker.log"),
    "G:L1xL5": ("G", "/tmp/gpswipe/obs_gps_l1.jsonl", 1575.42e6,
                "/tmp/gps_l5_gpu/obs_gps_l5.jsonl", 1176.45e6,
                "/tmp/gps_l1_broker.log", "/tmp/gps_l5_broker.log"),
    "E:E1xE5a": ("E", "/tmp/gpswipe/obs_gal_e1.jsonl", 1575.42e6,
                 "/tmp/gps_l5_gpu/obs_gal_e5a.jsonl", 1176.45e6,
                 "/tmp/gps_l1_broker_gal.log", "/tmp/gps_l5_broker_gal.log"),
    "C:B1CxB2a": ("C", "/tmp/gpswipe/obs_bds_b1c.jsonl", 1575.42e6,
                  "/tmp/gps_l5_gpu/obs_bds_b2a.jsonl", 1176.45e6,
                  "/tmp/gps_l1_broker_bds.log", "/tmp/gps_l5_broker_bds.log"),
}
C_L = 299792458.0
K_TEC = 40.308e16


def load_obs(path, t_cut, min_sig, t1=None):
    out = {}
    try:
        f = open(path)
    except Exception:
        return out
    for line in f:
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("t", 0)
        if t < t_cut or d.get("adr_cycles") is None:
            continue
        if t1 is not None and t > t1:
            continue
        if (d.get("sig") or 0) < min_sig:
            continue
        # Keep the TRUE (unrounded) capture time + this band's Doppler: the two bands'
        # airspys start independently, so their emit-epoch grids are offset by a sub-second
        # constant (measured 2026-07-20: L2C at frac .92 vs L1/L5 .00 -> ZERO exact common
        # epochs -> the flagship G:L1xL2C read "no arcs"). We resample band B onto band A's
        # true epochs Doppler-aware downstream, so exact alignment is no longer required.
        # v2 (slot-19 export): trim_cycles is the commanded-trim integral on the SAME arc.
        out.setdefault(d["prn"], []).append(
            (t, d["adr_cycles"], d.get("trim_cycles"), d.get("adr_arc"),
             d.get("doppler_hz") or 0.0))
    for v in out.values():
        v.sort()
    return out


def resample(pts, t, max_gap):
    """Doppler-aware ADR (+trim, arc) of a band at an arbitrary time t. pts = sorted
    (t, adr, trim, arc, dop). The carrier phase is dominated by the Doppler ramp, so we
    integrate the (linearly-varying) Doppler for the model phase and linearly interpolate
    ONLY the small model-removed residual: a sub-second inter-band epoch offset then costs
    ~jerk*dt^3 (sub-milliTECU) instead of the naive-linear Doppler*dt (tens of TECU -- an
    offset pair read 71 TECU under plain ADR interpolation). None outside a bracket or
    across a gap > max_gap. arc/trim are taken from the nearer endpoint (arc is an integer
    id -- never interpolate it)."""
    import bisect
    ts = [p[0] for p in pts]
    j = bisect.bisect_right(ts, t) - 1
    if j < 0 or j + 1 >= len(pts):
        return None
    t0, a0, _tr0, _ar0, d0 = pts[j]
    t1_, a1, _tr1, _ar1, d1 = pts[j + 1]
    span = t1_ - t0
    if span <= 0 or span > max_gap:
        return None
    dt = t - t0
    phi = d0 * dt + 0.5 * (d1 - d0) * dt * dt / span         # model carrier phase (cycles)
    resid_end = a1 - a0 - 0.5 * (d0 + d1) * span             # model-removed residual at t1
    adr = a0 + phi + resid_end * (dt / span)
    near = j if dt <= span / 2 else j + 1
    return adr, pts[near][2], pts[near][3]


def load_trims(log, t_cut):
    import re
    day0 = datetime.combine(date.today(), datetime.min.time()).timestamp()
    stamp = re.compile(r"^\[broker (\d\d):(\d\d):(\d\d\.\d+)\] (.*)$")
    car = re.compile(r"PRN (\d+) resid [+-][\d.]+ Hz trim ([+-][\d.]+)")
    series = {}
    try:
        f = open(log, errors="replace")
    except Exception:
        return series
    for line in f:
        m = stamp.match(line)
        if not m:
            continue
        t = day0 + int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        for pm in car.finditer(m.group(4)):
            series.setdefault(int(pm.group(1)), []).append((t, float(pm.group(2))))
    return series


def trim_cycles_at(series, prn, ts):
    """cumulative integral of trim_hz dt, evaluated at each t in ts (sorted)."""
    s = series.get(prn) or []
    out, tot, j, cur, tcur = [], 0.0, 0, None, None
    for t in ts:
        while j < len(s) and s[j][0] <= t:
            if cur is not None:
                tot += cur * (s[j][0] - tcur)
            cur, tcur = s[j][1], s[j][0]
            j += 1
        v = tot + (cur * (t - tcur) if cur is not None else 0.0)
        out.append(v)
    return out


def arc_scatter(ts, ys, max_gap, min_len, order=1):
    """split on gaps, detrend each arc (order 1 = linear, 2 = quadratic), return per-arc
    rms list. order=2 removes the arc's CURVATURE too -- a diagnostic: if the scatter
    collapses from order 1 to 2, the 'noise' was uncancelled 2nd-order (differential
    Doppler-rate / timing), not real high-frequency phase noise."""
    import numpy as np
    res = []
    i0 = 0
    for i in range(1, len(ts) + 1):
        if i == len(ts) or ts[i] - ts[i - 1] > max_gap:
            if ts[i - 1] - ts[i0] >= min_len:
                seg_t = np.array(ts[i0:i], float)
                seg_y = np.array(ys[i0:i], float)
                n = len(seg_t)
                if n > order:
                    x = seg_t - seg_t[0]
                    r = seg_y - np.polyval(np.polyfit(x, seg_y, order), x)
                    res.append((float(np.sqrt(np.mean(r * r))), n, float(seg_t[-1] - seg_t[0])))
            i0 = i
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--since", default=None)
    ap.add_argument("--t0", type=float, default=None,
                    help="absolute unix-epoch window start (overrides --since; forensics "
                         "on past eras, which the today-anchored --since cannot select)")
    ap.add_argument("--t1", type=float, default=None,
                    help="absolute unix-epoch window end (default: no upper bound)")
    ap.add_argument("--min-sig", type=float, default=15.0)
    ap.add_argument("--max-gap-s", type=float, default=10.0)
    ap.add_argument("--min-arc-s", type=float, default=60.0)
    ap.add_argument("--detrend-order", type=int, default=1,
                    help="per-arc detrend order: 1 = linear (default), 2 = quadratic. "
                         "Order 2 removes arc curvature -- if the scatter collapses 1->2, "
                         "the 'noise' was uncancelled 2nd-order (differential Doppler-rate / "
                         "resample-timing), not real phase noise.")
    ap.add_argument("--root", default=None,
                    help="read a ROTATED archive instead of live /tmp: replaces the '/tmp' "
                         "prefix of the obs/log paths with this dir (e.g. a "
                         "gnss_archive/run_<ts>/ that preserves the gpswipe/gps_l2c_gpu/"
                         "gps_l5_gpu subdirs). broker logs live under broker_logs/ there.")
    args = ap.parse_args()
    t_cut = 0.0
    if args.since:
        hh, mm = args.since.split(":")
        day0 = datetime.combine(date.today(), datetime.min.time()).timestamp()
        t_cut = day0 + int(hh) * 3600 + int(mm) * 60
    if args.t0 is not None:
        t_cut = args.t0

    for name in (args.pair or list(PAIRS)):
        if name not in PAIRS:
            print("unknown pair %s" % name)
            continue
        sysid, fa, freq_a, fb, freq_b, log_a, log_b = PAIRS[name]
        if args.root:  # redirect live /tmp paths into a rotated archive
            import os
            def _re(p):
                q = p.replace("/tmp", args.root, 1)
                if "broker" in os.path.basename(q):  # logs archived under broker_logs/
                    q = os.path.join(args.root, "broker_logs", os.path.basename(q))
                return q
            fa, fb, log_a, log_b = _re(fa), _re(fb), _re(log_a), _re(log_b)
        lam_a, lam_b = C_L / freq_a, C_L / freq_b
        fac = abs(K_TEC * (1.0 / freq_a ** 2 - 1.0 / freq_b ** 2))  # m per TECU
        oa, ob = (load_obs(fa, t_cut, args.min_sig, args.t1),
                  load_obs(fb, t_cut, args.min_sig, args.t1))
        ta_j, tb_j = load_trims(log_a, t_cut), load_trims(log_b, t_cut)
        # Resample band B onto band A's TRUE epochs (Doppler-aware), ONCE per pair.
        # rows: (t_a, adr_a, trim_a, arc_a, adr_b, trim_b, arc_b)
        paired = {}
        for prn in set(oa) & set(ob):
            rows = []
            for (t_a, adr_a, tr_a, ar_a, _d) in oa[prn]:
                rb = resample(ob[prn], t_a, args.max_gap_s)
                if rb is None:
                    continue
                rows.append((t_a, adr_a, tr_a, ar_a) + rb)
            if len(rows) >= 10:
                paired[prn] = rows

        results = {}
        for sign_a in (0, 1, -1):        # 0 = raw (no subtraction)
            tot = []
            for prn, rows in paired.items():
                # Split on COMBINER arc changes on either band (B1C's overlay re-anchors
                # reset adr/trim mid-time-arc; subtracting a resetting counter across the
                # boundary injects a step -- measured 2026-07-19: the C pair's subtraction
                # went 3.8 -> 75 TECU until this split). Then treat each piece separately.
                pieces, cur = [], []
                prev_arcs = None
                for r in rows:
                    arcs = (r[3], r[6])
                    if prev_arcs is not None and arcs != prev_arcs and cur:
                        pieces.append(cur)
                        cur = []
                    cur.append(r)
                    prev_arcs = arcs
                if cur:
                    pieces.append(cur)
                for piece in pieces:
                    if len(piece) < 10:
                        continue
                    common = [r[0] for r in piece]
                    ya = [r[1] for r in piece]
                    yb = [r[4] for r in piece]
                    if sign_a:
                        ea = [r[2] for r in piece]
                        eb = [r[5] for r in piece]
                        if all(v is not None for v in ea + eb):
                            ca, cb = ea, eb          # exact per-arc export (v2)
                        else:
                            ca = trim_cycles_at(ta_j, prn, common)
                            cb = trim_cycles_at(tb_j, prn, common)
                        # HIGH-PASS the trim (docs/adr_trim_subtraction.md): a STANDING
                        # trim slope is the loop absorbing real per-sat carrier content
                        # (young-chain model error, some iono rate) -- the raw gf's own
                        # detrend already handles it, and re-subtracting it injects
                        # independent inter-band noise (measured: the C pair went
                        # 12 -> 121 TECU under full subtraction). Only the trim's
                        # deviations from its per-piece line (the TRANSIENTS -- steps,
                        # re-pulls, walks) contaminate detrended arcs: subtract those.
                        def hp(vals):
                            n = len(vals)
                            mt = sum(common) / n
                            mv = sum(vals) / n
                            den = sum((t - mt) ** 2 for t in common) or 1.0
                            sl = sum((t - mt) * (v - mv)
                                     for t, v in zip(common, vals)) / den
                            return [v - (mv + sl * (t - mt))
                                    for t, v in zip(common, vals)]
                        ca, cb = hp(ca), hp(cb)
                        ya = [y - sign_a * c for y, c in zip(ya, ca)]
                        yb = [y - sign_a * c for y, c in zip(yb, cb)]
                    gf = [(lam_a * a - lam_b * b) / fac for a, b in zip(ya, yb)]  # TECU
                    tot += [r for r, n, s in arc_scatter(common, gf, args.max_gap_s,
                                                         args.min_arc_s, args.detrend_order)]
            results[sign_a] = tot
        def fmt(v):
            return ("%d arcs, scatter med %.2f p90 %.2f TECU"
                    % (len(v), statistics.median(v),
                       sorted(v)[int(len(v) * 0.9)] if len(v) > 1 else v[0])) \
                if v else "no arcs"
        print("%-10s raw: %s" % (name, fmt(results[0])))
        for s in (1, -1):
            print("%10s trim(sign %+d): %s" % ("", s, fmt(results[s])))


if __name__ == "__main__":
    main()
