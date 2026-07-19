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


def load_obs(path, t_cut, min_sig):
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
        if d.get("t", 0) < t_cut or d.get("adr_cycles") is None:
            continue
        if (d.get("sig") or 0) < min_sig:
            continue
        # v2 (slot-19 export): the combiner ships the commanded-trim integral on the
        # SAME arc as the ADR -- subtract exactly, no journal timing error. Rows
        # without it (old brokers/logs) fall back to the journal path downstream.
        out.setdefault(d["prn"], {})[round(d["t"], 1)] = (
            d["adr_cycles"], d.get("trim_cycles"))
    return out


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


def arc_scatter(ts, ys, max_gap, min_len):
    """split on gaps, linear-detrend each arc, return per-arc rms list"""
    res = []
    i0 = 0
    for i in range(1, len(ts) + 1):
        if i == len(ts) or ts[i] - ts[i - 1] > max_gap:
            if ts[i - 1] - ts[i0] >= min_len:
                seg_t, seg_y = ts[i0:i], ys[i0:i]
                n = len(seg_t)
                mt, my = sum(seg_t) / n, sum(seg_y) / n
                den = sum((t - mt) ** 2 for t in seg_t) or 1.0
                sl = sum((t - mt) * (y - my) for t, y in zip(seg_t, seg_y)) / den
                r = [y - (my + sl * (t - mt)) for t, y in zip(seg_t, seg_y)]
                res.append(((sum(x * x for x in r) / n) ** 0.5, n, seg_t[-1] - seg_t[0]))
            i0 = i
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--since", default=None)
    ap.add_argument("--min-sig", type=float, default=15.0)
    ap.add_argument("--max-gap-s", type=float, default=10.0)
    ap.add_argument("--min-arc-s", type=float, default=60.0)
    args = ap.parse_args()
    t_cut = 0.0
    if args.since:
        hh, mm = args.since.split(":")
        day0 = datetime.combine(date.today(), datetime.min.time()).timestamp()
        t_cut = day0 + int(hh) * 3600 + int(mm) * 60

    for name in (args.pair or list(PAIRS)):
        if name not in PAIRS:
            print("unknown pair %s" % name)
            continue
        sysid, fa, freq_a, fb, freq_b, log_a, log_b = PAIRS[name]
        lam_a, lam_b = C_L / freq_a, C_L / freq_b
        fac = abs(K_TEC * (1.0 / freq_a ** 2 - 1.0 / freq_b ** 2))  # m per TECU
        oa, ob = load_obs(fa, t_cut, args.min_sig), load_obs(fb, t_cut, args.min_sig)
        ta_j, tb_j = load_trims(log_a, t_cut), load_trims(log_b, t_cut)
        results = {}
        for sign_a in (0, 1, -1):        # 0 = raw (no subtraction)
            tot = []
            for prn in set(oa) & set(ob):
                common = sorted(set(oa[prn]) & set(ob[prn]))
                if len(common) < 10:
                    continue
                ya = [oa[prn][t][0] for t in common]
                yb = [ob[prn][t][0] for t in common]
                if sign_a:
                    ea = [oa[prn][t][1] for t in common]
                    eb = [ob[prn][t][1] for t in common]
                    if all(v is not None for v in ea + eb):
                        ca, cb = ea, eb          # exact per-arc export (v2)
                    else:
                        ca = trim_cycles_at(ta_j, prn, common)
                        cb = trim_cycles_at(tb_j, prn, common)
                    ya = [y - sign_a * c for y, c in zip(ya, ca)]
                    yb = [y - sign_a * c for y, c in zip(yb, cb)]
                gf = [(lam_a * a - lam_b * b) / fac for a, b in zip(ya, yb)]  # TECU
                tot += [r for r, n, s in arc_scatter(common, gf, args.max_gap_s,
                                                     args.min_arc_s)]
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
